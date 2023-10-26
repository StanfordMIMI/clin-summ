''' script containing data processing functions 

    currently organized chronologically into five sections
        1) pre-processing
        2) post-processing
        3) analysis
        4) file i/o 
        5) misc peripheral functions '''

import csv
import itertools
import json
import os
import re
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import default_data_collator

import constants
import summ_dataset


###############################################################################
### SECTION 1: pre-processing tasks
### e.g. generating in-context examples, creating data loader
###############################################################################


def prepend_icl_examples(args, list_prompt, list_inputs, task):
    ''' prepend in-context learning examples
        first, must run gen_faiss_idx.py for training set

        inputs:
            list_inputs: list of vanilla inputs
            list_prompt: list of prompts (inputs + formatting) '''

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except:
        raise NotImplementedError('in-context learning packages not installed')

    pre, suf = args.prefix, args.suffix
    icl_list_prompt, trn_inputs, trn_target = [], [], []

    # select in-context examples from training set, then run over test set
    trn_inputs_path = os.path.join(args.dir_data, constants.FN_INP_ICL)
    trn_target_path = os.path.join(args.dir_data, constants.FN_TGT_ICL)
    trn_inputs = read_csv_to_list(trn_inputs_path)
    trn_target = read_csv_to_list(trn_target_path)
    
    # load faiss indices pre-computed by gen_faiss_idx.py
    model = SentenceTransformer(constants.ICL_SENTENCE_TRANSFORMER)
    path_idcs = os.path.join(args.dir_data, 'icl', constants.FN_IDCS_ICL)
    if not os.path.exists(path_idcs):
        msg = 'must create faiss indices for dataset via src/gen_faiss_idx()'
        raise NotImplementedError(msg)
    cpu_index = faiss.read_index(path_idcs)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

    for i, element in tqdm(enumerate(list_inputs), total=len(list_inputs),
                           disable=False):
        
        # since if training, the train example is included in the trained index
        k_query = args.n_icl + 1 if task=='trn' else args.n_icl
        x_query = model.encode(element, convert_to_tensor=True)
        x_query = x_query.unsqueeze(0).cpu()
        x_query = x_query.numpy()
        example_dists, example_idcs = gpu_index.search(x=x_query, k=k_query)
        example_idcs = example_idcs[0]  # unnest single query
        example_idcs = example_idcs[1:] if task=='trn' else example_idcs
        example_idcs = example_idcs.tolist()

        # concatenate {n_icl} examples w original element
        new_element = '' 

        # format in-context examples via best practices
        # see example 5, https://shorturl.at/kuGN6
        for idx_idx, idx_ex in enumerate(example_idcs):
            inp_ex = f'{pre} {idx_idx+1}: {trn_inputs[idx_ex]}'.replace('\n', '')
            tgt_ex = f'{suf} {idx_idx+1}: {trn_target[idx_ex]}'.replace('\n', '')
            new_element += f'{inp_ex}\n{tgt_ex}\n##\n'

        # append formatted input sample (i.e. prompt)
        new_element += list_prompt[i] 

        icl_list_prompt.append(new_element)

    return icl_list_prompt


def preprocess_function_s2s(examples, tokenizer, len_pad=None):
    ''' preporcess function for seq2seq models '''
    
    # tokenize examples['sentence'] (inputs) and examples['text_label'] (target)
    # see e.g. https://huggingface.co/docs/transformers/model_doc/t5
    model_inputs = tokenizer(examples['sentence'],
                             max_length=len_pad,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")
    model_labels = tokenizer(examples['text_label'],
                             max_length=len_pad,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    # extract model_labels['input_ids'], move to one dict
    # replacing padding token id's by -100 so it's ignored by loss
    model_labels = model_labels["input_ids"]
    model_labels[model_labels == tokenizer.pad_token_id] = -100 
    model_inputs["labels"] = model_labels

    return model_inputs


def preprocess_function_gpt(examples, tokenizer, len_pad=None):
    ''' preprocess function for gpt-style models
        manually implemented left padding due to hugging face error 
            i.e. tokenizer(inputs, padding='max_length',
                           truncation=True, max_length=args.max_len_abs)
            gave nan loss while training '''
    
    # tokenize inputs, targets
    inputs = examples['sentence']
    batch_size = len(inputs)
    model_inputs = tokenizer(inputs)
    labels = tokenizer(examples['text_label'])

    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i] # get inputs
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id] # get labels

        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids # append labels to input
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids

        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = ([tokenizer.pad_token_id] * 
                                       (len_pad - len(sample_input_ids)) + 
                                       sample_input_ids)
        model_inputs["attention_mask"][i] = ([0] * (len_pad - len(sample_input_ids)) +
                                            model_inputs["attention_mask"][i])
        labels["input_ids"][i] = ([-100] * (len_pad - len(sample_input_ids)) +
                                 label_input_ids)
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:len_pad])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:len_pad])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:len_pad])

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def get_loader(args, dataset, tokenizer):
    ''' given list of input and target texts, return dataloader '''
        
    if args.model in constants.MODELS_W_CUSTOM_TOKENIZER:
        preprocess_function = preprocess_function_gpt
    else:
        preprocess_function = preprocess_function_s2s

    # determine length of padding based on tokenized input
    # i.e. only pad to longest input sequence --> allows faster training
    tokenized_inp = tokenizer(dataset['sentence'])
    n_seqs = len(tokenized_inp['input_ids'])
    n_toks = [len(tokenized_inp['input_ids'][i]) for i in range(n_seqs)]
    len_pad = max(n_toks)

    # map: transform each element of dataset by applying a lambda function
    # i.e. a wrapper on the preprocess function
    processed_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer,
                                             len_pad=len_pad),
        batched=True,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=['sentence', 'text_label'],
        desc="Running tokenizer on dataset",
    )
   
    loader = DataLoader(processed_dataset,
                        collate_fn=default_data_collator,
                        batch_size=args.batch_size, 
                        shuffle=True, pin_memory=True,
                        drop_last=False)

    return loader


###############################################################################
### SECTION 2: post-processing tasks
### e.g. post-processing for different architectures / datasets
###############################################################################

def postprocess(args, str_):
    ''' postprocess generated output '''

    if args.arch in ['oai-35', 'oai-4']:
        # post-process these tasks only
        tasks_to_postprocess = ['rrs', 'pls']
        if args.summ_task not in tasks_to_postprocess:
            return str_

        # set task-specific postprocess function
        if args.summ_task == 'pls':
            postprocess_fn = postprocess_pls
        elif args.summ_task == 'rrs':
            postprocess_fn = postprocess_rrs

        str_ = postprocess_fn(str_)

    # old post-processing used for gpt-neo models
    else:
        # remove everything in brackets and underscores
        str_ = re.sub("[\(\[].*?[\)\]]", "", str_)
        str_ = str_.replace('_', '')

        # remove all duplicate adjacent words
        str_ = ' '.join([k for k, g in itertools.groupby(str_.split())])

    return str_


def postprocess_pls(string):
    ''' postprocess openai output of pls task 
        remove all "n. " b/c gpt outputs things in numbered lists
        replace w semi-colons to separate problems '''

    string = string.replace('\n', '')

    sub_str = ' ; '
    string = re.sub(r'\d+\. ', sub_str, string)
    if string.startswith(sub_str):
        string = string[len(sub_str):].lstrip()

    return string


def postprocess_rrs(string):
    ''' postprocess openai output of rrs task 
        remove sub-strings "Impression: " '''

    sub_str_list = ['Impression: ', 'impression: ']
    for sub_str in sub_str_list:
        if string.startswith(sub_str):
            string = string[len(sub_str):].lstrip()

    return string


###############################################################################
### SECTION 3: analyze
### i.e. analyzing results for paper figures, etc.
###############################################################################

def load_dataset(args, purpose):
    ''' given expmt configs, return dataset object '''
    
    # whether to load input data or experiment result
    assert purpose in ['load_inputs', 'load_result']

    subdir = 'data' if purpose == 'load_inputs' else 'output'
    path_base = os.path.join(constants.DIR_PROJECT, subdir)
    args.modality = ''
   
    if purpose == 'load_inputs':
        args.dir_data = os.path.join(path_base, args.dataset)
    elif purpose == 'load_result':
        args.dir_out = os.path.join(path_base, args.dataset,
                                    args.model, f'case{args.case_id}')

    dataset = summ_dataset.SummDataset(args, task='test', purpose=purpose)

    return dataset


###############################################################################
### SECTION 4: file i/o 
### e.g. to/from list of strings (csv) or list of dicts (jsonl)
###############################################################################

def write_list_to_csv(fn_csv, list_, csv_action='w'):
    ''' write each element of 1d list to csv 
        can also append to existing file w csv_action="a" '''

    with open(fn_csv, csv_action) as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(list_)

    return


def read_csv_to_list(fn_csv, csv_action='r'):
    ''' given full path to csv file 
        load each element to list '''

    with open(fn_csv, csv_action) as f:
        lines = f.readlines()

    return lines


def write_list_to_jsonl(filename, data_list, action='w'):
    ''' given list, write each entry to separate lines in .jsonl file '''

    with open(filename, action) as file:
        for entry in data_list:
            json_line = json.dumps(entry)
            file.write(json_line + '\n')

    return


def read_jsonl_to_list(file_path):
    ''' given full path to jsonl file, load each element to list of dicts '''

    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            lines.append(data)

    return lines


###############################################################################
### SECTION 5: misc 
### section for containing clutter of peripheral functions 
###############################################################################

def generate_csvs_for_icl(list_inputs, list_target):
    ''' one-off generations of csv's for in-context learning '''
    
    write_list_to_csv('trn_inputs.csv', list_inputs)
    write_list_to_csv('trn_target.csv', list_target)

    return


def get_subdir(args):
    ''' given args.dir_out, get relevant sub-dir '''

    items = [item for item in os.listdir(args.dir_out)]
    contains_csv = any(['.csv' in ss for ss in items])

    if not contains_csv: # if dir doesn't have csv's for metric calc
        if not args.epoch_eval: # default: use last epoch for eval
            items = [item for item in items if os.path.isdir(
                             os.path.join(args.dir_out, item))]
            args.epoch_eval = get_highest_int(items)

        args.dir_out = os.path.join(args.dir_out, args.epoch_eval + '/')

    return args


def get_highest_int(strings):
    ''' return highest int given list of strings '''
    highest_int = None

    for string in strings:
        try:
            value = int(string)
            if highest_int is None or value > highest_int:
                highest_int = value
        except ValueError:
            continue

    assert highest_int != None
    return str(highest_int)