from datasets import Dataset
import itertools
import numpy as np
import os
import re
import sys
import tiktoken

import constants
import process


class SummDataset():
    ''' dataset class for summarization tasks '''

    def __init__(self, args, task='test', purpose='default'):
        ''' instantiate SummDataset class for one of three purposes
            default:        load+process inputs for training/generating
            load_inputs:    load inputs only, no processing
            load_result:    load experiment results only, no processing '''

        self.task = task
        self.load_data(args, purpose)

        # process data for training / generating
        if purpose == 'default':
            self.downsample_data(args)
            self.generate_prompt(args)
            self.remove_long_prompts(args)
            self.create_dataset_obj()
            self.set_max_new_toks(args)
            self.get_pregenerated_samples(args)
                
        self.list_idx = get_vals_by_key(self.data, key='idx')


    def load_data(self, args, purpose):
        ''' load data from jsonl file '''

        assert purpose in ['default', 'load_inputs',
                           'load_result', 'calc_max_new_toks']

        # get path to inputs
        if purpose in ['default', 'load_inputs', 'calc_max_new_toks']:
            task = 'trn' if purpose == 'calc_max_new_toks' else self.task 
            if task not in ['trn', 'val', 'test']:
                raise ValueError('must specify task to be either [trn, val, test]')
            task_dict = {'trn': 'train', 'val': 'validate', 'test': 'test'}
            path_data = os.path.join(args.dir_data, f'{task_dict[task]}.jsonl')

        # get path to results
        elif purpose == 'load_result':
            path_data = os.path.join(args.dir_out, constants.FN_RESULT)
            if not os.path.exists(path_data):
                raise FileNotFoundError(f'results not generated for {args.dir_out}')
            
        data = process.read_jsonl_to_list(path_data)
        data = sort_list_of_dicts(data)
       
        if purpose in ['default', 'load_inputs', 'load_result']:
            self.data = data

            # create class attributes for easy reference later
            if purpose == 'load_inputs':
                self.data = rm_incompletes(self.data, keys=constants.KEYS_INP)
                self.list_inputs = get_vals_by_key(self.data, key='inputs')
                self.list_target = get_vals_by_key(self.data, key='target')
            if purpose == 'load_result': # prompt, output already created
                self.data = rm_incompletes(self.data, keys=constants.KEYS_OUT)
                self.list_inputs = get_vals_by_key(self.data, key='inputs')
                self.list_target = get_vals_by_key(self.data, key='target')
                self.list_prompt = get_vals_by_key(self.data, key='prompt')
                self.list_output = get_vals_by_key(self.data, key='output')
                self.ratio_tok = {}
                for sample in self.data:
                    if 'ratio_tok' in sample:
                        self.ratio_tok[sample['idx']] = sample['ratio_tok']
        
        # don't update attributes
        elif purpose == 'calc_max_new_toks': 
            return data


    def downsample_data(self, args):
        ''' randomly select n_samples data points '''

        if self.task == 'trn':
            n_samps = args.n_trn_samples
        elif self.task == 'val':
            n_samps = args.n_val_samples
        else:
            n_samps = args.n_samples

        idcs_all = get_vals_by_key(lst=self.data, key='idx')
        if n_samps and n_samps < len(idcs_all):
            np.random.seed(0)
            idcs = np.random.choice(idcs_all, n_samps, replace=False)
            self.data = [ii for ii in self.data if ii['idx'] in idcs]


    def generate_prompt(self, args):
        ''' generate prompts from inputs '''

        list_inp = get_vals_by_key(lst=self.data, key='inputs')

        # create prompt
        list_idx, list_prompt, list_inp = [], [], []
        for sample in self.data:
            list_idx.append(sample['idx'])
            inp = sample['inputs'].replace('\n', '')
            list_inp.append(inp)

            # add example number if using examples
            ex_num = f' {args.n_icl+1}' if args.n_icl else ''

            # format input sample and leading word
            # e.g. finding {n_icl+1}: {finding} \n impression {n_icl+1}:
            prompt = f'{args.prefix}{ex_num}: {inp}\n'
            prompt += f'{args.suffix}{ex_num}:'
            list_prompt.append(prompt)

        # prepend in-context examples
        if args.n_icl > 0:
            list_prompt = process.prepend_icl_examples(args, list_prompt,
                                                       list_inp, self.task)

        # prepend prefix instruction; only relevant for in-house methods,
        # as chatgpt gets instruction via system prompt
        if args.use_instruction and args.case_id < 400:
            list_prompt = [f'{args.instruction}\n\n{pp}' for pp in list_prompt]

        # save to prompt
        for ii, idx in enumerate(list_idx):
            add_key_val_pair(self.data, idx=idx,
                             key='prompt', value=list_prompt[ii])


    def remove_long_prompts(self, args):
        ''' remove longest prompts according to a threshold '''

        if args.dataset in constants.DATASETS_W_MODALITIES:
            print('seqs not removed per modality, will affect distribution')

        ###### (1) filter out prompts above token threshold ######
        c_too_many_toks = 0
        list_idx = get_vals_by_key(lst=self.data, key='idx')
        list_prompt = get_vals_by_key(lst=self.data, key='prompt')
        n_toks = [n_tokens(pp) for pp in list_prompt]
        max_expmt = args.max_len_abs - args.n_toks_buffer
        for ii, idx in enumerate(list_idx):
            if n_toks[ii] > max_expmt:
                remove_sample(self.data, idx)
                c_too_many_toks += 1
        msg = f'removed {c_too_many_toks} sample(s) above token threshold'
        if c_too_many_toks == len(list_idx):
            raise RuntimeError(f'{msg}. none remaining.')
        
        ###### (2) filter out longest X% of prompts ######
        if args.thresh_seq_crop == 0:
            return
        list_idx = get_vals_by_key(lst=self.data, key='idx')
        list_prompt = get_vals_by_key(lst=self.data, key='prompt')

        # determine cutoff length based on threshold
        len_seqs = [len(pp) for pp in list_prompt]
        len_seqs.sort()
        num_seqs = len(list_prompt)
        len_cutoff = len_seqs[int((1 - args.thresh_seq_crop) * num_seqs)]
        # remove samples w prompts above threshold
        for idx, prompt in zip(list_idx, list_prompt):
            if len(prompt) > len_cutoff:
                remove_sample(self.data, idx)
        
        ###### (3) calc longest prompt after filtering ######
        list_prompt = get_vals_by_key(lst=self.data, key='prompt')
        if len(list_prompt) < constants.N_MIN_SAMPLES and not args.is_demo:
            msg_ = f'exiting experiment. require {constants.N_MIN_SAMPLES}'
            msg_ += f' samples, but have {len(list_prompt)} after filtering.'
            msg_ += ' consider increasing n_samples when calling script.'
            raise RuntimeError(msg_)
        n_toks = [n_tokens(pp) for pp in list_prompt]
        self.n_toks_longest = max(n_toks)

        msg += f' and removed longest {100*args.thresh_seq_crop}% of prompts.'
        msg += f' in total, processing {len(self.data)} samples,'
        print(f'{msg} w longest prompt of {self.n_toks_longest} tokens.')


    def create_dataset_obj(self):
        ''' create dataset dict object used for training/generating '''
            
        list_idx = get_vals_by_key(self.data, key='idx')
        list_prompt = get_vals_by_key(self.data, key='prompt')
        list_target = get_vals_by_key(self.data, key='target')

        data_list = [{'idx': idx,
                      'sentence': list_prompt[ii],
                      'text_label': list_target[ii]}
                    for ii, idx in enumerate(list_idx)]
        self.dataset_obj = Dataset.from_list(data_list)


    def set_max_new_toks(self, args):
        ''' determine max_new_tokens to generate
            by default, prohibits generating more tokens than
            the longest summary in our training set '''

        trn_dataset = self.load_data(args, purpose='calc_max_new_toks')
        list_tgt = get_vals_by_key(trn_dataset, key='target')
        n_toks = [n_tokens(tt) for tt in list_tgt]
        n_toks.sort()
        n_seqs = len(n_toks)

        if args.thresh_out_toks == 0: # no thresholding
            self.max_new_toks = n_toks[-1]
        else:
            self.max_new_toks = n_toks[int((1 - args.thresh_out_toks) * n_seqs)]


    def get_pregenerated_samples(self, args):
        ''' get sample indices whose output has already been generated
            hence enables us to avoid redundant generation

            new attr: idcs_pregen: list of sample idcs to remove from loaded dataset
                      data_pregen: list of dicts to stash for saving step '''

        if os.path.exists(os.path.join(args.dir_out, constants.FN_RESULT)):
            data_pregen = SummDataset(args, purpose='load_result')
            self.idcs_pregen = []

            for idx in data_pregen.list_idx:
                sample = extract_dict_by_val(data_pregen.data, 'idx', idx)

                if 'output' in sample and bool(sample['output']):
                    self.idcs_pregen.append(idx)
                else:
                    remove_sample(data_pregen.data, idx)

            self.data_pregen = data_pregen.data

        else:
            self.idcs_pregen, self.data_pregen = [], []


    def postprocess_append_output(self, args, list_idx, list_out,
                                  purpose=None):
        ''' given generated output and corresponding indices
            postprocess output, append to dataset '''

        for idx, out in zip(list_idx, list_out):

            sample = extract_dict_by_val(lst=self.data, key='idx', value=idx)

            # remove prompts from output for in-house gpt models
            if 'gpt' in args.arch: 
                out = out.replace(sample['prompt'], '')

            # postprocess output specific to model, dataset
            out = process.postprocess(args, out)

            add_key_val_pair(lst=self.data, idx=idx,
                             key='output', value=out)

            # append ratio_tok = n_tok(out) / n_tok(tgt)
            tgt = sample['target']
            try:
                r_tok = round(n_tokens(out) / n_tokens(tgt), 3)
            except ZeroDivisionError:
                r_tok = 0
            add_key_val_pair(self.data, idx=idx,
                             key='ratio_tok', value=r_tok)

        self.data = sort_list_of_dicts(self.data)


    def append_scores(self, idx, scores):
        ''' given metric scores (dict) and idx (int) of sample
            append scores to dataset sample '''

        add_key_val_pair(lst=self.data, idx=idx,
                         key='scores', value=scores)
        self.data = sort_list_of_dicts(self.data)


    def append_pregenerated_data(self):
        ''' given data_pregen, a list of dicts not operated upon b/c
                the samples have already been generated,
            append to newly generated data '''
   
        self.data += self.data_pregen
        self.data = sort_list_of_dicts(self.data)


    def save_data(self, args, fn_out=constants.FN_RESULT, append_pregen=False):
        ''' save data object to jsonl file '''
        
        if append_pregen: # include pre-generated data in file write
            self.append_pregenerated_data()
            
        path = os.path.join(args.dir_out, fn_out)
        process.write_list_to_jsonl(path, self.data)#, key=None)

        if 'result' in fn_out:
            print(f'results generated in {args.dir_out}')


###############################################################################
### helper functions below
###############################################################################

def n_tokens(string):
    ''' returns the number of tokens in a text string '''

    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    num_tokens = len(encoding.encode(string))

    return num_tokens


def extract_dict_by_val(lst, key, value):
    ''' given a list of dicts
        extract the dict containing a specified key/value pair
        assumed values are unique across list of dicts '''
    for d in lst:
        if d.get(key) == value:
            return d
    return None


def add_key_val_pair(lst, idx, key, value):
    ''' given a list of dicts and an idx value to identify dict 
        add a new key/value pair into that dict, save to list '''
    
    # extract dict according to idx
    d = extract_dict_by_val(lst, 'idx', idx)
    
    # remove dict from list so we can replace it
    lst.remove(d)
    
    # add key/value pair, append to lst
    d[key] = value
    lst.append(d)
    
    return lst    


def get_vals_by_key(lst, key):
    ''' given list of dicts
        extract all values pertaining to a particular key 
        return as list '''
    values = []
    for d in lst:
        if key in d:
            values.append(d[key])
    return values


def remove_sample(lst, idx):
    ''' remove sample containing a particular index '''
    
    d = extract_dict_by_val(lst, 'idx', idx)
    lst.remove(d)
    
    return lst


def sort_list_of_dicts(lst, key='idx'):
    ''' given list of dicts all containing the key "idx" 
        return a sorted list of dicts using that key '''
    return sorted(lst, key=lambda x: x['idx'])


def rm_incompletes(lst, keys):
    ''' given list of dicts, remove all dicts which don't contain every key '''
    return [d for d in lst if all(key in d for key in keys)]
