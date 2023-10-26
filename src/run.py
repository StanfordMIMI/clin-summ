import copy
import os
from peft import PeftModel
import time
import torch
from tqdm import tqdm

import parser
import process
import summ_dataset
import train


def main():

    # parse arguments. set paths based on expmt params
    args = parser.get_parser()

    # load data
    dataset = summ_dataset.SummDataset(args, task='test')

    # filter out pre-generated samples for this experimental configuration
    for sample in copy.deepcopy(dataset.data):
        if sample['idx'] in dataset.idcs_pregen:
            summ_dataset.remove_sample(dataset.data, sample['idx'])

    # load model, tokenizer
    model, tokenizer = load_model_and_tokenizer_wrapper(args)

    list_out, list_idx = [], []
    t0 = time.time()

    if args.arch in ['gpt', 'gptq']:

        for sample in (tqdm(dataset.data)):

            input_tok = tokenizer(sample['prompt'], return_tensors='pt')
            input_tok = {k: v.to(args.device) for k, v in input_tok.items()}

            # assert max_length > inp_length b/c gpt prepends input to output
            inp_length = input_tok['input_ids'].shape[-1]
            max_length = dataset.max_new_toks + inp_length
            
            with torch.no_grad():
                out_tok = model.generate(input_ids=input_tok['input_ids'],
                                         attention_mask=input_tok['attention_mask'],
                                         do_sample=False,
                                         max_length=max_length,
                                         stopping_criteria=stopping_criteria,
                                         pad_token_id=tokenizer.pad_token_id)

            out_txt = tokenizer.batch_decode(out_tok.detach().cpu().numpy(),
                                             skip_special_tokens=True)[0]
            list_out.append(out_txt)
            list_idx.append(sample['idx'])

    elif 's2s' in args.arch:
    
        loader = process.get_loader(args, dataset.dataset_obj, tokenizer)

        for step, batch in enumerate(tqdm(loader)):
           
            list_idx.extend(batch['idx']) # idcs preserve order of input/output
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            with torch.no_grad():
                output = model.generate(input_ids=batch['input_ids'],
                                        attention_mask=batch['attention_mask'],
                                        do_sample=False,
                                        max_new_tokens=dataset.max_new_toks,
                                        pad_token_id=tokenizer.eos_token_id)
       
            list_out.extend(tokenizer.batch_decode(output,
                                                   skip_special_tokens=True))

    print('generated {} samples for {} expmt in {} sec'.format(\
          len(list_idx), args.expmt_name, time.time() - t0))

    # add output to dataset, save result.jsonl
    dataset.postprocess_append_output(args, list_idx, list_out)
    dataset.save_data(args, append_pregen=True)


def get_finetuned_model(model, args):
    ''' load model weights which were fine-tuned in-house '''

    if args.epoch_eval == None: # if not specified, get highest epoch in folder
        subdirs = [ii[0].split('/')[-1] for ii in os.walk(args.dir_models_tuned)]
        epochs_all = [int(ii) for ii in subdirs if ii.isdigit()]
        args.epoch_eval = max(epochs_all)

    dir_model_peft = os.path.join(args.dir_models_tuned, f'{args.epoch_eval}')
    print(f'evaluating model: {dir_model_peft}')
    model = PeftModel.from_pretrained(model, dir_model_peft)
    model.eval()

    return model


def load_model_and_tokenizer_wrapper(args):
    ''' wrapper for loading model and tokenizer '''

    model, tokenizer = train.load_model_and_tokenizer(args)
    tokenizer.eos_token_id = 1
    if args.case_id >= 100: # if not discrete prompting
        model = get_finetuned_model(model, args)
    if args.model != 'flan-ul2':
        model.to(args.device)

    return model, tokenizer


if __name__ == '__main__':
    main()
