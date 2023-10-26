from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import math
import os
import peft
import random
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
)

import constants
import parser
import process
from summ_dataset import SummDataset


def main():
   
    # set preliminaries, model, tokenizer
    args, writer = set_preliminaries()
    model, tokenizer = load_model_and_tokenizer(args)
    model = get_tunable_model(model, args)

    # load data
    trn_dataset = SummDataset(args, task='trn').dataset_obj
    trn_loader = process.get_loader(args, trn_dataset, tokenizer)
    val_dataset = SummDataset(args, task='trn').dataset_obj
    val_loader = process.get_loader(args, val_dataset, tokenizer)
    args.steps_per_epoch = len(trn_loader)
    print(f'{len(trn_dataset)} samples w batch size {args.batch_size}, ' \
          f'hence {args.steps_per_epoch} gradient steps per epoch')
  
    # define optimizer, lr scheduler
    num_trn_steps = len(trn_loader) * args.max_trn_epochs
    optimizer, lr_scheduler = define_optimizer(args, model, num_trn_steps)

    model.train()
    best_val_loss = math.inf
    patience = constants.PATIENCE # early stop if loss doesn't reach new min in consec epochs
    n_steps = 0 # track number of steps taken
    trn_losses = []
    print('begin training!')

    for epoch in range(args.max_trn_epochs):
        with tqdm(total=len(trn_loader)) as pbar: # progress bar
            for idx_b, batch in enumerate(trn_loader):
                n_steps += 1

                # forward pass 
                batch = prep_batch(args, batch)
                outputs = model(**batch)

                # compute loss, gradient step 
                loss = outputs.loss / args.grad_accum_steps
                loss.backward()

                # optimizer step/zero after grad_accum_steps steps
                if (n_steps % args.grad_accum_steps == 0) or (n_steps == len(trn_loader)):
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()

                detached_loss = loss.detach().float()
                trn_losses.append(detached_loss)
                writer.add_scalar('trn_loss', detached_loss, n_steps)
                writer.add_scalar('trn_perplexity', torch.exp(detached_loss), n_steps)
                pbar.update(1)

        # calculate validation loss
        with tqdm(total=len(val_loader)) as pbar: # progress bar
            val_losses = []
            for batch in val_loader:
                batch = prep_batch(args, batch)
                with torch.no_grad():
                    outputs_val = model(**batch) 
                    val_losses.append(outputs_val.loss.detach().float())
                pbar.update(1)

        trn_loss_epoch = sum(trn_losses) / len(trn_losses)
        val_loss_epoch = sum(val_losses) / len(val_losses)
        trn_perplexity_epoch = torch.exp(trn_loss_epoch)
        val_perplexity_epoch = torch.exp(val_loss_epoch)

        writer.add_scalar('lr', lr_scheduler.get_lr()[0], epoch)
        writer.add_scalar('trn_loss_epoch', trn_loss_epoch, epoch)
        writer.add_scalar('val_loss_epoch', val_loss_epoch, epoch)
        writer.add_scalar('trn_perplexity_epoch', trn_perplexity_epoch, epoch)
        writer.add_scalar('val_perplexity_epoch', val_perplexity_epoch, epoch)
        
        print(f"epoch: {epoch}/{args.max_trn_epochs}, "
              f"trn_loss_epoch: {trn_loss_epoch}, "
              f"trn_perplexity_epoch: {trn_perplexity_epoch}, "
              f"val_loss_epoch: {val_loss_epoch}, "
              f"val_perplexity_epoch: {val_perplexity_epoch}, "
              f"lr: {lr_scheduler.get_lr()[0]}")

        # save model at each epoch
        model_save_dir = os.path.join(args.dir_models_tuned, f'{epoch}')
        model.save_pretrained(model_save_dir)

        # early stopping
        if val_loss_epoch > best_val_loss:
            if patience == 0:
                print(f'stopping early at epoch {epoch}!')
                break
            else:
                patience -= 1
        else:
            patience = constants.PATIENCE
            best_val_loss = val_loss_epoch


def set_preliminaries():
    ''' parse args, set paths, create dirs, basic checks '''

    # parse arguments, set paths based on expmt params
    args = parser.get_parser()

    # basic checks
    assert args.case_id >= 100 # case_id's 0-99 reserved for discrete prompting 
    assert constants.cases[args.case_id]['method'] in constants.METHODS

    # create tb dir
    dir_tb_log = os.path.join(args.dir_models_tuned, 'logs')
    if not os.path.exists(dir_tb_log):
        os.makedirs(dir_tb_log)
    
    # init tb writer. via cl: tensorboard --logdir=args.dir_out --port=8888
    writer = SummaryWriter(dir_tb_log) 

    return args, writer


def define_optimizer(args, model, num_trn_steps):
    ''' given parameters
        define optimizer '''
    
    # extract learning rate params
    case = constants.cases[args.case_id]
    lr0 = case['lr0'] # initial learning rate

    # define optimizer, lr_scheduler
    optimizer = transformers.AdamW(model.parameters(), lr=lr0,
                                   no_deprecation_warning=True)

    if case['lr_schedule'] == 'polynomial_decay':

        lrn = case['lrn'] # final learning rate
        lr_decay_power = case['lr_decay_power'] # rate of polynomial decay
        str_ = f'using polynomial decay scheduler with lr0 {lr0}, '
        str_ += f'lrn {lrn}, power {lr_decay_power},'

        lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            lr_end=lrn, 
            power=lr_decay_power,
            num_warmup_steps=args.lr_num_warmup_steps,
            num_training_steps=num_trn_steps,
        )
    
    elif case['lr_schedule'] == 'linear_decay':

        str_ = f'using linear scheduler with lr0 {lr0},'
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_num_warmup_steps,
            num_training_steps=num_trn_steps,
        )

    elif case['lr_schedule'] == 'constant':
    
        str_ = f'using constant learning rate {lr0},'
        lr_scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_num_warmup_steps,
        )

    else:
        raise NotImplementedError('learning rate method not implemented')
   
    str_ += f' and {args.lr_num_warmup_steps} warm-up steps!' 
    print(str_)

    return optimizer, lr_scheduler


def load_model_and_tokenizer(args):
    ''' load model and tokenizer '''

    # set quantization configs if using qlora
    try: 
        if constants.cases[args.case_id]['method'] == 'qlora':
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            quantization_config = None
    except:
        quantization_config = None

    # read model path, basename from constants.py 
    if 'gptq' in args.arch:
        model_path = constants.MODELS[args.arch][args.model]['path']
        model_basename = constants.MODELS[args.arch][args.model]['basename']
    else:
        model_path = constants.MODELS[args.arch][args.model]

    # define model and tokenizer
    if args.arch == 'gpt': # gpt-style models
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  padding_side='left')
        
        if tokenizer.pad_token_id is None: # gpt-2 pad token not set by default
            tokenizer.pad_token_id = tokenizer.eos_token_id 

    elif 's2s' in args.arch: # encoder-decoder (seq-2-seq) architectures

        if args.model == 'flan-ul2':
            model = T5ForConditionalGeneration.from_pretrained(model_path,
                                        device_map="auto", load_in_8bit=True)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path,
                                                          quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    elif 'gptq' in args.arch: # gpt-style models w quant

        try:
           quantize_config = BaseQuantizeConfig.from_pretrained(model_path)
        except:
            quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=True)

        model = AutoGPTQForCausalLM.from_quantized(model_path,
                                                   model_basename=model_basename,
                                                   use_safetensors=True,
                                                   trust_remote_code=False,
                                                   use_triton=False,
                                                   quantize_config=quantize_config)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    else:
        raise NotImplementedError('model not defined')

    return model, tokenizer


def get_tunable_model(model, args):
    ''' prep model for param-efficient fine-tuning '''

    if 'gpt' in args.arch: # gpt-style models
        task_type = peft.TaskType.CAUSAL_LM
    elif 's2s' in args.arch: # encoder-decoder (seq-2-seq) architectures
        task_type = peft.TaskType.SEQ_2_SEQ_LM 
    
    if constants.cases[args.case_id]['method'] == 'qlora':

        # prepare for k-bit training
        model = peft.prepare_model_for_kbit_training(model) 
        
        # get peft configs based on architecture (task_type) and fine-tuning method
        config = peft.LoraConfig(task_type=task_type, inference_mode=False,
                                 r=constants.LORA_R, lora_alpha=constants.LORA_ALPHA,
                                 lora_dropout=constants.LORA_DROPOUT)

    # wrap model w peft configs
    model = peft.get_peft_model(model, config).to(args.device)
    model.print_trainable_parameters()

    return model


def prep_batch(args, batch):
    ''' remove irrelevant dict keys needed for training
        move to device '''
    
    for key in list(batch.keys()):
        if key not in constants.KEYS_TRN:
            batch.pop(key)
    batch = {k: v.to(args.device) for k, v in batch.items()}

    return batch


if __name__ == '__main__':
    main()