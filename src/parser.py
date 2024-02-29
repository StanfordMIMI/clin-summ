import argparse
import os

import constants
from summ_dataset import n_tokens


def get_parser(purpose=None):
    ''' parse arguments '''

    # alternative purpose(s) for calling this parser function
    if purpose:
        assert purpose in ['openai']
    required = not purpose  # if not alt purpose, some args required

    parser = argparse.ArgumentParser()

    ############### main args ###############
    parser.add_argument("--model", required=True,
                        help="model name")
    parser.add_argument("--case_id", type=int, required=required, default=0,
                        help="case id number (int) per constants.py")
    parser.add_argument('--dataset', type=str,
                        help='dataset to use', required=required,
                        choices=constants.DATASETS_IMPLEMENTED)

    ############### misc args ###############
    parser.add_argument('--epoch_eval', type=int, default=None,
                        help='epoch at which to evaluate model')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='number of samples to evaluate')
    parser.add_argument('--is_demo', action='store_true',
                        help='allows one to bypass N_MIN_SAMPLES')
    parser.add_argument('--use_finetuned', action='store_true', default=True,
                        help='allows one to also use a non-finetuned model for case id 300')

    args = parser.parse_args()

    args = set_args(args, purpose=purpose)

    args.device = 'cuda:0'

    return args


def set_args(args, purpose=None):
    ''' set args based on parser, constants.py 
        written separately for modularity '''

    # define architecture (gpt or seq-2-seq)
    try:
        args.arch = next(key for key, value in constants.MODELS.items()
                         if args.model in value)
    except:
        raise NotImplementedError('model architecture not defined')
    args.max_len_abs = constants.MAX_LEN[args.arch]

    # define input data directory
    args.dir_data = get_dir(args, 'data')

    # define output directory for data + tuned models
    args.expmt_name = f'{args.model}/case{args.case_id}'
    args = set_args_dir_out(args)
    args = set_args_dir_model(args)

    # transport args from dict in constants for convenience
    # set prefix, suffix for constructing prompts
    args = transport_args(args)

    return args


def get_dir(args, dir_name):
    ''' get directory for either output data or tuned models '''

    assert dir_name in ['data', 'output', 'models_tuned']

    dir_ = os.path.join(constants.DIR_PROJECT, dir_name, args.dataset)

    if dir_name != 'data':  # if not input data, dir is for expmt output
        dir_ = os.path.join(dir_, args.expmt_name + '/')

    return dir_


def mkdir(dir_):
    ''' make directory if one doesn't already exist '''
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def set_args_dir_out(args):
    ''' create directory for output data 
        separate dir from output data for ood cases '''

    args.dir_out = get_dir(args, dir_name='output')

    # save output to epoch-specific sub-dir
    if args.epoch_eval is not None:
        args.dir_out = os.path.join(args.dir_out, str(args.epoch_eval) + '/')

    mkdir(args.dir_out)

    return args


def set_args_dir_model(args):
    ''' create directory for tuned models 
        separate dir from output data for ood cases '''

    if args.case_id >= 100:  # no tuned model for case_id < 100
        args.dir_models_tuned = get_dir(args, dir_name='models_tuned')
        mkdir(args.dir_models_tuned)

    return args


def transport_args(args):
    ''' transport args from constants.cases[case_id]
        to args for convenience '''

    case = constants.cases[args.case_id]

    # relevant for in-house expmts
    args.batch_size = case['batch_size']
    args.thresh_seq_crop = case['thresh_seq_crop']
    assert args.thresh_seq_crop >= 0 and args.thresh_seq_crop < 1
    args.thresh_out_toks = case['thresh_out_toks']
    args.n_icl = case['n_icl']
    args.use_instruction = case['use_instruction']

    args = set_prompt_component(args)

    # args relevant for fine-tuning
    if args.case_id >= 100 and args.case_id <= 399:
        args.max_trn_epochs = case['max_trn_epochs']
        args.n_trn_samples = case['n_trn_samples']
        args.n_val_samples = case['n_val_samples']
        args.grad_accum_steps = case['grad_accum_steps']
        args.lr_num_warmup_steps = case['lr_num_warmup_steps']

    # relevant for openai queries
    if args.case_id >= 400 and args.case_id <= 499:
        args.gpt_temp = case['gpt_temp']

    return args


def set_prompt_component(args):
    ''' define insert_suffix, insert_prefix, prefix_instruction
        given expmt configs '''

    # get dataset-specific prompt components
    args.summ_task = ('rrs' if args.dataset in constants.DATASETS_RRS
                      else args.dataset)
    task_dict = constants.PROMPT_COMPONENT[args.summ_task]
    args.prefix = task_dict['prefix']
    args.suffix = task_dict['suffix']

    # get task-specific instructions
    args.instruction = task_dict['instruction']
    if args.case_id >= 400:  # direct chatgpt re its expertise
        args.instruction = constants.CHATGPT_EXPERTISE + args.instruction
    if args.n_icl:
        args.instruction += constants.INSTRUCTION_ICL

    # set n_toks_buffer to account for system prompt (chatgpt) in filtering
    n_ = n_tokens(args.instruction)
    args.n_toks_buffer = n_ if args.case_id >= 400 else 0

    # overwrite default if manually specified in constants.cases
    case = constants.cases[args.case_id]
    args.prefix = case['prefix'] if 'prefix' in case else args.prefix
    args.suffix = case['suffix'] if 'suffix' in case else args.suffix
    args.instruction = (case['instruction'] if 'instruction'
                        in case else args.instruction)

    return args
