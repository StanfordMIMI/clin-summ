import copy
import datetime
import os
import sys
import time
import timeout_decorator
from timeout_decorator import timeout

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, '../src/')
import constants
import parser
import process
from summ_dataset import (
    add_key_val_pair,
    n_tokens,
    remove_sample,
    SummDataset,
)

LEN_TIMEOUT = 30 # seconds for api timeout
    

def main():
    ''' calls openai via microsoft azure '''

    args, dataset = load_data_()

    # set delay timer (in seconds) based on rate limit
    tok_per_sec = constants.TOK_PER_MIN[args.oai_model] / 60
    tok_per_sample = dataset.n_toks_longest
    time_per_sample = round((tok_per_sample / tok_per_sec), 2)
    print(f'set delay of {time_per_sample} seconds per sample')

    # files across all samples
    fn_log = os.path.join(args.dir_out, 'log.txt')
    fn_result = os.path.join(args.dir_out, 'result.jsonl')
    fn_naughty = os.path.join(args.dir_out, 'naughty.csv')
    naughty_lst = get_naughty_list(fn_naughty)

    # directories for individual samples
    dir_indiv = os.path.join(args.dir_out, 'indiv')
    dir_inp = os.path.join(dir_indiv, 'inp')
    dir_inp_proc = os.path.join(dir_indiv, 'inp_proc')
    dir_out_ = os.path.join(dir_indiv, 'out')
    dir_result = os.path.join(dir_indiv, 'results')
    for dir_ in [dir_inp, dir_inp_proc, dir_out_, dir_result]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    # filter out pre-generated samples, add system prompt
    idcs_pregen = get_completed_idcs(dir_result)
    for sample in copy.deepcopy(dataset.data):
        if sample['idx'] in idcs_pregen:
            remove_sample(dataset.data, sample['idx'])
        else:    
            add_key_val_pair(dataset.data, idx=sample['idx'],
                             key='system_prompt', value=args.instruction)
    if len(idcs_pregen):
        msg = f'filtered out {len(idcs_pregen)} pre-generated samples. '
        msg += f'querying openai model for {len(dataset.data)} remaining samples'
        print(msg)

    for sample in dataset.data:
       
        t0 = time.time()
        idx = sample['idx']
        tgt = sample['target']
        fn_ = f'{idx}.jsonl'

        if idx in naughty_lst: # sample indices previously rejected by openai
            continue

        # get individual filenames
        fn_inp_ = get_path(dir_inp, fn_, rm=True)
        fn_inp_proc_ = get_path(dir_inp_proc, fn_, rm=True)
        fn_out = get_path(dir_out_, fn_, rm=True)
        fn_result_ = os.path.join(dir_result, fn_)

        sample = [sample] # convert to list so we can use built-ins
        process.write_list_to_jsonl(fn_inp_, sample, key=None)

        # pre-process inputs for azure
        cmd = f'python preprocess.py {fn_inp_} {fn_inp_proc_}'
        cmd += f' --system_prompt "{args.instruction}"'
        cmd += f' --temperature {args.gpt_temp}'
        os.system(cmd)

        # query openai api
        try:
            t0 = time.time()
            out = call_api_wrapper(args, fn_inp_proc_, fn_out)
            t1 = round(time.time() - t0, 1)
            now = str(datetime.datetime.now()).split('.')[0]
            if not out: # if openai gave error
                msg = f'sample {idx} exited w error in {t1} seconds, {now}'
                log_progress(fn_log, msg)
                continue
            msg = f'sample {idx} processed in {t1} seconds, {now}'
            log_progress(fn_log, msg)

        except timeout_decorator.timeout_decorator.TimeoutError:
            now = str(datetime.datetime.now()).split('.')[0]
            msg = f'sample {idx} timed out given {LEN_TIMEOUT} seconds, {now}'
            log_progress(fn_log, msg)
            with open(fn_naughty, 'a') as f:
                f.write(f'{idx}\n')
            continue
       
        # postprocess output
        out = process.postprocess(args, out)
        add_key_val_pair(lst=sample, idx=idx, key='output', value=out)
        r_tok = round(n_tokens(out) / n_tokens(tgt), 3)
        add_key_val_pair(sample, idx=idx, key='ratio_tok', value=r_tok)

        # save indiv file
        process.write_list_to_jsonl(fn_result_, sample, key=None)

        # delay to avoid overloading api
        time_elapsed = time.time() - t0
        if time_elapsed < time_per_sample:
            time.sleep(time_per_sample - time_elapsed)

    # load all indiv files, save to one for expmt
    fn_out_lst = get_files(dir_result)
    results_all = []
    for fn_out in fn_out_lst:
        results_all.append(process.read_jsonl_to_list(fn_out)[0])
    process.write_list_to_jsonl(fn_result, results_all, key=None)
    print(f'results generated in {args.dir_out}\n\n\n\n\n')


def load_data_():
    ''' load data for api call
        determine which samples, add system prompt '''

    args = parser.get_parser(purpose='openai')

    # get oai configs (gpt-x v. gpt-x-long, openai api key, max_len_abs)
    args = get_oai_configs(args)
  
    # load inputs
    dataset = SummDataset(args, task='test')
    dataset.save_data(args, fn_out='inputs.jsonl')

    return args, dataset


@timeout(LEN_TIMEOUT, use_signals=False)
def call_api_wrapper(args, fn_inp_proc_, fn_out):
    ''' wrapper to call openai api, save intermediate jsonl output
        log exit reasons for failed requests
        return valid requests as list of tuples i.e. (inp, out) '''
    
    if not constants.RESOURCE:
        raise NotImplementedError('define azure RESOURCE in constants.py')

    cmd = f'python call_api.py {fn_inp_proc_} {fn_out}'
    cmd += f' --request_url {constants.URL_MODEL[args.oai_model]}'
    cmd += f' --api_key {args.oai_key}'
    cmd += f' --max_tok_per_min {constants.TOK_PER_MIN[args.oai_model]}'
    os.system(cmd)
    out_gpt = process.read_jsonl_to_list(fn_out)
    
    # track exit indices
    exits = {'content_filter': [], 'error': [], 'unknown': []}

    list_idx, list_out = [], []
    for ii, out in enumerate(out_gpt):
       
        idx_sample = out[2]['idx']

        try: # try extracting output

            # running list of successful outputs, indices
            out_ = out[1]['choices'][0]['message']['content']
            list_out.append(out_)
            list_idx.append(idx_sample)

        except: # track errors
            try: # check if request exited due to content filter
                code = oo[1]['choices'][0]['finish_reason']
                if code == 'content_filter':
                    exits[code].append(idx_sample)
            except:
                try: # check if request exited due to content error
                    code = oo[1][0][2:7]
                    if code == 'error':
                        exits[code].append(idx_sample)
                except: # unknown reason for exit request
                    exits['unknown'].append(idx_sample)
   
    log_exits(args, exits, out_gpt)
    
    if list_out:
        return list_out[0] # only processing one sample
    else:
        return None # i.e. got error


def get_oai_configs(args):
    ''' based on manually determined limints for n_icl
        decide whether to use longer context model '''
   
    if not constants.API_KEY:
        raise NotImplementedError('define azure API_KEY in constants.py')
    args.oai_key = constants.API_KEY

    # get model-, dataset-specific n_icl limit
    # for both gpt-x, gpt-x-long
    n_icl_limits = constants.N_ICL_LIMITS[args.model][args.dataset]
    if args.n_icl <= n_icl_limits[0]:
        args.oai_model = args.model
    elif args.n_icl <= n_icl_limits[1]:
        args.oai_model = f'{args.model}-long'
    else:
        raise NotImplementedError('n_icl above manually chosen limit')

    # overwrite args.max_len_abs based on oai_model
    args.max_len_abs = constants.MAX_LEN_OAI[args.oai_model]

    return args


def log_exits(args, exits, out_gpt):
    ''' given dict w reasons and indices for exits, save to csv '''

    exits_lst = []
    for key, idcs in exits.items():
        exits_lst.append(f'exit code: {key}')
        for ii, idx in enumerate(idcs):
            exits_lst.append(idx)
            exits_lst.append(out_gpt[ii])
        exits_lst.append('\n')

    fn_exit = get_path(args.dir_out, 'exit_log.csv', rm=False)
    process.write_list_to_csv(fn_exit, exits_lst)

    return


def get_naughty_list(fn_naughty):
    ''' list of samples idcs which openai previously rejected
        track s.t. we avoid them in future runs '''
    
    if not os.path.exists(fn_naughty):
        return []

    with open(fn_naughty, 'r') as f:
        lines = f.readlines()
    naughty_lst = [line.strip() for line in lines if line.strip()]

    return naughty_lst


def get_path(dir_, fn, rm=False):

    path = os.path.join(dir_, fn)
   
    # delete old version of that file
    if rm:
        try:
            os.remove(path)
        except OSError:
            pass

    return path


def log_progress(fn_log, msg):
    with open(fn_log, 'a') as f:
        f.write(msg + '\n')


def get_files(dir_, abs_path=True):
    ''' get all files in a directory '''
    f_lst = [f for f in os.listdir(dir_) if os.path.isfile(os.path.join(dir_, f))]
    if abs_path:
        f_lst = [os.path.join(dir_, f) for f in f_lst]
    return f_lst


def get_completed_idcs(dir_):
    ''' get indices of files in a directory, i.e. xxx from xxx.jsonl '''
    f_lst = get_files(dir_, abs_path=False)
    idx_lst = [int(f.split('.')[0]) for f in f_lst] 
    return idx_lst


if __name__ == '__main__':
    main()