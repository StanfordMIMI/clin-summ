import evaluate 
from f1chexbert import F1CheXbert
import io
import json
import numpy as np
import os
from radgraph import F1RadGraph
import sys
# from UMLSScorer import UMLSScorer 

import constants
import parser
import process
import summ_dataset

CALC_REDUNDANT = False # re-calculate, even if scores already exist 

def main():

    # parse arguments, set data paths
    args = parser.get_parser()
    is_cxr = True if args.dataset in ['cxr', 'opi'] else False
    if (not CALC_REDUNDANT and
        os.path.exists(os.path.join(args.dir_out, constants.FN_METRICS_TXT))):
        print(f'metrics already calculated for {args.dir_out}')
        sys.exit()
   
    # load data
    dataset = summ_dataset.SummDataset(args, task=None, purpose='load_result')
    if len(dataset.data) == 0:
        print('no data loaded')
        sys.exit()
    lst_tgt = dataset.list_target
    lst_out = dataset.list_output
    lst_idx = dataset.list_idx

    # load metrics
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')
    f1radgraph = F1RadGraph(reward_level='partial')
    f1chexbert = F1CheXbert(device='cuda') if is_cxr else None
    # medcon = UMLSScorer()
    metrics = (bleu, rouge, bertscore,
               f1radgraph, f1chexbert)#, medcon)

    # compute scores of each sample across entire dataset
    scores_all = {}
    for tgt, out, idx in zip(lst_tgt, lst_out, lst_idx):

        # get sub-dict containing scores for each metric
        scores = compute_scores(tgt, out, metrics, is_cxr)

        # append to master dict, dataset object
        scores_all[idx] = scores
        dataset.append_scores(idx, scores)

    # save averaged scores across entire dataset
    write_all_scores(args, scores_all)
    
    dataset.save_data(args)


def compute_scores(tgt, out, metrics, is_cxr):
    ''' given output(s), target(s), and a tuple of metrics
        return a scores dict ''' 
   
    # unpack tuple of pre-loaded metrics
    bleu, rouge, bertscore, f1radgraph, f1chexbert = metrics #, medcon

    # compute medcon umls metric over individual string
    # score_medcon = medcon(tgt, out)

    # convert single sample to list
    tgt, out = wrap_str_in_lst(tgt), wrap_str_in_lst(out)

    # compute hugging face scores
    try:
        scores_bleu = bleu.compute(predictions=out, references=tgt)
    except: # division by zero
        print('bleu not computed correctly')
        scores_bleu = {'bleu': 0}
    scores_rouge = rouge.compute(predictions=out, references=tgt)
    scores_bert = bertscore.compute(predictions=out, references=tgt, lang='en')

    # compute f1-radgraph, f1-chexbert
    try:
        score_radgraph, _, _, _ = f1radgraph(hyps=out, refs=tgt)
    except:
        print('radgraph not computed correctly')
        score_radgraph = 0
    if is_cxr:
        _, _, class_report, _ = f1chexbert(hyps=out, refs=tgt)
        score_chexbert = class_report['micro avg']['f1-score']
    else:
        score_chexbert = 0.

    scores = {
        'BLEU': scores_bleu['bleu'],
        'ROUGE-1': scores_rouge['rouge1'],
        'ROUGE-2': scores_rouge['rouge2'],
        'ROUGE-L': scores_rouge['rougeL'],
        'BERT': np.mean(scores_bert['f1']), 
        'F1-CheXbert': score_chexbert,
        'F1-Radgraph': score_radgraph,
        # 'MEDCON': score_medcon,
    }

    # scale scores to be on [0,100] instead of [0,1]
    for key in scores:
        scores[key] *= 100.
        scores[key] = round(scores[key], 2)

    return scores


def write_all_scores(args, scores_all): 
    ''' write all scores across dataset to json file 
        redundantly write to txt for copy-paste into overleaf '''

    validate_keys(scores_all) # sanity check

    # compute avg, std across all samples. write to json
    scores_avg_std = avg_across_samples(scores_all)
    fn_scores_json = os.path.join(args.dir_out, constants.FN_METRICS_JSON)
    with open(fn_scores_json, 'w') as f:
        f.write(json.dumps(scores_avg_std))

    # extract avg, write to txt file
    scores_avg = extract_avg_only(scores_avg_std)
    ss = scores_avg
    txt_out = []
    for key, val in scores_avg.items():
        ss[key] = round(ss[key], 1)
    header = 'BLEU & ROUGE-L & BERT & Radgraph & CheXbert & MEDCON'
    txt_out.append(header)
    str_txt = f'{ss["BLEU"]} & {ss["ROUGE-L"]} & {ss["BERT"]}'
    str_txt += f' & {ss["F1-Radgraph"]} & {ss["F1-CheXbert"]}'
    # str_txt += f' & {ss["MEDCON"]} '
    txt_out.append(str_txt)
    fn_scores_txt = os.path.join(args.dir_out, constants.FN_METRICS_TXT)
    process.write_list_to_csv(fn_scores_txt, txt_out)

    return


def avg_across_samples(scores_all):
    ''' average across individual sample scores (sub-dicts) '''

    scores_avg_std = {} 
    keys_to_avg = constants.METRICS

    for key in keys_to_avg:
        values = [sub_dict[key] for sub_dict in scores_all.values()]
        avg_std = {'avg': round(np.mean(values), 2),
                   'std': round(np.std(values), 2)}
        scores_avg_std[key] = avg_std

    return scores_avg_std


def extract_avg_only(scores_avg_std):
    ''' extract only values from sub-dict key avg '''
    scores_avg = {}
    for idx in scores_avg_std:
        scores_avg[idx] = scores_avg_std[idx]['avg']
    return scores_avg


def validate_keys(my_dict):
    ''' given dict w sub-dict, validate all sub-dicts have same keys '''
    
    sub_dict_keys = None
    for sub_dict in my_dict.values():
        if sub_dict_keys is None:
            sub_dict_keys = set(sub_dict.keys())
        else:
            msg = 'sub-dicts do not contain same keys'
            assert set(sub_dict.keys()) == sub_dict_keys, msg 

    return
   

def wrap_str_in_lst(var):
    if isinstance(var, str):
        return [var]
    return var


if __name__ == '__main__':
    main()
