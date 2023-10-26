import os


def main():
    ''' wrapper to call run_azure.py using a dictionary
        of dataset : case_id_list pairs '''

    # TODO: manually set these params for each set of expmts
    task = 'run' # 'run' (query openai), 'calc' (metrics)
    expmt_name_list = ['demo'] # per expmts defined in get_expmt_configs()
    is_demo = True # enable runs over 1 sample (else constants.N_MIN_SAMPLES)

    for expmt_name in expmt_name_list:

        case_ids, model, n_samples = get_expmt_configs(expmt_name)
        dataset_list = ['opi', 'chq']
        assert task in ['run', 'calc']
        script = 'run_expmt.py' 
        
        if task == 'run':
            cmd = f'python {script} --n_samples {n_samples} --model {model}'
            if is_demo:
                cmd += ' --is_demo'
        elif task == 'calc':
            cmd = f'python ../src/calc_metrics.py --n_samples 9999 --model {model}'

        for dataset in dataset_list:
            
            case_id_list = case_ids[dataset]
            cmd += f' --dataset {dataset}'
            
            for case_id in case_id_list:

                cmd += f' --case_id {case_id}'
                
                os.system(cmd)
            

def get_expmt_configs(expmt_name):
    ''' TODO: define experimental configurations, i.e.
            model (gpt-35 or gpt-4)
            case_ids: corresponding to configs in constants.cases
            datasets, e.g. opi or chq '''

    assert expmt_name in ['demo']
    
    if expmt_name == 'demo':
        n_samples = 1 # number of samples to run
        model = 'gpt-35' # choose gpt-3.5
        case_ids = {'opi': [400], 'chq': []} # case_id=400 for opi dataset

    else:
        raise NotImplementedError


    return case_ids, model, n_samples


if __name__ == '__main__':
    main()
