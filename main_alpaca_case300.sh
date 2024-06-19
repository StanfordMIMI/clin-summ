#!/bin/bash

###############################################################################
### README ####################################################################
# main script for training models, generating output, calculating metrics
# step 1: set one of the following to "true"
#           DO_TRN: fine-tune a model w qlora (train.py)
#           DO_RUN: generate outputs from a model (run.py)
#           DO_CALC: calculate metrics on output (calc_metrics.py) 
# step 2: uncomment desired models, datasets, case_ids (defined in constants.py)
# step 3: save file, run script
###############################################################################

export CUDA_VISIBLE_DEVICES=$1
if [ -z "${VAR}" ]; then
  export CUDA_VISIBLE_DEVICES="0"
  echo "Defaultly set CUDA_VISIBLE_DEVICES=0"
fi

export DIR_PROJECT="./_out"
DO_TRN=true
DO_RUN=false
DO_CALC=false
n_samples=250 # n_samples to run, calculate metrics

model_list=(
    #flan-t5-xl
    #flan-ul2
    #vicuna-7b
    alpaca-7b
    #med-alpaca-7b
    #llama2-7b
    #llama2-13b
)

dataset_list=(
    opi
    #cxr
    #iii
    #chq
    #pls
    #d2n
)

case_id_list=(
    #0
    #10
    #11
    #12
    #13
    #14
    #15
    #16
    300
)

for j in "${!model_list[@]}"; do
    model="${model_list[j]}"
    
    for i in "${!dataset_list[@]}"; do
        dataset="${dataset_list[i]}"
            
        for k in "${!case_id_list[@]}"; do
            case_id="${case_id_list[k]}"

            ### train model
            if $DO_TRN; then 
                python src/train.py --model $model \
                                    --case_id $case_id \
                                    --dataset $dataset
            fi
            
            ### calculate metrics
            if $DO_CALC; then 
                python src/calc_metrics.py --model $model \
                                            --case_id $case_id \
                                            --dataset $dataset \
                                            --n_samples 999999 
            fi

            ### generate output 
            if $DO_RUN; then 
                python src/run.py --model $model \
                                    --case_id $case_id \
                                    --dataset $dataset \
                                    --n_samples $n_samples
            fi

        done
    done
done
