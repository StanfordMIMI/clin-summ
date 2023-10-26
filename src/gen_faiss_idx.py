import faiss
import os 
from sentence_transformers import SentenceTransformer

import constants
import parser
import process


def main():

    # TODO: select your dataset for faiss indexing
    dataset = 'chq'

    # define directories
    dir_data = os.path.join(constants.DIR_PROJECT, 'data', dataset)
    dir_icl = os.path.join(dir_data, 'icl')
    parser.mkdir(dir_icl)

    # training samples into list
    trn_inputs_path = os.path.join(dir_data, constants.FN_INP_ICL)
    sentences = process.read_csv_to_list(trn_inputs_path)

    #compute x_index
    model = SentenceTransformer(constants.ICL_SENTENCE_TRANSFORMER)
    x_index = model.encode(sentences, convert_to_tensor=True)
    x_index = x_index.cpu().detach().numpy()

    # train faiss index. ref: https://github.com/matsui528/faiss_tips
    cpu_index = faiss.IndexFlatL2(x_index.shape[1])
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(x_index)
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)

    # save indices
    faiss_index_path = os.path.join(dir_icl, constants.FN_IDCS_ICL)
    faiss.write_index(cpu_index, faiss_index_path)

    print(f'faiss indices generated in {faiss_index_path}')


if __name__ == '__main__':
    main()
