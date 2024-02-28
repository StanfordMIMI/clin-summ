import os


##############################################################
### project directory ########################################

# TODO: set DIR_PROJECT as location for all data and models (outside git repo)
DIR_PROJECT = "/home/tim/clin_summ_results"
assert os.path.exists(DIR_PROJECT), "please enter valid directory"


##############################################################
### models ###################################################

MODELS = {

    's2s-t5': {  # seq2seq architectures
        "t5-xl": "t5-3b",
        "flan-t5-xl": "google/flan-t5-xl",
    },
    's2s-ul2': {
        "flan-ul2": "google/flan-ul2",
    },

    'gpt': {  # gpt-style architectures

        "vicuna-7b": "AlekseyKorshuk/vicuna-7b",
        "alpaca-7b": "chavinlo/alpaca-native",
        "med-alpaca-7b": "medalpaca/medalpaca-7b",

    },

    'gptq': {  # gpt-style architectures w quant
        'llama2-7b': {
            'path': 'TheBloke/Llama-2-7B-GPTQ',
            'basename': 'model',
        },
        'llama2-13b': {
            'path': 'TheBloke/Llama-2-13B-GPTQ',
            'basename': 'model',
        },
    },

    # openai models: leave blank
    'oai-35': {'gpt-35': '', 'gpt-35-16k': ''},
    'oai-4': {'gpt-4': '', 'gpt-4-32k': ''},

}


##############################################################
### prompt components ########################################

# instructions
INSTRUCTION_RRS = (
    'summarize the radiology report findings into an impression'
    ' with minimal text'
)
INSTRUCTION_CHQ = (
    'summarize the patient health query into one question'
    ' of 15 words or less'
)
INSTRUCTION_PLS = (
    'based on the progress note, generate a list of 3-7 problems'
    ' (a few words each) ranked in order of importance'
)
INSTRUCTION_D2N = (
    'summarize the patient/doctor dialogue into an assessment and plan'
)
INSTRUCTION_ICL = ', using the provided examples to guide word choice.'
CHATGPT_EXPERTISE = 'you are a knowledgeable medical professional. '

# define task-dependent prompt components
PROMPT_COMPONENT = {
    'rrs': {
            'prefix': 'finding',
            'suffix': 'impression',
            'instruction': INSTRUCTION_RRS,
           },
    'chq': {
            'prefix': 'query',
            'suffix': 'summarized question',
            'instruction': INSTRUCTION_CHQ,
           },
    'pls': {
            'prefix': 'progress note',
            'suffix': 'problem list',
            'instruction': INSTRUCTION_PLS,
           },
    'd2n': {
            #'prefix': 'patient/provider dialogue',
            'prefix': 'patient/doctor dialogue',
            'suffix': 'assessment and plan',
            'instruction': INSTRUCTION_D2N,
           },
}


##############################################################
### expmt parameters #########################################

# append DEFAULTS keys to cases[case_id] only if key dne in cases[case_id]
# NOTE: to override variable defaults, include the key in cases[case_id]
DEFAULTS = {
    'n_icl': 0, # number of in-context examples
    'use_instruction': True, # include prefix instruction
    'modalities': 'all', # only relevant for datasets w modalities (iii)
    'thresh_seq_crop': 0, # threshold to remove longest X% of sequences 
    'thresh_out_toks': 0, # threshold to set max_new_tokens to longest Y% summary 
    'batch_size': 6,

    # only relevant if fine-tuning, i.e. not for discrete prompting
    'n_trn_samples': None,
    'n_val_samples': None,
    'batch_size': 6,
    'grad_accum_steps': 4, 
    'lr0': 1e-3,
    'lr_schedule': 'linear_decay', 
    'lr_num_warmup_steps': 100, 
    'max_trn_epochs': 5,
    
    # only relevant if querying openai model
    'gpt_temp': 0.1, 
}

# define cases i.e. experiment configurations, modifying params above
cases = { 

    ### cases 0-99 reserved for discrete prompting open-source models ###
    0: {'use_instruction': False}, # zero-shot null prompt
    10: {'n_icl': 1}, # in-context learning with 1 example
    11: {'n_icl': 2},
    12: {'n_icl': 4},
    13: {'n_icl': 8},
    14: {'n_icl': 16},
    15: {'n_icl': 32},
    16: {'n_icl': 64},

    ### cases 200-399 reserved for lora/qlora with open-source models ###
    300: {},

    ### cases 400-499 reserved for proprietary models ###
    400: {}, # zero-shot prompt
    410: {'n_icl': 1}, # in-context learning with 1 example
    411: {'n_icl': 2},
    412: {'n_icl': 4},
    413: {'n_icl': 8},
    414: {'n_icl': 16},
    415: {'n_icl': 32}, 
    416: {'n_icl': 64},
    417: {'n_icl': 128},
    418: {'n_icl': 256},

}

# append DEFAULTS keys to cases[case_id] only if key dne in cases[case_id]
TRN_ONLY = ['grad_accum_steps', 'lr0', 'lr_schedule',
            'lr_num_warmup_steps', 'max_trn_epochs',
            'n_trn_samples', 'n_val_samples']
for case_id in cases:
    for key in DEFAULTS:
        if key not in cases[case_id]:
            if key in TRN_ONLY: # params specific to fine-tuning
                if case_id >= 100:
                    cases[case_id][key] = DEFAULTS[key]
            else: # params relevant for every expmt
                cases[case_id][key] = DEFAULTS[key]
    if case_id >= 200 and case_id < 300:
        cases[case_id]['method'] = 'lora'
    if case_id >= 300 and case_id < 400:
        cases[case_id]['method'] = 'qlora'


##############################################################
### misc #####################################################

# datasets which have been implemented to date
DATASETS_IMPLEMENTED = ['iii', 'chq', 'pls', 'opi', 'cxr', 'd2n']

# datasets of radiology reports w modality subdirs
DATASETS_W_MODALITIES = ['iii']

# datasets of radiology reports (all)
DATASETS_RRS = DATASETS_W_MODALITIES + ['opi', 'cxr'] 

# allowable keys when training model
KEYS_TRN = ['input_ids', 'attention_mask', 'labels']

# implemented peft methods
METHODS = ['qlora']

# models which require custom tokenizer (gpt-neo family, deprecated)
MODELS_W_CUSTOM_TOKENIZER = []

# architecture : max_len of input tokens
MAX_LEN_OAI = {'gpt-35': 4096, 'gpt-35-long': 16384,
               'gpt-4': 8192, 'gpt-4-long': 32768}
MAX_LEN = {
    's2s-t5': 512, 's2s-ul2': 2048, 'gpt': 2048, 'gptq': 4096,
    'oai-35': MAX_LEN_OAI['gpt-35'], # will be overwritten if openai
    'oai-4': MAX_LEN_OAI['gpt-4'], # will be overwritten if openai
    }

# maximum number of in-context examples by dataset
# tuple is max n_icl for (gpt-x, gpt-x-long)
N_ICL_LIMITS = {
    'gpt-35' : {
        'opi': (32, 128),
        'cxr': (16, 96),
        'iii': (8, 32),
        'chq': (16, 96),
        'pls': (2, 8),
        'd2n': (1, 8),
    },
    'gpt-4' : {
        'opi': (64, 256),
        'cxr': (48, 192),
        'iii': (16, 80),
        'chq': (48, 208),
        'pls': (4, 24),
        'd2n': (2, 16),
    },
}

# minimum number of samples required to run expmt
N_MIN_SAMPLES = 50

# misc training hyperparameters
LORA_R = 8 # rank decomposn value (w higher r, more capacity + less efficiency)
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
PATIENCE = 5

# in-context learning (icl) prompts
ICL_SENTENCE_TRANSFORMER = 'pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb'

##############################################################
### openai ###################################################

# set request url based on endpoint + names of deployed models
def get_url(resource, deployment):
    ''' get request url given azure endpoint, deployment '''
    endpt = f'https://{resource}.openai.azure.com/'
    URL_PREFIX = 'openai/deployments/' 
    URL_SUFFIX = '/chat/completions?api-version=2023-03-15-preview'
    return endpt + URL_PREFIX + deployment + URL_SUFFIX

# TODO: (1) define your RESOURCE, API_KEY 
#       (2) ensure your model deployments match those in URL_MODEL
RESOURCE = None # name of your azure resource
API_KEY = None # your azure api key
URL_MODEL = { # your url for individual model deployments
    'gpt-35': get_url(RESOURCE, 'gpt35-model'),
    'gpt-35-long': get_url(RESOURCE, 'gpt35-16k-model'),
    'gpt-4': get_url(RESOURCE, 'gpt4-model'),
    'gpt-4-long': get_url(RESOURCE, 'gpt4-32k-model'),
}

# rate limits per model
TOK_PER_MIN = { 
    'gpt-35': 120000, 'gpt-35-long': 120000,
    'gpt-4': 10000, 'gpt-4-long': 30000,
}

##############################################################
### misc filenames ###########################################

FN_INPUTS = 'inputs.csv'
FN_TARGET = 'target.csv' # target
FN_OUTPUT = 'output.csv' # generated output
FN_METRICS_JSON = 'metrics.json' 
FN_METRICS_TXT = 'metrics.txt'
FN_INP_ICL = 'train.inputs.tok'
FN_TGT_ICL = 'train.target.tok'
FN_IDCS_ICL = 'trn_inputs_index.bin'
FN_RESULT = 'result.jsonl'
FN_TST = 'test' 

# keys which should be present when loading data
KEYS_INP = ['idx', 'inputs', 'target']
KEYS_OUT = ['idx', 'inputs', 'target', 'prompt', 'output']
METRICS = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L',
           'BERT', 'F1-CheXbert', 'F1-Radgraph']#, 'MEDCON']
