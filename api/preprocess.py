#!/usr/bin/env python3
"""
API REQUEST PARALLEL PROCESSOR
    previously preprocess_for_batch_api.py

This script preprocesses input JSONL files and generates API request JSONL files.
The resulting files can then be used with the API Request Parallel Processor:
https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py

Features:
- Parses input files line by line to avoid running out of memory for large jobs
- Creates API requests suitable for the OpenAI GPT-4 model
- Preserves original data in the 'metadata' field of the request

Example command to call script:
```
python preprocess.py \
  --input_file data/medqa_open/train_with_ids.jsonl \
  --output_file data/medqa_open/train_with_ids_preprocessed.jsonl \
  --task medqa_open \
  --model gpt-4 \
  --temperature 0.0
```

Another example command to call script:
```
python preprocess.py \
  --input_file data/ehr_sample/ehr_512_256.jsonl \
  --output_file data/medqa_open/ehr_512_256_requests.jsonl \
  --task medqa_open \
  --model gpt-4 \
  --temperature 0.0
```

Inputs:
- input_file : str
    - Path to the input JSONL file.
- output_file : str
    - Path to the preprocessed output JSONL file.
- model : str, optional
    - The model to be used for the API request. If omitted, will default to "gpt-4".
- temperature : float, optional
    - The sampling temperature to be used for the API request. If omitted, will default to 0.0.

The script is structured as follows:
    - Imports
    - Define helper functions
        - generate_medqa_open_prompt (generates the user and system prompts for the API request)
        - generate_api_request (generates API request)
    - Define main function, preprocess
        - Opens the input file and output file
        - Processes each line of the input file, generating an API request
        - Writes the API request to the output file
    - If script is run (not imported), parse command line arguments and call preprocess
"""

import argparse
import json
from collections import OrderedDict
from typing import Dict, Tuple


def generate_api_request(args, json_obj): 
    
    user_prompt = json_obj["prompt"]

    return {
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": args.temperature,
    }

def preprocess_jsonl(args) -> None:
    """
    Preprocesses the input JSONL file and outputs to a new file.

    The request object `request_obj` that is written to file should
    conform to the format specified in the documentation for the
    OpenAI API:
    https://platform.openai.com/docs/api-reference/chat/create
    Specifically, first-level keys should be "model", "messages",
    and optionally "metadata". The "model" key should map to a string
    specifying the model to be used for the API request (e.g., "gpt-4).
    The "messages" key should map to a list of dictionaries, where each
    dictionary has keys "role" and "content". The "role" key should map
    to a string specifying the role of the message (e.g., "system" or
    "user"). The "content" key should map to a string specifying the
    content of the message. The "metadata" key should map to a dictionary
    containing any additional metadata to be included in the request.

    Parameters
    ----------
    fn_inp : str
        The path to the input JSONL file.

    fn_out : str
        The path to the output JSONL file.

    model : str, default="gpt-4"
        The model to be used for the API request.

    task : str, default="medqa_open"
        The task for which the preprocessing is being done. This will
        be dataset specific (e.g., "medqa_open" for the MedQA Open dataset,
        which originally takes the form of a JSONL file).

    Raises
    ------
    ValueError
        If the task is not recognized.

    Examples
    --------
    >>> fn_inp = "data/medqa_open/train_with_ids.jsonl"
    >>> fn_out = "data/medqa_open/train_with_ids_preprocessed.jsonl"
    >>> preprocess(fn_inp, fn_out, model="gpt-4", task="medqa_open")
    """
    with open(args.fn_inp, "r") as inp, open(args.fn_out, "w") as out:
        for index, line in enumerate(inp):
            json_obj = json.loads(line.strip(), object_pairs_hook=OrderedDict)
            request_obj = generate_api_request(args, json_obj)
            request_obj["metadata"] = json_obj
            out.write(json.dumps(request_obj) + "\n")

def main():
    
    SYS_PROMPT = "You are an expert doctor who consults electronic health"
    SYS_PROMPT += " records to answer questions about patients"
    
    parser = argparse.ArgumentParser(
        description="Preprocess JSONL files for use with the API Request Parallel Processor."
    )
    parser.add_argument("fn_inp", type=str,
                        help="Path to the input JSONL file.")
    parser.add_argument("fn_out", type=str,
                        help="Path to the output JSONL file.")
    parser.add_argument("--task", type=str, default="radadapt",
                        help="Name of the task.")
    parser.add_argument("--model", type=str, default="gpt-35-turbo",
                        help="Name of the model to use.")
    parser.add_argument("--temperature", type=float, default=0.1, #default=0.7,
                        help="Temperature for sampling.")
    parser.add_argument("--system_prompt", type=str, default=SYS_PROMPT,
                        help="system prompt applied to each sample")

    args = parser.parse_args()

    preprocess_jsonl(args)

if __name__ == "__main__":
    main()
