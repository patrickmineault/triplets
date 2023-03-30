import asyncio
import click
import json
import logging
import numpy as np
import openai
import pandas as pd
import time
from typing import Optional
from src.api_request_parallel_processor import process_api_requests_from_file


from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

openai.api_key_path = "/home/pmin/Documents/nma-match/.openai-key"

prompts = {
            'direct': ("""Report which of these objects, 1, 2 or 3, is the least """
                     """similar to the other two. 1: {word1}, 2: {word2}, 3: {word3}. """
                     """Focus your judgement on the objects. Give a one-digit answer. """),
            'direct_completion': ("""Which of these objects, 1, 2 or 3, is the least """
                     """similar to the other two? 1: {word1}, 2: {word2}, 3: {word3}. """
                     """The answer is :"""),
            'chain': ("""Report which of these objects, 1, 2 or 3, is the least """
                     """similar to the other two. 1: {word1}, 2: {word2}, 3: {word3}. """
                     """Focus your judgement on the objects. Think step-by-step. Finish """
                     """with the phrase: The answer is *digit*."""),
            'visual': ("""Report which of these objects, 1, 2 or 3, is the least """
                       """similar to the other two. 1: {word1}, 1: {word2}, 3: {word3}. """
                       """Focus your judgement on the objects and their visual attributes. Think step-by-step. Finish """
                       """with the phrase: The answer is *digit*."""),
            'pairs': ("""Report which of these pairs of objects, 1, 2 or 3, is the most """
                     """similar. 1: {word2}-{word3}, 2: {word1}-{word3}, 3: {word1}-{word2}. """
                     """Focus your judgement on the objects. Think step-by-step. Finish """
                     """with the phrase: The answer is *digit*."""),
            'rubric': ("""Report which of these objects, 1, 2 or 3, is the least """
                       """similar to the other two. 1: {word1}, 1: {word2}, 3: {word3}. """
                       """Focus your judgement on the objects. Consider aspects """
                       """such as the visual attributes of the objects (color, shape), """
                       """their size, whether they're organic or man-made. """
                       """Think step-by-step. Finish """
                       """with the phrase: The answer is *digit*."""),
          }

def generate_prompt(selected_prompt, 
                    word1, 
                    word2, 
                    word3, 
                    model="gpt-3.5-turbo") -> Optional[int]:
    words = {'word1': word1, 'word2': word2, 'word3': word3}
    prompt = prompts[selected_prompt].format(**words)
    return prompt
    
def run_requests_parallel(requests, model='gpt-3.5-turbo', model_type='chat', word_lists=None):
    # Save requests to file.
    input_file = 'data/interim/prompts-input.jsonl'
    output_file = 'data/interim/prompts-outputs.jsonl'

    keys = [json.dumps(x) for x in requests]
    with open(input_file, 'w') as f:
        f.write('\n'.join(keys))

    with open(output_file, 'w') as f:
        pass

    with open(openai.api_key_path, 'r') as f:
        api_key = f.read().strip()

    if model_type == 'chat':
        endpoint = 'https://api.openai.com/v1/chat/completions'
    else:
        endpoint = 'https://api.openai.com/v1/completions'

    if model == 'gpt-4':
        max_requests_per_minute = 200
    else:
        max_requests_per_minute = 900

    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=input_file,
            save_filepath=output_file,
            request_url=endpoint,
            api_key=api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=90000,
            token_encoding_name='cl100k_base',
            max_attempts=5,
            logging_level=logging.INFO,
        )
    )

    with open(output_file, 'r') as f:
        outputs = [json.loads(x) for x in f.read().splitlines()]

    # Reorder outputs.
    output_map = {k: i for i, k in enumerate(keys)}
    print(list(output_map.keys())[0])
    for i, o in enumerate(outputs):
        if model_type == 'chat':
            output_map[json.dumps(o[0])] = o[1]["choices"][0]["message"]["content"]
        else:
            output_map[json.dumps(o[0])] = o[1]["choices"][0]["text"]

    choices = []
    reasonings = []
    for i, content in enumerate(output_map.values()):
        positions = [content.rfind('1'), content.rfind('2'), content.rfind('3')]
        the_max = max(positions)
        if the_max == -1:
            choices.append(-1)
        elif the_max == positions[0]:
            choices.append(0)
        elif the_max == positions[1]:
            choices.append(1)
        elif the_max == positions[2]:
            choices.append(2)

        if choices[-1] == -1:
            # Especially with weaker models, sometimes they will report the word
            # only.
            content = content.lower()
            word1, word2, word3 = word_lists[i]
            positions = [content.rfind(word1), content.rfind(word2), content.rfind(word3)]

            the_max = max(positions)
            if the_max == -1:
                choices[-1] = -1
            elif the_max == positions[0]:
                choices[-1] = 0
            elif the_max == positions[1]:
                choices[-1] = 1
            elif the_max == positions[2]:
                choices[-1] = 2
        
        reasonings.append(content)

    return choices, reasonings


@click.command()
@click.option('--dataset', default='data/interim/testset1.csv')
@click.option('--model', default='gpt-3.5-turbo')
@click.option('--model_type', default='chat')
@click.option('--prompt', default='direct')
@click.option('--test', default=True)
@click.option('--ordering', default=0)
def fit_model(dataset, model, model_type, prompt, test, ordering=0):
    df = pd.read_csv(dataset)
    if test:
        # No use in testing on the whole dataset.
        df = df.loc[:100]

    dataset_short = dataset.split('/')[-1].split('.')[0]

    output_name = f'results/{dataset_short}_{model_type}_{model}_{prompt}_{ordering}.csv'
    df[model] = [None] * len(df)
    df[f"{model}_reasoning"] = [None] * len(df)

    orderings = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0)]
    chosen_ordering = orderings[ordering]
    requests = []
    word_lists = []
    for i, row in df.iterrows():
        words = [row['word0'], row['word1'], row['word2']]
        words_reordered = words[chosen_ordering[0]], words[chosen_ordering[1]], words[chosen_ordering[2]]
        word_lists.append(words_reordered)
        prompt_text = generate_prompt(prompt, *words_reordered)

        if model_type == 'chat':
            requests.append(
                {'messages': [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. You always follow instructions.",
                    },
                    {"role": "user", "content": prompt_text}
                ],
                'model': model,
                }
            )
        else:
            requests.append({'prompt': prompt_text, 'model': model, 'max_tokens': 512})

    choices, reasonings = run_requests_parallel(requests, model, model_type, word_lists)

    choices = [chosen_ordering[x] for x in choices]

    # Now run the requests
    df[model] = choices
    df[f"{model}_reasoning"] = reasonings
    df.to_csv(output_name)    


if __name__ == "__main__":
    fit_model()