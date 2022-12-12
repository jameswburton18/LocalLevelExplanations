from transformers import AutoFeatureExtractor, AutoTokenizer
import torch
from typing import List
from dataclasses import dataclass
import re
from datasets.arrow_dataset import Dataset
import inflect


def linearise_input(data_row, method, max_fts=15, data_only=False):
    """Linearise data row to be in chosen form."""
    
    # Linearising the data
    chosen_class = data_row["predicted_class"]
    classes_dict = eval(data_row["classes_dict"])
    other_classes = "&& ".join([f"{k} {v}" for k,v in classes_dict.items() if k != chosen_class])
    
    feature_nums = data_row['feature_nums'][:max_fts]
    sign = data_row['sign'][:max_fts]
    values = data_row['values'][:max_fts]

    fts_and_signs = "&& ".join([f'{a} {b} ' for a, b in zip(feature_nums, sign)])
    fts_and_pos = "&& ".join([f'{a} {b} ' for a, b in zip(feature_nums, sign) if b == 'positive'])
    fts_and_nega = "&& ".join([f'{a} {b} ' for a, b in zip(feature_nums, sign) if b == 'negative'])
    fts_and_negl = "&& ".join([f'{a} {b} ' for a, b in zip(feature_nums, sign) if b == 'negligible'])
    fts_and_negl = 'None' if fts_and_negl == '' else fts_and_negl
    
    fts = "&& ".join([f'{a} ' for a in feature_nums])
    pos_fts = "&& ".join([f'{a} ' for a, b in zip(feature_nums, sign) if b == 'positive'])
    nega_fts = "&& ".join([f'{a} ' for a, b in zip(feature_nums, sign) if b == 'negative'])
    negl_fts = "&& ".join([f'{a} ' for a, b in zip(feature_nums, sign) if b == 'negligible'])
    negl_fts = 'None' if negl_fts == '' else negl_fts

    essel_input = f'| predicted class | {chosen_class} {classes_dict[chosen_class]} | other classes | {other_classes} | \
    features | {fts}| postive features | {pos_fts} | negative features | {nega_fts} | \
    negligible features | {negl_fts} |'

    p = inflect.engine()

    ordinals = [p.ordinal(i+1) for i in range(len(feature_nums))]

    features = ' '.join([f'| {o} | {f} {s} {v}' for o, f, s, v in 
                         zip(ordinals, feature_nums, sign, 
                             values)])
        
    ord_first_input = f'| predicted class | {chosen_class} {classes_dict[chosen_class]} | other classes | {other_classes} {features} |'

    ft_first_input = ' '.join([f'| {f} | {o} {s} {v}' for o, f, s, v in zip(ordinals, feature_nums, sign, values)])

    if data_only:
        preamble = ''
        questions = ''
    else:
        # Preamble
        preamble = "\n <br> <br> Using the above information, answer the following \
            in detail: <br> <br> "
        questions = '\n'.join([f'{idx+1}. {q}' for idx, q in 
                            enumerate(data_row['narrative_questions'])])

    if method == 'essel':
        data_row['input'] = essel_input + preamble + questions
    elif method == 'ord_first':
        data_row['input'] = ord_first_input + preamble + questions
    elif method == 'ft_first':
        data_row['input'] = ft_first_input + preamble + questions
    else:
        raise ValueError('method must be one of essel, ord_first or ft_first')

    return data_row

def form_stepwise_input(data_row, method, max_fts):
    # Linearising the data
    data_row = linearise_input(data_row, method=method, max_fts=max_fts, data_only=True)
    preamble = "\n <br> <br> Using the above information, answer the following \
            in detail: <br> <br> "
    data_row['input'] = [data_row['input'] + preamble + q for q in data_row['narrative_questions']]
    return data_row

def form_qa_input_output(data_row, method, max_fts=15):
    """Combining the quesiton with the linearised data and a preamble. Also
    renaming answer as narration so as to match `convert_to_features()`."""
    # Linearising the data
    data_row = linearise_input(data_row, method=method, max_fts=max_fts, data_only=True)
    # Preamble
    preamble = "\n <br> <br> Using the above information, answer the following \
        question: <br> <br> "
    
    data_row['input'] = data_row['input'] + preamble + data_row['question']
    data_row['narration'] = data_row['answer']
    return data_row

def convert_to_features(batch, tokenizer, max_input_length=400, max_output_length=350):
    if type(batch['input'][0]) == list:
        input_encodings = [tokenizer(i, padding="max_length", truncation=True, max_length=max_input_length) for i in batch['input']]
        input_ids = [i['input_ids'] for i in input_encodings]
        attention_mask = [i['attention_mask'] for i in input_encodings]
    else:
        input_encodings = tokenizer(batch['input'], padding="max_length", truncation=True, max_length=max_input_length)
        input_ids = input_encodings['input_ids']
        attention_mask = input_encodings['attention_mask']
    target_encodings = tokenizer(batch['narration'], padding="max_length", truncation=True, max_length=max_output_length)
    encodings = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': target_encodings['input_ids'],
    }
    return encodings

