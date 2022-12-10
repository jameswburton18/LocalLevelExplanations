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

def form_question_input(data_row, q_idx):
    """Form question input."""
    # Linearising the data
    chosen_class = data_row["predicted_class"]
    # classes_dict = eval(data_row

def convert_to_features(batch, tokenizer, max_input_length=400, max_output_length=350):
    input_encodings = tokenizer(batch['input'], padding="max_length", truncation=True, max_length=max_input_length)
    target_encodings = tokenizer(batch['narration'], padding="max_length", truncation=True, max_length=max_output_length)
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids'],
    }
    return encodings

@dataclass
class DataCollator:
    
    def tokenize_question(self, questions: List[str]):
        question_encoding = self.tokenizer(questions, 
                                           return_attention_mask=True,
                                           max_length=self.max_question_len,
                                           padding='max_length',
                                           add_special_tokens=True,
                                           truncation=True,
                                           return_tensors='pt')
        return {
            "question_tokens": question_encoding['input_ids'].squeeze(),
            "question_att_mask": question_encoding['attention_mask'].squeeze(),
        }

    def tokenize_answer(self, answers: List[str]):
        split_answers = [clean_and_split(answer) for answer in answers]
        answers = [re.sub(r"'", "", ans) for ans in answers]
        answers = [re.sub(r"\[", "", ans) for ans in answers]
        answers = [re.sub(r"\]", "", ans) for ans in answers]
        answers = ['None' if ans == '[]' or ans =='' else ans for ans in answers]
        
        chosen_answers = split_answers if self.use_split_answers else answers
        
        target_encoding = self.tokenizer(chosen_answers, 
                                         max_length=self.max_answer_len,
                                         padding='max_length',
                                         truncation=True,
                                         return_attention_mask=False,
                                         add_special_tokens=True,
                                         is_split_into_words=self.use_split_answers,
                                         return_tensors='pt'
                                         )
        
        return {
            "ans_tokens": target_encoding['input_ids'].squeeze(),
            # "labels_attention_mask": target_encoding['attention_mask'].squeeze()
        }

    def preprocess_images(self, image_path_batch: List[str]):
        processed_images = self.preprocessor(
            images=[Image.open(im).convert('RGB') for im in image_path_batch],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images['pixel_values'].squeeze(),
        }
    
    def preprocess_ques_type(self, ques_types: List[str]):
        ids = [QUES_TYPE_DICT[ques_type] for ques_type in ques_types]
        return {
            "ques_type_ids": torch.tensor(ids),
        }    
            
    def __call__(self, raw_batch_dict):
        print()
        output_dict = {
            **self.tokenize_question(
                raw_batch_dict['question']
                if isinstance(raw_batch_dict, dict) else
                [i['question'] for i in raw_batch_dict]
            ),
            **self.tokenize_answer(
                raw_batch_dict['answer']
                if isinstance(raw_batch_dict, dict) else
                [i['answer'] for i in raw_batch_dict]
            ),
            **self.preprocess_images(
                raw_batch_dict['image_path']
                if isinstance(raw_batch_dict, dict) else
                [i['image_path'] for i in raw_batch_dict]
            ),
            # "ques_type_ids": QUES_TYPE_DICT[raw_batch_dict['ques_type']]
            #     if isinstance(raw_batch_dict, dict) else
            #     [QUES_TYPE_DICT[i['ques_type']] for i in raw_batch_dict]
            # ,
            **self.preprocess_ques_type(
                raw_batch_dict['ques_type']
                if isinstance(raw_batch_dict, dict) else
                [i['ques_type'] for i in raw_batch_dict]
            ),
        }
        return output_dict
 

