from transformers import AutoFeatureExtractor, AutoTokenizer
import torch
from typing import List
from dataclasses import dataclass
import re

@dataclass
class DataCollator:
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor
    max_question_len: int = 80
    max_answer_len: int = 80
    use_split_answers: bool = True
    
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
    
@dataclass
class TabularRepCollator:
    tokenizer: AutoTokenizer
    max_question_len: int = 80
    max_answer_len: int = 80
    use_split_answers: bool = True
    
    def join_inputs(self, tab_rep, question):
        return f'{tab_rep}<[TAB_QUES_SEP]>{question}'
    
    def tokenize_input(self, joined_inputs: List[str]):
        input_encoding = self.tokenizer(joined_inputs, 
                                           return_attention_mask=True,
                                           max_length=self.max_question_len,
                                           padding='max_length',
                                           add_special_tokens=True,
                                           truncation=True,
                                           return_tensors='pt')
        return {
            "input_ids": input_encoding['input_ids'].squeeze(),
            "attention_mask": input_encoding['attention_mask'].squeeze(),
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
            "labels": target_encoding['input_ids'].squeeze(),
            # "labels_attention_mask": target_encoding['attention_mask'].squeeze()
        }
    
    def preprocess_ques_type(self, ques_types: List[str]):
        ids = [QUES_TYPE_DICT[ques_type] for ques_type in ques_types]
        return {
            "ques_type_ids": torch.tensor(ids),
        } 
    
    def __call__(self, raw_batch_dict):
        output_dict = {
            **self.tokenize_input(
                self.join_inputs(raw_batch_dict['tab_rep'], raw_batch_dict['question'])
                if isinstance(raw_batch_dict, dict) else
                [self.join_inputs(i['tab_rep'], i['question']) for i in raw_batch_dict]
            ),
            **self.tokenize_answer(
                raw_batch_dict['answer']
                if isinstance(raw_batch_dict, dict) else
                [i['answer'] for i in raw_batch_dict]
            ),
            **self.preprocess_ques_type(
                raw_batch_dict['ques_type']
                if isinstance(raw_batch_dict, dict) else
                [i['ques_type'] for i in raw_batch_dict]
            ),
        }
        return output_dict

    

    

