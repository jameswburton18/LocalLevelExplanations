import copy
import functools
import glob
import gzip
from dataclasses import dataclass, field
from functools import partial
import re
from typing import Dict, List, Optional
from xmlrpc.client import boolean

import numpy as np
import torch
from transformers import (BartTokenizer, BartTokenizerFast, DataCollator,
                          GPT2TokenizerFast, PreTrainedModel,
                          PreTrainedTokenizer, RobertaTokenizerFast,
                          T5Tokenizer, T5TokenizerFast)
from nltk.tokenize import sent_tokenize
# handling all operations about the dataset


def setupTokenizer(modelbase):

    additional_vocab = new_tokens = ['[EON]', '[NLS]', '[N9S]',
                                     '[N10S]', '[PFS]', '[NFS]', '[IFS]',
                                     '[N4S]', '[N5S]', '[N8S]', '[N6S]',
                                     '[N7S]', '[N1S]', '[N2S]','[CON]',
                                     '[N0S]', '[N3S]']  # +['predictionlabel', 'predictionrankA','predictionrankB', 'predictionrankC', 'predictionrankD', 'predictionrankE']
    specials = ['<positives>', '<neutrals>', '<negatives>',
                '</positives>', '</negatives>', '</neutrals>',
                '<mentions>', "</mentions>"]
    special_tokens = ['&&',
                      '<|>',
                      '<full_explain>',
                      '<full_narration>',
                      "<next_sequence>",
                      "</next_sequence>",
                      '<explain>']+specials
    if 't5' in modelbase:
        tokenizer_ = T5TokenizerFast.from_pretrained(modelbase,)
    elif 'bart' in modelbase:
        tokenizer_ = BartTokenizerFast.from_pretrained(modelbase)
    elif 'robert' in modelbase:
        tokenizer_ = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif 'gpt' in modelbase:
        tokenizer_ = tokenizer = GPT2TokenizerFast.from_pretrained(modelbase,
                                                                   bos_token='<|startoftext|>',
                                                                   eos_token='<|endoftext|>',
                                                                   pad_token='<|pad|>')
        tokenizer_.add_special_tokens({'sep_token': '<|section-sep|>',  # 'eos_token':'<eos>',
                                       'additional_special_tokens': special_tokens+['<nextsection>']})

    if 't5' in modelbase:
        tokenizer_.add_special_tokens({'sep_token': '<|section-sep|>',
                                       'additional_special_tokens': special_tokens})
    elif 'bart' in modelbase:
        tokenizer_.add_special_tokens(
            {'additional_special_tokens': special_tokens+['<|section-sep|>']})
    elif 'robert' in modelbase:
        tokenizer_.add_special_tokens(
            {'additional_special_tokens': special_tokens+['<|section-sep|>']})

    tokenizer_.add_tokens(additional_vocab)
    return tokenizer_


@dataclass
class Features:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    decoder_attention_mask: Optional[List]


class RDFDataSetForLinearisedStructured(torch.utils.data.Dataset):
    def __init__(self, tokenizer,
                 data_pack,
                 modelbase,
                 feature_token_len=65,
                 preamble_choice=2,
                 nb_feature_attributions=32,
                 max_preamble_len=280,
                 max_narration_len=280,
                 step_continue=False,
                 ):
        super().__init__()
        self.modelbase = modelbase
        self.tokenizer = tokenizer
        self.data_pack = data_pack
        self.max_preamble_len = max_preamble_len
        self.max_narration_len = max_narration_len
        self.preamble_choice = preamble_choice
        self.nb_feature_attributions = nb_feature_attributions
        self.step_continue = step_continue
        self.preamble_tokenizer = lambda x: self.tokenizer.encode_plus(x,
                                                                       return_attention_mask=True,
                                                                       add_special_tokens=False,
                                                                       truncation=True,
                                                                       return_tensors='pt')

        self.feature_tokenizer = lambda x: self.tokenizer.encode_plus(x,
                                                                      return_attention_mask=True,
                                                                      truncation=True,
                                                                      is_split_into_words=True,
                                                                      return_tensors='pt')

        self.target_tokenizer = lambda x: self.tokenizer.encode_plus(x,
                                                                     truncation=True,
                                                                     return_attention_mask=True,
                                                                     add_special_tokens=True,
                                                                     return_tensors='pt'
                                                                     )

    def processTableInfoStepContinue(self, data_row,is_inference=False):

        attr = data_row['attributions']
        data_target = data_row['outputs'] if 'outputs' in data_row.keys(
        ) else data_row['output']

        pr_c = self.preamble_choice  # 2
        prem_key = f'new_preamble_{pr_c}'

        prev_sentence = data_row['prev_seq'].replace('<prem>', '')
        if '<full_narration>' in prev_sentence:
            if not is_inference:
                prev_sentence = ' <full_narration> ' #+ data_target.replace('" pred_label"', '"pred_label"').strip()
            else:
                prev_sentence = ' <full_narration> ' 


        data_preamble = (data_row[prem_key]+' [N0S] '+ prev_sentence).replace('" pred_label"', '"pred_label"')

        preamble_encoding = self.preamble_tokenizer(data_preamble.strip())
        preamble_tokens = preamble_encoding['input_ids']
        preamble_attention_mask = preamble_encoding['attention_mask']

        # <next_sequence>
        next_sequence = data_row['prev_seq'].replace('<prem>', '').replace('<full_narration>', '').replace(
            '" pred_label"', '"pred_label"') + self.tokenizer.pad_token+ '<next_sequence>' + self.tokenizer.pad_token + data_target.replace('" pred_label"', '"pred_label"').strip() + ' </next_sequence> '
        
        target_encoding = self.target_tokenizer(next_sequence )
        previous = self.target_tokenizer(data_row['prev_seq'].strip())
        labels = target_encoding['input_ids']
        prev_labels = previous['input_ids']

        if 't5' in self.modelbase:
            labels[labels == 0] = -100
            prev_labels[prev_labels == 0] = -100
        elif 'bart' in self.modelbase or 'robert' in self.modelbase:
            labels[labels == 1] = -100
            prev_labels[prev_labels == 1] = -100
        # elif 'robert' in modelbase:

        return Features(
            input_ids=preamble_tokens.flatten(),
            attention_mask=preamble_attention_mask.flatten(),
            labels=labels.flatten(),
            decoder_attention_mask=target_encoding['attention_mask'].flatten(),
        )

    def processTableInfo(self, data_row):

        attr = data_row['attributions']
        data_target = data_row['outputs'] if 'outputs' in data_row.keys(
        ) else data_row['output']

        pr_c = self.preamble_choice  # 2
        prem_key = f'new_preamble_{pr_c}'
        data_preamble = (data_row[prem_key]+' [N0S] '+data_row['prev_seq'].replace(
            '<prem>', '')).replace('" pred_label"', '"pred_label"')

        preamble_encoding = self.preamble_tokenizer(data_preamble.strip())
        preamble_tokens = preamble_encoding['input_ids']
        preamble_attention_mask = preamble_encoding['attention_mask']

        target_encoding = self.target_tokenizer(
            data_target.replace('" pred_label"', '"pred_label"').strip())
        previous = self.target_tokenizer(data_row['prev_seq'].strip())
        labels = target_encoding['input_ids']
        prev_labels = previous['input_ids']

        if 't5' in self.modelbase:
            labels[labels == 0] = -100
            prev_labels[prev_labels == 0] = -100
        elif 'bart' in self.modelbase or 'robert' in self.modelbase:
            labels[labels == 1] = -100
            prev_labels[prev_labels == 1] = -100
        # elif 'robert' in modelbase:

        return Features(
            input_ids=preamble_tokens.flatten(),
            attention_mask=preamble_attention_mask.flatten(),
            labels=labels.flatten(),
            decoder_attention_mask=target_encoding['attention_mask'].flatten(),
        )

    def processTableInfoGPT(self, data_row):
        def cleanOutput(passages):
            placeholders = {t: ' ' for t in ['[EON]', '[N9S]',
                                             '[N10S]',
                                             '[N4S]', '[N5S]', '[N8S]', '[N6S]',
                                             '[N7S]', '[N1S]', '[N2S]',
                                             '[N0S]', '[N3S]']}

            passages = copy.deepcopy(passages)
            dd = [functools.reduce(lambda a, kv: a.replace(*kv), placeholders.items(),
                                   re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [passages]][0]
            dd = ' '.join(sent_tokenize(dd))
            return dd
        data_target = data_row['outputs'] if 'outputs' in data_row.keys(
        ) else data_row['output']

        pr_c = self.preamble_choice  # 2
        prem_key = f'new_preamble_{pr_c}'
        data_preamble = (data_row[prem_key]+' [N0S] '+data_row['prev_seq'].replace(
            '<prem>', '')).replace('" pred_label"', '"pred_label"') +\
            data_target.replace(
                '" pred_label"', '"pred_label"').strip() + ' <|endoftext|>'
        data_preamble = cleanOutput(data_preamble)
        preamble_encoding = target_encoding = self.target_tokenizer(
            data_preamble.strip())
        preamble_tokens = preamble_encoding['input_ids']
        preamble_attention_mask = preamble_encoding['attention_mask']
        labels = target_encoding['input_ids']

        return Features(
            input_ids=preamble_tokens.flatten(),
            attention_mask=preamble_attention_mask.flatten(),
            labels=labels.flatten(),
            decoder_attention_mask=None,
        )

    def __len__(self,):
        return len(self.data_pack)

    def __getitem__(self, idx):
        data_row = self.data_pack[idx]
        if 'gpt' not in self.modelbase:
            if not self.step_continue:
                return self.processTableInfo(data_row)
            else:
                return self.processTableInfoStepContinue(data_row)
        else:
            return self.processTableInfoGPT(data_row)


def pad_seq(seq: List[int], max_batch_len: int, pad_value: int, verbose=False) -> List[int]:
    if len(seq) > max_batch_len:
        seq = seq.to(torch.long).unsqueeze(0)[:, :max_batch_len]
        return seq
    pads = torch.from_numpy(np.array([pad_value]*(max_batch_len - len(seq))))
    out = torch.concat([seq, pads], -1).to(torch.long).unsqueeze(0)
    return out


@dataclass
class SmartCollator:
    pad_token_id: int
    label_pad_token_id: int = -100
    is_gpt: boolean = False

    def __call__(self, batch: List[Features]) -> Dict[str, torch.Tensor]:
        batch_inputs = list()
        batch_attention_masks = list()
        decoder_attention_mask = list()
        labels = list()
        prev_labels = list()
        decoder_attention_mask = list()
        input_marker = list()
        max_size = max([len(ex.input_ids) for ex in batch])
        max_size_output = max([len(ex.labels) for ex in batch])

        for item in batch:
            batch_inputs += [pad_seq(item.input_ids,
                                     max_size, self.pad_token_id)]
            batch_attention_masks += [
                pad_seq(item.attention_mask, max_size, 0)]

            if not self.is_gpt:
                decoder_attention_mask += [
                    pad_seq(item.decoder_attention_mask, max_size_output, 0)]
            labels += [pad_seq(item.labels, max_size_output,
                               self.label_pad_token_id)]
        if not self.is_gpt:
            return dict(
                input_ids=torch.concat(batch_inputs, 0),
                attention_mask=torch.concat(batch_attention_masks, 0),
                labels=torch.concat(labels, 0),
                decoder_attention_mask=torch.concat(decoder_attention_mask, 0),)
        else:
            return dict(
                input_ids=torch.concat(batch_inputs, 0),
                attention_mask=torch.concat(batch_attention_masks, 0),
                labels=torch.concat(labels, 0),)
