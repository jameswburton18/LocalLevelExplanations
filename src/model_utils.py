import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (BartForConditionalGeneration, EncoderDecoderModel, GPT2Config, GPT2LMHeadModel,
                          T5ForConditionalGeneration)


class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )

        k_t = self.key(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len,
                                  self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)

        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)

        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out) + x

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class NarratorAttributor(nn.Module):
    def __init__(self, vocab_size, modelbase,):
        super(NarratorAttributor, self).__init__()
        #print(modelbase)
        self.modelbase = modelbase
        if 't5' in self.modelbase:
            self.generator = T5ForConditionalGeneration.from_pretrained(
                self.modelbase,)
            self.generator.resize_token_embeddings(vocab_size)

        elif 'bart' in self.modelbase:
            self.generator = BartForConditionalGeneration.from_pretrained(
                self.modelbase,)

            self.generator.resize_token_embeddings(vocab_size)
        elif 'roberta' in self.modelbase:
            self.generator = EncoderDecoderModel.from_encoder_decoder_pretrained(self.modelbase, self.modelbase, tie_encoder_decoder=True)
            self.generator.encoder.resize_token_embeddings(vocab_size)
            self.generator.decoder.resize_token_embedding(vocab_size)
        elif "gpt" in self.modelbase:
            configuration = GPT2Config.from_pretrained(modelbase, output_hidden_states=False)
            # instantiate the model
            self.generator = GPT2LMHeadModel.from_pretrained(modelbase, config=configuration)
            self.generator.resize_token_embeddings(vocab_size)

        

        self.config = self.generator.config

    def forward(self, input_ids):
        input_ids['output_hidden_states'] = False
        comp_input = dict((key, value)
                          for key, value in input_ids.items() if key != 'input_marker')
        generator_output = self.generator(**comp_input)
        return generator_output


def get_basic_model(dataset):
    def getModel():
        return NarratorAttributor(len(dataset.tokenizer_), dataset.modelbase,)
    return getModel

def buildRoberta(dataset):
    roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained(dataset.modelbase, dataset.modelbase, tie_encoder_decoder=True)
    roberta_shared.decoder.resize_token_embeddings(len(dataset.tokenizer_))
    roberta_shared.encoder.resize_token_embeddings(len(dataset.tokenizer_))
    roberta_shared.config.decoder_start_token_id = dataset.tokenizer_.bos_token_id                                             
    roberta_shared.config.eos_token_id = dataset.tokenizer_.eos_token_id
    roberta_shared.config.pad_token_id = dataset.tokenizer_.pad_token_id
    roberta_shared.config.max_length = 600



