from dataclasses import dataclass
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import math

class CustomTrainer(Trainer):
    vocab_size: int
    scale_loss: bool = False
    prev_seq_loss = []
    ce_loss = []
    total_loss =[]
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        comp_input =  inputs
        if 'prev_labels' in inputs.keys():
            comp_input = dict((key,value) for key, value in inputs.items() if key != 'prev_labels')
            comp_input['output_hidden_states'] = False
        outputs = model(comp_input)
        
        scl = math.log(self.vocab_size) if self.scale_loss else 1   
        loss = outputs.loss/scl
        return (loss, outputs) if return_outputs else loss

def getTrainingArguments(arg_dict):
    return TrainingArguments(
        **arg_dict,
        overwrite_output_dir=True,
        adafactor =True,
        # load_best_model_at_end=True,
        save_total_limit = 1,
        # disable_tqdm=True,
        )