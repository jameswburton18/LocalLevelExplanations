from datasets import load_dataset
from src.utils import linearise_input
import re
from src.utils import linearise_input
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from transformers.trainer_callback import EarlyStoppingCallback
import yaml
import argparse
import wandb
import os

def find_mentioned_fts(data_row):
    matches = re.finditer(r"F\d+\b", data_row['narration'], re.MULTILINE)
    data_row['mentioned_fts'] = sorted(list(set([match.group() for match in matches])))
    options =  [f'F{i}' for i in range(1,51)]
    data_row['labels'] = torch.tensor([1.0 if option in data_row['mentioned_fts'] else 0.0 for option in options])
    return data_row

def convert_to_features_clf(batch, tokenizer, max_input_length=400, max_output_length=350):
    if type(batch['input'][0]) == list:
        input_encodings = [tokenizer(i, padding="max_length", truncation=True, max_length=max_input_length) for i in batch['input']]
        input_ids = [i['input_ids'] for i in input_encodings]
        attention_mask = [i['attention_mask'] for i in input_encodings]
    else:
        input_encodings = tokenizer(batch['input'], padding="max_length", truncation=True, max_length=max_input_length)
        input_ids = input_encodings['input_ids']
        attention_mask = input_encodings['attention_mask']
    encodings = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': batch['labels'],
    }
    return encodings

    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='default',
                    help='Name of config from the the multi_config.yaml file')
config_type = parser.parse_args().config

def main():
    # import yaml file
    with open('configs/which_exps_default.yaml') as f:
        args = yaml.safe_load(f)
    
    # Update default args with chosen config
    if config_type != 'default':
        with open('configs/which_exps_configs.yaml') as f:
            yaml_configs = yaml.safe_load_all(f)
            yaml_args = next(conf for conf in yaml_configs if conf['config'] == config_type)
        args.update(yaml_args)
        print(f'Updating with:\n{yaml_args}\n')
    print(f'\n{args}\n')
    
    model = AutoModelForSequenceClassification.from_pretrained(args['model_base'], num_labels=50,problem_type="multi_label_classification")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset = load_dataset("james-burton/aug-text-exps")
    dataset = dataset.map(
            lambda x: linearise_input(x, args['linearisation'], args['max_features']),
            ) 
    dataset = dataset.map(
        lambda x: find_mentioned_fts(x),
    )
    dataset = dataset.map(
        lambda x: convert_to_features_clf(x, tokenizer, args['max_input_len']), 
        batched=True, load_from_cache_file=False
        )

    # Fast dev run if want to run quickly and not save to wandb
    if args['fast_dev_run']:
        args['num_epochs'] = 1
        args['tags'].append("fast-dev-run")
        dataset['train'] = dataset['train'].select(range(50))
        dataset['test'] = dataset['test'].select(range(10))
        output_dir = os.path.join(args['output_root'], 'testing')
        print("\n######################    Running in fast dev mode    #######################\n")

    # If not, initialize wandb
    if not args['fast_dev_run']:
        wandb.init(
            project="Which exps mentioned classifier",
            tags=args['tags'],
            save_code=True, 
            config={'my_args/'+k: v for k, v in args.items()},
        )
        # wandb.log(args)
        os.environ['WANDB_LOG_MODEL'] = 'True'
        output_dir = os.path.join(args['output_root'], wandb.run.name)
        print(f'Results will be saved @: {output_dir}')



    # Initialise training arguments and trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args['num_epochs'],
        per_device_train_batch_size=args['batch_size'],
        per_device_eval_batch_size=args['batch_size'],
        logging_steps=args['logging_steps'],
        do_train=args['do_train'],
        do_predict=args['do_predict'],
        resume_from_checkpoint=args['resume_from_checkpoint'],
        report_to="wandb" if not args['fast_dev_run'] else "none",
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=args['save_total_limit'],
        load_best_model_at_end=True,     
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        callbacks=[EarlyStoppingCallback(args['early_stopping_patience'])] if args['early_stopping_patience'] > 0 else [],
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == '__main__':
    main()



