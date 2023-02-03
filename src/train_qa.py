from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, 
    TrainingArguments, Trainer
)
from datasets import load_dataset
from evaluate import load
from src.utils import (
    linearise_input, convert_to_features, label_qs,
    simplify_narr_question, nums_to_names, form_qa_input_output
)
import wandb
import os
import numpy as np
import torch.nn.functional as F
import torch
from tqdm import tqdm
import json
import yaml
import argparse
from transformers.trainer_callback import EarlyStoppingCallback

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='essel_test',
                    help='Name of config from the the multi_config.yaml file')
config_type = parser.parse_args().config

def main():
    # Import yaml file
    with open('configs/qa_default.yaml') as f:
        args = yaml.safe_load(f)
    
    # Update default args with chosen config
    if config_type != 'default':
        with open('configs/qa_configs.yaml') as f:
            yaml_configs = yaml.safe_load_all(f)
            yaml_args = next(
                conf for conf in yaml_configs if conf['config'] == config_type)
        args.update(yaml_args)
        print(f'Updating with:\n{yaml_args}\n')
    print(f'\n{args}\n')
    
    model = AutoModelForSeq2SeqLM.from_pretrained(args['model_base'], return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(args['model_base'])

    dataset = load_dataset("james-burton/text-exp-qa") if args['version'] == 'normal' \
        else load_dataset("james-burton/text-exp-qa-hard")
    dataset = dataset.map(
        lambda x: form_qa_input_output(x, args['linearisation'], args['max_features']),
        load_from_cache_file=True #False
        )
    dataset = dataset.map(
        lambda x: convert_to_features(x, tokenizer, args['max_input_len'], args['max_output_len']), 
        batch_size=args['batch_size'],load_from_cache_file=True #False 
        )
    
    # Fast dev run if want to run quickly and not save to wandb
    if args['fast_dev_run']:
        args['num_epochs'] = 1
        args['tags'].append("fast-dev-run")
        dataset['train'] = dataset['train'].select(range(50))
        dataset['test'] = dataset['test'].select(range(50))
        output_dir = os.path.join(args['output_root'], 'testing')
        print("\n######################    Running in fast dev mode    #######################\n")

    # If not, initialize wandb
    if not args['fast_dev_run']:
        wandb.init(
            project="Local Level Explanations QA",
            tags=args['tags'],
            save_code=True, 
            config={'my_args/'+k: v for k, v in args.items()},
        )
        # wandb.log(args)
        os.environ['WANDB_LOG_MODEL'] = 'True'
        output_dir = os.path.join(args['output_root'], wandb.run.name)
        print(f'Results will be saved @: {output_dir}')

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save args to json file
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args, f)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        yaml.dump(args, f)

    # Set up training arguments

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args['num_epochs'],
        per_device_train_batch_size=args['batch_size'],
        per_device_eval_batch_size=args['batch_size'],
        logging_steps=args['logging_steps'],
        learning_rate=args['lr'],
        weight_decay=args['weight_decay'],
        gradient_accumulation_steps=args['grad_accumulation_steps'],
        warmup_ratio=args['warmup_ratio'],
        lr_scheduler_type=args['lr_scheduler'],
        dataloader_num_workers=args['num_workers'], 
        do_train=args['do_train'],
        do_predict=args['do_predict'],
        resume_from_checkpoint=args['resume_from_checkpoint'],
        eval_accumulation_steps=args['eval_accumulation_steps'],
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
        callbacks=[EarlyStoppingCallback(
            args['early_stopping_patience'])] if args['early_stopping_patience'] > 0 else []
    )

    if args['do_train']:
        print('Training...')
        trainer.train()
        print('Training complete')
    
    if args['do_predict']:
        print("***** Running Prediction *****")
        input_ids = torch.tensor(dataset['test']['input_ids']).to(model.device)
        attention_mask = torch.tensor(dataset['test']['attention_mask']).to(model.device)
        all_preds = []
        for i in tqdm(range(0,input_ids.shape[0],args['predict_batch_size'])):
            output_tokens = model.generate(input_ids=input_ids[i:i+args['predict_batch_size']],
                                                 attention_mask=attention_mask[i:i+args['predict_batch_size']],
                                                 num_beams=args['num_beams'],
                                                 repetition_penalty=args["repetition_penalty"],
                                                 length_penalty=args["length_penalty"],
                                                 max_length=args['max_output_len'],
                                                 no_repeat_ngram_size=2,
                                                 num_return_sequences=1,
                                                 do_sample=True,
                                                 early_stopping=True,
                                                 use_cache=True)
            preds = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
            all_preds.extend(preds)
        
        # Save the predictions
        readable_predictions = ['.\n'.join(pred.split('. ')) for pred in all_preds]
        print(f'Saving predictions to {output_dir}')
        with open(os.path.join(output_dir, 'test_predictions_readable.txt'), 'w') as f:
            f.write('\n\n'.join([f'Q: {q}\nA: {a}\nPred: {p}' for q,a,p in 
                                 zip(dataset['test']['question'], 
                                     dataset['test']['answer'], 
                                     readable_predictions
                                     )
                                 ]))
        with open(os.path.join(output_dir, 'test_predictions.txt'), 'w') as f:
            f.write('\n'.join(all_preds))
            
        # Evaluate the predictions
        bleurt = load('bleurt',checkpoint="bleurt-base-512")
        bleurt_results = bleurt.compute(predictions=all_preds, 
                                        references=dataset['test']['narration'])
        bleu = load('bleu')
        bleu_results = bleu.compute(predictions=all_preds, 
                                    references=[[r] for r in dataset['test']['narration']])
        meteor = load('meteor')
        meteor_results = meteor.compute(predictions=all_preds, 
                                        references=[[r] for r in dataset['test']['narration']])
        
        pred_eq_ans = [a.strip()==p.strip() for a, p in zip(dataset['test']['narration'], all_preds)]
        q_ids = dataset['test']['question_id']
        Q_ID_DICT = (
            {
                0: "val_of_F",
                1: "FA_minus_FB",
                2: "xth_most_important",
                3: "top_x_pos",
                4: "top_x_neg",
            }
            if args["version"] == "normal"
            else {
                0: "of_top_pos",
                1: "of_top_neg",
                2: "of_these_support",
                3: "of_these_against",
                4: "fts_>_x",
                5: "val_of_F",
                6: "least_important_fts",
                7: "val_of_C",
            }
        )
        # check accuracy per question id
        q_id_acc = {}
        for qid, eq in zip(q_ids, pred_eq_ans):
            qid = Q_ID_DICT[qid]
            if qid not in q_id_acc:
                q_id_acc[qid] = []
            q_id_acc[qid].append(eq)
        q_id_acc = {k+'_acc':np.mean(v) for k,v in q_id_acc.items()}
        
        # Log the results
        results = {'bleurt': np.mean(bleurt_results['scores']),
                   'bleu': bleu_results['bleu'],
                   'meteor': meteor_results['meteor'],
                   'acc': np.mean(pred_eq_ans),
                   **q_id_acc
                   }
        with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
            f.write(str(results))
        if not args['fast_dev_run']:
            wandb.log(results)
            
            
    print('Predictions complete')
    

    
    
if __name__ == '__main__':
    main()
