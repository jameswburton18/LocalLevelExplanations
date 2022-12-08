from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, pipeline
# import lmap
from datasets import load_dataset
from evaluate import load
from src.data_collator import linearise_input, convert_to_features
import wandb
import os
import numpy as np
import torch.nn.functional as F
import torch
from tqdm import tqdm
import json


def main():
    args = {
        "fast_dev_run": True,
        "do_train": False,
        "do_predict": True,
        "tags": ["t5-base"],
        "batch_size": 4, # default PC: 32, ncc: 100
        
        "linearisation": "essel", # ['essel', 'ord_first', 'ft_first']
        "max_features": 50,
        "model_base": "t5-base",
        "output_root": 'models/t5-base/',
        "max_input_len": 500,
        "lr": 5e-5,
        "weight_decay": 0.3,
        "num_epochs": 50 ,
        "early_stopping_patience": 3, # -1 for no early stopping
        "grad_accumulation_steps": 1,
        "seed": 43,
        "logging_steps": 10,
        "lr_scheduler": "linear",
        "warmup_ratio": 0.1,
        "device": "cuda",
        "num_workers": 1,
        "resume_from_checkpoint": False, #'models/bart-base/iconic-darkness-1/checkpoint-13360',
        "eval_accumulation_steps": None,
        
        "num_beams": 4,
        "repetition_penalty": 3.5,
        "length_penalty": 1.5,
        "max_output_len": 250,
        "predict_batch_size": 4,
        }
    print(f'\n{args}\n')
    
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base', return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    dataset = load_dataset("james-burton/textual-explanations")
    # dataset.push_to_hub('james-burton/textual-explanations')
    dataset = dataset.map(
        lambda x: linearise_input(x, args['linearisation'], args['max_features']),
        load_from_cache_file=False)
    # dataset = linearise_input(dataset['train'][99], args['linearisation'])
    dataset = dataset.map(
        lambda x: convert_to_features(x, tokenizer, args['max_input_len']), 
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
            project="Local Level Explanations",
            tags=args['tags'],
            save_code=True, 
            config={'my_args/'+k: v for k, v in args.items()},
        )
        # wandb.log(args)
        os.environ['WANDB_LOG_MODEL'] = 'True'
        output_dir = os.path.join(args['output_root'], wandb.run.name)
        print(f'Results will be saved @: {output_dir}')

    # Save args to json file
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args, f)

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
        save_total_limit=3,
        load_best_model_at_end=True,   
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
    )

    if args['do_train']:
        print('Training...')
        trainer.train()
        print('Training complete')
    # Predict on the test set
    if args['do_predict']:
        input_ids = torch.tensor(dataset['test']['input_ids']).to(model.device)
        attention_mask = torch.tensor(dataset['test']['attention_mask']).to(model.device)
        all_preds = []
        for i in tqdm(range(0,input_ids.shape[0],args['predict_batch_size'])):
            sample_outputs = model.generate(input_ids=input_ids[i:i+4],
                                                 attention_mask=attention_mask[i:i+args['predict_batch_size']],
                                                 num_beams=args['num_beams'],
                                                 repetition_penalty=args["repetition_penalty"],
                                                 length_penalty=args["length_penalty"],
                                                 max_length=args['max_output_len'],
                                                 no_repeat_ngram_size=2,
                                                 num_return_sequences=1,
                                                 do_sample=True,
                                                 early_stopping=True,
                                                 use_cache=False)
            preds = tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
            all_preds.extend(preds)
        
        
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
        # Log the results
        results = {'bleurt': np.mean(bleurt_results['scores']),
                   'bleu': bleu_results['bleu'],
                   'meteor': meteor_results['meteor']}
        wandb.log(results)
        
        # Save the predictions
        readable_predictions = ['.\n'.join(pred.split('. ')) for pred in all_preds]
        print(f'Saving predictions to {output_dir}')
        with open(os.path.join(output_dir, 'test_predictions_readable.txt'), 'w') as f:
            f.write('\n\n'.join(readable_predictions))
        with open(os.path.join(output_dir, 'test_predictions.txt'), 'w') as f:
            f.write('\n'.join(all_preds))
        with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
            f.write(str(results))
            
            
    print('Predictions complete')
    
    
if __name__ == '__main__':
    main()
