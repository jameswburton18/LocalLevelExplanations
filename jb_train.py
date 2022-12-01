from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
from src.data_collator import linearise_input, convert_to_features
import wandb
import os

def main():
    args = {
        "fast_dev_run": True,
        "do_train": True,
        "do_predict": True,
        "tags": ["t5-base"],
        "batch_size": 1, # default PC: 32, ncc: 100
        
        "linearisation": "essel", # ['essel', 'ord_first', 'ft_first']
        "max_features": 20,
        "model_base": "t5-base",
        "output_root": 'models/t5-base/',
        "max_input_len": 300,
        "lr": 5e-5,
        "weight_decay": 0.3,
        "num_epochs": 20 ,
        "early_stopping_patience": 3, # -1 for no early stopping
        "grad_accumulation_steps": 1,
        "seed": 43,
        "logging_steps": 10,
        "lr_scheduler": "linear",
        "warmup_ratio": 0.1,
        "device": "cuda",
        "num_workers": 1,
        "resume_from_checkpoint": False, #'models/bart-base/iconic-darkness-1/checkpoint-13360',
        "eval_accumulation_steps": None
        }
    print(f'\n{args}\n')
    
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base', return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    dataset = load_dataset("src/dataset_builder.py", download_mode="force_redownload")
    dataset = dataset.map(
        lambda x: linearise_input(x, args['linearisation'], args['max_features']))
    # dataset = linearise_input(dataset['train'][99], args['linearisation'])
    dataset = dataset.map(
        lambda x: convert_to_features(x, tokenizer, args['max_input_len']), 
        batched=True
        )
    
    # Fast dev run if want to run quickly and not save to wandb
    if args['fast_dev_run']:
        args['num_epochs'] = 1
        args['tags'].append("fast-dev-run")
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

    # Metrics
    ####################################################################################################
        
    # Processing functions for calculating metrics
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    def postprocess_text(preds, labels):
        preds = [pred.strip().split(',') for pred in preds]
        labels = [[label.strip().split(',')] for label in labels]
        return preds, labels
    
    # Metrics
    # bleu_metric = evaluate.load("bleu")
    # ppl_metric = evaluate.load("perplexity")
    bleurt_metric = evaluate.load("bleurt-tiny-512")
    # met_metric = evaluate.load("meteor")

    
    def compute_metrics(eval_tuple):#): Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        # Fetching labels and decoding
        preds, grouped_labels = eval_tuple
        labels, ques_type_ids = grouped_labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # Compute metrics
        bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        ordered_accs = [[qtype, ordered_acc(pred, lab)] for pred, lab, qtype in zip(preds, labels, ques_type_ids)]
        unordered_accs = [[qtype, unordered_acc(pred, lab)]for pred, lab, qtype in zip(preds, labels, ques_type_ids)]
        
        qtype_list, acc_list = zip(*ordered_accs)
        _, unordered_acc_list = zip(*unordered_accs)
        
        result = {"bleu": bleu["score"]}
        result["ordered_acc"] = np.mean(acc_list)
        result["unordered_acc"] = np.mean(unordered_acc_list)
        
        # Computing metrics per question type
        unique_qtypes = list(set(qtype_list))
        for i in unique_qtypes:
            result[f'ordered_acc_{ID_TO_QTYPE_DICT[str(i)]}'] = np.mean(
                [acc_list[j] for j in range(len(acc_list)) if qtype_list[j] == unique_qtypes[i]])
        for i in unique_qtypes:
            result[f'unordered_acc_{ID_TO_QTYPE_DICT[str(i)]}'] = np.mean(
                [unordered_acc_list[j] for j in range(len(unordered_acc_list)) if qtype_list[j] == unique_qtypes[i]])
        
        
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     num_train_epochs=args['num_epochs'],
    #     per_device_train_batch_size=args['batch_size'],
    #     per_device_eval_batch_size=args['batch_size'],
    #     logging_steps=args['logging_steps'],
    #     learning_rate=args['lr'],
    #     weight_decay=args['weight_decay'],
    #     gradient_accumulation_steps=args['grad_accumulation_steps'],
    #     warmup_ratio=args['warmup_ratio'],
    #     lr_scheduler_type=args['lr_scheduler'],
    #     dataloader_num_workers=args['num_workers'], 
    #     do_train=args['do_train'],
    #     do_predict=args['do_predict'],
    #     resume_from_checkpoint=args['resume_from_checkpoint'],
    #     eval_accumulation_steps=args['eval_accumulation_steps'],
    #     report_to="wandb" if not args['fast_dev_run'] else "none",
    #     evaluation_strategy='epoch',
    #     save_strategy='epoch',
    #     save_total_limit=3,
    #     load_best_model_at_end=True,   
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     tokenizer=tokenizer,
    #     train_dataset=dataset['train'],
    #     eval_dataset=dataset['validation'],
    #     compute_metrics=compute_metrics,
    # )

    # trainer.train()
    
if __name__ == '__main__':
    main()
