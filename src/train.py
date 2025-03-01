from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from evaluate import load
from src.utils import (
    linearise_input,
    convert_to_features,
    label_qs,
    simplify_narr_question,
    nums_to_names,
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
parser.add_argument(
    "--config",
    type=str,
    default="essel_test",
    help="Name of config from the the multi_config.yaml file",
)
config_type = parser.parse_args().config


def main():
    # Import yaml file
    with open("configs/train_default.yaml") as f:
        args = yaml.safe_load(f)

    # Update default args with chosen config
    if config_type != "default":
        with open("configs/train_configs.yaml") as f:
            yaml_configs = yaml.safe_load_all(f)
            yaml_args = next(
                conf for conf in yaml_configs if conf["config"] == config_type
            )
        args.update(yaml_args)
        print(f"Updating with:\n{yaml_args}\n")
    print(f"\n{args}\n")

    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args["model_base"], return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(args["model_base"])

    # Dataset
    if args["augmented_ds"]:
        if args["big_test_set"]:
            # ds_path = 'james-burton/aug-text-exps-702010'
            ds_path = "james-burton/aug-text-exps-v3"
        else:
            ds_path = "james-burton/aug-text-exps"
    else:
        if args["big_test_set"]:
            ds_path = "james-burton/textual-explanations-702010"
        else:
            ds_path = "james-burton/textual-explanations"
    dataset = load_dataset(ds_path)

    if args["simplify_narr_qs"]:
        dataset = dataset.map(
            lambda x: simplify_narr_question(label_qs(x)), load_from_cache_file=False
        )

    # Form the linearised or stepwise (and linearised) input
    dataset = dataset.map(
        lambda x: linearise_input(x, args["linearisation"], args["max_features"]),
        load_from_cache_file=False,
    )

    # Convert to tokens
    dataset = dataset.map(
        lambda x: convert_to_features(x, tokenizer, args["max_input_len"]),
        batched=True,
        load_from_cache_file=False,
    )

    # Fast dev run if want to run quickly and not save to wandb
    if args["fast_dev_run"]:
        args["num_epochs"] = 1
        args["tags"].append("fast-dev-run")
        dataset["train"] = dataset["train"].select(range(50))
        dataset["test"] = dataset["test"].select(range(10))
        output_dir = os.path.join(args["output_root"], "testing")
        print(
            "\n######################    Running in fast dev mode    #######################\n"
        )

    # If not, initialize wandb
    else:
        wandb.init(
            project="Local Level Explanations",
            tags=args["tags"],
            save_code=True,
            config={"my_args/" + k: v for k, v in args.items()},
        )
        os.environ["WANDB_LOG_MODEL"] = "True"
        output_dir = os.path.join(args["output_root"], wandb.run.name)
        print(f"Results will be saved @: {output_dir}")

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save args file
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(args, f)
    with open(os.path.join(output_dir, "args.yaml"), "w") as f:
        yaml.dump(args, f)

    # Initialise training arguments and trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args["num_epochs"],
        per_device_train_batch_size=args["batch_size"],
        per_device_eval_batch_size=args["batch_size"],
        logging_steps=args["logging_steps"],
        learning_rate=args["lr"],
        weight_decay=args["weight_decay"],
        gradient_accumulation_steps=args["grad_accumulation_steps"],
        warmup_ratio=args["warmup_ratio"],
        lr_scheduler_type=args["lr_scheduler"],
        dataloader_num_workers=args["num_workers"],
        do_train=args["do_train"],
        do_predict=args["do_predict"],
        resume_from_checkpoint=args["resume_from_checkpoint"],
        eval_accumulation_steps=args["eval_accumulation_steps"],
        report_to="wandb" if not args["fast_dev_run"] else "none",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args["save_total_limit"],
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[EarlyStoppingCallback(args["early_stopping_patience"])]
        if args["early_stopping_patience"] > 0
        else [],
    )

    # Train model
    if args["do_train"]:
        print("Training...")
        trainer.train()
        print("Training complete")
    # Predict on the test set
    if args["do_predict"]:
        print("***** Running Prediction *****")
        input_ids = torch.tensor(dataset["test"]["input_ids"]).to(model.device)
        attention_mask = torch.tensor(dataset["test"]["attention_mask"]).to(
            model.device
        )
        all_preds = []
        for i in tqdm(range(0, input_ids.shape[0], args["predict_batch_size"])):
            sample_outputs = model.generate(
                input_ids=input_ids[i : i + args["predict_batch_size"]],
                attention_mask=attention_mask[i : i + args["predict_batch_size"]],
                num_beams=args["num_beams"],
                repetition_penalty=args["repetition_penalty"],
                length_penalty=args["length_penalty"],
                max_length=args["max_output_len"],
                no_repeat_ngram_size=2,
                num_return_sequences=1,
                do_sample=True,
                early_stopping=True,
                use_cache=True,
            )
            preds = tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
            all_preds.extend(preds)

        all_preds_w_names = [
            nums_to_names(pred, eval(c2s), eval(f2s))
            for pred, c2s, f2s in zip(
                all_preds, dataset["test"]["class2name"], dataset["test"]["ft_num2name"]
            )
        ]
        narrs_w_names = [
            nums_to_names(narr, eval(c2s), eval(f2s))
            for narr, c2s, f2s in zip(
                dataset["test"]["narration"],
                dataset["test"]["class2name"],
                dataset["test"]["ft_num2name"],
            )
        ]
        input_w_names = [
            nums_to_names(inp, eval(c2s), eval(f2s))
            for inp, c2s, f2s in zip(
                dataset["test"]["input"],
                dataset["test"]["class2name"],
                dataset["test"]["ft_num2name"],
            )
        ]

        # Evaluate the predictions
        bleurt = load("bleurt", checkpoint="bleurt-base-512")
        bleu = load("bleu")
        meteor = load("meteor")

        results = {}
        for names in ["w_names", "w/o_names"]:
            if names == "w_names":
                preds = all_preds_w_names
                refs = narrs_w_names
                suffix = "_w_names"
            else:
                preds = all_preds
                refs = dataset["test"]["narration"]
                suffix = ""

            bleurt_results = bleurt.compute(predictions=preds, references=refs)
            bleu_results = bleu.compute(
                predictions=preds, references=[[r] for r in refs]
            )
            meteor_results = meteor.compute(
                predictions=preds, references=[[r] for r in refs]
            )
            # Log the results
            results.update(
                {
                    f"bleurt{suffix}": np.mean(bleurt_results["scores"]),
                    f"bleu{suffix}": bleu_results["bleu"],
                    f"meteor{suffix}": meteor_results["meteor"],
                }
            )

        """
        if args['simplify_narr_qs']:
            # Results by narr_q_label_group
            grp_lens = {}
            results = {}
            for grp in set(dataset['test']['narr_q_label_group']):
                # Select only the data predictions for this group
                grp_data = dataset['test'].filter(
                    lambda x: x['narr_q_label_group'] == grp)['narration']
                grp_all_preds = [p for x, p in zip(
                    [lab_grp == grp for lab_grp in dataset['test']['narr_q_label_group']], all_preds) if x]
                
                grp_bleurt_results = bleurt.compute(predictions=grp_all_preds,
                                                    references=grp_data)
                grp_bleu_results = bleu.compute(predictions=grp_all_preds,
                                                references=[[r] for r in grp_data])
                grp_meteor_results = meteor.compute(predictions=grp_all_preds,
                                                    references=[[r] for r in grp_data])
                grp_lens[grp] = len(grp_data)
                
                results[f'bleurt_{grp}'] = np.mean(grp_bleurt_results['scores'])
                results[f'bleu_{grp}'] = grp_bleu_results['bleu']
                results[f'meteor_{grp}'] = grp_meteor_results['meteor']
            
            results['bleurt'] = sum(
                [results[f'bleurt_{grp}'] * grp_lens[grp] for grp in grp_lens]) / sum(grp_lens.values())
            results['bleu'] = sum(
                [results[f'bleu_{grp}'] * grp_lens[grp] for grp in grp_lens]) / sum(grp_lens.values())
            results['meteor'] = sum(
                [results[f'meteor_{grp}'] * grp_lens[grp] for grp in grp_lens]) / sum(grp_lens.values())
        
        else:
            bleurt_results = bleurt.compute(predictions=all_preds,
                                            references=dataset['test']['narration'])
            bleu_results = bleu.compute(predictions=all_preds,
                                        references=[[r] for r in dataset['test']['narration']])
            meteor_results = meteor.compute(predictions=all_preds,
                                            references=[[r] for r in dataset['test']['narration']])
            # Log the results
            results = {'bleurt': np.mean(bleurt_results['scores']),
                    'bleu': bleu_results['bleu'],
                    'meteor': meteor_results['meteor']}
        """

        # Save the predictions
        readable_predictions = [".\n".join(pred.split(". ")) for pred in all_preds]
        readable_preds_w_names = [
            ".\n".join(pred.split(". ")) for pred in all_preds_w_names
        ]
        print(f"Saving predictions to {output_dir}")
        with open(os.path.join(output_dir, "test_predictions_readable.txt"), "w") as f:
            for ans, input, pred in zip(
                dataset["test"]["narration"],
                dataset["test"]["input"],
                readable_predictions,
            ):
                f.write(f"INPUT: {input} \n\n")
                f.write(f"GOLD: {ans} \n\n")
                f.write(f"OUTPUT: {pred} \n\n")
        with open(
            os.path.join(output_dir, "test_predictions_readable_w_names.txt"), "w"
        ) as f:
            for ans, input, pred in zip(
                narrs_w_names, input_w_names, readable_preds_w_names
            ):
                f.write(f"INPUT: {input} \n\n")
                f.write(f"GOLD: {ans} \n\n")
                f.write(f"OUTPUT: {pred} \n\n")
        with open(os.path.join(output_dir, "test_predictions.txt"), "w") as f:
            f.write("\n".join(all_preds))
        with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
            f.write(str(results))
        if not args["fast_dev_run"]:
            wandb.log(results)

    print("Predictions complete")


if __name__ == "__main__":
    main()
