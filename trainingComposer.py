from ast import arg
import json
import os
from argparse import ArgumentParser

from pytorch_lightning import seed_everything
import torch
from transformers import EarlyStoppingCallback

from src.datasetComposer import DatasetBuilder, composed_train_path, composed_test_path, compactComposer, test_path,train_path,test_path
from src.inference_utils import InferenceGenerator
from src.datasetHandlers import SmartCollator
from src.model_utils import get_basic_model
from src.trainerArgs import CustomTrainer, getTrainingArguments

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


args = ArgumentParser("Arguments for the models")

args.add_argument('--modelbase', '-mbase',
                  default='facebook/bart-base', type=str)
args.add_argument('--run_id', '-run_id', type=str, default='')
args.add_argument('--iterative_gen', '-itg', action='store_true', )
args.add_argument('--output_dir', '-output_dir', type=str, required=True)
args.add_argument('--inference_dir', '-inf_dir', type=str, required=True)

args.add_argument('--warmup_ratio', '-wr', default=0.2, type=float)
args.add_argument('--weight_decay', '-weight_decay', type=float, default=0.3)
args.add_argument('--per_device_train_batch_size',
                  '-train_bs', type=int, default=4,)
args.add_argument('--per_device_eval_batch_size',
                  '-eval_bs', type=int, default=4,)
args.add_argument('--num_train_epochs', '-nb_epochs', default=20, type=int)
args.add_argument('--lr_scheduler_type', '-lr_scheduler',
                  default='cosine', type=str)
args.add_argument('--learning_rate', '-lr', default=5e-5, type=float)
args.add_argument('--evaluation_strategy',
                  '-evaluation_strategy', default="steps", )
args.add_argument('--logging_steps', '-logging_steps', default=500,)
args.add_argument('--seed', '-seed', type=int, default=43)
args.add_argument('--max_full_len', '-max_full_len', type=int, default=300)
args.add_argument('--inf_sample', '-inf_sample', action='store_true')


parsed_args = args.parse_args()
param_dict = vars(parsed_args)

train_arguments = {k: v for k, v in param_dict.items() if k not in [
    'iterative_gen', 'modelbase', 'inference_dir', 'inf_sample', 'run_id','max_full_len']}

zz = parsed_args.modelbase.replace('/', '_').replace('-', '_')
train_arguments['output_dir'] += f'/{zz}_seed{parsed_args.seed}{parsed_args.run_id}/'

output_path = train_arguments['output_dir']

inf_path = train_arguments['output_dir']+'/inference_output/'
if not os.path.exists(inf_path):
    os.makedirs(inf_path)


# Build the dataset manager with the already cleaned dataset saved at the "composed_train_path" and "composed_test_path"
composed_already = True
if not parsed_args.iterative_gen:
    composed_already = False

if parsed_args.iterative_gen:
    experiments_dataset = DatasetBuilder(train_data_path=composed_train_path,
                                     test_data_path=composed_test_path,
                                     modelbase=parsed_args.modelbase,
                                     iterative_mode=parsed_args.iterative_gen,
                                     composed_already=composed_already)
else:
    experiments_dataset = DatasetBuilder(train_data_path=train_path,
                                     test_data_path=test_path,
                                     modelbase=parsed_args.modelbase,
                                     iterative_mode=parsed_args.iterative_gen,
                                     composed_already=False)
experiments_dataset.fit()

print(parsed_args.modelbase)
print('Dataset built')
print(f"Training size: {len(experiments_dataset.train_dataset)}")
print(f"Test size: {len(experiments_dataset.test_dataset)}")

seed_everything(parsed_args.seed)
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# Build actual trainingArgument object
training_arguments = getTrainingArguments(train_arguments)

# Get the model
getModel = get_basic_model(experiments_dataset)


trainer = CustomTrainer(model_init=getModel,
                        data_collator=SmartCollator(
                            pad_token_id=experiments_dataset.tokenizer_.pad_token_id),
                        args=training_arguments,
                        train_dataset=experiments_dataset.train_dataset,
                        eval_dataset=experiments_dataset.test_dataset,
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)])

trainer.train()
inference_model = trainer.model.generator.to(device)
inference_model.eval()
print("-- Traininfg Completed ----")


# Perform the inference with the best model from the training
# Extract the test dataset
tests = json.load(open(test_path, encoding='utf-8'))
# ,iterative_mode=False,force_section=False
if parsed_args.iterative_gen:
    test_examples = compactComposer(
        tests, iterative_mode=parsed_args.iterative_gen, force_section=True)
    test_examples2 = compactComposer(
        tests, iterative_mode=parsed_args.iterative_gen, force_section=False)
else:
    test_examples = compactComposer(tests, iterative_mode=parsed_args.iterative_gen, force_section=False)

# Instantiate the inference routine
iterativeGen = InferenceGenerator(inference_model,
                                  experiments_dataset,
                                  device,
                                  max_iter=8,
                                  sampling=False, verbose=False)

if parsed_args.iterative_gen:
    print("Explanation Generation Iteratively")
    iterativeGen.sampling = False
    #output_sentences = iterativeGen.MultipleIterativeGeneratorJoint( test_examples, parsed_args.seed,)
    output_sentences = iterativeGen.MultipleIterativeGeneratorJoint(
        test_examples, parsed_args.seed, length_penalty=1.6,max_length=269)
    output_sentences2 = iterativeGen.MultipleIterativeGeneratorJoint(
        test_examples2, parsed_args.seed, length_penalty=1.6,max_length=269)
else:
    print("Full Explanation Generation")
    output_sentences = iterativeGen.MultipleFullGeneratorJoint(
        test_examples, parsed_args.seed, max_length=parsed_args.max_full_len)

output_sentences_sampled = []
if parsed_args.inf_sample and parsed_args.iterative_gen:
    iterativeGen.sampling = True
    output_sentences_sampled = iterativeGen.MultipleIterativeGeneratorJoint(
        test_examples,  parsed_args.seed)

# Save the generated output
if parsed_args.iterative_gen:
    json.dump({'beam_search_stepwise':output_sentences,
           'beam_search_random':output_sentences2},
          open(inf_path+f'model_output_{parsed_args.seed}.json','w'))
else:
    json.dump({'beam_search_full':output_sentences},
          open(inf_path+f'model_output_{parsed_args.seed}.json','w'))
