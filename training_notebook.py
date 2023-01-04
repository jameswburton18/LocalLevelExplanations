# %%
import json
import os
import torch
from argparse import ArgumentParser

from pytorch_lightning import seed_everything

from transformers import EarlyStoppingCallback

from src.datasetComposer import DatasetBuilder, composed_train_path, composed_test_path, compactComposer, test_path, train_path, test_path
from src.inference_routine import InferenceGenerator
from src.datasetHandlers import SmartCollator
from src.model_utils import get_basic_model
from src.trainerArgs import CustomTrainer, getTrainingArguments
from transformers import TrainingArguments
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%

iterative_gen = False
composed_already = True

# Define the parameters used to set up the models
modeltype = 'iterative' if iterative_gen else 'normal'  # either baseline or 'earlyfusion'

# either t5-small,t5-base, t5-large, facebook/bart-base, or facebook/bart-large
modelbase = 't5-base'#'facebook/bart-base'

# we will use the above variables to set up the folder to save our model
pre_trained_model_name = modelbase.split(
    '/')[1] if 'bart' in modelbase else modelbase

# where the trained model will be saved
output_path = 'TrainModels/' + modeltype + '/'+pre_trained_model_name+'/'

#tests = json.load(open(test_path,encoding='utf-8'))

# %% [markdown]
# ## Dataset 
# Only the data for the iterative generation have been processed already to reduce the loading and processing time. 
# 

# %%
# Load the dataset
if iterative_gen:
    experiments_dataset = DatasetBuilder(train_data_path=composed_train_path,#train_path,#
                                     test_data_path=composed_test_path,#test_path,#
                                     modelbase= modelbase,
                                     iterative_mode= iterative_gen,
                                     composed_already=composed_already)
else:
    experiments_dataset = DatasetBuilder(train_data_path=train_path,
                                     test_data_path=test_path,
                                     modelbase= modelbase,
                                     iterative_mode= False,
                                     composed_already=False)


experiments_dataset.fit()

print('Dataset built')
print(f"Training size: {len(experiments_dataset.train_dataset)}")
print(f"Test size: {len(experiments_dataset.test_dataset)}")


# # Model
# Below we create the narration model

rand_seed = 453
seed_everything(rand_seed)
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

arguments = train_arguments = {'output_dir': output_path,
                               'warmup_ratio': 0.2,
                               #'disable_tqdm':False,
                               'per_device_train_batch_size': 8,
                               'num_train_epochs': 6,
                               'lr_scheduler_type': 'cosine',
                               'learning_rate': 5e-5,
                            #    'evaluation_strategy': 'steps',
                               'logging_steps': 20,
                               
                               'seed': rand_seed}

# Build actual trainingArgument object
training_arguments = getTrainingArguments(train_arguments)

# %%
# Define the model
getModel = get_basic_model(experiments_dataset)


trainer = CustomTrainer(model_init=getModel,
                        data_collator=SmartCollator(
                            pad_token_id=experiments_dataset.tokenizer_.pad_token_id),
                        args=training_arguments,
                        train_dataset=experiments_dataset.train_dataset,
                        # eval_dataset=experiments_dataset.test_dataset,
                        # callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
                        )

# %%
trainer.train()

# Save the model with the lowest evaluation loss
trainer.save_model()
trainer.save_state()

# get the best checkpoint
best_check_point = trainer.state.best_model_checkpoint


params_dict = train_arguments

params_dict['best_check_point'] = best_check_point
params_dict['output_path'] = output_path
json.dump(params_dict, open(f'{output_path}/parameters.json', 'w'))

# %% [markdown]
# # Inference

# %%
inference_model = trainer.model.generator.to(device)
inference_model.eval();

# %%
tests = json.load(open(test_path, encoding='utf-8'))
# ,iterative_mode=False,force_section=False
if iterative_gen:
    test_examples = compactComposer(
        tests, iterative_mode=iterative_gen, force_section=True)
    test_examples2 = compactComposer(tests, iterative_mode=iterative_gen, force_section=False)
else:
    test_examples = compactComposer(tests, iterative_mode=iterative_gen, force_section=False)

# Instantiate the inference routine
iterativeGen = InferenceGenerator(inference_model,
                                  experiments_dataset,
                                  device,
                                  max_iter=8,
                                  sampling=False, verbose=False)
iterative_gen

# %%


# %%
max_full_len = 300
'''
During the inference, the model is still passed as input/preamble, only features which
it has already seen are part of the answer.

'''
if iterative_gen:
    print("Explanation Generation Iteratively")
    iterativeGen.sampling = False
    #output_sentences = iterativeGen.MultipleIterativeGeneratorJoint( test_examples, parsed_args.seed,)
    output_sentences = iterativeGen.MultipleIterativeGeneratorJoint(
        test_examples[:1], rand_seed, length_penalty=1.6,max_length=269)
    output_sentences2 = iterativeGen.MultipleIterativeGeneratorJoint(
        test_examples2[:1], rand_seed, length_penalty=1.6,max_length=269)
else:
    print("Full Explanation Generation")
    output_sentences = iterativeGen.MultipleFullGeneratorJoint(
        test_examples, rand_seed, max_length=max_full_len)

# %%
print(test_examples[:1][0])

# %%
print(output_sentences)
