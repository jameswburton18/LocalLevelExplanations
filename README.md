# Natural Language Explanations for Machine Learning Classification Decisions

## Setup

To setup run:

```bash
pip install -e .
pip install -r requirements.txt
```

## Datasets

* Running `src/prepare_datasets.py` will create regular, augmented and QnA json files saving them to `data/processed/`
* `src/dataset_builders/` contains scripts to create huggingface datasets from the json files

Alternatively, the datasets can be download directly from huggingface:

```python
from datasets import load_dataset

text_exp_dataset = load_dataset("james-burton/textual-explanations-702010")
aug_text_exp_dataset = load_dataset("james-burton/aug-text-exps-v3")
qa_dataset = load_dataset("james-burton/text-exp-qa-hard")
```

Note that the datasets will need to be processed: example in `notebooks/inference_example.ipynb`

## Models

Training the models is done using `src/train.py` which will save the model to `models/` in subdirectories named after the model type and the model used. Weights and biases is used to log the training process and to generate the folder name.

The model configurations are defined in `yaml` files in `configs/`. `src/train.py` will load the default config from `configs/train_default.yaml` or `configs/qa_default.yaml` and then update from the default depending on the config bash argument that is passed to the script. This will load the correspoinding config file from `configs/train_configs.yaml` or `configs/qa_configs.yaml`.

To train the models with the configs in the paper, run:

```bash
# Textual Explanation model: T5
python src/train.py --config text_3_bigtest # base-20
python src/train.py --config text_17 # base-20-Aug
python src/train.py --config text_25 # base-10
python src/train.py --config text_29 # base-10-Aug
python src/train.py --config text_21 # large-20
python src/train.py --config text_22 # large-20-Aug
python src/train.py --config text_34 # large-10
python src/train.py --config text_33 # large-10-Aug

# Textual Explanation model: BART
python src/train.py --config text_4_bigtest # base-20
python src/train.py --config text_18 # base-20-Aug
python src/train.py --config text_26 # base-10
python src/train.py --config text_30 # base-10-Aug
python src/train.py --config text_23 # large-20
python src/train.py --config text_24 # large-20-Aug
python src/train.py --config text_31 # large-10
python src/train.py --config text_32 # large-10-Aug

# QA model: T5
python src/train_qa.py --config text_3

# QA model: BART
python src/train_qa.py --config text_4

```

We have also uploaded the trained models to huggingface:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

t5_model_names = [
    "text-exps-t5-20",
    "text-exps-t5-20-aug",
    "text-exps-t5-10",
    "text-exps-t5-10-aug",
    "text-exps-t5-large-20",
    "text-exps-t5-large-20-aug",
    "text-exps-t5-large-10",
    "text-exps-t5-large-10-aug",
]

bart_model_names = [
    "text-exps-bart-20",
    "text-exps-bart-20-aug",
    "text-exps-bart-10",
    "text-exps-bart-10-aug",
    "text-exps-bart-large-20",
    "text-exps-bart-large-20-aug",
    "text-exps-bart-large-10",
    "text-exps-bart-large-10-aug",
]

qa_model_names = [
    "text-exps-qa-t5",
    "text-exps-qa-bart",
]



# Load any of the models in the following fashion:
model_name = t5_model_names[0]
model = AutoModelForSeq2SeqLM.from_pretrained("james-burton/" + model_name)
tokenizer = AutoTokenizer.from_pretrained("james-burton/" + model_name)

# The following 
```

The `-aug` models are trained using the `aug_text_exp_dataset`, non-augmented models are trained using the `text_exp_dataset` and the QA models are trained using the `qa_dataset`.
