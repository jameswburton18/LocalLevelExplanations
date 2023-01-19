import csv
import datasets
import json
from typing import Dict
from datasets import set_caching_enabled

DESCRIPTION = """\
    This is a collection of 20000 quesiton answer pairs generated to mimic the
    information contained in the textual explanations dataset. 
    """
    

    
_TRAIN_DOWNLOAD_URL = "jb_data/qa_train.json"
_DEV_DOWNLOAD_URL = "jb_data/qa_val.json"
_TEST_DOWNLOAD_URL = "jb_data/qa_test.json"

class QADatasetBuilder(datasets.GeneratorBasedBuilder):
    """Question Answering Dataset based on textual explanations."""
    
    VERSION = datasets.Version("2.7.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="text-exp-qa",
            version=VERSION,
            description="Question Answering Dataset based on textual explanations",
        ),
    ]

    def __init__(self, data_dir=None, **kwargs):
        super().__init__()
    
    def _info(self):
        # classes_dict is a dictionary
        return datasets.DatasetInfo(
            description=DESCRIPTION,
            features=datasets.Features(
                {
                    "predicted_class": datasets.Value("string"),
                    "classes_dict": datasets.Value("string"),
                    "feature_nums": datasets.Sequence(datasets.Value("string")),
                    "sign": datasets.Sequence(datasets.Value("string")),
                    "values": datasets.Sequence(datasets.Value("string")),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "id": datasets.Value("int32"),
                    "question_id": datasets.Value("int32"),
                }
            ),
            supervised_keys=None,
        )
        
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        dev_path = dl_manager.download_and_extract(_DEV_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]
        
    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for id_, row in enumerate(data):
                yield id_, {
                    "predicted_class": row["predicted_class"],
                    "classes_dict": row["classes_dict"],
                    "feature_nums": row["feature_nums"],
                    "sign": row["sign"],
                    "values": row["values"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "id": row["id"],
                    "question_id": row["question_id"],
                }

if __name__ == "__main__":
    # Note that if changes are made to the dataset then it will raise a ChecksumError.
    # To fix this you need to delete the cached files in ~/.cache/huggingface/datasets/qa_dataset_builder/
    dataset = datasets.load_dataset("src/dataset_builders/qa_dataset_builder.py", download_mode="force_redownload", load_from_cache_file=False)#, ignore_verifications=True)
    
    # Save the dataset to huggingface
    dataset.push_to_hub("text-exp-qa", private=True)
