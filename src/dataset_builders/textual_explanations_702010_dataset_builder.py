import csv
import datasets
import json
from typing import Dict
from datasets import set_caching_enabled

DESCRIPTION = """\
    This is a collection of 469 textual explanations paired with the raw data that
    was used to generate a visualization. The explanations were created by computer
    science experts. Explanations themselves comes from LIME, SHAP and gradient
    based methods. The explanations are to explain classification results from a 
    selection of machine learning models; there are 40 different tasks and 9 
    different models that were used. Anonymised and randomised feature names were
    used to ensure that there is no data leakage.
    
    There are several features that are dictionaries
    that have been stored as strings as the keys for said diciotnaries are not the same 
    across all examples. These features are:
        - classes_dict
        - ft_num2name
        - old2new_ft_nums
        - old2new_classes
        - class2name
    """


_TRAIN_DOWNLOAD_URL = "jb_data/train_70-20-10.json"
_DEV_DOWNLOAD_URL = "jb_data/val.json"
_TEST_DOWNLOAD_URL = "jb_data/test_70-20-10.json"


class TextualExplanation702010DatasetBuilder(datasets.GeneratorBasedBuilder):
    """TextualExplanationDataset dataset."""

    VERSION = datasets.Version("2.7.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="textual-explanations",
            version=VERSION,
            description="TextualExplanationDataset dataset",
        ),
    ]

    def __init__(self, data_dir=None, **kwargs):
        super().__init__()

    def _info(self):
        return datasets.DatasetInfo(
            description=DESCRIPTION,
            features=datasets.Features(
                {
                    "model_name": datasets.Value("string"),
                    "predicted_class": datasets.Value("string"),
                    "task_name": datasets.Value("string"),
                    "narration": datasets.Value("string"),
                    "values": datasets.Sequence(datasets.Value("string")),
                    "sign": datasets.Sequence(datasets.Value("string")),
                    "narrative_id": datasets.Value("int32"),
                    "unique_id": datasets.Value("int32"),
                    "classes_dict": datasets.Value("string"),
                    "narrative_questions": datasets.Sequence(datasets.Value("string")),
                    "feature_nums": datasets.Sequence(datasets.Value("string")),
                    "ft_num2name": datasets.Value("string"),
                    "old2new_ft_nums": datasets.Value("string"),
                    "old2new_classes": datasets.Value("string"),
                    "predicted_class_label": datasets.Value("string"),
                    "class2name": datasets.Value("string"),
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
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_path}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for id_, row in enumerate(data):
                yield id_, {
                    "model_name": row["model_name"],
                    "predicted_class": row["predicted_class"],
                    "task_name": row["task_name"],
                    "narration": row["narration"],
                    "values": row["values"],
                    "sign": row["sign"],
                    "narrative_id": row["narrative_id"],
                    "unique_id": row["unique_id"],
                    "classes_dict": row["classes_dict"],
                    "narrative_questions": row["narrative_questions"],
                    "feature_nums": row["feature_nums"],
                    "ft_num2name": row["ft_num2name"],
                    "old2new_ft_nums": row["old2new_ft_nums"],
                    "old2new_classes": row["old2new_classes"],
                    "predicted_class_label": row["predicted_class_label"],
                    "class2name": row["class2name"],
                }


if __name__ == "__main__":
    # Note that if changes are made to the dataset then it will raise a ChecksumError.
    # To fix this you need to delete the cached files in ~/.cache/huggingface/datasets/
    dataset = datasets.load_dataset(
        "src/dataset_builders/textual_explanations_702010_dataset_builder.py",
        download_mode="force_redownload",
    )  # , ignore_verifications=True)

    # Save the dataset to huggingface
    dataset.push_to_hub("textual-explanations-702010", private=True)
    print()
