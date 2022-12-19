# Functions to process and return the training and test pytorch Dataset
import json
import pickle as pk

from typing import List

from .datasetHandlers import (RDFDataSetForLinearisedStructured, SmartCollator,
                              setupTokenizer)
from .NarrationDatautils_new import *

# replace these urls with the paths to your files
main_path = 'raw_data/'
train_path = main_path+"all_train.json"
test_path = main_path+"test_set_new.json"

composed_train_path = main_path+"processed_data/train_pack4.dat"
composed_test_path = main_path+"processed_data/test_pack4.dat"


####################################################


# Apply the different preprocessing methods
def composeDataset(data, iterative_mode=True, force_section=False, include_full_set=False):
    """
    This method take as input the attribution information in the form of dictionary with fields:
        'predicted_class_label' : the assigned label
        'narration': textual explanations. (Empty when generating for test set)
        'prediction_confidence_level': confidence levels across the classes
        'feature_division': Attribution information from lime or shap or LRP or feature importance
        
        
    The iterative dataset has random choice of 1, 2 or 3 sentences for the initial generation. 
    From there one sentence at a time is chosen for the next split. Each split is
    marked with a special token, such as [N1S], [N2S], [N3S] etc.
    """
    def pass_through(x): return iterNarationDataGenerators(
        x, ignore=not iterative_mode, force_section=force_section, include_full_set=include_full_set)
    if type(data) is not list:
        data = [data]
    full_set = []
    for p in data:
        full_set += pass_through(p)

    final_data = [simpleMapper(reformulateInput(finalProcessor(
        p))) for p in composeTrainingData(full_set, force_consistency=True, shrink=None)]
    return final_data


def compactComposer(data, iterative_mode=True, force_consistency=True, force_section=False, include_full_set=False):
    if type(data) is not list:
        data = [data]

    def pass_through(x): return iterNarationDataGenerators(
        x, ignore=not iterative_mode, force_section=force_section, include_full_set=include_full_set)
    epre_test = [[simpleMapper(reformulateInput(finalProcessor(p_))) for p_ in composeTrainingData(pass_through(p),
                                                                                                   force_consistency=force_consistency, shrink=None)] for p in data]
    return epre_test


class InferenceDatasetBuilder:
    def __init__(self, modelbase,
                 preamble_choice=2,
                 iterative_mode=True,
                 composed_already=False,
                 step_continue=False,
                 include_full_set=False,
                 ):
        self.modelbase = modelbase
        self.preamble_choice = preamble_choice

        self.tokenizer_ = setupTokenizer(modelbase)
        self.step_continue = step_continue

        

class DatasetBuilder:
    train_data_raw: List
    test_data_raw: List

    def __init__(self, modelbase,
                 train_data_path=train_path,
                 test_data_path=test_path,
                 preamble_choice=2,
                 iterative_mode=True,
                 composed_already=False,
                 step_continue=False,
                 include_full_set=False,
                 ):
        self.modelbase = modelbase
        self.preamble_choice = preamble_choice
        if not iterative_mode:
            composed_already = False
        if not composed_already:
            self.train_data_raw = composeDataset(json.load(open(train_data_path,encoding='utf-8')), iterative_mode=iterative_mode,include_full_set=include_full_set)
            self.test_data_raw = composeDataset(json.load(open(test_data_path, encoding='utf-8')), iterative_mode=iterative_mode,include_full_set=include_full_set)
        else:
            self.train_data_raw = pk.load(open(train_data_path, 'rb'))
            self.test_data_raw = pk.load(open(test_data_path, 'rb'))

        self.tokenizer_ = setupTokenizer(modelbase)
        self.step_continue = step_continue
    def build_default(self,):
        
        # Build the dataset framework for inference purposes
        self.base_dataset =RDFDataSetForLinearisedStructured(
            self.tokenizer_,[], self.modelbase, step_continue=self.step_continue)
    
    def dataset_fit(self, dataset):
        self.base_dataset = RDFDataSetForLinearisedStructured(
            self.tokenizer_,dataset, self.modelbase, step_continue=self.step_continue)
    
    def fit(self,):
        # creates pytorch dataset for the training and testing sets
        eprocessed_train = self.train_data_raw
        eprocessed_test = self.test_data_raw
        random.shuffle(eprocessed_train)
        random.shuffle(eprocessed_train)
        self.train_dataset = RDFDataSetForLinearisedStructured(
            self.tokenizer_, eprocessed_train, self.modelbase, step_continue=self.step_continue)

        self.test_dataset= self.base_dataset = val_dataset = RDFDataSetForLinearisedStructured(self.tokenizer_, eprocessed_test,
                                                                            self.modelbase, step_continue=self.step_continue)
    
    def transform(self, data):
        if not self.step_continue:
            return self.base_dataset.processTableInfo(data)
        else:
            return self.base_dataset.processTableInfoStepContinue(data)
