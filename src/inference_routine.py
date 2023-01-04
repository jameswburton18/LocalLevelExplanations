import copy
import functools
import json
import random
import re
from types import SimpleNamespace

import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from pytorch_lightning import seed_everything

from src.datasetHandlers import RDFDataSetForLinearisedStructured, setupTokenizer
from src.model_utils import get_basic_model
from src.NarrationDatautils_new import (finalProcessor,
                                    linearisedFeaturesAttributions,
                                    processPredictionProbabilities2,
                                    reformulateInput, simpleMapper)

# Takes care of processing the explanation output and text generation instructions


def inferenceIterativeGenerator(pack,
                                pr_c=1,
                                ignore=False,
                                force_section=False,
                                include_full_set=False,
                                ):
    pack = pack_x = copy.deepcopy(pack)
    pack['narration'] = ' , '.join([''.join(t) for t in pack['steps']])
    outputs = pack['steps']

    max_init = 1
    results = []

    if len(outputs[max_init:]) > 0:
        sofar = ' '.join(outputs[:max_init])+' [N1S]'+' '
        pack_x = copy.deepcopy(pack)

        pack_x['next_sequence'] = sofar
        pack_x['prev_seq'] = '<prem>'
        results.append(pack_x)
    prev = copy.deepcopy(sofar)

    for idx, sent in enumerate(outputs[max_init:-1]):
        # print(idx+1)
        lotto = [0, 1, 1, 0, 0, 1]
        random.shuffle(lotto)
        pack_x = copy.deepcopy(pack)

        pack_x['next_sequence'] = sent+f' [N{idx+2}S]'
        pack_x['prev_seq'] = sofar
        results.append(pack_x)

        sofar += sent+f' [N{idx+2}S]'+' '
    pack_x = copy.deepcopy(pack)
    pack_x['prev_seq'] = sofar+' [EON]'
    pack_x['next_sequence'] = outputs[-1]+' [EON]'
    results.append(pack_x)

    if include_full_set:
        pack_f = copy.deepcopy(pack)
        pack_f['prev_seq'] = '<full_narration>'
        pack_f['next_sequence'] = sofar+' [EON] ' + \
            outputs[-1]+' [CON]'  # outputs[-1]+' [EON]'
        results.append(pack_f)

    results.append(sofar+' [EON] ')
    return results


def processFeatureAttributions(attributions, narration, force_consistency=True, nb_base=None):

    #  nb_base: specifies the number of features to consider in the output text when full-text generation mode is used

    feature_division = attributions

    contradict, support, ignore = feature_division[
        'negatives'], feature_division['positives'], feature_division['ignore']

    output = {'features': [], 'order': [], 'direction': []}
    nat = set([c.strip() for c in word_tokenize(narration)])

    ordered_features = attributions['explanation_order']
    nb_features = len(ordered_features)

    for pos, feat in enumerate(ordered_features):
        include = False

        # check if the user specified if this feat should be in the output text

        if feat in nat and force_consistency:
            include = True
        elif feat not in nat and force_consistency:
            include = False
        elif not force_consistency:
            include = True

        if nb_base is not None and nb_features < nb_base:
            include = True
        if include:
            output['features'].append(feat)
            output['order'].append(pos)
            if feat in contradict:
                output['direction'].append(-1)
            elif feat in support:
                output['direction'].append(1)
            else:
                output['direction'].append(-2)
        else:
            pass
    return output


# The post-processing function to clean out invalid tokens as well as replace *placeholders* with their correct values
def cleanOutput(passages, label_holder):
    placeholders = {t: ' ' for t in ['[EON]', '[N9S]',
                                     '[N10S]', '[CON]',
                                     '[N4S]', '[N5S]', '[N8S]', '[N6S]',
                                     '[N7S]', '[N1S]', '[N2S]',
                                     '[N0S]', '[N3S]']}
    placeholders.update(label_holder)
    passages = copy.deepcopy(passages)
    dd = [functools.reduce(lambda a, kv: a.replace(*kv), placeholders.items(),
                           re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [passages]][0]
    dd = ' '.join(sent_tokenize(dd))
    return dd


def jaccard_similarity(query, document):
    #print("Query: ",query)
    #print('Previ: ',document)
    placeholders = {t: ' ' for t in ['[EON]', '[N9S]',
                                     '[N10S]', '[CON]',
                                     '[N4S]', '[N5S]', '[N8S]', '[N6S]',
                                     '[N7S]', '[N1S]', '[N2S]',
                                     '[N0S]', '[N3S]']}
    query, document = [functools.reduce(lambda a, kv: a.replace(*kv), placeholders.items(),
                                        re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [query, document]]
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    score = len(intersection)/len(union)
    # print(score)
    return score


def filterOutAlreadyGenerated(past_generations, current_sentence, label_placeholder, threshold=0.9):
    if len(past_generations) == 0:
        # print('Nopes')
        return False
    oop = [jaccard_similarity(cleanOutput(copy.deepcopy(current_sentence), label_placeholder), cleanOutput(copy.deepcopy(o),
                                                                                                           label_placeholder)) for o in past_generations]

    #print('scores: ',oop)
    ch_max = max(oop)
    if ch_max >= threshold:
        #print('is high')
        return True
    else:
        #print('is Low')
        return False


def cleanFeatureHallucinations(feature_map, text):
    rev_map = feature_map
    feat_regex = {re.compile(
        r"[\s+]?\w+"+f.split('feat')[-1]+"[\s+]?"): ' '+f+' ' for f in rev_map.keys()}
    feat_regex2 = {re.compile(
        f'\((['+f.split('feat')[-1]+'\)]+)\)'): ' '+f+' ' for f in rev_map.keys()}
    feat_regex.update(feat_regex2)
    feat_regex[re.compile(' \.')] = '.'
    feat_regex[re.compile(' ,')] = ','
    return [functools.reduce(lambda a, kv: kv[0].sub(kv[1], a).replace(' ,', ','),  feat_regex.items(), ss) for ss in [text]][0]


class InferenceGenerator:
    def __init__(self, model,
                 experiments_dataset,
                 device,
                 max_iter=5,
                 sampling=True, verbose=False) -> None:
        self.model = model.to(device)
        self.experiments_dataset = experiments_dataset
        self.vectorizer = vectorizer = lambda x: experiments_dataset.base_dataset.processTableInfo(
            x)
        self.max_iter = max_iter
        self.sampling = sampling
        self.verbose = verbose
        self.device = device

    def fullNarration(self, example, seed, label_placeholder, max_length=260,
                      length_penalty=1.6, beam_size=10, repetition_penalty=3.56, return_top_beams=4):
        seed_everything(seed)
        example = copy.deepcopy(example)

        sample_too = self.sampling
        bs = beam_size

        completed = False
        final_output = ""
        steps = 0
        sampling_helper = {} if not sample_too else dict(top_k=50, top_p=0.95,)

        if not completed:
            batch = self.vectorizer(example)
            preamble_tokens = batch.input_ids.unsqueeze(0).to(self.device)
            preamble_attention_mask = batch.attention_mask.unsqueeze(
                0).to(self.device)

            sample_outputs = self.model.generate(input_ids=preamble_tokens,  **sampling_helper,
                                                 attention_mask=preamble_attention_mask,
                                                 num_beams=bs,
                                                 repetition_penalty=repetition_penalty,
                                                 length_penalty=length_penalty,
                                                 early_stopping=True,
                                                 use_cache=True,
                                                 max_length=max_length,
                                                 no_repeat_ngram_size=2,
                                                 num_return_sequences=return_top_beams,
                                                 do_sample=sample_too,
                                                 eos_token_id=self.experiments_dataset.tokenizer_.eos_token_id,)
            # Convert the generated output to sentences
            oop = [self.experiments_dataset.tokenizer_.decode(sample_outputs[idx],
                                                              skip_special_tokens=True,
                                                              clean_up_tokenization_spaces=True) for idx in range(return_top_beams)]
            rev_map = example['reverse_map']
            return cleanOutput(cleanFeatureHallucinations(rev_map, oop[0]), label_placeholder)

    def iterativeGeneratorJoint(self, example, seed, label_placeholder, max_length=150,
                                length_penalty=1.6, beam_size=10, repetition_penalty=3.5, return_top_beams=4, no_repeat_ngram_size=2):
        seed_everything(seed)
        example = copy.deepcopy(example)
        sample_too = self.sampling
        bs = beam_size

        completed = False
        final_output = ""
        sofar = "<prem>"
        steps = 0

        previous_step = ""
        past_gens = []

        self.model.eval()
        rev_map = example[0]['reverse_map']
        with torch.no_grad():
            while not completed:
                if steps < len(example):
                    dat = example[steps]
                    dat['prev_seq'] = sofar
                    # print(dat['prev_seq'])
                else:
                    print('All facts presented')
                    break

                batch = self.vectorizer(dat)
                preamble_tokens = batch.input_ids.unsqueeze(0).to(self.device)
                preamble_attention_mask = batch.attention_mask.unsqueeze(0).to(
                    self.device)
                tokenizer_ = self.experiments_dataset.tokenizer_
                sampling_helper = {} if not sample_too else dict(
                    top_k=50, top_p=0.95,)
                sample_outputs = self.model.generate(input_ids=preamble_tokens,  **sampling_helper,
                                                     attention_mask=preamble_attention_mask,
                                                     num_beams=bs,
                                                     repetition_penalty=repetition_penalty,
                                                     length_penalty=length_penalty,
                                                     early_stopping=True,
                                                     use_cache=True,
                                                     max_length=max_length,
                                                     no_repeat_ngram_size=no_repeat_ngram_size,
                                                     num_return_sequences=return_top_beams,
                                                     do_sample=sample_too,
                                                     eos_token_id=tokenizer_.eos_token_id,)
                # Convert the generated output to sentences
                oop = [tokenizer_.decode(sample_outputs[idx],
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True) for idx in range(return_top_beams)]
                # print(oop)

                # Because it is iterative, we expect to have the proper next sentence
                expected_terminal = f'N{steps+1}'

                # Get the sentences that correctly follow the order of information based on the next sentence tokens
                oop = [cleanFeatureHallucinations(
                    rev_map, o) for o in oop if expected_terminal in o or '[EON]' in o]

                # print(oop)

                oop = [o for o in oop if not filterOutAlreadyGenerated(
                    past_gens, o, label_placeholder)]

                #print('Filtered ',oop)

                # print('Nexts')

                oop = [(o, jaccard_similarity(cleanOutput(copy.deepcopy(o), label_placeholder),
                                              cleanOutput(copy.deepcopy(previous_step), label_placeholder))) for o in oop]
                oop = [o[0] for o in sorted(
                    oop, key=lambda x: x[-1], reverse=True)[-3:]]

                if len(oop) < 1:
                    break

                # Pick one of the best sentences for the next step
                best_choice = random.choice(oop)

                #print('This is here ',best_choice)

                # Check if the end of narrative token has been generated, if yes then we terminate and return the output paragraph
                completed = '[EON]' in best_choice
                sofar = sofar+' '+best_choice+' '

                dat['preamble'] = sofar  # +' [NLS] '+prema
                dat['prev_seq'] = sofar
                final_output += ' '+best_choice
                previous_step = best_choice
                past_gens.append(best_choice)

                if self.verbose:
                    print(f' The text generated so far is: {sofar}')
                    print(f'Generation Step {steps+1}')

                if completed:
                    print('Completed')
                    break

                if steps > self.max_iter:
                    # print('Heyys')
                    break

                steps += 1

            return cleanOutput(final_output, label_placeholder)

    def MultipleIterativeGeneratorJoint(self, examples, seed, max_length=150,
                                        length_penalty=1.6, beam_size=10, repetition_penalty=3.5, return_top_beams=4):

        generatedOutput = []

        if type(examples) is not list:
            examples = [examples]

        for example in examples:
            label_placeholder = {p: ' '+l for l,
                                 p in example[0]['label_placeholders'].items()}
            rev_map = example[0]['reverse_map']
            label_placeholder.update(rev_map)
            label_placeholder[' percent'] = '%'
            label_placeholder['predictionlabel'] = label_placeholder[' predictionlabel']
            label_placeholder['predictionslabel'] = label_placeholder[' predictionlabel']
            label_placeholder.update({'predictionranking'+k.split('rank')
                                     [-1]: v for k, v in label_placeholder.items() if 'rank' in k})
            label_placeholder.update(
                {'predictions'+k.split('prediction')[-1]: v for k, v in label_placeholder.items()})
            label_placeholder[', or'] = ' and '
            label_placeholder['predictionLabel'] = label_placeholder[' predictionlabel']
            label_placeholder.update(
                {'predicted'+k.split('prediction')[-1]: v for k, v in label_placeholder.items()})
            label_placeholder.update(
                {'feature'+k.split('feat')[-1]: v for k, v in rev_map.items()})

            # print(example,'\n')
            # print(len(example))
            output_sentence = self.iterativeGeneratorJoint(
                example, seed, label_placeholder, max_length, length_penalty, beam_size, repetition_penalty, return_top_beams)

            generatedOutput.append(output_sentence)
        return generatedOutput

    def MultipleFullGeneratorJoint(self, examples, seed,  max_length=260,
                                   length_penalty=1.6, beam_size=10, repetition_penalty=1.54, return_top_beams=4):

        generatedOutput = []

        if type(examples) is not list:
            examples = [examples]

        for example in examples:
            label_placeholder = {p: ' '+l for l,
                                 p in example[0]['label_placeholders'].items()}
            rev_map = example[0]['reverse_map']
            label_placeholder.update(rev_map)
            label_placeholder[' percent'] = '%'
            label_placeholder['predictionlabel'] = label_placeholder[' predictionlabel']
            label_placeholder['predictionslabel'] = label_placeholder[' predictionlabel']
            label_placeholder.update({'predictionranking'+k.split('rank')
                                     [-1]: v for k, v in label_placeholder.items() if 'rank' in k})
            label_placeholder.update(
                {'predictions'+k.split('prediction')[-1]: v for k, v in label_placeholder.items()})
            label_placeholder.update(
                {'predicted'+k.split('prediction')[-1]: v for k, v in label_placeholder.items()})
            label_placeholder[', or'] = ' and '
            label_placeholder['predictionLabel'] = label_placeholder[' predictionlabel']
            label_placeholder.update(
                {'feature'+k.split('feat')[-1]: v for k, v in rev_map.items()})
            output_sentence = self.fullNarration(
                example[0], seed, label_placeholder, max_length, length_penalty, beam_size, repetition_penalty, return_top_beams)

            generatedOutput.append(output_sentence)
        return generatedOutput

# an object of this class loads the fine-tuned pre-trained model


class NarratorUtils:
    def __init__(self, modelbase, trained_model_path):
        self.modelbase = modelbase
        self.trained_model_path = trained_model_path
        # Setting up tokenizer
        self.tokenizer_ = setupTokenizer(modelbase)

        self.local_dict = SimpleNamespace(modelbase=modelbase,
                                          tokenizer_=self.tokenizer_)
        self.setup_performed = False
        self.base_dataset = RDFDataSetForLinearisedStructured(
            self.tokenizer_, [], self.modelbase, step_continue=False)

        print(' Dont forget to call initialise_Model() before running any inference')

    def initialise_Model(self):
        classification_explanator = get_basic_model(self.local_dict)()
        # Set up the model along with the tokenizers and other important stuff required to run the generation
        params_dict = json.load(
            open(self.trained_model_path+'/parameters.json'))
        #state_dict = json.load(open(args.model_base_dir+'/parameters.json'))
        best_check_point = 'TrainModels/iterative/t5-base/checkpoint-1500' #params_dict['best_check_point']
        best_check_point_model = best_check_point + '/pytorch_model.bin'

        state_dict = torch.load(best_check_point_model)
        classification_explanator.load_state_dict(state_dict)

        classification_explanator.eval()
        self.setup_performed = True

        return classification_explanator


class ExplanationRecord():
    def __init__(self, ml_task_name, feature_names, prediction_probabilities, attributions, iterative_mode=False) -> None:
        self.input_record = SimpleNamespace()
        classes = list(prediction_probabilities.keys())
        self.input_record.classes = classes
        self.input_record.feature_names = feature_names
        self.input_record.attributions = attributions
        self.input_record.ml_task_name = ml_task_name
        self.input_record.prediction_as_string = ' , '.join(
            [f'{k}:{round(v*100,2)}% ' for k, v in prediction_probabilities.items()])

        predicted_label = max(prediction_probabilities,
                              key=prediction_probabilities.get)
        self.input_record.predicted_class = predicted_label
        print(f"The ML model predicted the label : {predicted_label}")

        self.iterative_mode = iterative_mode

    def processStep(self, instruction, shrink=None, randomise_preds=False):
        def cleanNarrations(x): return x
        instance = instruction

        # print(instance)
        narration = cleanNarrations(copy.deepcopy(instance['next_sequence']))
        prev_seq = cleanNarrations(copy.deepcopy(instance['prev_seq']))
        full_narra = cleanNarrations(copy.deepcopy(instance['narration']))

        # get the rankings information for all the features
        feature_ranks = processFeatureAttributions(
            self.input_record.attributions, narration+prev_seq+full_narra, force_consistency=True)

        # get the ranking information for only the features that should be present in the next text to be generated
        feature_ranks2 = processFeatureAttributions(
            self.input_record.attributions, narration, force_consistency=True)

        feature_desc, [pf, nf, neu_f], directions = linearisedFeaturesAttributions(
            feature_ranks, shrink=shrink)
        feature_desc2, [pf2, nf2, neu_f2], directions2 = linearisedFeaturesAttributions(
            feature_ranks2, shrink=shrink)

        preamble, place_holderss = processPredictionProbabilities2(instance['prediction_confidence_level'],
                                                                   instance['predicted_class'], randomise=randomise_preds)
        preamble = preamble+' <|section-sep|> '+feature_desc
        class_dict = place_holderss

        extended1 = preamble

        # replace the class labels with their corresponding placeholder
        narr, prev = [functools.reduce(lambda a, kv: a.replace(*kv), class_dict.items(),
                                       re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [narration, prev_seq]]

        return {'rele_feat': [pf2, nf2, neu_f2], 'directions': directions, 'label_placeholders': place_holderss, 'preamble': extended1, 'narration': narr, 'prev_seq': prev,
                'positives': pf, 'negatives': nf, 'neutral': neu_f, 'pred_label': class_dict[instance['predicted_class']]}

    def setup_generation_steps(self, generation_steps, randomise_preds=True, shrink=None):
        if not self.iterative_mode:
            print(' The texts will be generated in the default full-text mode')
        self.input_record.generation_steps_as_string = list(
            [' '.join(t) for t in generation_steps.values()])

        generation_instuctions = inferenceIterativeGenerator({'steps': self.input_record.generation_steps_as_string,
                                                              'prediction_confidence_level': self.input_record.prediction_as_string,
                                                              'predicted_class': self.input_record.predicted_class

                                                              }, ignore=not self.iterative_mode)

        processed_instructions = []

        for instruction in generation_instuctions[:-1]:
            examples = self.processStep(instruction)
            # preambles.append(zz)
            examples_ = copy.deepcopy(examples)
            examples_['preamble'] = examples['preamble']+' <explain>'
            narr = examples['narration']
            examples_['output'] = narr
            # processed_instructions(examples_)

            # apply the final touches to the data
            # print(examples_['positives'])
            processed_generation_step = simpleMapper(
                reformulateInput(finalProcessor(examples_)))

            processed_instructions.append(processed_generation_step)
        return [processed_instructions]


class LocalLevelExplanationNarration:
    def generateTexts(self, processed_data):
        if self.iterative_mode:
            output_sentences = self.inferenceGen.MultipleIterativeGeneratorJoint(processed_data,
                                                                                 self.seed, beam_size=self.beam_size,
                                                                                 length_penalty=self.length_penalty,
                                                                                 repetition_penalty=self.repetition_penalty,
                                                                                 max_length=self.max_len_trg)
        else:
            output_sentences = self.inferenceGen.MultipleFullGeneratorJoint(
                processed_data, self.seed, max_length=self.max_len_trg, beam_size=self.beam_size)

        return output_sentences

    def __init__(self, classification_explanator,
                 narrator_utils,
                 device,
                 iterative_mode=True,
                 max_preamble_len=160,
                 max_len_trg=185,

                 length_penalty=8.6,
                 beam_size=10,
                 repetition_penalty=1.5,
                 return_top_beams=1,
                 lower_narrations=True,
                 process_target=True, random_state=None):
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.beam_size = beam_size
        self.return_top_beams = return_top_beams
        self.max_preamble_len = max_preamble_len
        self.max_len_trg = max_len_trg
        self.max_rate_toks = 8
        self.lower_narrations = lower_narrations
        self.process_target = process_target

        if random_state is None:
            random_state = 456

        self.seed = random_state
        self.random_state = random_state
        self.setup_performed = False

        self.iterative_mode = iterative_mode

        self.inferenceGen = InferenceGenerator(classification_explanator.generator,
                                               narrator_utils,
                                               device,
                                               max_iter=10,
                                               sampling=False, verbose=False)
