import copy
import functools
import random
import re

from pytorch_lightning import seed_everything
import torch
from nltk.tokenize import sent_tokenize


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
    return len(intersection)/len(union)


def filterOutAlreadyGenerated(past_generations, current_sentence, label_placeholder, threshold=0.9):
    if len(past_generations) == 0:
        return False
    oop = [jaccard_similarity(cleanOutput(copy.deepcopy(current_sentence), label_placeholder), cleanOutput(copy.deepcopy(o),
                                                                                                           label_placeholder)) for o in past_generations]
    ch_max = max(oop)
    if ch_max >= threshold:
        return True
    else:
        return False


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


class InferenceGeneratorPop:
    def __init__(self, model,
                 experiments_dataset,
                 device,
                 max_iter=5,
                 sampling=True, verbose=False) -> None:
        self.model = model.to(device)
        self.experiments_dataset = experiments_dataset
        self.vectorizer = vectorizer = lambda x: experiments_dataset.test_dataset.processTableInfo(
            x)
        self.max_iter = max_iter
        self.sampling = sampling
        self.verbose = verbose
        self.device = device

    def fullNarration(self, example, seed, label_placeholder, max_length=200,
                      length_penalty=1.6, beam_size=10, return_top_beams=4):
        seed_everything(seed)
        example = copy.deepcopy(example)

    def iterativeGeneratorJoint(self, example, seed, label_placeholder, max_length=150,
                                length_penalty=1.6, beam_size=10, return_top_beams=4):
        seed_everything(seed)
        example = copy.deepcopy(example)
        sample_too = self.sampling
        bs = beam_size

        completed = False
        final_output = ""
        sofar = ""
        steps = 0

        previous_step = ""
        past_gens = []

        self.model.eval()
        with torch.no_grad():
            while not completed:
                if steps < len(example):
                    dat = example[steps]
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
                                                     repetition_penalty=3.5,
                                                     length_penalty=length_penalty,
                                                     early_stopping=True,
                                                     use_cache=True,
                                                     max_length=max_length,
                                                     no_repeat_ngram_size=2,
                                                     num_return_sequences=return_top_beams,
                                                     do_sample=sample_too,
                                                     eos_token_id=tokenizer_.eos_token_id,)
                # Convert the generated output to sentences
                oop = [tokenizer_.decode(sample_outputs[idx],
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True) for idx in range(return_top_beams)]

                # Because it is iterative, we expect to have the proper next sentence
                expected_terminal = f'N{steps+1}'

                # Get the sentences that correctly follow the order of information based on the next sentence tokens
                oop = [o for o in oop if expected_terminal in o or '[EON]' in o]

                oop = [o for o in oop if not filterOutAlreadyGenerated(
                    past_gens, o, label_placeholder)]
                oop = [(o, jaccard_similarity(cleanOutput(copy.deepcopy(o), label_placeholder),
                                              cleanOutput(copy.deepcopy(previous_step), label_placeholder))) for o in oop]
                oop = [o[0] for o in sorted(
                    oop, key=lambda x: x[-1], reverse=True)[-3:]]

                if len(oop) < 1:
                    break

                # Pick one of the best sentences for the next step
                best_choice = random.choice(oop)

                # Check if the end of narrative token has been generated, if yes then we terminate and return the output paragraph
                completed = '[EON]' in best_choice
                sofar = sofar+' '+best_choice+' '
                dat['preamble'] = sofar  # +' [NLS] '+prema
                final_output += ' '+best_choice
                previous_step = best_choice
                past_gens.append(best_choice)
                dat['prev_seq'] = sofar

                if completed:
                    print('Completed')
                    break

                if steps > self.max_iter:
                    break
                if self.verbose:
                    print(f'Generation Step {steps+1}')
                steps += 1
            return cleanOutput(final_output, label_placeholder)

    def MultipleIterativeGeneratorJoint(self, examples, seed, max_length=150,
                                        length_penalty=1.6, beam_size=10, return_top_beams=4):

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
            label_placeholder.update(
                {'feature'+k.split('feat')[-1]: v for k, v in rev_map.items()})
            label_placeholder['predictionslabel'] = label_placeholder[' predictionlabel']
            label_placeholder.update({'predictionranking'+k.split('rank')
                                     [-1]: v for k, v in label_placeholder.items() if 'rank' in k})
            label_placeholder.update(
                {'predictions'+k.split('prediction')[-1]: v for k, v in label_placeholder.items()})
            label_placeholder[', or'] = ' and '

            output_sentence = self.iterativeGeneratorJoint(
                example, seed, label_placeholder, max_length, length_penalty, beam_size, return_top_beams)

            generatedOutput.append(output_sentence)
        return generatedOutput


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
        self.vectorizer = vectorizer = lambda x: experiments_dataset.test_dataset.processTableInfo(
            x)
        self.max_iter = max_iter
        self.sampling = sampling
        self.verbose = verbose
        self.device = device

    def fullNarration(self, example, seed, label_placeholder, max_length=260,
                      length_penalty=1.6, beam_size=10, return_top_beams=4):
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
                                                 repetition_penalty=3.5,
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
                                length_penalty=1.6, beam_size=10, return_top_beams=4):
        seed_everything(seed)
        example = copy.deepcopy(example)
        sample_too = self.sampling
        bs = beam_size

        completed = False
        final_output = ""
        sofar = ""
        steps = 0

        previous_step = ""
        past_gens = []

        self.model.eval()
        rev_map = example[0]['reverse_map']
        with torch.no_grad():
            while not completed:
                if steps < len(example):
                    dat = example[steps]
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
                                                     repetition_penalty=3.5,
                                                     length_penalty=length_penalty,
                                                     early_stopping=True,
                                                     use_cache=True,
                                                     max_length=max_length,
                                                     no_repeat_ngram_size=2,
                                                     num_return_sequences=return_top_beams,
                                                     do_sample=sample_too,
                                                     eos_token_id=tokenizer_.eos_token_id,)
                # Convert the generated output to sentences
                oop = [tokenizer_.decode(sample_outputs[idx],
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True) for idx in range(return_top_beams)]

                # Because it is iterative, we expect to have the proper next sentence
                expected_terminal = f'N{steps+1}'

                # Get the sentences that correctly follow the order of information based on the next sentence tokens
                oop = [cleanFeatureHallucinations(
                    rev_map, o) for o in oop if expected_terminal in o or '[EON]' in o]

                oop = [o for o in oop if not filterOutAlreadyGenerated(
                    past_gens, o, label_placeholder)]
                oop = [(o, jaccard_similarity(cleanOutput(copy.deepcopy(o), label_placeholder),
                                              cleanOutput(copy.deepcopy(previous_step), label_placeholder))) for o in oop]
                oop = [o[0] for o in sorted(
                    oop, key=lambda x: x[-1], reverse=True)[-3:]]

                if len(oop) < 1:
                    break

                # Pick one of the best sentences for the next step
                best_choice = random.choice(oop)

                # Check if the end of narrative token has been generated, if yes then we terminate and return the output paragraph
                completed = '[EON]' in best_choice
                sofar = sofar+' '+best_choice+' '
                dat['preamble'] = sofar  # +' [NLS] '+prema
                final_output += ' '+best_choice
                previous_step = best_choice
                past_gens.append(best_choice)
                dat['prev_seq'] = sofar

                if completed:
                    print('Completed')
                    break

                if steps > self.max_iter:
                    break
                if self.verbose:
                    print(f'Generation Step {steps+1}')
                steps += 1
            return cleanOutput(final_output, label_placeholder)

    def MultipleIterativeGeneratorJoint(self, examples, seed, max_length=150,
                                        length_penalty=1.6, beam_size=10, return_top_beams=4):

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
            output_sentence = self.iterativeGeneratorJoint(
                example, seed, label_placeholder, max_length, length_penalty, beam_size, return_top_beams)

            generatedOutput.append(output_sentence)
        return generatedOutput

    def MultipleFullGeneratorJoint(self, examples, seed,  max_length=260,
                      length_penalty=1.6, beam_size=10, return_top_beams=4):

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
                example[0], seed, label_placeholder, max_length, length_penalty, beam_size, return_top_beams)

            generatedOutput.append(output_sentence)
        return generatedOutput
