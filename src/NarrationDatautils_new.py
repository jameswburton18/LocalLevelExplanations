import copy
import functools
import json
import os
import random
import re

import numpy as np
import sacrebleu
from nltk.tokenize import sent_tokenize, word_tokenize

from .attribution_sentence_templates import *


def readSentences(file, lower=False):
    with open(file, 'r', encoding="utf-8") as o_file:
        sentennces = []
        for s in o_file.readlines():
            ss = s.strip()  # .lower() if  lower else s.strip()
            sentennces.append(ss)
    return sentennces


def normalize_text(s):
    # pylint: disable=unnecessary-lambda
    def tokenize_fn(x): return sacrebleu.tokenizers.Tokenizer13a()(x)
    return tokenize_fn(s.strip().lower())


def writeToFile(content, filename):
    fil = filename+'.txt'
    if os.path.exists(fil):
        os.remove(fil)
    with open(fil, 'x') as fwrite:
        fwrite.writelines("%s\n" % s for s in content)
    print('Done')
    return


def stringifyFeats(feats):
    if len(feats) > 1:
        return ', '.join(feats[:-1]) + ' and ' + feats[-1]
    elif len(feats) == 1:
        return feats[0]
    else:
        return ''


def fillTheBlanks(sentence, tag, options):
    assert tag in sentence, f'Error {tag} not found in {sentence}'
    tag_options = {tag: options}
    extended1 = [functools.reduce(lambda a, kv: a.replace(*kv), tag_options.items(),
                                  re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [sentence]][0]
    return extended1


def generateRandomSentences(sentence_list, pred_class, features, feature_maps=['#fnegatives', '#fnegative_top', '#fnegative_others']):
    # {'singular':['#fnegatives'],'more':['#fnegatives','#fnegative_top','#fnegative_others']}
    placeholders = {'#pred_label': pred_class}

    if len(features) == 1:
        # get only the singles
        for m in feature_maps:
            placeholders[m] = features[0]
    elif len(features) == 2:
        placeholders[feature_maps[0]] = stringifyFeats(features)
        placeholders[feature_maps[1]] = stringifyFeats([features[0]])
        placeholders[feature_maps[2]] = stringifyFeats([features[1]])
        placeholders[feature_maps[3]] = stringifyFeats([features[1]])
    elif len(features) > 2:
        placeholders[feature_maps[0]] = stringifyFeats(features)
        placeholders[feature_maps[1]] = stringifyFeats(features[:2])
        placeholders[feature_maps[2]] = stringifyFeats(features[2:])
        placeholders[feature_maps[3]] = stringifyFeats(features[2:])

    sentence_list = [s if type(s) is not list else s[0] for s in sentence_list]
    sentence_list_ = [random.choice(sentence_list)]

    extended1 = [functools.reduce(lambda a, kv: a.replace(*kv), placeholders.items(),
                                  re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in sentence_list_]

    return extended1[0]


def processPredictionProbabilities1(pred_str, predicted_class):
    confidence_levels = [c.strip().split(':') for c in pred_str.split(',')]
    statement_end = f' <|section-sep|>  current_case | predicted_class | {predicted_class}'
    s = []
    for c in confidence_levels:
        s.append(f'{c[0].strip()} | pred_prob | {c[1].strip()} ')
    s = '&& '.join(s[:])
    return '<PRED_INFO> '+s + statement_end+' <|section-sep|> <FEATURES_ATTRIBUTIONS> '


def linearizePredictionStatement90(pred_str, predicted_class,):
    confidence_levels = [c.strip().split(':') for c in pred_str.split(',')]

    # if both labels are equally probable, we will just return the statement below
    if 'or' in predicted_class.split():
        preamble = f"case_label | predicted | {predicted_class.strip()} && "
        s = []
        for c in confidence_levels:
            s.append(f'{c[0].strip()} | not_prob_less | {c[1].strip()} ')
        s = preamble+' <|section-sep|> '+'&& '.join(s[:])
        return s
        # '{predicted_class.strip()} | pred_prob | {pred_confidence.strip()}

    zz = sorted(confidence_levels, key=lambda x: float(
        x[1].replace('%', '').strip()), reverse=True)

    pred_rank = {f'{c[0].strip()}': idx for idx, c in enumerate(zz)}
    nb_classes = len(confidence_levels)
    confidence_levels = {
        f'{c[0].strip()}': f"{c[1].strip()}" for c in confidence_levels}
    filler = {'pred_label': predicted_class,
              'pred_label_prob': confidence_levels[predicted_class]}
    pred_confidence = confidence_levels[predicted_class]

    preamble = f"case_label | predicted | {predicted_class.strip()} && {predicted_class.strip()} | pred_prob | {pred_confidence.strip()} "
    s = []

    for idx, (c, v) in enumerate(pred_rank.items()):
        if c.strip() != predicted_class.strip():
            s.append(
                f'{c.strip()} | not_prob_less | {confidence_levels[c].strip()} ')

    s = '&& '.join(s[:])+' <|section-sep|> ' + preamble
    return s


def linearizePredictionStatement8080(pred_str, predicted_class,):
    confidence_levels = [c.strip().split(':') for c in pred_str.split(',')]

    # if both labels are equally probable, we will just return the statement below
    if 'or' in predicted_class.split():
        preamble = f"case_label | predicted | {predicted_class.strip()} && "
        s = []
        for c in confidence_levels:
            s.append(f'{c[0].strip()} | not_prob_less | {c[1].strip()} ')
        s = preamble+' <|section-sep|> '+'&& '.join(s[:])
        return s
        # '{predicted_class.strip()} | pred_prob | {pred_confidence.strip()}

    zz = sorted(confidence_levels, key=lambda x: float(
        x[1].replace('%', '').strip()), reverse=True)

    pred_rank = {f'{c[0].strip()}': idx for idx, c in enumerate(zz)}
    nb_classes = len(confidence_levels)
    confidence_levels = {
        f'{c[0].strip()}': f"{c[1].strip()}" for c in confidence_levels}
    filler = {'pred_label': predicted_class,
              'pred_label_prob': confidence_levels[predicted_class]}
    pred_confidence = confidence_levels[predicted_class]

    preamble = f"prediction:{predicted_class.strip()} &&  {predicted_class.strip()}:{pred_confidence.strip()}"
    s = []

    for idx, (c, v) in enumerate(pred_rank.items()):
        if c.strip() != predicted_class.strip():
            s.append(f'{c.strip()}:{confidence_levels[c].strip()} ')

    s = preamble+' && '+'&& '.join(s[:])
    return s


def linearizePredictionStatement(pred_str, predicted_class,):
    confidence_levels = [c.strip().split(':') for c in pred_str.split(',')]

    placeholders = {predicted_class.strip(): ' predictionlabel', }
    labels = ['A', 'B', 'C', 'D', 'E', 'F']

    # if both labels are equally probable, we will just return the statement below
    if 'or' in predicted_class.split():
        # preamble= f"case_label | predicted | {predicted_class.strip()} && "
        preamble = f"prediction: predictionlabel && predictionlabel: {confidence_levels[0][1].strip()} "
        s = []
        idx_ = 1
        for idx, c in enumerate(confidence_levels):
            # s.append(f'{c[0].strip()} | not_prob_less | {c[1].strip()} ')
            idx_ = labels[idx_+1]
            s.append(f'predictionrank{idx_}: {c[1].strip()} ')
            # c[0].strip()
            placeholders[c[0].strip()] = f' predictionrank{idx_}'

        s = preamble+' <|section-sep|> '+'&& '.join(s[:])
        return s, placeholders
        # '{predicted_class.strip()} | pred_prob | {pred_confidence.strip()}

    zz = sorted(confidence_levels, key=lambda x: float(
        x[1].replace('%', '').strip()), reverse=True)

    pred_rank = {f'{c[0].strip()}': labels[idx] for idx, c in enumerate(zz)}
    nb_classes = len(confidence_levels)
    confidence_levels = {
        f'{c[0].strip()}': f"{c[1].strip()}" for c in confidence_levels}
    filler = {'predictionlabel': predicted_class,
              'predictionlabel_prob': confidence_levels[predicted_class]}
    pred_confidence = confidence_levels[predicted_class]

    preamble = f"prediction: predictionlabel && predictionlabel: {pred_confidence.strip()} "

    s = []

    for idx, (c, v) in enumerate(pred_rank.items()):
        if c.strip() != predicted_class.strip():

            s.append(f'predictionrank{v}: {confidence_levels[c].strip()} ')
            placeholders[c] = f'predictionrank{v}'  # c.strip()

    s = preamble+' && '+'&& '.join(s[:])
    return s, placeholders


def linearizePredictionStatementRt(pred_str, predicted_class,):
    confidence_levels = [c.strip().split(':') for c in pred_str.split(',')]

    placeholders = {predicted_class.strip(): ' prediction_label', }

    # if both labels are equally probable, we will just return the statement below
    if 'or' in predicted_class.split():
        # preamble= f"case_label | predicted | {predicted_class.strip()} && "
        preamble = f"prediction: prediction_label && prediction_label: {confidence_levels[0][1].strip()} "
        s = []
        idx_ = 1
        for idx, c in enumerate(confidence_levels):
            # s.append(f'{c[0].strip()} | not_prob_less | {c[1].strip()} ')
            idx_ = idx_+1
            s.append(f'rank{idx_}_predict: {c[1].strip()} ')
            placeholders[c[0].strip()] = f' rank{idx_}_predict'  # c[0].strip()

        s = preamble+' <|section-sep|> '+'&& '.join(s[:])
        return s, placeholders
        # '{predicted_class.strip()} | pred_prob | {pred_confidence.strip()}

    zz = sorted(confidence_levels, key=lambda x: float(
        x[1].replace('%', '').strip()), reverse=True)

    pred_rank = {f'{c[0].strip()}': idx+1 for idx, c in enumerate(zz)}
    nb_classes = len(confidence_levels)
    confidence_levels = {
        f'{c[0].strip()}': f"{c[1].strip()}" for c in confidence_levels}
    filler = {'prediction_label': predicted_class,
              'prediction_label_prob': confidence_levels[predicted_class]}
    pred_confidence = confidence_levels[predicted_class]

    preamble = f"prediction: prediction_label && prediction_label: {pred_confidence.strip()} "
    s = []

    for idx, (c, v) in enumerate(pred_rank.items()):
        if c.strip() != predicted_class.strip():
            s.append(f'rank{v}_predict: {confidence_levels[c].strip()} ')
            placeholders[c] = f'rank{v}_predict'  # c.strip()

    s = preamble+' && '+'&& '.join(s[:])
    return s, placeholders


def processPredictionProbabilities2(pred_str, predicted_class, randomise=False):
    return linearizePredictionStatement(pred_str, predicted_class)


def PredictionProbabilityPrep(pred_str, predicted_class, randomise=False):
    confidence_levels = [c.strip().split(':') for c in pred_str.split(',')]
    preamble = f'<probable_class> {predicted_class} <|section-sep|> <pred_probs> '


def cleanFeature(s):
    js = ''.join(s.replace('<|', '').replace('|>', '').split('#')).capitalize()
    return js








def processFeatureRanks(feature_division, narration, force_consistency=True, nb_base=None):
    contradict, support, ignore = feature_division[
        'contradict'], feature_division['support'], feature_division['ignore']
    output = {'features': [], 'order': [], 'direction': []}
    nat = set([c.strip() for c in word_tokenize(narration)])
    pack = feature_division['rank'] if 'rank' in feature_division.keys(
    ) else feature_division['ranks']

    # If there are only less than 25 features, we disable the force_consistency flag
    nb_features = len(pack)

    for f, pos in pack:
        f_ = cleanFeature(f)
        include = False
        if f_ in nat and force_consistency:
            include = True
        elif f_ not in nat and force_consistency:
            include = False
        elif not force_consistency:
            include = True  # random.choice([True,False])
            # print(include)
        # print(f_)
        # include=True
        if nb_base is not None and nb_features < nb_base:
            include = True
        if include:
            output['features'].append(f_)
            output['order'].append(pos)
            if f in contradict:
                output['direction'].append(-1)
            elif f in support:
                output['direction'].append(1)
            else:
                output['direction'].append(-2)
        else:
            pass
    return output


def cleanNarrations(narration):
    narration = narration.replace('<#>', '').strip()
    ss = narration.replace('\u200b', '').replace('\n', '').strip()
    narration = ''.join(c for c in ss if c.isprintable())
    narration = " ".join(narration.split())
    return narration


def getClassLabels(nb_classes):
    # The class label token is represented as #C{chr(i+97).upper()}
    classes = []
    for i in range(nb_classes):
        cl = '#C'+chr(i+97).upper()
        classes.append(cl)
    return classes


def linearisedFeaturesAttributions56(feature_ranks):
    flist, ford, fdir = list(feature_ranks.values())
    preamble = ''
    s = []
    irrelevants = []

    positive_features = []
    negative_features = []
    neutral_features = []

    for fl, fo, fd in zip(flist, ford, fdir):
        dir_ = 'NEUTRAL'
        if fd == 1:
            dir_ = 'POSITIVE'
            s.append(f'{fl} | attr_direction | {dir_} ')
            positive_features.append(fl)
        elif fd == -1:
            dir_ = 'NEGATIVE'
            s.append(f'{fl} | attr_direction | {dir_} ')
            negative_features.append(fl)
        else:
            s.append(f'{fl} | attr_direction | {dir_} ')
            neutral_features.append(fl)
            # irrelevants.append(f'{fl} ')

    s = '&& '.join(s[:])

    irrelevant = '&& '.join(irrelevants[:])

    if len(irrelevants) < 1:
        return s+' <|section-sep|> ', [positive_features, negative_features, neutral_features]
    else:
        return s+f' <|section-sep|> <IRRELEVANT_FEATURES> {irrelevant} <|section-sep|> ', [positive_features, negative_features, neutral_features]


def linearisedFeaturesAttributions(feature_ranks, shrink=None):
    flist, ford, fdir = list(feature_ranks.values())
    preamble = ''
    s = []
    irrelevants = []

    positive_features = []
    negative_features = []
    neutral_features = []

    directions = []
    features_ = []

    for idx, (fl, fo, fd) in enumerate(zip(flist, ford, fdir)):
        if shrink is not None:
            if idx > shrink + 2:
                break
        directions.append(fd)
        features_.append(fl)
        dir_ = '@'
        if fd == 1:
            dir_ = '+'
            s.append(f'{fl}:{dir_} ')
            positive_features.append(fl)
        elif fd == -1:
            dir_ = '-'
            s.append(f'{fl}:{dir_} ')
            negative_features.append(fl)
        else:
            s.append(f'{fl}:{dir_} ')
            neutral_features.append(fl)
    s = '&& '.join(s[:])

    irrelevant = '&& '.join(irrelevants[:])

    # negative_features = stringifyFeats(negative_features)
    # positive_features = stringifyFeats(positive_features)
    # neutral_features = stringifyFeats(neutral_features)

    if len(irrelevants) < 1:
        return s+' <|section-sep|> ', [positive_features, negative_features, neutral_features], [features_, directions]
    else:
        return s+f' <|section-sep|> <IRRELEVANT_FEATURES> {irrelevant} <|section-sep|> ',\
            [positive_features, negative_features,
                neutral_features], [features_, directions]

def  lineariseExplanation(data, attributions, randomise_preds=False, force_consistency=True, shrink=None):
    instance = data
    narration = cleanNarrations(copy.deepcopy(instance['next_sequence']))
    prev_seq = cleanNarrations(copy.deepcopy(instance['prev_seq']))
    full_narra = cleanNarrations(copy.deepcopy(instance['narration']))

    if len(narration) < 1:
        force_consistency = False
    feature_ranks =processFeatureAttributions(attributions, narration+prev_seq+full_narra, force_consistency=force_consistency)
    feature_ranks2 = processFeatureAttributions(attributions, narration, force_consistency=force_consistency)
    #processFeatureAttributions(attributions,nars[-1],force_consistency=iterative_gen)

    feature_desc, [pf, nf, neu_f], directions = linearisedFeaturesAttributions(
        feature_ranks, shrink=shrink)
    feature_desc2, [pf2, nf2, neu_f2], directions2 = linearisedFeaturesAttributions(
        feature_ranks2, shrink=shrink)

    preamble, place_holderss = processPredictionProbabilities2(instance['prediction_confidence_level'],
                                                               instance['predicted_class'], randomise=randomise_preds)

    preamble = preamble+' <|section-sep|> '+feature_desc

    class_labels = getClassLabels(7)
    class_dict = {f'C{i+1}': c for i, c in enumerate(class_labels)}
    class_dict['C1 or C2'] = '#CA or #CB'
    class_dict['C2 or C1'] = '#CB or #CA'
    class_dict.update({f'c{i+1}': c for i, c in enumerate(class_labels)})

    class_dict = place_holderss

    # [functools.reduce(lambda a, kv: a.replace(*kv), class_dict.items(),re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [preamble]][0]
    extended1 = preamble
    narr, prev = [functools.reduce(lambda a, kv: a.replace(*kv), class_dict.items(),
                                   re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [narration, prev_seq]]

    return {'rele_feat': [pf2, nf2, neu_f2], 'directions': directions, 'label_placeholders': place_holderss, 'preamble': extended1, 'narration': narr, 'prev_seq': prev,
            'positives': pf, 'negatives': nf, 'neutral': neu_f, 'pred_label': class_dict[instance['predicted_class']]}
    



def processDataLinearized(data, randomise_preds=False, force_consistency=True, shrink=None):
    instance = data
    narration = cleanNarrations(copy.deepcopy(instance['next_sequence']))
    prev_seq = cleanNarrations(copy.deepcopy(instance['prev_seq']))
    full_narra = cleanNarrations(copy.deepcopy(instance['narration']))
    # print(instance.keys())
    try:
        feat_div = json.loads(instance['feature_division'])
    except:
        feat_div = instance['feature_division']

    if len(narration) < 1:
        force_consistency = False
    feature_ranks = processFeatureRanks(
        feat_div, narration+prev_seq+full_narra, force_consistency=force_consistency)
    feature_ranks2 = processFeatureRanks(
        feat_div, narration, force_consistency=force_consistency)
    feature_desc, [pf, nf, neu_f], directions = linearisedFeaturesAttributions(
        feature_ranks, shrink=shrink)
    feature_desc2, [pf2, nf2, neu_f2], directions2 = linearisedFeaturesAttributions(
        feature_ranks2, shrink=shrink)

    preamble, place_holderss = processPredictionProbabilities2(instance['prediction_confidence_level'],
                                                               instance['predicted_class'], randomise=randomise_preds)

    preamble = preamble+' <|section-sep|> '+feature_desc

    class_labels = getClassLabels(7)
    class_dict = {f'C{i+1}': c for i, c in enumerate(class_labels)}
    class_dict['C1 or C2'] = '#CA or #CB'
    class_dict['C2 or C1'] = '#CB or #CA'
    class_dict.update({f'c{i+1}': c for i, c in enumerate(class_labels)})

    class_dict = place_holderss

    # [functools.reduce(lambda a, kv: a.replace(*kv), class_dict.items(),re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [preamble]][0]
    extended1 = preamble
    narr, prev = [functools.reduce(lambda a, kv: a.replace(*kv), class_dict.items(),
                                   re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [narration, prev_seq]]

    return {'rele_feat': [pf2, nf2, neu_f2], 'directions': directions, 'label_placeholders': place_holderss, 'preamble': extended1, 'narration': narr, 'prev_seq': prev,
            'positives': pf, 'negatives': nf, 'neutral': neu_f, 'pred_label': class_dict[instance['predicted_class']]}


def composeTrainingData(data, force_consistency=True, shrink=None):
    aug_train = []
    preambles = []
    processed_train = []
    positive_preambles = []
    negative_preambles = []
    neutral_preambles = []
    for dat in data:
        examples = processDataLinearized(
            copy.deepcopy(dat), randomise_preds=False, force_consistency=force_consistency, shrink=shrink)
        tt = examples['preamble']  # +examples['narration']
        zz = examples['preamble']+examples['narration']
        preambles.append(zz)
        examples_ = copy.deepcopy(examples)
        examples_['preamble'] = examples['preamble']+' <explain>'
        narr = examples['narration']
        positives = examples['positives']
        negatives = examples['negatives']
        neutrals = examples['neutral']
        examples_['output'] = narr
        processed_train.append(examples_)
    return processed_train


def composeGetPredictionStatements(data, include_randomise_preds=False, randomise_count=1, force_consistency=False):
    preambles = []
    processed_train = []
    for dat in data:
        examples = processDataLinearized(
            dat, randomise_preds=False, force_consistency=force_consistency)
        tt = examples['preamble']  # +examples['narration']
        zz = examples['preamble']+examples['narration']
        preambles.append(zz)
        examples_ = copy.deepcopy(examples)
        examples_['preamble'] = examples['preamble'].replace(
            '<RELEVANT_FEATURES_ATTRIBUTIONS> <|section-sep|>', '')+' <prediction_statement> '
        narr = examples['narration']
        examples_['output'] = narr
        processed_train.append(examples_)
    if include_randomise_preds:
        for _ in range(randomise_count):
            for dat in data:
                examples = processDataLinearized(
                    dat, randomise_preds=True, force_consistency=force_consistency)
                tt = examples['preamble']  # +examples['narration']
                zz = examples['preamble']+examples['narration']
                if zz not in set(preambles):
                    preambles.append(zz)
                    examples_ = copy.deepcopy(examples)
                    examples_['preamble'] = examples['preamble'].replace(
                        '<RELEVANT_FEATURES_ATTRIBUTIONS> <|section-sep|>', '')+' <prediction_statement> '
                    narr = examples['narration']
                    examples_['output'] = narr
                    processed_train.append(examples_)
    return processed_train


def composeRandomFeatureAttributionsStatements(data, force_consistency=True):
    aug_train = []
    preambles = []
    processed_train = []
    positive_preambles = []
    negative_preambles = []
    neutral_preambles = []
    for dat in data:
        examples = processDataLinearized(dat, randomise_preds=False)

        tt = examples['preamble']  # +examples['narration']
        zz = examples['preamble']+examples['narration']
        preambles.append(zz)
        examples = copy.deepcopy(examples)
        preambles.append(tt)
        pred_label = examples['pred_label'].strip()
        if len(examples['negatives']) > 0:
            examples_n = copy.deepcopy(examples)
            if len(examples['negatives']) > 2:
                statements = negative_statements
            elif len(examples['negatives']) == 2:
                statements = negative_double
            else:
                statements = negative_singular

            examples_n['preamble'] = examples_n['preamble']+' <negative>'
            bd = copy.deepcopy(examples['negatives'])
            # random.shuffle(bd)
            sentence = generateRandomSentences(statements,
                                               pred_label, bd,
                                               feature_maps=['#fnegatives', '#fnegative_top', '#fnegative_others', '#fnegative_other'])
            examples_n['output'] = sentence
            if tt+examples_n['output'] not in set(negative_preambles):
                processed_train.append(examples_n)
                negative_preambles.append(tt+examples_n['output'])
        if len(examples['positives']) > 0:
            if len(examples['positives']) > 2:
                statements = positive_statements
            elif len(examples['positives']) == 2:
                statements = positive_double
            else:
                statements = positive_singular
            examples_p = copy.deepcopy(examples)
            examples_p['preamble'] = examples_p['preamble']+' <positives>'
            bd = copy.deepcopy(examples['positives'])
            # random.shuffle(bd)
            sentence = generateRandomSentences(statements,
                                               pred_label, bd,
                                               feature_maps=['#fpositives', '#fpositive_top', '#fpositive_others', '#fpositive_other'])
            examples_p['output'] = sentence
            if tt+examples_p['output'] not in set(positive_preambles):
                processed_train.append(examples_p)
                positive_preambles.append(tt+examples_p['output'])

        if len(examples['neutral']) > 0:
            examples_nl = copy.deepcopy(examples)
            if len(examples['neutral']) > 1:
                statements = neutral_statements
            else:
                statements = neutral_statements_single
            examples_nl['preamble'] = examples_nl['preamble']+' <neutral>'
            examples_p['preamble'] = examples_p['preamble']+' <positives>'
            bd = copy.deepcopy(examples['neutral'])
            # random.shuffle(bd)
            sentence = generateRandomSentences(statements,
                                               pred_label, bd,
                                               feature_maps=['#fneutral', '#fneutral_top', '#fneutral_others', '#ffneutral_other'])
            examples_nl['output'] = sentence
            if tt+examples_nl['output'] not in set(neutral_preambles):
                processed_train.append(examples_nl)
                neutral_preambles.append(tt+examples_nl['output'])
    return processed_train


def finalProcessor(package, ignore=False):
    if ignore:
        return package

    def replace_percentage(x): return re.sub('%', ' percent', x)
    preamble = package['preamble']
    output_narration = replace_percentage(package['output'])
    prev_narr = replace_percentage(package['prev_seq'])
    features = package['directions'][0]
    package['directions_org'] = copy.deepcopy(package['directions'])

    def clear_pred_space(x): return re.sub('"\s+p', '"p', x)

    fidx = sorted([int(f[1:]) for f in features], reverse=True)
    feature_map = {f'F{idx}': f'feat{idx}value' for idx in fidx}
    package['preamble'] = [functools.reduce(lambda a, kv: a.replace(*kv), feature_map.items(),
                                            re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [preamble]][0]
    package['output'] = clear_pred_space([functools.reduce(lambda a, kv: a.replace(*kv), feature_map.items(),
                                                           re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [output_narration]][0])
    package['prev_seq'] = clear_pred_space([functools.reduce(lambda a, kv: a.replace(*kv), feature_map.items(),
                                                             re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [prev_narr]][0])
    package['directions'][0] = [feature_map[k] for k in features]
    package['positives'] = [feature_map[k] for k in package['positives']]
    package['negatives'] = [feature_map[k] for k in package['negatives']]
    package['neutral'] = [feature_map[k] for k in package['neutral']]

    pf2, nf2, neu_f2 = package['rele_feat']
    def ll(x): return [feature_map[k] for k in x]
    package['rele_feat'] = [ll(pf2), ll(nf2), ll(neu_f2)]
    package['feature_map'] = {v: k for k, v in feature_map.items()}

    return package


def cleanNarrations(narration):
    def repl(x): return re.sub('-', ' ', x)
    narration = narration.replace('<#>', '').strip()
    ss = narration.replace('\u200b', '').replace('\n', '').strip()
    narration = ''.join(c for c in ss if c.isprintable())
    narration = " ".join(repl(narration).split())
    return narration


def inferenceIterativeGenerator(pack, 
                               pr_c=1,
                               ignore=False,
                               force_section=False,
                               include_full_set=False,
                               ):
    pack = pack_x = copy.deepcopy(pack)
    outputs = pack['steps']

    max_init = 1



def iterNarationDataGenerators(pack, pr_c=1,
                               ignore=False,
                               force_section=False,
                               include_full_set=False,
                               ):
    pack = pack_x = copy.deepcopy(pack)
    outputs = sent_tokenize(cleanNarrations(pack['narration']))
    max_init = random.choice([3, 3, 2, 2, 2, 3, 1]) if not ignore else len(
        outputs) if not force_section else 1
    results = []
    if len(outputs) <= max_init:
        oop = [s+f'[N{i+1}S] ' for i, s in enumerate(outputs[:max_init])]
        sofar = ' '.join(oop)+' [EON]'
        pack_x = copy.deepcopy(pack)
        pack_x['next_sequence'] = sofar
        pack_x['prev_seq'] = '<prem>'
        return [pack_x]

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
        pack_f['next_sequence'] = sofar+' [EON] '+ outputs[-1]+' [CON]' #outputs[-1]+' [EON]'
        results.append(pack_f)

    return results


def reformulateInput22(pack, order_invariant=True, first_ins=0):
    pp = copy.deepcopy(pack)
    predict = pp['preamble'].split('<|section-sep|>')[0]
    nf = random.choice([10, 19, 16, 12, 17, 18, 20])
    feat_string = ' '.join(pp['directions'][0][:])
    pf2, nf2, neu_f2 = pp['rele_feat']
    positives = '<positives> '+' '.join(pf2)+' </positives>'
    negatives = "<negatives> "+" ".join(nf2)+" </negatives>"
    neutral = "<neutrals> "+" ".join(neu_f2)+" </neutrals>"

    res = predict+' <|section-sep|> ' + feat_string
    attributions = []

    # if len(pp['positives'])>0:
    attributions.append(' <|section-sep|> '+positives)
    # if len(pp['negatives'])>0:
    attributions.append(' <|section-sep|> '+negatives)
    # if len(pp['neutral'])>0:

    preamble_1 = res + ' <|section-sep|> <mentions> '+positives + ' <|section-sep|> ' +\
        negatives + ' <|section-sep|> '+neutral + ' <|section-sep|> <explain> '
    preamble_2 = res + ' <|section-sep|> <mentions> '+negatives + ' <|section-sep|> '+positives + ' <|section-sep|> ' +\
        neutral + ' <|section-sep|> <explain> '

    pack['new_preamble_1'] = preamble_1
    pack['new_preamble_2'] = preamble_2

    attributions = np.array(pack['directions'][-1])
    attributions[attributions == -1] = 2  # Negatives
    attributions[attributions == -2] = 0
    pack['attributions'] = attributions

    return pack


def reformulateInput(pack, order_invariant=True, first_ins=0):
    pp = copy.deepcopy(pack)
    predict = pp['preamble'].split('<|section-sep|>')[0]
    nf = random.choice([10, 19, 16, 12, 17, 18, 20])
    feat_string = ' '.join(pp['directions'][0][:])
    pf2, nf2, neu_f2 = pp['rele_feat']
    positives = '<positives> '+' '.join(pf2)+' </positives>'
    negatives = "<negatives> "+" ".join(nf2)+" </negatives>"
    neutral = "<neutrals> "+" ".join(neu_f2)+" </neutrals>"

    mentioned_features = ' <|section-sep|> '.join(
        [' '.join(pf2), " ".join(nf2), " ".join(neu_f2)])

    combined = [f'{f}:+' for f in pf2]+[f'{f}:-' for f in nf2]
    mentioned_features3 = ' <|section-sep|> '.join([' '.join([f'{f}:+' for f in pf2]),
                                                    " ".join([f'{f}:-' for f in nf2]), " ".join([f'{f}:@' for f in neu_f2])])

    if mentioned_features.strip() == "<mentions> <|section-sep|> <|section-sep|>  <|section-sep|> </mentions>":
        mentioned_features = "<mentions> </mentions>"

    res = predict+' <|section-sep|> ' + feat_string
    attributions = []

    # if len(pp['positives'])>0:
    attributions.append(' <|section-sep|> '+positives)
    # if len(pp['negatives'])>0:
    attributions.append(' <|section-sep|> '+negatives)
    # if len(pp['neutral'])>0:

    preamble_1 = res + ' <|section-sep|> <mentions> '+positives + ' <|section-sep|> ' +\
        negatives + ' <|section-sep|> '+neutral + ' <|section-sep|> <explain> '

    preamble_2 = res + ' <|section-sep|> <mentions> ' + \
        mentioned_features+' </mentions>  <|section-sep|> <explain> '

    preamble_3 = res + ' <|section-sep|> <mentions> ' + \
        mentioned_features3+' </mentions>  <|section-sep|> <explain> '

    preamble_4 = res + ' <|section-sep|> <mentions> '+negatives + ' <|section-sep|> '+positives + ' <|section-sep|> ' +\
        neutral + ' <|section-sep|> <explain> '

    pack['new_preamble_1'] = preamble_1
    pack['new_preamble_2'] = preamble_2
    pack['new_preamble_3'] = preamble_3
    pack['new_preamble_4'] = preamble_4

    attributions = np.array(pack['directions'][-1])
    attributions[attributions == -1] = 2  # Negatives
    attributions[attributions == -2] = 0
    pack['attributions'] = attributions

    return pack


def simpleMapper(pack):
    pack = copy.deepcopy(pack)
    mappers = dict()
    rev_mapper = dict()
    pos_idx, neg_idx, neut_idx = [0]*3
    feat_list = []
    feat_code = ['A',  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                 'K', 'L', 'M', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                 'X', 'Y', 'Z', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH',
                 'BI', 'BK', 'BS', 'BZ', ]

    for idx, f in enumerate(pack['directions'][0]):
        idx_code = feat_code[idx]
        key = ''
        if f in pack['positives']:
            key = f'feat{idx_code}P'
            pos_idx += 1
        elif f in pack['negatives']:
            key = f'feat{idx_code}N'
            neg_idx += 1
        elif f in pack['neutral']:
            key = f'feat{idx_code}O'
            neut_idx += 1
        mappers[f] = key
        rev_mapper[key] = pack['feature_map'][f]
        feat_list.append(key)

    # fix the preambles
    preamble_1, preamble_2, preamble_3 = [functools.reduce(lambda a, kv: a.replace(*kv), mappers.items(),
                                                           re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [pack['new_preamble_1'], pack['new_preamble_2'], pack['new_preamble_3']]]
    pack['output'], pack['prev_seq'] = [functools.reduce(lambda a, kv: a.replace(*kv), mappers.items(),
                                                         re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [pack['output'], pack['prev_seq']]]
    pack['new_preamble_1'] = preamble_1
    pack['new_preamble_2'] = preamble_2
    pack['new_preamble_3'] = preamble_3
    pack['new_preamble_4'] = [functools.reduce(lambda a, kv: a.replace(*kv), mappers.items(),
                                               re.sub('\s+', ' ', ss.strip().replace('\n',
                                                                                     ' '))) for ss in [pack['new_preamble_4']]][0]
    pack['reverse_map'] = rev_mapper
    pack['input_feats'] = feat_list

    return pack
