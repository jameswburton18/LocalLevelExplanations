from attribution_sentence_templates import *
import copy
import functools
import re
import random
from nltk.tokenize import word_tokenize
import json


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
    confidence_levels = [c.strip().split(':') for c in pred_str.split(',')]#
    statement_end = f' <|section-sep|>  current_case | predicted_class | {predicted_class}'
    s = []
    for c in confidence_levels:
        s.append(f'{c[0].strip()} | pred_prob | {c[1].strip()} ')
    s = '&& '.join(s[:])
    return '<PRED_INFO> '+s + statement_end+' <|section-sep|> <FEATURES_ATTRIBUTIONS> '


def processPredictionProbabilities2(pred_str, predicted_class, randomise=False):
    confidence_levels = [c.strip().split(':') for c in pred_str.split(',')]
    preamble = f'<probable_class> {predicted_class} <|section-sep|> <pred_probs> '
    s = []
    if randomise:
        random.shuffle(confidence_levels)
    for c in confidence_levels:
        s.append(f'{c[1].strip()} for {c[0].strip()} ')

    s = '&& '.join(s[:])
    return preamble+s+' <|section-sep|> <RELEVANT_FEATURES_ATTRIBUTIONS> '


def cleanFeature(s):
    js = ''.join(s.replace('<|', '').replace('|>', '').split('#')).capitalize()
    return js


def processFeatureRanks(feature_division, narration, force_consistency=True):
    contradict, support, ignore = feature_division[
        'contradict'], feature_division['support'], feature_division['ignore']
    output = {'features': [], 'order': [], 'direction': []}
    nat = set([c.strip() for c in word_tokenize(narration)])
    pack = feature_division['rank'] if 'rank' in feature_division.keys() else feature_division['ranks']
    for f, pos in pack:
    #for f, pos in feature_division['rank']:
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
        if include:
            output['features'].append(f_)
            output['order'].append(pos)
            if f in contradict:
                output['direction'].append(-1)
            elif f in support:
                output['direction'].append(1)
            else:
                output['direction'].append(0)
        else:
            pass
    return output


def cleanNarrations(narration):
    narration = narration.replace('<#>', '').strip()
    ss = narration.replace('\u200b', '').replace('\n', '').strip()
    narration = ''.join(c for c in ss if c.isprintable())
    return narration


def getClassLabels(nb_classes):
    # The class label token is represented as #C{chr(i+97).upper()}
    classes = []
    for i in range(nb_classes):
        cl = '#C'+chr(i+97).upper()
        classes.append(cl)
    return classes


def linearisedFeaturesAttributions(feature_ranks):
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
            #irrelevants.append(f'{fl} ')

    s = '&& '.join(s[:])

    irrelevant = '&& '.join(irrelevants[:])

    #negative_features = stringifyFeats(negative_features)
    #positive_features = stringifyFeats(positive_features)
    #neutral_features = stringifyFeats(neutral_features)

    if len(irrelevants) < 1:
        return s+' <|section-sep|> ', [positive_features, negative_features, neutral_features]
    else:
        return s+f' <|section-sep|> <IRRELEVANT_FEATURES> {irrelevant} <|section-sep|> ', [positive_features, negative_features, neutral_features]


def processDataLinearized(data, randomise_preds=False, force_consistency=True):
    instance = data
    narration = cleanNarrations(instance['narration'])
    try:
        feat_div = json.loads(instance['feature_division'])
    except:
        feat_div = instance['feature_division']

    if len(narration) < 1:
        force_consistency = False
    feature_ranks = processFeatureRanks(
        feat_div, narration, force_consistency=force_consistency)
    feature_desc, [pf, nf, neu_f] = linearisedFeaturesAttributions(
        feature_ranks)
    preamble = processPredictionProbabilities2(instance['prediction_confidence_level'],
                                               instance['predicted_class'], randomise=randomise_preds)+feature_desc

    class_labels = getClassLabels(7)
    class_dict = {f'C{i+1}': c for i, c in enumerate(class_labels)}
    class_dict['C1 or C2']= '#CA or #CB'
    class_dict['C2 or C1']= '#CB or #CA'
    class_dict.update({f'c{i+1}': c for i, c in enumerate(class_labels)})

    extended1 = [functools.reduce(lambda a, kv: a.replace(*kv), class_dict.items(),
                                  re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [preamble]][0]
    extended2 = [functools.reduce(lambda a, kv: a.replace(*kv), class_dict.items(),
                                  re.sub('\s+', ' ', ss.strip().replace('\n', ' '))) for ss in [narration]][0]

    return {'preamble': extended1, 'narration': extended2, 'positives': pf, 'negatives': nf, 'neutral': neu_f, 'pred_label': class_dict[instance['predicted_class']]}


def composeTrainingData(data, force_consistency=True):
    aug_train = []
    preambles = []
    processed_train = []
    positive_preambles = []
    negative_preambles = []
    neutral_preambles = []
    for dat in data:
        examples = processDataLinearized(
            dat, randomise_preds=False, force_consistency=force_consistency)
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

def composeGetPredictionStatements(data,force_consistency=False):
    preambles = []
    processed_train = []
    for dat in data:
        examples = processDataLinearized(
            dat, randomise_preds=False, force_consistency=force_consistency)
        tt = examples['preamble']  # +examples['narration']
        zz = examples['preamble']+examples['narration']
        preambles.append(zz)
        examples_ = copy.deepcopy(examples)
        examples_['preamble'] = examples['preamble'].replace('<RELEVANT_FEATURES_ATTRIBUTIONS> <|section-sep|>','')+' <prediction_statement>'
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
        pred_label = examples['pred_label']
        if len(examples['negatives']) > 0:
            examples_n = copy.deepcopy(examples)
            if len(examples['negatives']) > 2:
                statements = negative_statements
            elif len(examples['negatives']) == 2:
                statements = negative_double
            else:
                statements = negative_singular

            examples_n['preamble'] = examples_n['preamble']+' <negative>'
            sentence = generateRandomSentences(statements,
                                               pred_label, examples['negatives'],
                                               feature_maps=['#fnegatives', '#fnegative_top', '#fnegative_others', '#fnegative_other'])
            examples_n['output'] =sentence
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
            sentence = generateRandomSentences(statements,
                                               pred_label, examples['positives'],
                                               feature_maps=['#fpositives','#fpositive_top','#fpositive_others','#fpositive_other'])
            examples_p['output'] =sentence
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
            sentence = generateRandomSentences(statements,
                                               pred_label, examples['neutral'],
                                               feature_maps=['#fneutral','#fneutral_top','#fneutral_others','#ffneutral_other'])
            examples_nl['output'] =sentence
            if tt+examples_nl['output'] not in set(neutral_preambles):
                processed_train.append(examples_nl)
                neutral_preambles.append(tt+examples_nl['output'])
    return processed_train
