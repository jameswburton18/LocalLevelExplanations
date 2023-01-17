import inflect
import re

def linearise_input(data_row, method, max_fts=15, data_only=False):
    """Linearise data row to be in chosen form."""
    sign_dict = {'positive': 'pos', 'negative': 'neg', 'negligible': 'null'}
    
    
    # Linearising the data
    chosen_class = data_row["predicted_class"]
    classes_dict = eval(data_row["classes_dict"])
    other_classes = "& ".join([f"{k} {v}" for k,v in classes_dict.items() if k != chosen_class])
    
    feature_nums = data_row['feature_nums'][:max_fts]
    sign = [sign_dict[s] for s in data_row['sign']][:max_fts]
    values = data_row['values'][:max_fts]

    fts = "& ".join([f'{a} ' for a in feature_nums])
    pos_fts = "& ".join([f'{a} ' for a, b in zip(feature_nums, sign) if b == 'pos'])
    nega_fts = "& ".join([f'{a} ' for a, b in zip(feature_nums, sign) if b == 'neg'])
    negl_fts = "& ".join([f'{a} ' for a, b in zip(data_row['feature_nums'], sign) if b == 'null'])
    negl_fts = 'None' if negl_fts == '' else negl_fts

    essel_input = f'| predicted class | {chosen_class} {classes_dict[chosen_class]} | other classes | {other_classes} '\
        f'| features | {fts}| postive features | {pos_fts} | negative features | {nega_fts} | negligible features | {negl_fts} |'

    p = inflect.engine()

    ordinals = [p.ordinal(i+1) for i in range(len(feature_nums))]

    ord_first_fts = ' '.join([f'| {o} | {f} {s} {v}' for o, f, s, v in 
                              zip(ordinals, feature_nums, sign, values)])
    ft_first_fts = ' '.join([f'| {f} | {o} {s} {v}' for o, f, s, v in
                             zip(ordinals, feature_nums, sign, values)])
        
    ord_first_input = f'| predicted class | {chosen_class} {classes_dict[chosen_class]} | other classes | {other_classes} {ord_first_fts} |'

    ft_first_input = f'| predicted class | {chosen_class} {classes_dict[chosen_class]} | other classes | {other_classes} {ft_first_fts} |'
    
    #### Text input
    text_negl_fts = data_row['feature_nums'][-5:]
    text_input = f'Predicted class is {chosen_class}, value of {classes_dict[chosen_class]}. '\
        f'Other classes and values are {other_classes}. '\
        f'Top features are [{commas_and_and(feature_nums)}], with values [{commas_and_and(values)}]. '\
        f'Postive features are [{commas_and_and([f for f, s in zip(feature_nums, sign) if s == "pos"])}]. '\
        f'Negative features are [{commas_and_and([f for f, s in zip(feature_nums, sign) if s == "neg"])}]. '\
        f'Lowest impact features are [{commas_and_and(data_row["feature_nums"][-5:])}] with values [{commas_and_and(data_row["values"][-5:])}].'


    if data_only:
        preamble = ''
        questions = ''
    else:
        # Preamble
        preamble = " | Questions | "
        questions = ' '.join([f'{idx+1}. {q}' for idx, q in 
                            enumerate(data_row['narrative_questions'])])

    if method == 'essel':
        data_row['input'] = essel_input + preamble + questions
    elif method == 'ord_first':
        data_row['input'] = ord_first_input + preamble + questions
    elif method == 'ft_first':
        data_row['input'] = ft_first_input + preamble + questions
    elif method == 'text':
        data_row['input'] = text_input
    else:
        raise ValueError('method must be one of essel, ord_first or ft_first')

    return data_row

def form_stepwise_input(data_row, method, max_fts):
    # Linearising the data
    data_row = linearise_input(data_row, method=method, max_fts=max_fts, data_only=True)
    preamble = "\n <br> <br> Using the above information, answer the following \
            in detail: <br> <br> "
    data_row['input'] = [data_row['input'] + preamble + q for q in data_row['narrative_questions']]
    return data_row

def form_qa_input_output(data_row, method, max_fts=15):
    """Combining the quesiton with the linearised data and a preamble. Also
    renaming answer as narration so as to match `convert_to_features()`."""
    # Linearising the data
    data_row = linearise_input(data_row, method=method, max_fts=max_fts, data_only=True)
    # Preamble
    preamble = " Answer the following question: "
    
    data_row['input'] = data_row['input'] + preamble + data_row['question']
    data_row['narration'] = data_row['answer']
    return data_row

def convert_to_features(batch, tokenizer, max_input_length=400, max_output_length=350):
    if type(batch['input'][0]) == list:
        # input_encodings = [tokenizer(i, padding="max_length", truncation=True, max_length=max_input_length) for i in batch['input']]
        input_encodings = [tokenizer(i, padding=True, truncation=True) for i in batch['input']]
        input_ids = [i['input_ids'] for i in input_encodings]
        attention_mask = [i['attention_mask'] for i in input_encodings]
    else:
        # input_encodings = tokenizer(batch['input'], padding="max_length", truncation=True, max_length=max_input_length)
        input_encodings = tokenizer(batch['input'], padding=True, truncation=True)
        input_ids = input_encodings['input_ids']
        attention_mask = input_encodings['attention_mask']
    # target_encodings = tokenizer(batch['narration'], padding="max_length", truncation=True, max_length=max_output_length)
    target_encodings = tokenizer(batch['narration'], padding=True, truncation=True)
    encodings = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': target_encodings['input_ids'],
    }
    return encodings

def simplify_feat_names(row):
    '''Simplifying the feature names to make them more readable. We do this
    by removing the cases where it says (value equal to V0)
    '''
    row['narrative_questions'] = [re.sub(r' \(([^()]*\bequal[^()]*)\)', '', q) for q in row['narrative_questions']]
    return row

def label_qs(row):
    narr_qs = row['narrative_questions']
    if narr_qs[0] == 'In a single sentence, state the prediction output of the model for the selected test case along with the confidence level of the prediction (if applicable).':
        label = 'A'
    elif narr_qs[0] == "For this test instance, provide information on the predicted label along with the confidence level of the model's decision.":
        label = 'B'
    elif narr_qs[0] == 'Summarize the prediction for the given test example?' and narr_qs[1] == "In two sentences, provide a brief overview of the features with a higher impact on the model's output prediction.":
        label = 'C'
    elif narr_qs[0] == 'Summarize the prediction for the given test example?' and narr_qs[1] == "For this test case, summarize the top features influencing the model's decision.": 
        label = 'D'
    elif narr_qs[0] == "Summarize the prediction made for the test under consideration along with the likelihood of the different possible class labels." and narr_qs[1] == 'Provide a statement summarizing the ranking of the features as shown in the feature impact plot.':
        label = 'E'
    elif narr_qs[0] == 'Provide a statement summarizing the prediction made for the test case.':
        label = 'F'
    elif narr_qs[0] == 'Provide a statement summarizing the ranking of the features as shown in the feature impact plot.':
        label = 'G'
    elif narr_qs[0] == 'Summarize the prediction made for the test under consideration along with the likelihood of the different possible class labels.' and narr_qs[1][:53] == 'Summarize the direction of influence of the variables':
        label = 'H'
    else:
        raise ValueError('Unknown narrative question')
    row['narr_q_label'] = label
    grouping = {'A': 'A-B', 'B': 'A-B', 'C': 'C-E', 'D': 'C-E', 
                'E': 'C-E', 'F': 'F-H', 'G': 'F-H', 'H': 'F-H'}
    row['narr_q_label_group'] = grouping[label]
    return row

def commas_and_and(fts_list):
    if len(fts_list) == 0:
        return ' '
    elif len(fts_list) == 1:
        return fts_list[0]
    elif len(fts_list) == 2:
        return f'{fts_list[0]} and {fts_list[1]}'
    else:
        return f'{", ".join(fts_list[:-1])}, and {fts_list[-1]}'

def simplify_narr_question(row):
    label = row['narr_q_label']
    reg = re.compile(r'F\d+')
    mentioned_fts = [reg.findall(n) for n in row['narrative_questions']]
    q1 = "Summarise the prediction."
    
    if label in ['A', 'B', 'C', 'D']:
        q2 = 'Summarise the top features.'
    elif label == 'E':
        q2 = f'Summarise these top features ({commas_and_and(mentioned_fts[2])}).'
    else: # label in ['F', 'G', 'H']
        q2 = f'Summarise these top features ({commas_and_and(mentioned_fts[1])}).'
        
    if label in ['A', 'B', 'C', 'F', 'G', 'H']:
        q3 = f'Summarise these moderate features ({commas_and_and(mentioned_fts[2])}).'
    elif label ==  'D':
        q3 = '' # Ds have no q3 
    else: # label ==  'E'
        q3 = f'Summarise these moderate features ({commas_and_and(mentioned_fts[3])}).'
    
    if label in ['A', 'B']:
        q4 = ''
    elif label in ['C', 'D', 'E']:
        q4 = f'Summarise the negligible features.'
    else: # label in ['F', 'G', 'H']
        q4 = f'Summarise these negligible features ({commas_and_and(mentioned_fts[3])}).'
        
    row['original_narrative_questions'] = row['narrative_questions']
    row['narrative_questions'] = [q1, q2, q3, q4]
    return row

def old_linearise_input(data_row, method, max_fts=15, data_only=False):
    """Linearise data row to be in chosen form. 
    This is the old version being kept just in case. the new 'linearise_input'
    is slimmed down and includes the class values which were mistakenly
    missed out here"""
    
    # Linearising the data
    chosen_class = data_row["predicted_class"]
    classes_dict = eval(data_row["classes_dict"])
    other_classes = "&& ".join([f"{k} {v}" for k,v in classes_dict.items() if k != chosen_class])
    
    feature_nums = data_row['feature_nums'][:max_fts]
    sign = data_row['sign'][:max_fts]
    values = data_row['values'][:max_fts]
    
    fts = "&& ".join([f'{a} ' for a in feature_nums])
    pos_fts = "&& ".join([f'{a} ' for a, b in zip(feature_nums, sign) if b == 'positive'])
    nega_fts = "&& ".join([f'{a} ' for a, b in zip(feature_nums, sign) if b == 'negative'])
    negl_fts = "&& ".join([f'{a} ' for a, b in zip(feature_nums, sign) if b == 'negligible'])
    negl_fts = 'None' if negl_fts == '' else negl_fts

    essel_input = f'| predicted class | {chosen_class} {classes_dict[chosen_class]} | other classes | {other_classes} | \
    features | {fts}| postive features | {pos_fts} | negative features | {nega_fts} | \
    negligible features | {negl_fts} |'

    p = inflect.engine()

    ordinals = [p.ordinal(i+1) for i in range(len(feature_nums))]

    features = ' '.join([f'| {o} | {f} {s} {v}' for o, f, s, v in 
                         zip(ordinals, feature_nums, sign, 
                             values)])
        
    ord_first_input = f'| predicted class | {chosen_class} {classes_dict[chosen_class]} | other classes | {other_classes} {features} |'

    ft_first_input = ' '.join([f'| {f} | {o} {s} {v}' for o, f, s, v in zip(ordinals, feature_nums, sign, values)])

    if data_only:
        preamble = ''
        questions = ''
    else:
        # Preamble
        preamble = "\n <br> <br> Using the above information, answer the following in detail: <br> <br> "
        questions = '\n'.join([f'{idx+1}. {q}' for idx, q in 
                            enumerate(data_row['narrative_questions'])])

    if method == 'essel':
        data_row['input'] = essel_input + preamble + questions
    elif method == 'ord_first':
        data_row['input'] = ord_first_input + preamble + questions
    elif method == 'ft_first':
        data_row['input'] = ft_first_input + preamble + questions
    else:
        raise ValueError('method must be one of essel, ord_first or ft_first')

    return data_row

