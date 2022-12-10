import numpy as np
import json
import random
import inflect

def create_classes_dict():
    sign_dict = {1: 'positive', -1: 'negative', 0: 'negligible'}
    x = dict()
    classes = ['C1', 'C2']
    random.shuffle(['C1', 'C2'])
    pct = random.randint(500,9500)/100
    other_pct = round(100 - pct, 2)
    class_dict  = {classes[0]: format(pct, '.2f'), classes[1]: format(other_pct, '.2f')}
    x['predicted_class'] = max(class_dict, key=class_dict.get)
    x['classes_dict'] = str({f'{k}': f'{v}%' for k,v in class_dict.items()})
    x['feature_nums'] = [f'F{i}' for i in random.sample(range(1, 80), 15)]
    values = sorted([random.randint(-50,50)/100 for i in range(15)], key=abs, reverse=True)
    x['sign'] = [sign_dict[np.sign(i)] for i in values]
    x['values'] =  x['values'] = [format(v, '.2f') for v in values]
    return x

def question_generator(dict):
    # 'What is the value of FA?'
    choice = random.randint(0,len(dict['feature_nums'])-1)
    feat = dict['feature_nums'][choice]
    q1 = f'What is the value of {feat}?'
    val = dict['values'][choice]
    a1 = val
    
    # What is FA - FB?
    choice1 = random.randint(0,len(dict['feature_nums'])-1)
    feat1 = dict['feature_nums'][choice1]
    val1 = dict['values'][choice1]
    choice2 = random.randint(0,len(dict['feature_nums'])-1)
    feat2 = dict['feature_nums'][choice2]
    val2 = dict['values'][choice2]
    q2 = f'What is the difference between {feat1} and {feat2}?'
    a2 = format(float(val1) - float(val2), '.2f')
    
    # What is the xth most important feature?
    p = inflect.engine()
    choice = random.randint(0,len(dict['feature_nums'])-1)
    feat = dict['feature_nums'][choice]
    q3 = f'What is the {p.ordinal(choice+1)} most important feature?'
    a3 = feat
    
    # Top x postive features: 
    x = random.randint(1,5)
    top_x_pos_fts = [ft for ft, sign in zip(dict['feature_nums'],dict['sign']) if sign == 'positive'][:x]
    q4 = f'What are the top {x} positive features?'
    a4 = ', '.join(top_x_pos_fts)
    
    # Top x negative features: 
    x = random.randint(1,5)
    top_x_neg_fts = [ft for ft, sign in zip(dict['feature_nums'],dict['sign']) if sign == 'negative'][:x]
    q4 = f'What are the top {x} negative features?'
    a4 = ', '.join(top_x_neg_fts)
    
    dict['questions'] = [q1, q2, q3, q4]
    dict['answers'] = [a1, a2, a3, a4]
    
    return dict

random.seed(77)
qa_data = [question_generator(create_classes_dict()) for i in range(20000)]
# Split into train, test, val 80:10:10
random.shuffle(qa_data)
train = qa_data[:int(0.8*len(qa_data))]
test = qa_data[int(0.8*len(qa_data)):int(0.9*len(qa_data))]
val = qa_data[int(0.9*len(qa_data)):]

with open('jb_data/qa_train.json', 'w') as f:
    json.dump(train, f)
with open('jb_data/qa_test.json', 'w') as f:
    json.dump(test, f)
with open('jb_data/qa_val.json', 'w') as f:
    json.dump(val, f)