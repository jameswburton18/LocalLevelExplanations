import json
import re
from collections import Counter
import inflect
import random

random.seed(42)

all_train = json.load(open('raw_data/all_train.json',encoding='utf-8'))
test = json.load(open('raw_data/test_set_new.json',encoding='utf-8'))
all = all_train + test
no_task = [x for x in all if x.get('task_name', None) == None]
all = [x for x in all if x.get('task_name', None) != None]

sign_dict = {'red': 'negative', 'green': 'positive', 'yellow': 'negligible'}

for i in range(len(all)):
    # Some of the data is in string form, eval() is to convert it to dict
    try:
        all[i]['feature_division'] = eval(all[i]['feature_division'])
    except:
        all[i]['feature_division'] = all[i]['feature_division']
    all[i]['feature_division']['explainable_df'] = eval(all[i]['feature_division']['explainable_df'])
    
    # Some of the fields we want are inside the feature_division dict, moving them to the top level
    all[i]['values'] = [format(val, '.2f') for val in all[i]['feature_division']['explainable_df']['Values'].values()]
    ft_nums =[re.search('F\d*', val).group() for val in list(all[i]['feature_division']['explainable_df']['annotate_placeholder'].values())]
    ft_names = list(all[i]['feature_division']['explainable_df']['Variable'].values())
    all[i]['sign'] = [sign_dict[x] for x in all[i]['feature_division']['explainable_df']['Sign'].values()]
    all[i]['narrative_id'] = all[i].pop('id')
    all[i]['unique_id'] = i
    all[i]['classes_dict'] = {v[0].strip(): v[1].strip() for v in [y.split(':') for y in [x for x in all[i]['prediction_confidence_level'].split(',')]]}
    all[i]['narrative_questions'] = all[i]['narrative_question'].strip('<ul><li>/ ').split(' </li> <li> ')
    
    # Shuffle feature names to ensure separation of train and test
    new_ft_nums = ft_nums.copy()
    random.shuffle(new_ft_nums)
    old2new_ft_nums = dict(zip(ft_nums, new_ft_nums))
    ft_ptn = re.compile("|".join([f'{k}\\b' for k in old2new_ft_nums.keys()]))
    
    all[i]['feature_nums'] = new_ft_nums
    all[i]['ft_num_to_name'] = str(dict(zip(new_ft_nums, ft_names)))
    all[i]['old2new_ft_nums'] = str(old2new_ft_nums)
    all[i]['narration'] = ft_ptn.sub(lambda m: old2new_ft_nums[re.escape(m.group(0))], all[i]['narration'])
    all[i]['narrative_questions'] = [ft_ptn.sub(lambda m: old2new_ft_nums[re.escape(m.group(0))], x) for x in all[i]['narrative_questions']]
    
    # # Shuffle class names too
    new_classes = list(all[i]['classes_dict'].keys()).copy()
    random.shuffle(new_classes)
    old2new_classes = dict(zip(list(all[i]['classes_dict'].keys()), new_classes))
    cls_ptn = re.compile("|".join([f'{k}\\b' for k in old2new_classes.keys()]))
    
    all[i]['predicted_class'] = cls_ptn.sub(lambda m: old2new_classes[re.escape(m.group(0))], all[i]['predicted_class'])
    all[i]['narration'] = cls_ptn.sub(lambda m: old2new_classes[re.escape(m.group(0))], all[i]['narration'])
    all[i]['classes_dict'] = str({old2new_classes[k]: v for k, v in all[i]['classes_dict'].items()})
    all[i]['narrative_questions'] = [cls_ptn.sub(lambda m: old2new_classes[re.escape(m.group(0))], x) for x in all[i]['narrative_questions']]
    all[i]['old2new_classes'] = str(old2new_classes)

    
    for key in ['deleted', 'mturk_id','narrative_status', 'predicted_class_label', 'date_submitted',
                'date_approved', 'features_placeholder', 'is_paid', 'redeem_code', 'narrator',
                'user_ip','feature_division', 'narrative_question', 'prediction_confidence_level',
                'test_instance', "prediction_confidence"]:
        try:
            all[i].pop(key)
        except:
            pass
# no_task = prepare_all(no_task)
json.dump(all, open('jb_data/all.json', 'w', encoding='utf-8'), indent=4)

# Split into train, test, val 80:10:10
random.shuffle(all)
train = all[:int(0.8*len(all))]
test = all[int(0.8*len(all)):int(0.9*len(all))]
val = all[int(0.9*len(all)):]

json.dump(train, open('jb_data/train.json', 'w', encoding='utf-8'), indent=4)
json.dump(test, open('jb_data/test.json', 'w', encoding='utf-8'), indent=4)
json.dump(val, open('jb_data/val.json', 'w', encoding='utf-8'), indent=4)
random.seed(2)

for ds in ['train', 'val']:
    num_repeats = 10
    new = []
    for j in range(num_repeats):
        original = json.load(open(f'jb_data/{ds}.json', 'r', encoding='utf-8'))
        for i in range(len(original)):
            new.append(original[i].copy())
            # Shuffle feature names to ensure separation of train and test
            new_ft_nums = new[i+j*len(original)]['feature_nums'].copy()
            old_ft_nums = eval(new[i+j*len(original)]['old2new_ft_nums']).keys()
            # old_new are the 'new' feature numbers from the original train set creation
            old_new_ft_nums = eval(new[i+j*len(original)]['old2new_ft_nums']).values()
            ft_names = eval(new[i+j*len(original)]['ft_num_to_name']).values()
            random.shuffle(new_ft_nums)
            old2new_ft_nums = dict(zip(old_ft_nums, new_ft_nums))
            old_new2new_ft_nums = dict(zip(old_new_ft_nums, new_ft_nums))
            ft_ptn = re.compile("|".join([f'{k}\\b' for k in old_new2new_ft_nums.keys()]))
            
            new[i+j*len(original)]['feature_nums'] = new_ft_nums
            new[i+j*len(original)]['ft_num_to_name'] = str(dict(zip(new_ft_nums, ft_names)))
            new[i+j*len(original)]['old2new_ft_nums'] = str(old2new_ft_nums)
            new[i+j*len(original)]['narration'] = ft_ptn.sub(lambda m: old_new2new_ft_nums[re.escape(m.group(0))], new[i+j*len(original)]['narration'])
            new[i+j*len(original)]['narrative_questions'] = [ft_ptn.sub(lambda m: old_new2new_ft_nums[re.escape(m.group(0))], x) for x in new[i+j*len(original)]['narrative_questions']]
            
            # # Shuffle class names too
            new[i+j*len(original)]['classes_dict'] = eval(new[i+j*len(original)]['classes_dict'])
            new_classes = list(new[i+j*len(original)]['classes_dict'].keys()).copy()
            random.shuffle(new_classes)
            # old_new are the 'new' class names from the original train set creation
            # We maintain old2new as the original raw data classes to the newly shuffled classes
            old_new2new_classes = dict(zip(list(eval(new[i+j*len(original)]['old2new_classes']).values()), new_classes))
            old2new_classes = dict(zip(list(new[i+j*len(original)]['classes_dict'].keys()), new_classes))
            cls_ptn = re.compile("|".join([f'{k}\\b' for k in old_new2new_classes.keys()]))
            
            new[i+j*len(original)]['predicted_class'] = cls_ptn.sub(lambda m: old_new2new_classes[re.escape(m.group(0))], new[i+j*len(original)]['predicted_class'])
            new[i+j*len(original)]['narration'] = cls_ptn.sub(lambda m: old_new2new_classes[re.escape(m.group(0))], new[i+j*len(original)]['narration'])
            new[i+j*len(original)]['classes_dict'] = str({old_new2new_classes[k]: v for k, v in new[i+j*len(original)]['classes_dict'].items()})
            new[i+j*len(original)]['narrative_questions'] = [cls_ptn.sub(lambda m: old_new2new_classes[re.escape(m.group(0))], x) for x in new[i+j*len(original)]['narrative_questions']]
            new[i+j*len(original)]['old2new_classes'] = str(old2new_classes)
            new[i+j*len(original)]['unique_id'] = original[i]['unique_id'] + j*len(original)

    json.dump(new, open(f'jb_data/{ds}_augmented.json', 'w', encoding='utf-8'), indent=4)
print()