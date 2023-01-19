import json
import re
import random

def prepare_dataset():
    # This part of the code is the format the data in a nice way, from the raw data
    random.seed(42)

    all_train = json.load(open('raw_data/all_train.json',encoding='utf-8'))
    test = json.load(open('raw_data/test_set_new.json',encoding='utf-8'))
    ds = all_train + test
    no_task = [x for x in ds if x.get('task_name', None) == None]
    ds = [x for x in ds if x.get('task_name', None) != None]

    sign_dict = {'red': 'negative', 'green': 'positive', 'yellow': 'negligible'}

    tasknames = set([(a['task_name'], a['predicted_class'], a['predicted_class_label']) for a in ds])
    task2name_dict = {f'{t}_{c}': name for (t, c, name) in tasknames}
    task2name_dict.update({'Air Quality Prediction_C4': 'Other',
                        'Cab Surge Pricing System_C1': 'Low',
                        'Cab Surge Pricing System_C2': 'Medium',
                        'Cab Surge Pricing System_C3': 'High',
                        'Car Acceptability Valuation_C3': 'Other A',
                        'Car Acceptability Valuation_C4': 'Other B',
                        'Concrete Strength Classification_C3': 'Other',
                        'Customer Churn Modelling_C3': 'Other',
                        'Flight Price-Range Classification_C4': 'Special',
                        'Food Ordering Customer Churn Prediction_C3': 'Accept',
                        'German Credit Evaluation_C3': 'Other',
                        'Mobile Price-Range Classification_C3': 'r3',
                        'Suspicious Bidding Identification_C2': 'Suspicious',
                        'Used Cars Price-Range Prediction_C3': 'Medium',
                        'Wine Quality Prediction_C1': 'low_quality',
                        })

    for i in range(len(ds)):
        # Some of the data is in string form, eval() is to convert it to dict
        try:
            ds[i]['feature_division'] = eval(ds[i]['feature_division'])
        except:
            ds[i]['feature_division'] = ds[i]['feature_division']
        ds[i]['feature_division']['explainable_df'] = eval(
            ds[i]['feature_division']['explainable_df'])

        # Some of the fields we want are inside the feature_division dict, moving them to the top level
        ds[i]['values'] = [format(val, '.2f') for val in ds[i]
                            ['feature_division']['explainable_df']['Values'].values()]
        ft_nums = [re.search('F\d*', val).group() for val in list(
            ds[i]['feature_division']['explainable_df']['annotate_placeholder'].values())]
        ft_names = list(ds[i]['feature_division']
                        ['explainable_df']['Variable'].values())
        ds[i]['sign'] = [sign_dict[x]
                        for x in ds[i]['feature_division']['explainable_df']['Sign'].values()]
        ds[i]['narrative_id'] = ds[i].pop('id')
        ds[i]['unique_id'] = i
        ds[i]['classes_dict'] = {v[0].strip(): v[1].strip() for v in [y.split(
            ':') for y in [x for x in ds[i]['prediction_confidence_level'].split(',')]]}
        ds[i]['narrative_questions'] = ds[i]['narrative_question'].strip(
            '<ul><li>/ ').split(' </li> <li> ')
        
        # Shuffle feature names to ensure separation of train and test
        new_ft_nums = ft_nums.copy()
        random.shuffle(new_ft_nums)
        old2new_ft_nums = dict(zip(ft_nums, new_ft_nums))
        ft_ptn = re.compile("|".join([f'{k}\\b' for k in old2new_ft_nums.keys()]))
        
        ds[i]['feature_nums'] = new_ft_nums
        ds[i]['ft_num2name'] = str(dict(zip(new_ft_nums, ft_names)))
        ds[i]['old2new_ft_nums'] = str(old2new_ft_nums)
        ds[i]['narration'] = ft_ptn.sub(
            lambda m: old2new_ft_nums[re.escape(m.group(0))], ds[i]['narration'])
        ds[i]['narrative_questions'] = [ft_ptn.sub(lambda m: old2new_ft_nums[re.escape(
            m.group(0))], x) for x in ds[i]['narrative_questions']]
        
        # # Shuffle class names too
        new_classes = list(ds[i]['classes_dict'].keys()).copy()
        random.shuffle(new_classes)
        old2new_classes = dict(zip(list(ds[i]['classes_dict'].keys()), new_classes))
        cls_ptn = re.compile("|".join([f'{k}\\b' for k in old2new_classes.keys()]))
        
        ds[i]['predicted_class'] = cls_ptn.sub(
            lambda m: old2new_classes[re.escape(m.group(0))], ds[i]['predicted_class'])
        ds[i]['narration'] = cls_ptn.sub(
            lambda m: old2new_classes[re.escape(m.group(0))], ds[i]['narration'])
        ds[i]['classes_dict'] = str(
            {old2new_classes[k]: v for k, v in ds[i]['classes_dict'].items()})
        ds[i]['narrative_questions'] = [cls_ptn.sub(lambda m: old2new_classes[re.escape(
            m.group(0))], x) for x in ds[i]['narrative_questions']]
        ds[i]['old2new_classes'] = str(old2new_classes)
        
        new2old_classes = {v: k for k, v in old2new_classes.items()}
        task_classes = [
            f"{ds[i]['task_name']}_{new2old_classes[c]}" for c in eval(ds[i]['classes_dict']).keys()]
        ds[i]['class2name'] = str({c: task2name_dict[t_c] for c, t_c in zip(
            eval(ds[i]['classes_dict']).keys(), task_classes)})

        
        for key in ['deleted', 'mturk_id','narrative_status', 'date_submitted',
                    'date_approved', 'features_placeholder', 'is_paid', 'redeem_code', 'narrator',
                    'user_ip','feature_division', 'narrative_question', 'prediction_confidence_level',
                    'test_instance', "prediction_confidence"]:
            try:
                ds[i].pop(key)
            except:
                pass
    # no_task = prepare_all(no_task)
    json.dump(ds, open('jb_data/all.json', 'w', encoding='utf-8'), indent=4)

    # Split into train, test, val 80:10:10
    random.shuffle(ds)
    train = ds[:int(0.8*len(ds))]
    test = ds[int(0.8*len(ds)):int(0.9*len(ds))]
    val = ds[int(0.9*len(ds)):]

    json.dump(train, open('jb_data/train.json', 'w', encoding='utf-8'), indent=4)
    json.dump(test, open('jb_data/test.json', 'w', encoding='utf-8'), indent=4)
    json.dump(val, open('jb_data/val.json', 'w', encoding='utf-8'), indent=4)


# This second part of the code is to create the augmented datasets
def prepare_aug_dataset():
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
                ft_names = eval(new[i+j*len(original)]['ft_num2name']).values()
                random.shuffle(new_ft_nums)
                old2new_ft_nums = dict(zip(old_ft_nums, new_ft_nums))
                old_new2new_ft_nums = dict(zip(old_new_ft_nums, new_ft_nums))
                ft_ptn = re.compile("|".join([f'{k}\\b' for k in old_new2new_ft_nums.keys()]))
                
                new[i+j*len(original)]['feature_nums'] = new_ft_nums
                new[i+j*len(original)
                    ]['ft_num2name'] = str(dict(zip(new_ft_nums, ft_names)))
                new[i+j*len(original)]['old2new_ft_nums'] = str(old2new_ft_nums)
                new[i+j*len(original)]['narration'] = ft_ptn.sub(
                    lambda m: old_new2new_ft_nums[re.escape(m.group(0))], new[i+j*len(original)]['narration'])
                new[i+j*len(original)]['narrative_questions'] = [ft_ptn.sub(lambda m: old_new2new_ft_nums[re.escape(m.group(0))], x)
                                                                for x in new[i+j*len(original)]['narrative_questions']]
                
                # # Shuffle class names too
                new[i+j*len(original)]['classes_dict'] = eval(new[i +
                                                                j*len(original)]['classes_dict'])
                new_classes = list(new[i+j*len(original)]
                                ['classes_dict'].keys()).copy()
                random.shuffle(new_classes)
                # old_new are the 'new' class names from the original train set creation
                # We maintain old2new as the original raw data classes to the newly shuffled classes
                old_new2new_classes = dict(
                    zip(list(eval(new[i+j*len(original)]['old2new_classes']).values()), new_classes))
                old2new_classes = dict(
                    zip(list(new[i+j*len(original)]['classes_dict'].keys()), new_classes))
                cls_ptn = re.compile(
                    "|".join([f'{k}\\b' for k in old_new2new_classes.keys()]))
                
                new[i+j*len(original)]['predicted_class'] = cls_ptn.sub(
                    lambda m: old_new2new_classes[re.escape(m.group(0))], new[i+j*len(original)]['predicted_class'])
                new[i+j*len(original)]['narration'] = cls_ptn.sub(
                    lambda m: old_new2new_classes[re.escape(m.group(0))], new[i+j*len(original)]['narration'])
                new[i+j*len(original)]['classes_dict'] = str(
                    {old_new2new_classes[k]: v for k, v in new[i+j*len(original)]['classes_dict'].items()})
                new[i+j*len(original)]['narrative_questions'] = [cls_ptn.sub(lambda m: old_new2new_classes[re.escape(
                    m.group(0))], x) for x in new[i+j*len(original)]['narrative_questions']]
                new[i+j*len(original)]['old2new_classes'] = str(old2new_classes)
                new[i+j*len(original)]['unique_id'] = original[i]['unique_id'] + \
                    j*len(original)
                new[i+j*len(original)]['class2name'] = str(
                    {old_new2new_classes[k]: v for k, v in eval(new[i+j*len(original)]['class2name']).items()})

        json.dump(new, open(f'jb_data/{ds}_augmented.json', 'w', encoding='utf-8'), indent=4)

if __name__ == "__main__":
    prepare_dataset()
    prepare_aug_dataset()