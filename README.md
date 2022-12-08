This readme is me trying to piece together what is happening in this repo

Dataset:
* Creating a DatasetBuilder and then calling fit() will make a .train_dataset and .test_dataset, which are ready to be input into PyTorch, ie has input ids, attention masks, labels, etc.
* The default is that is has been 'composed' already, which I am 
* There are 530 datapoints in all_train.json
* Of these 109 cases where the task or the model is not specified.
* Of these 109 they have the following information: 'feature_division, narration, predicted_class, prediction_confidence_level'
* Of the remaining 421, there are 40 unique tasks (median=11.0 datapoints per task) and 21 unique models (median = 10.5 datapoints per model)
* In 176 cases the 

* Across all_train.json and test_set_new.json there are 578 datapoints
* Of these 109 cases where the task or the model is not specified.
* Of the remaining 469, there are 40 unique tasks and 20 unique models 

in datasetHandlers.py Essel covers the special tokens that are used
On multiple yamls :https://crumb.sh/3FkJXZuuK4H

This is a datapoint

    {
        "model_name": "LogisticRegression",
        "deleted": false,
        "mturk_id": "Basic",
        "predicted_class": "C1",
        "narrative_status": 1,
        "predicted_class_label": "dissatisfied",
        "date_submitted": "06/10/2021",
        "id": 1,
        "feature_division": "{\"rank\": [[\"<|f#3|>\", 0], [\"<|f#4|>\", 1], [\"<|f#6|>\", 2], [\"<|f#12|>\", 3], [\"<|f#11|>\", 4], [\"<|f#14|>\", 5], [\"<|f#13|>\", 6], [\"<|f#9|>\", 7], [\"<|f#10|>\", 8], [\"<|f#15|>\", 9], [\"<|f#5|>\", 10], [\"<|f#7|>\", 11], [\"<|f#2|>\", 12], [\"<|f#8|>\", 13], [\"<|f#1|>\", 14]], \"annotate_code\": [\"F3-V0\", \"F4-V0\", \"F6\", \"F12\", \"F11\", \"F14\", \"F13\", \"F9\", \"F10\", \"F15\", \"F5\", \"F7\", \"F2-V0\", \"F8\", \"F1-V0\"], \"explainable_df\": \"{\\\"Values\\\":{\\\"2\\\":-0.2954524482,\\\"3\\\":-0.2525100076,\\\"5\\\":0.2296850087,\\\"11\\\":0.1492687389,\\\"10\\\":0.0917511915,\\\"13\\\":0.0863306218,\\\"12\\\":-0.0746328313,\\\"8\\\":0.065316989,\\\"9\\\":-0.0643314259,\\\"14\\\":0.0543745922,\\\"4\\\":0.047533335,\\\"6\\\":0.0183182452,\\\"1\\\":0.0165591439,\\\"7\\\":-0.0132520169,\\\"0\\\":0.0093566921},\\\"Variable\\\":{\\\"2\\\":\\\"Type of Travel\\\",\\\"3\\\":\\\"Type Of Booking\\\",\\\"5\\\":\\\"Hotel wifi service\\\",\\\"11\\\":\\\"Common Room entertainment\\\",\\\"10\\\":\\\"Stay comfort\\\",\\\"13\\\":\\\"Other service\\\",\\\"12\\\":\\\"Checkin\\\\/Checkout service\\\",\\\"8\\\":\\\"Hotel location\\\",\\\"9\\\":\\\"Food and drink\\\",\\\"14\\\":\\\"Cleanliness\\\",\\\"4\\\":\\\"Age\\\",\\\"6\\\":\\\"Departure\\\\/Arrival  convenience\\\",\\\"1\\\":\\\"purpose_of_travel\\\",\\\"7\\\":\\\"Ease of Online booking\\\",\\\"0\\\":\\\"Gender\\\"},\\\"effect_abs\\\":{\\\"2\\\":0.2954524482,\\\"3\\\":0.2525100076,\\\"5\\\":0.2296850087,\\\"11\\\":0.1492687389,\\\"10\\\":0.0917511915,\\\"13\\\":0.0863306218,\\\"12\\\":0.0746328313,\\\"8\\\":0.065316989,\\\"9\\\":0.0643314259,\\\"14\\\":0.0543745922,\\\"4\\\":0.047533335,\\\"6\\\":0.0183182452,\\\"1\\\":0.0165591439,\\\"7\\\":0.0132520169,\\\"0\\\":0.0093566921},\\\"placeholder\\\":{\\\"2\\\":\\\"<|f#3|>\\\",\\\"3\\\":\\\"<|f#4|>\\\",\\\"5\\\":\\\"<|f#6|>\\\",\\\"11\\\":\\\"<|f#12|>\\\",\\\"10\\\":\\\"<|f#11|>\\\",\\\"13\\\":\\\"<|f#14|>\\\",\\\"12\\\":\\\"<|f#13|>\\\",\\\"8\\\":\\\"<|f#9|>\\\",\\\"9\\\":\\\"<|f#10|>\\\",\\\"14\\\":\\\"<|f#15|>\\\",\\\"4\\\":\\\"<|f#5|>\\\",\\\"6\\\":\\\"<|f#7|>\\\",\\\"1\\\":\\\"<|f#2|>\\\",\\\"7\\\":\\\"<|f#8|>\\\",\\\"0\\\":\\\"<|f#1|>\\\"},\\\"annotate_placeholder\\\":{\\\"2\\\":\\\"F3-V0\\\",\\\"3\\\":\\\"F4-V0\\\",\\\"5\\\":\\\"F6\\\",\\\"11\\\":\\\"F12\\\",\\\"10\\\":\\\"F11\\\",\\\"13\\\":\\\"F14\\\",\\\"12\\\":\\\"F13\\\",\\\"8\\\":\\\"F9\\\",\\\"9\\\":\\\"F10\\\",\\\"14\\\":\\\"F15\\\",\\\"4\\\":\\\"F5\\\",\\\"6\\\":\\\"F7\\\",\\\"1\\\":\\\"F2-V0\\\",\\\"7\\\":\\\"F8\\\",\\\"0\\\":\\\"F1-V0\\\"},\\\"local_rank\\\":{\\\"2\\\":0,\\\"3\\\":1,\\\"5\\\":2,\\\"11\\\":3,\\\"10\\\":4,\\\"13\\\":5,\\\"12\\\":6,\\\"8\\\":7,\\\"9\\\":8,\\\"14\\\":9,\\\"4\\\":10,\\\"6\\\":11,\\\"1\\\":12,\\\"7\\\":13,\\\"0\\\":14},\\\"local_normalize_scores\\\":{\\\"2\\\":-1.0,\\\"3\\\":-0.86,\\\"5\\\":0.79,\\\"11\\\":0.54,\\\"10\\\":0.36,\\\"13\\\":0.34,\\\"12\\\":-0.31,\\\"8\\\":0.28,\\\"9\\\":-0.27,\\\"14\\\":0.24,\\\"4\\\":0.22,\\\"6\\\":0.13,\\\"1\\\":0.12,\\\"7\\\":-0.11,\\\"0\\\":0.1},\\\"Sign\\\":{\\\"2\\\":\\\"red\\\",\\\"3\\\":\\\"red\\\",\\\"5\\\":\\\"green\\\",\\\"11\\\":\\\"green\\\",\\\"10\\\":\\\"green\\\",\\\"13\\\":\\\"green\\\",\\\"12\\\":\\\"red\\\",\\\"8\\\":\\\"green\\\",\\\"9\\\":\\\"red\\\",\\\"14\\\":\\\"green\\\",\\\"4\\\":\\\"green\\\",\\\"6\\\":\\\"green\\\",\\\"1\\\":\\\"green\\\",\\\"7\\\":\\\"red\\\",\\\"0\\\":\\\"green\\\"},\\\"local_impact\\\":{\\\"2\\\":2,\\\"3\\\":2,\\\"5\\\":1,\\\"11\\\":1,\\\"10\\\":1,\\\"13\\\":1,\\\"12\\\":2,\\\"8\\\":1,\\\"9\\\":2,\\\"14\\\":1,\\\"4\\\":1,\\\"6\\\":1,\\\"1\\\":1,\\\"7\\\":2,\\\"0\\\":1},\\\"annotate_placeholder_display\\\":{\\\"2\\\":\\\"F3-V0\\\",\\\"3\\\":\\\"F4-V0\\\",\\\"5\\\":\\\"F6\\\",\\\"11\\\":\\\"F12\\\",\\\"10\\\":\\\"F11\\\",\\\"13\\\":\\\"F14\\\",\\\"12\\\":\\\"F13\\\",\\\"8\\\":\\\"F9\\\",\\\"9\\\":\\\"F10\\\",\\\"14\\\":\\\"F15\\\",\\\"4\\\":\\\"F5\\\",\\\"6\\\":\\\"F7\\\",\\\"1\\\":\\\"F2-V0\\\",\\\"7\\\":\\\"F8\\\",\\\"0\\\":\\\"F1-V0\\\"},\\\"annotate_placeholder_code\\\":{\\\"2\\\":\\\"F3-V0\\\",\\\"3\\\":\\\"F4-V0\\\",\\\"5\\\":\\\"F6\\\",\\\"11\\\":\\\"F12\\\",\\\"10\\\":\\\"F11\\\",\\\"13\\\":\\\"F14\\\",\\\"12\\\":\\\"F13\\\",\\\"8\\\":\\\"F9\\\",\\\"9\\\":\\\"F10\\\",\\\"14\\\":\\\"F15\\\",\\\"4\\\":\\\"F5\\\",\\\"6\\\":\\\"F7\\\",\\\"1\\\":\\\"F2-V0\\\",\\\"7\\\":\\\"F8\\\",\\\"0\\\":\\\"F1-V0\\\"},\\\"ftype\\\":{\\\"2\\\":\\\"categorical\\\",\\\"3\\\":\\\"categorical\\\",\\\"5\\\":\\\"numeric\\\",\\\"11\\\":\\\"numeric\\\",\\\"10\\\":\\\"numeric\\\",\\\"13\\\":\\\"numeric\\\",\\\"12\\\":\\\"numeric\\\",\\\"8\\\":\\\"numeric\\\",\\\"9\\\":\\\"numeric\\\",\\\"14\\\":\\\"numeric\\\",\\\"4\\\":\\\"numeric\\\",\\\"6\\\":\\\"numeric\\\",\\\"1\\\":\\\"categorical\\\",\\\"7\\\":\\\"numeric\\\",\\\"0\\\":\\\"categorical\\\"}}\", \"feature_type\": [\"categorical\", \"categorical\", \"numeric\", \"numeric\", \"numeric\", \"numeric\", \"numeric\", \"numeric\", \"numeric\", \"numeric\", \"numeric\", \"numeric\", \"categorical\", \"numeric\", \"categorical\"], \"contradict\": [\"<|f#3|>\", \"<|f#4|>\", \"<|f#13|>\", \"<|f#10|>\", \"<|f#8|>\"], \"support\": [\"<|f#6|>\", \"<|f#12|>\", \"<|f#11|>\", \"<|f#14|>\", \"<|f#9|>\", \"<|f#15|>\", \"<|f#5|>\", \"<|f#7|>\", \"<|f#2|>\", \"<|f#1|>\"], \"ignore\": []}",
        "date_approved": "01-01-1970",
        "test_instance": 4,
        "features_placeholder": "[{\"Type of Travel\": \"<|f#3|>\", \"Type Of Booking\": \"<|f#4|>\", \"Hotel wifi service\": \"<|f#6|>\", \"Common Room entertainment\": \"<|f#12|>\", \"Stay comfort\": \"<|f#11|>\", \"Other service\": \"<|f#14|>\", \"Checkin/Checkout service\": \"<|f#13|>\", \"Hotel location\": \"<|f#9|>\", \"Food and drink\": \"<|f#10|>\", \"Cleanliness\": \"<|f#15|>\", \"Age\": \"<|f#5|>\", \"Departure/Arrival  convenience\": \"<|f#7|>\", \"purpose_of_travel\": \"<|f#2|>\", \"Ease of Online booking\": \"<|f#8|>\", \"Gender\": \"<|f#1|>\"}, {\"Type of Travel\": \"F3-V0\", \"Type Of Booking\": \"F4-V0\", \"Hotel wifi service\": \"F6\", \"Common Room entertainment\": \"F12\", \"Stay comfort\": \"F11\", \"Other service\": \"F14\", \"Checkin/Checkout service\": \"F13\", \"Hotel location\": \"F9\", \"Food and drink\": \"F10\", \"Cleanliness\": \"F15\", \"Age\": \"F5\", \"Departure/Arrival  convenience\": \"F7\", \"purpose_of_travel\": \"F2-V0\", \"Ease of Online booking\": \"F8\", \"Gender\": \"F1-V0\"}]",
        "is_paid": 2,
        "task_name": "Hotel Satisfaction",
        "prediction_confidence": "91.36%",
        "redeem_code": "Y16U-M@AW-7HRV-VK7Q-1-ALC",
        "narrator": 45,
        "narration": "The model prediction for the test case is C1 and the confidence level of this prediction decision is 91.36%, while the predicted probability of C2 is only 8.64%. According to the attribution analysis, we can see that the features F3 and F4 have negative attributions, pushing the prediction decision towards the alternative label, C2. Conversely, the F6, F12, F11, and F14 have values with a positive impact, shifting the classification decision towards label C1. Furthermore, while the attributes F13 and F10 contradict the prediction made, F9 and F15 have values that support the prediction from the model for the test case under consideration. Finally, F7, F2, F1, and F8 are the least ranked features, and among them, only F8 has a negative influence that contributes marginally to the shift away from labelling the case as C1.",
        "user_ip": "81.100.22.24",
        "narrative_question": "<ul><li> Provide a statement summarizing the prediction made for the test case. </li> <li> For the current test instance, describe the direction of influence of the following features: F3 (value equal to  V0) and F4 (with a value equal to  V0). </li> <li> Compare and contrast the impact of the following features  (F6, F12, F11 and F14) on the model\u2019s prediction of C1. </li> <li> Describe the degree of impact of the following features: F13, F9, F10 and F15? </li></ul>",
        "prediction_confidence_level": "C1: 91.36%, C2: 8.64%"
    },

    Actuaully need:
    "annotate_code": ["F3", "F4", "F6", "F12", "F11", "F14", "F13", "F9", "F10", "F15", "F5", "F7", "F2", "F8", "F1"],
    "Values":{"2":-0.2954524482,"3":-0.2525100076,"5":0.2296850087,"11":0.1492687389,"10":0.0917511915,"13":0.0863306218,"12":-0.0746328313,"8":0.065316989,"9":-0.0643314259,"14":0.0543745922,"4":0.047533335,"6":0.0183182452,"1":0.0165591439,"7":-0.0132520169,"0":0.0093566921}
    "prediction_confidence_level": "C1: 91.36%, C2: 8.64%"

    "narrative_question": " Provide a statement summarizing the prediction made for the test case.  For the current test instance, describe the direction of influence of the following features: F3 (value equal to  V0) and F4 (with a value equal to  V0).  Compare and contrast the impact of the following features  (F6, F12, F11 and F14) on the model's prediction of C1.  Describe the degree of impact of the following features: F13, F9, F10 and F15?",

    "narration": "The model prediction for the test case is C1 and the confidence level of this prediction decision is 91.36%, while the predicted probability of C2 is only 8.64%. According to the attribution analysis, we can see that the features F3 and F4 have negative attributions, pushing the prediction decision towards the alternative label, C2. Conversely, the F6, F12, F11, and F14 have values with a positive impact, shifting the classification decision towards label C1. Furthermore, while the attributes F13 and F10 contradict the prediction made, F9 and F15 have values that support the prediction from the model for the test case under consideration. Finally, F7, F2, F1, and F8 are the least ranked features, and among them, only F8 has a negative influence that contributes marginally to the shift away from labelling the case as C1.",

    "features_placeholder"{"Type of Travel": "F3-V0", "Type Of Booking": "F4-V0", "Hotel wifi service": "F6", "Common Room entertainment": "F12", "Stay comfort": "F11", "Other service": "F14", "Checkin/Checkout service": "F13", "Hotel location": "F9", "Food and drink": "F10", "Cleanliness": "F15", "Age": "F5", "Departure/Arrival  convenience": "F7", "purpose_of_travel": "F2-V0", "Ease of Online booking": "F8", "Gender": "F1-V0"}]


## On Essels part
There is nowhere where the evaluation happens. This could be seperate? Just need to do it again.

BLEU
METEOR
BLEURT

Let's look back at my other work and see how I did the metrics

# IDEAS
Can follow https://arxiv.org/pdf/2004.04487.pdf and generate training data such that the model learns to identify the correct number.

Eg generate inputs such as this:
| F5 | 1st positive 0.05 | F8 | 2nd positive 0.03 | F1 | 3rd positive 0.03 | F11 | 4th negative -0.03 | F7 | 5th positive 0.02 | F6 | 6th negative -0.02 | F16 | 7th positive 0.02 | F15 | 8th positive 0.02 | F13 | 9th negative -0.02 | F3 | 10th positive 0.02 | F14 | 11th positive 0.01 | F9 | 12th positive 0.01 | F10 | 13th positive 0.00 | F2 | 14th negative -0.00 | F12 | 15th negative -0.00 | F4 | 16th negative -0.00

And get the model to answer:
Q: Which feature had the highest positive impact?
A: F5

Q: Which feature had the highest negative impact?
A: F11

Q: What was the feature importance of F13?
A: -0.02

Could also do a contrastive loss where I say that the answer (with correct numbers) is correct, contrasted with the same answer but with the wrong numbers, which would be incorrect.

