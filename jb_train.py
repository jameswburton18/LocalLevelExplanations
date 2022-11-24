from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from transformers.optimization import  Adafactor 
from datasets import load_dataset
import warnings
import json
from src.dataCollator import DataCollator
warnings.filterwarnings('ignore')


train = json.load(open('jb_data/train.json'))
batch_size=8
# num_of_batches=len(train_df)/batch_size
# num_of_epochs=4
# num_of_batches=int(num_of_batches)

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)

collator = DataCollator(tokenizer, max_question_len=80, max_answer_len=80,
                        use_split_answers=True)

data_files = {'train': 'jb_data/train.json', 
              'val': 'jb_data/val.json',
              'test': 'jb_data/test.json'}
dataset = load_dataset('json', data_files=data_files)

training_args = TrainingArguments(
    output_dir='models/T5'
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    eval_dataset=dataset['val'],
    data_collator=collator
)

trainer.train()
'''

#moving the model to device(GPU/CPU)
model.to('cuda')


optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False
)

#Sets the module in training mode
model.train()

loss_per_10_steps=[]
for epoch in range(1,num_of_epochs+1):
  print('Running epoch: {}'.format(epoch))
  
  running_loss=0

for i in range(num_of_batches):
    inputbatch=[]
    labelbatch=[]
    new_df=train_df[i*batch_size:i*batch_size+batch_size]
    for indx,row in new_df.iterrows():
        input = 'WebNLG: '+row['input_text']+'</s>' 
        labels = row['target_text']+'</s>'   
        inputbatch.append(input)
        labelbatch.append(labels)
    inputbatch=tokenizer.batch_encode_plus(inputbatch,padding=True,max_length=400,return_tensors='pt')["input_ids"]
    labelbatch=tokenizer.batch_encode_plus(labelbatch,padding=True,max_length=400,return_tensors="pt") ["input_ids"]
    inputbatch=inputbatch.to(dev)
    labelbatch=labelbatch.to(dev)

    # clear out the gradients of all Variables 
    optimizer.zero_grad()

    # Forward propogation
    outputs = model(input_ids=inputbatch, labels=labelbatch)
    loss = outputs.loss
    loss_num=loss.item()
    logits = outputs.logits
    running_loss+=loss_num
    if i%10 ==0:      
        loss_per_10_steps.append(loss_num)

    # calculating the gradients
    loss.backward()

    #updating the params
    optimizer.step()
    
running_loss=running_loss/int(num_of_batches)
print('Epoch: {} , Running loss: {}'.format(epoch,running_loss))
  
model.eval()
input_ids = tokenizer.encode("WebNLG: sidharth | hometown | Delhi && sidharth | play |  football </s>", return_tensors="pt")  # Batch size 1
input_ids=input_ids.to(dev)
outputs = model.generate(input_ids)
tokenizer.decode(outputs[0])
torch.save(model.state_dict(),'data_to_text_model.bin')
'''