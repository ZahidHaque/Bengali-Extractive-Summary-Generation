# -*- coding: utf-8 -*-

!pip install transformers
!pip3 install googletrans==3.1.0a0

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data from csv file
data = pd.read_csv('/content/drive/Shareddrives/Dante 2/Samit/dataset_main.csv')

# Split data into training and validation sets
train_data = data.sample(frac=0.8, random_state=42)
val_data = data.drop(train_data.index)

# Define a custom dataset
class SummaryDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        document = row['Documents']
        summary = row['Summary']
        encoding = self.tokenizer(document, summary, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        encoding['labels'] = encoding['input_ids'].clone()
        encoding['labels'][encoding['attention_mask'] == 0] = -100
        return {key: val.squeeze() for key, val in encoding.items()}
def conv(text,num):
  translator = Translator()
  if(num==1) : 
    result = translator.translate(text, src='bn', dest='en')
  else :
    result = translator.translate(text, src='en', dest='bn')
  return result.text

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('t5-small')
model = AutoModelForSeq2SeqLM.from_pretrained('t5-small').to(device)

# Create data loaders
train_dataset = SummaryDataset(train_data, tokenizer, max_length=256)
val_dataset = SummaryDataset(val_data, tokenizer, max_length=256)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Evaluate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f'Epoch: {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

# Save the trained model
model.save_pretrained('/content/model')

# Save the tokenizer
tokenizer.save_pretrained('/content/model')

# Load the saved model
model = AutoModelForSeq2SeqLM.from_pretrained('/content/drive/Shareddrives/Dante 2/Samit/model').to(device)

# Load the saved tokenizer
tokenizer = AutoTokenizer.from_pretrained('/content/drive/Shareddrives/Dante 2/Samit/model')

# Define a function to generate summaries
def generate_summary(document, max_length=150):
    document = conv(document,1)
    input_ids = tokenizer.encode(document, return_tensors='pt').to(device)
    summary_ids = model.generate(input_ids, max_length=max_length)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = conv(summary,2)
    return summary

# Generate a summary for a custom Bengali passage
document = 'উত্তপ্ত বাতাসের সঙ্গে ঠান্ডা হাওয়ার একটি সংমিশ্রণ এখন কাঙ্ক্ষিত। এই সংমিশ্রণ কেমন করে হতে পারে, এর ব্যাখ্যা দিয়েছেন আবহাওয়া অধিদপ্তরের আবহাওয়াবিদ মো. বজলুর রশীদ। তিনি বলেন, ঠান্ডা বাতাস থাকে ঊর্ধ্বাকাশে। এটি যদি নিচের দিকে নেমে আসে, তাহলে গরম ও আর্দ্রতাযুক্ত বাতাসের সঙ্গে মিশ্রণ ঘটবে। তৈরি হবে ঘূর্ণন। লাটিম যেমন ঘোরে, তেমনি করে তৈরি হবে ঘূর্ণন। আর্দ্রতামুক্ত বাতাস যখন ওপর দিকে উঠবে, তখন ওপরের ঠান্ডা বাতাসের সঙ্গে এর সংঘাত হবে। এতে মেঘ সৃষ্টি হয়ে বৃষ্টি হবে।'
summary = generate_summary(document)
print(summary)