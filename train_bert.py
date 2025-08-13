import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# --- Helper function to compute evaluation metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

# --- Data loading and processing ---
# Load dataset from Hugging Face Hub
ds = load_dataset("ShreyaR/DepressionDetection")

# The dataset has only one split ('train'), so we take that and split it
ds_split = ds['train'].train_test_split(test_size=0.2, seed=42)
train_dataset = ds_split['train']
test_dataset = ds_split['test']

print("Label distribution in the dataset:")
print(train_dataset.to_pandas()['is_depression'].value_counts())

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Tokenize function
def tokenize(batch):
    return tokenizer(batch["clean_text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Rename the label column before setting the format
train_dataset = train_dataset.rename_column("is_depression", "labels")
test_dataset = test_dataset.rename_column("is_depression", "labels")

# Set the format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# --- Training configuration ---
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1.5e-5,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    fp16=True,
    push_to_hub=False
)

# --- Initialize the Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# --- Train and save the model ---
trainer.train()

# Save model
model.save_pretrained("./bert_depression_model")
tokenizer.save_pretrained("./bert_depression_model")