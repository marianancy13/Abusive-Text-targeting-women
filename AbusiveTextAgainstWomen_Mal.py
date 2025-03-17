import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import classification_report
from torch import nn
from tqdm import tqdm

# 1. Load and preprocess data
train_file = 'AWM_dev.xlsx'  # Path to your training file
test_file = 'AWM_test_without_labels.xlsx'  # Path to your test file

# Load data
train_data = pd.read_excel(train_file)
test_data = pd.read_excel(test_file)

# Map labels to integers
label_mapping = {'Non-Abusive': 0, 'Abusive': 1}

# Tokenizer
model_name = "bert-base-multilingual-cased"  # You can change this to mBERT or other multilingual models
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Create Dataset class for PyTorch
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Prepare training and test datasets
train_labels = [label_mapping[label] for label in train_data['Class']]
train_dataset = TextDataset(train_data['Text'].tolist(), train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TextDataset(test_data['Text'].tolist(), tokenizer=tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 3. Model and optimizer setup
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels: Abusive, Non-Abusive
optimizer = AdamW(model.parameters(), lr=2e-5)

# 4. Class weights for imbalanced data (adjust based on data distribution)
class_weights = torch.tensor([1.0, 2.0])  # Adjust class weights if necessary
criterion = nn.CrossEntropyLoss(weight=class_weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. Training Loop
epochs = 3

for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item()}")

# 6. Model Evaluation
model.eval()
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(outputs.logits, dim=1)
        
        # Get predicted labels (choose the class with the highest probability)
        predicted_labels = torch.argmax(probs, dim=1)
        
        predictions.extend(predicted_labels.cpu().numpy())

# 7. Save predictions to CSV with id and Labels columns
test_data['Labels'] = predictions
output_df = test_data[['id', 'Labels']]
output_df['Labels'] = output_df['Labels'].map({0: 'Non-Abusive', 1: 'Abusive'})
output_df.to_csv("predictions_Mal.csv", index=False)

print("Predictions saved to 'predictions_Mal.csv'.")
