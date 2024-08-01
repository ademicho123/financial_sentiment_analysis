# Import necessary libraries
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
with open('requirements.txt', 'r') as f:
    packages = f.read().splitlines()
for package in packages:
    install(package)

import argparse
import json
import os
import sys
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PyPDF2
import re
import nltk
import emoji
import shap
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
import logging
import traceback

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('omw-1.4')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the model name
MODEL_NAME = 'soleimanian/financial-roberta-large-sentiment'

# Function to extract sentences from PDFs using PyPDF2
def read_pdf_sentences(file_path):
    sentences = []
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    sentences.extend(sent_tokenize(text))
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {str(e)}")
    return sentences

def extract_and_merge(pdf_path, csv_path):
    try:
        print(f"Attempting to read PDF files from: {pdf_path}")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF directory not found: {pdf_path}")
        
        pdf_files = [os.path.join(pdf_path, file) for file in os.listdir(pdf_path) if file.endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files")
        
        pdf_sentences = []
        for file in pdf_files:
            pdf_sentences.extend(read_pdf_sentences(file))
        
        print(f"Extracted {len(pdf_sentences)} sentences from PDF files")
        pdf_df = pd.DataFrame({'content': pdf_sentences})
        
        print(f"Attempting to read CSV file: {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        news_data = pd.read_csv(csv_path, encoding='latin1')
        content_column = next((col for col in news_data.columns if col.lower().strip() == 'content'), None)
        if content_column is None:
            raise KeyError(f"No 'content' column found in the CSV file: {csv_path}")
    
        news_data_paragraphs = []
        for content in news_data[content_column].dropna():
            paragraphs = content.split('\n\n')
            news_data_paragraphs.extend(paragraphs)
        
        print(f"Extracted {len(news_data_paragraphs)} paragraphs from CSV file")
        news_df = pd.DataFrame({'content': news_data_paragraphs})
        
        merged_data = pd.concat([pdf_df, news_df], ignore_index=True)
        print(f"Merged data shape: {merged_data.shape}")
        
        return merged_data
    
    except Exception as e:
        print(f"Error in extract_and_merge: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Contents of current directory: {os.listdir('.')}")
        if os.path.exists(pdf_path):
            print(f"Contents of PDF directory: {os.listdir(pdf_path)}")
        raise

# Assign Sentiment Analyzer Score
sid = SentimentIntensityAnalyzer()

def assign_sentiment_scores(text):
    scores = sid.polarity_scores(text)
    return scores['compound']

def assign_scores(data):
    data['sentiment'] = data['content'].apply(assign_sentiment_scores)
    return data

# Function to assign direction and new_direction based on sentiment scores
def assign_directions(data):
    data['direction'] = data['sentiment'].apply(lambda x: 'bearish' if x < 0.0 else ('neutral' if 0.0 <= x < 0.4 else 'bullish'))
    data['new_direction'] = data['sentiment'].apply(lambda x: 2 if x < 0.0 else (1 if 0.0 <= x < 0.4 else 0))
    return data

# Function to preprocess individual text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Lowercase the text
    text = text.lower()

    # Remove emojis
    text = emoji.replace_emoji(text, '')

    # Remove emoticons (this is a basic implementation, might need refinement)
    text = re.sub(r'[:;=]-?[()DPp]', '', text)

    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    try:
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    except LookupError:
        # If lemmatization fails, just use the original tokens
        tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)

# Function to preprocess the entire DataFrame
def preprocess_data(df):
    df_cleaned = df.copy()
    df_cleaned['content'] = df_cleaned['content'].apply(preprocess_text)
    return df_cleaned

# Count the number of bearish, bullish, and neutral sentiments
def sentiment_counts(data):
    return data['direction'].value_counts()

# Download the model and tokenizer at runtime
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Prepare Dataset Function
def prepare_dataset(data, tokenizer, sample_frac=0.10, random_state=42, chunk_size=100):
    print("Preparing dataset...")
    data = data.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    
    def tokenize_function(examples):
        tokenized = tokenizer(examples['content'], truncation=True, padding='max_length', max_length=512)
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
    
    datasets = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        dataset_dict = Dataset.from_pandas(chunk)
        dataset_dict = dataset_dict.map(tokenize_function, batched=True, remove_columns=['content', 'sentiment'])
        datasets.append(dataset_dict)
        del chunk, dataset_dict
        gc.collect()
    
    dataset = concatenate_datasets(datasets)
    del datasets
    gc.collect()
    
    dataset = dataset.rename_column('new_direction', 'labels')
    dataset = dataset.with_format("torch")
    
    train_testvalid = dataset.train_test_split(test_size=0.3, seed=random_state)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=random_state)
    
    train_dataset = train_testvalid['train']
    test_dataset = test_valid['test']
    
    y_train = train_dataset['labels']
    
    print("Dataset prepared with train and test splits.")
    return train_dataset, test_dataset, y_train

# Compute Class Weights
def compute_class_weights(y_train):
    y_train = np.array(y_train)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    return class_weights

# Training and Evaluation Function
def train_and_evaluate(model_name, train_dataset, test_dataset, class_weights):
    print("Training and evaluating model...")
    
    try:
        accelerator = Accelerator()
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        
        def collate_fn(batch):
            return {
                'input_ids': torch.stack([item['input_ids'].squeeze() for item in batch]),
                'attention_mask': torch.stack([item['attention_mask'].squeeze() for item in batch]),
                'labels': torch.tensor([item['labels'] for item in batch])
            }
        
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)
        
        model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, test_dataloader
        )

        print("Starting training...")
        model.train()
        accumulation_steps = 4  # Accumulate gradients over 4 batches
        for epoch in range(3):  # 3 epochs
            for i, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps
                accelerator.backward(loss)
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item()*accumulation_steps}")
            
            # Garbage collection after each epoch
            gc.collect()
            torch.cuda.empty_cache()
        
        print("Training complete.")
        print("Evaluating model...")
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_dataloader:
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(accelerator.gather(preds).cpu().numpy())
                all_labels.extend(accelerator.gather(batch['labels']).cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, output_dict=True)
        
        return model, {'accuracy': accuracy}, report
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None, None
        
def compute_shap_values(model, dataset, num_samples=100):
    # Create a small subset of the dataset for SHAP analysis
    subset = dataset.shuffle(seed=42).select(range(num_samples))
    
    # Create an explainer
    explainer = shap.Explainer(model)
    
    # Compute SHAP values
    shap_values = explainer(subset)
    
    return shap_values, subset

def plot_shap_summary(shap_values, feature_names):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar")
    plt.tight_layout()
    plt.show()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    try:
        print(f"Running main with args: {args}")
        
        # Extract and merge data
        data = extract_and_merge(args.pdf_path, args.csv_path)
        
        # Assign sentiment scores
        data = assign_scores(data)
        
        # Assign directions
        data = assign_directions(data)
        
        # Preprocess the data
        data_cleaned = preprocess_data(data)
        
        # Prepare dataset
        train_dataset, test_dataset, y_train = prepare_dataset(data_cleaned, tokenizer)
        
        # Compute class weights
        class_weights = compute_class_weights(y_train)
        
        # Train and evaluate
        train_and_evaluate(MODEL_NAME, train_dataset, test_dataset, class_weights)
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Define paths for each company
    companies = {
        'Lloyds': {
            'pdf_path': 'data/lloyds',
            'csv_path': 'data/lloyds/lloyds_news.csv'
        },
        'IAG': {
            'pdf_path': 'data/iag',
            'csv_path': 'data/iag/iag_news.csv'
        },
        'Vodafone': {
            'pdf_path': 'data/vodafone',
            'csv_path': 'data/vodafone/vodafone_news.csv'
        }
    }

    for company_name, paths in companies.items():
        main(company_name, paths['pdf_path'], paths['csv_path'])