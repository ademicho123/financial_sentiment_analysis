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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from nlpaug.augmenter.word import SynonymAug
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
MODEL_NAME = 'huawei-noah/TinyBERT_General_4L_312D'

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
def prepare_dataset(data, tokenizer, sample_frac=0.1, random_state=42, chunk_size=100):
    print("Preparing dataset...")
    data = data.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    
    # Data augmentation
    aug = SynonymAug(aug_src='wordnet')
    augmented_data = []
    for _, row in data.iterrows():
        augmented_text = aug.augment(row['content'])
        if isinstance(augmented_text, list):
            augmented_text = ' '.join(augmented_text)  # Join the list into a single string
        augmented_data.append({'content': augmented_text, 'new_direction': row['new_direction']})
    
    augmented_df = pd.DataFrame(augmented_data)
    data = pd.concat([data, augmented_df], ignore_index=True)
    
    def tokenize_function(examples):
        tokenized = tokenizer(examples['content'], truncation=True, padding='max_length', max_length=256)
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }
    
    datasets = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        if not chunk.empty:
            dataset_dict = Dataset.from_pandas(chunk)
            dataset_dict = dataset_dict.map(tokenize_function, batched=True, remove_columns=['content', 'sentiment'])
            datasets.append(dataset_dict)
        del chunk
        gc.collect()
    
    if not datasets:
        raise ValueError("No valid datasets created. Check your input data.")
    
    dataset = concatenate_datasets(datasets)
    del datasets
    gc.collect()
    
    dataset = dataset.rename_column('new_direction', 'labels')
    dataset = dataset.with_format("torch")
    
    # Ensure correct data types
    dataset = dataset.map(lambda example: {
        'input_ids': [int(x) for x in example['input_ids']],
        'attention_mask': [int(x) for x in example['attention_mask']],
        'labels': int(example['labels'])
    })
    
    train_testvalid = dataset.train_test_split(test_size=0.3, seed=random_state)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=random_state)
    
    train_dataset = train_testvalid['train']
    test_dataset = test_valid['test']
    
    # Convert to numpy arrays for resampling
    X_train = np.array(train_dataset['input_ids'])
    y_train = np.array(train_dataset['labels'])
    attention_masks = np.array(train_dataset['attention_mask'])
    
    # Combine input_ids and attention_masks
    X_combined = np.column_stack((X_train, attention_masks))
    
    # Define resampling strategy
    over = SMOTE(sampling_strategy='auto', random_state=random_state)
    under = RandomUnderSampler(sampling_strategy='auto', random_state=random_state)
    
    # Create a pipeline with SMOTE and RandomUnderSampler
    resampling = Pipeline([('over', over), ('under', under)])
    
    # Apply resampling
    X_resampled, y_resampled = resampling.fit_resample(X_combined, y_train)
    
    # Split X_resampled back into input_ids and attention_masks
    X_resampled_input_ids = X_resampled[:, :256]  # Assuming max_length is 256
    X_resampled_attention_masks = X_resampled[:, 256:]
    
    # Create a new dataset with resampled data
    resampled_train_dataset = Dataset.from_dict({
        'input_ids': X_resampled_input_ids.tolist(),
        'attention_mask': X_resampled_attention_masks.tolist(),
        'labels': y_resampled.tolist()
    })
    
    print(f"Dataset prepared with train size: {len(resampled_train_dataset)} and test size: {len(test_dataset)}")
    return resampled_train_dataset, test_dataset

def classification_report_to_dict(report):
    """Convert the classification report string to a dictionary."""
    lines = report.split('\n')
    res = {}
    for line in lines[2:-3]:  # skip header and footer
        line = line.strip()
        if line:
            row = line.split()
            if len(row) == 5:  # it's a regular row
                class_name, precision, recall, f1_score, support = row
                res[class_name] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1-score': float(f1_score),
                    'support': int(support)
                }
            elif len(row) == 4:  # it's the micro/macro avg row
                class_name = ' '.join(row[:-3])
                precision, recall, f1_score = row[-3:]
                res[class_name] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1-score': float(f1_score)
                }
    return res

def train_and_evaluate(model_name, train_dataset, test_dataset):
    print("Training and evaluating model...")
    
    try:
        accelerator = Accelerator()
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        optimizer = AdamW(model.parameters(), lr=2e-5)  # Slightly increased learning rate
        
        # Use CosineAnnealingWarmRestarts for better learning rate scheduling
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        loss_fn = nn.CrossEntropyLoss()
        
        def collate_fn(batch):
            try:
                input_ids = []
                attention_mask = []
                labels = []
                
                for item in batch:
                    if isinstance(item['input_ids'], torch.Tensor):
                        input_ids.append(item['input_ids'].tolist())
                    else:
                        input_ids.append(item['input_ids'])
                    
                    if isinstance(item['attention_mask'], torch.Tensor):
                        attention_mask.append(item['attention_mask'].tolist())
                    else:
                        attention_mask.append(item['attention_mask'])
                    
                    labels.append(item['labels'].item() if isinstance(item['labels'], torch.Tensor) else item['labels'])
                
                return {
                    'input_ids': torch.tensor(input_ids),
                    'attention_mask': torch.tensor(attention_mask),
                    'labels': torch.tensor(labels)
                }
            except Exception as e:
                print(f"Error in collate_fn: {str(e)}")
                print(f"Problematic batch: {batch}")
                raise

        # Increase batch size
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)
        
        model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, test_dataloader, scheduler
        )

        print("Starting training...")
        model.train()
        num_epochs = 5  # Increased number of epochs
        accumulation_steps = 2  # Reduced accumulation steps due to increased batch size
        for epoch in range(num_epochs):
            total_loss = 0
            for i, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, batch['labels']) / accumulation_steps
                accelerator.backward(loss)
                total_loss += loss.item() * accumulation_steps
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                if i % 100 == 0:
                    print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item()*accumulation_steps}")
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss}")
            
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
    
        # Get unique labels from the predictions and true labels
        unique_labels = sorted(set(all_labels) | set(all_preds))
        label_names = ['bullish', 'neutral', 'bearish']
        target_names = [label_names[i] for i in unique_labels if i < len(label_names)]
    
        report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
    
        return model, {'accuracy': accuracy}, report_df, all_labels, all_preds
    
    except Exception as e:
        print(f"An error occurred in train_and_evaluate: {str(e)}")
        print(f"Error details: {traceback.format_exc()}")
        return None

def create_comprehensive_report(company_name, metrics, report_df, all_labels, all_preds):
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=['True Bullish', 'True Neutral', 'True Bearish'],
                         columns=['Pred Bullish', 'Pred Neutral', 'Pred Bearish'])
    
    # Prepare data for the comprehensive report
    report_data = {
        'Company': company_name,
        'Accuracy': metrics['accuracy'],
        'Confusion Matrix': cm_df.to_json(),
    }
    
    # Add precision, recall, and F1-score for each class if available
    for class_label, class_name in zip(['0', '1', '2'], ['Bullish', 'Neutral', 'Bearish']):
        if class_label in report_df.index:
            report_data.update({
                f'Precision ({class_name})': report_df.loc[class_label, 'precision'],
                f'Recall ({class_name})': report_df.loc[class_label, 'recall'],
                f'F1-Score ({class_name})': report_df.loc[class_label, 'f1-score'],
            })
        else:
            report_data.update({
                f'Precision ({class_name})': None,
                f'Recall ({class_name})': None,
                f'F1-Score ({class_name})': None,
            })
    
    return pd.DataFrame([report_data])

def main(company_name, pdf_path, csv_path):
    try:
        logger.info(f"Processing {company_name}...")
        
        # Load and preprocess data
        raw_data = extract_and_merge(pdf_path, csv_path)
        data_with_sentiment = assign_scores(raw_data)
        data_with_directions = assign_directions(data_with_sentiment)
        cleaned_data = preprocess_data(data_with_directions)

        # Display sentiment counts
        counts = sentiment_counts(cleaned_data)
        logger.info(f"{company_name} Sentiment Counts:")
        logger.info(counts)

        # Prepare dataset
        train_dataset, test_dataset = prepare_dataset(cleaned_data, tokenizer)

        # Train and evaluate
        result = train_and_evaluate(MODEL_NAME, train_dataset, test_dataset)
        
        if result is None:
            logger.error(f"Training and evaluation failed for {company_name}")
            return None

        model, metrics, report, all_labels, all_preds = result

        # Display the evaluation metrics
        logger.info(f"Evaluation Metrics for {company_name}:")
        logger.info(f"Accuracy: {metrics['accuracy']}")
        
        # Display the classification report
        logger.info(f"Classification Report for {company_name}:")
        logger.info(report)

        # Convert the report to a DataFrame
        report_dict = classification_report_to_dict(report)
        report_df = pd.DataFrame(report_dict).transpose()

        # Create comprehensive report
        comprehensive_report = create_comprehensive_report(company_name, metrics, report_df, all_labels, all_preds)

        return comprehensive_report

    except Exception as e:
        logger.error(f"Error processing {company_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

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

    all_reports = []

    for company_name, paths in companies.items():
        try:
            logger.info(f"Starting processing for {company_name}")
            company_report = main(company_name, paths['pdf_path'], paths['csv_path'])
            
            if company_report is not None:
                all_reports.append(company_report)
            
        except Exception as e:
            logger.error(f"Failed to process {company_name}: {str(e)}")

    # Combine all reports into a single DataFrame
    if all_reports:
        combined_report = pd.concat(all_reports, ignore_index=True)
        combined_report.to_csv('comprehensive_classification_report.csv', index=False)
        logger.info("Comprehensive classification report for all companies saved to CSV.")
    else:
        logger.warning("No reports were generated.")