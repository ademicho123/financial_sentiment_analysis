import pytest
import os
import pandas as pd
from transformers import AutoTokenizer
from financial_sentiment_analysis import *

# Assuming this script is in the root directory
sample_data_folder = os.path.join(os.getcwd(), "sample_data")

# Paths for the files
pdf_file_paths = [
    os.path.join(sample_data_folder, "pdf_2.pdf"),
    os.path.join(sample_data_folder, "pdf_3.pdf")
]
csv_file_path = os.path.join(sample_data_folder, "iag_sample_csv.csv")

def test_read_pdf_sentences_valid():
    sentences = []
    for pdf_file_path in pdf_file_paths:
        sentences.extend(read_pdf_sentences(pdf_file_path))
    assert len(sentences) > 0

def test_read_pdf_sentences_invalid_path():
    invalid_pdf_path = "invalid_path.pdf"
    sentences = read_pdf_sentences(invalid_pdf_path)
    assert len(sentences) == 0

def test_extract_and_merge_valid():
    data = extract_and_merge(sample_data_folder, csv_file_path)
    assert len(data) > 0
    assert 'content' in data.columns

def test_extract_and_merge_missing_files():
    with pytest.raises(FileNotFoundError):
        extract_and_merge(sample_data_folder, "sample_data/missing_file.csv")

def test_assign_sentiment_scores_positive():
    score = assign_sentiment_scores("This is great!")
    assert score > 0

def test_assign_sentiment_scores_negative():
    score = assign_sentiment_scores("This is terrible!")
    assert score < 0

def test_assign_sentiment_scores_neutral():
    neutral_text = "This is a sentence."
    score = assign_sentiment_scores(neutral_text)
    assert abs(score) < 0.1

def test_assign_directions_bearish():
    df = pd.DataFrame({'sentiment': [-0.5]})
    result = assign_directions(df)
    assert result['direction'][0] == 'bearish'

def test_assign_directions_bullish():
    df = pd.DataFrame({'sentiment': [0.5]})
    result = assign_directions(df)
    assert result['direction'][0] == 'bullish'

def test_assign_directions_neutral():
    df = pd.DataFrame({'sentiment': [0.0]})
    result = assign_directions(df)
    assert result['direction'][0] == 'neutral'

def test_preprocess_text_lowercase():
    result = preprocess_text("Hello WORLD")
    assert result == "hello world"

def test_preprocess_text_remove_numbers():
    text_with_numbers = "This is 2023!"
    result = preprocess_text(text_with_numbers)
    assert not any(char.isdigit() for char in result)

def test_preprocess_data_valid():
    df = pd.DataFrame({'content': ["Sample TEXT", "More TEXT"]})
    cleaned_df = preprocess_data(df)
    assert cleaned_df['content'][0] == "sample text"
    assert cleaned_df['content'][1] == "text"  # "More" is likely removed as a stopword

def test_prepare_dataset_valid():
    # Create a larger sample DataFrame with the required columns
    data = pd.DataFrame({
        'content': ["Sample text for sentiment analysis"] * 60,  # Increased to 60 samples
        'sentiment': [0.5] * 60,
        'new_direction': [0, 1, 2] * 20  # Ensure all classes are represented equally
    })
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Wrap the function call in a try-except block to catch potential ValueError
    try:
        train_dataset, test_dataset = prepare_dataset(data, tokenizer, sample_frac=1.0, chunk_size=60)  # Use all data, increased chunk size
        
        # Check if datasets are not empty
        assert len(train_dataset) > 0, "Train dataset is empty"
        assert len(test_dataset) > 0, "Test dataset is empty"
        
        # Check if the datasets have the expected features
        expected_features = ['input_ids', 'attention_mask', 'labels']
        assert all(feature in train_dataset.features for feature in expected_features), "Train dataset missing expected features"
        assert all(feature in test_dataset.features for feature in expected_features), "Test dataset missing expected features"
        
    except ValueError as e:
        pytest.fail(f"prepare_dataset raised ValueError: {str(e)}")
    except Exception as e:
        pytest.fail(f"prepare_dataset raised an unexpected exception: {str(e)}")