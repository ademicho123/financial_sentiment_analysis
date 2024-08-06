import pandas as pd
import torch
import shap
from financial_sentiment_analysis import (
    preprocess_text,
    assign_sentiment_scores,
    assign_directions,
    MODEL_NAME,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Create a SHAP explainer
def model_wrapper(x):
    return model(**tokenizer(x, return_tensors="pt", padding=True, truncation=True)).logits

explainer = shap.Explainer(model_wrapper, tokenizer)

def predict_sentiment(input_text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(input_text)
    
    # Tokenize the preprocessed text
    inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
    
    # Map the predicted class to sentiment
    sentiment_map = {0: 'bullish', 1: 'neutral', 2: 'bearish'}
    predicted_sentiment = sentiment_map[predicted_class]
    
    # Get the sentiment score
    sentiment_score = assign_sentiment_scores(preprocessed_text)
    
    # Assign direction based on the score
    direction_df = pd.DataFrame({'sentiment': [sentiment_score]})
    direction_df = assign_directions(direction_df)
    direction = direction_df['direction'].iloc[0]
    
    # Generate SHAP values
    shap_values = explainer([preprocessed_text])
    
    return {
        'sentiment': predicted_sentiment,
        'score': sentiment_score,
        'direction': direction,
        'preprocessed_text': preprocessed_text,
        'shap_values': shap_values
    }

def analyze_text(input_text):
    results = {}
    for company in ['Lloyds', 'IAG', 'Vodafone']:
        # In a real scenario, you might want to use company-specific models here
        results[company] = predict_sentiment(input_text)
    return results