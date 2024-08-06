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

def model_predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.squeeze().numpy()

def predict_sentiment(input_text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(input_text)
    
    # Make prediction
    logits = model_predict(preprocessed_text)
    predictions = torch.softmax(torch.tensor(logits), dim=0)
    predicted_class = torch.argmax(predictions).item()
    
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
    explainer = shap.Explainer(model_predict, tokenizer)
    
    # Adjust how the input is provided to the SHAP explainer
    shap_values = explainer([preprocessed_text])
    
    return {
        'sentiment': predicted_sentiment,
        'score': sentiment_score,
        'direction': direction,
        'preprocessed_text': preprocessed_text,
        'shap_values': shap_values
    }

def analyze_text(input_text):
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError("Input text must be a non-empty string.")
    
    return predict_sentiment(input_text)
