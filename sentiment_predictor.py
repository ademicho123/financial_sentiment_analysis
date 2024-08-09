import pandas as pd
import torch
import shap
from attention_visualization import visualize_attention
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

def model_predict(texts):
    texts = [str(text) for text in texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.numpy()

def predict_sentiment(input_text):
    preprocessed_text = preprocess_text(input_text)
    
    logits = model_predict([preprocessed_text])
    predictions = torch.softmax(torch.tensor(logits), dim=1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    
    sentiment_map = {0: 'bullish', 1: 'neutral', 2: 'bearish'}
    predicted_sentiment = sentiment_map[predicted_class]
    
    sentiment_score = assign_sentiment_scores(preprocessed_text)
    
    direction_df = pd.DataFrame({'sentiment': [sentiment_score]})
    direction_df = assign_directions(direction_df)
    direction = direction_df['direction'].iloc[0]
    
    explainer = shap.Explainer(model_predict, tokenizer)
    shap_values = explainer([str(preprocessed_text)])
    
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

def predict_sentiment(input_text):
    preprocessed_text = preprocess_text(input_text)
    
    inputs = tokenizer(preprocessed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    logits = outputs.logits
    attention_weights = outputs.attentions
    
    predictions = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    
    sentiment_map = {0: 'bullish', 1: 'neutral', 2: 'bearish'}
    predicted_sentiment = sentiment_map[predicted_class]
    
    sentiment_score = assign_sentiment_scores(preprocessed_text)
    
    direction_df = pd.DataFrame({'sentiment': [sentiment_score]})
    direction_df = assign_directions(direction_df)
    direction = direction_df['direction'].iloc[0]
    
    explainer = shap.Explainer(model_predict, tokenizer)
    shap_values = explainer([str(preprocessed_text)])
    
    # Generate attention visualization
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attention_plot = visualize_attention(tokens, attention_weights)
    
    return {
        'sentiment': predicted_sentiment,
        'score': sentiment_score,
        'direction': direction,
        'preprocessed_text': preprocessed_text,
        'shap_values': shap_values,
        'attention_plot': attention_plot
    }