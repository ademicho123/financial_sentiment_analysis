"""Inference utilities for financial sentiment analysis."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import emoji
import torch
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = Path(__file__).resolve().parent / "fin_sentiment" / "model"

if not MODEL_NAME.is_dir():
    raise RuntimeError(
        "Trained model not found. Run: python training/train_tinybert.py"
    )

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_NAME), local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_NAME), local_files_only=True
)
model.eval()
_sentiment_analyzer: SentimentIntensityAnalyzer | None = None
_lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=1)
def _load_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk_download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def _lemmatize(tokens: list[str]) -> list[str]:
    try:
        return [_lemmatizer.lemmatize(token) for token in tokens]
    except LookupError:
        nltk_download("wordnet", quiet=True)
        return [_lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(text: str) -> str:
    """Normalize text for model inference without importing training code."""
    text = emoji.replace_emoji(text.lower(), "")
    text = re.sub(r"[:;=]-?[()DPp]", "", text)
    text = re.sub(r"[^\w\s]|\d+", " ", text)
    tokens = re.sub(r"\s+", " ", text).strip().split()
    tokens = [token for token in tokens if token not in _load_stopwords()]
    return " ".join(_lemmatize(tokens))


def assign_sentiment_score(text: str) -> float:
    """Return the VADER compound score for text."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            _sentiment_analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk_download("vader_lexicon", quiet=True)
            _sentiment_analyzer = SentimentIntensityAnalyzer()
    return float(_sentiment_analyzer.polarity_scores(text)["compound"])


def score_to_direction(score: float) -> str:
    if score < 0.0:
        return "bearish"
    if score < 0.4:
        return "neutral"
    return "bullish"


def model_predict(texts: list[str]) -> Any:
    """Return model logits for SHAP and internal inference."""
    inputs = tokenizer(
        [str(text) for text in texts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.detach().cpu().numpy()


def predict_sentiment(input_text: str, explain: bool = True) -> dict[str, Any]:
    """Predict sentiment, optionally including SHAP and attention output."""
    preprocessed_text = preprocess_text(input_text)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=explain)

    predicted_class = torch.argmax(torch.softmax(outputs.logits, dim=1), dim=1).item()
    predicted_sentiment = model.config.id2label[predicted_class].lower()
    sentiment_score = assign_sentiment_score(input_text)
    result: dict[str, Any] = {
        "sentiment": predicted_sentiment,
        "score": sentiment_score,
        "direction": score_to_direction(sentiment_score),
        "preprocessed_text": preprocessed_text,
    }

    if explain:
        import shap

        from attention_visualization import visualize_attention

        explainer = shap.Explainer(model_predict, tokenizer)
        result["shap_values"] = explainer([str(input_text)])
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        result["attention_plot"] = visualize_attention(tokens, outputs.attentions)

    return result


def analyze_text(input_text: str, explain: bool = True) -> dict[str, Any]:
    """Validate and analyze a non-empty string."""
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError("Input text must be a non-empty string.")
    return predict_sentiment(input_text, explain=explain)
