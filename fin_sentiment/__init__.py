"""Public interface for the financial sentiment analyzer."""

from importlib import import_module
from typing import Any

__version__ = "0.1.0"
__all__ = [
    "analyze_text",
    "analyze_article",
    "analyze_file",
    "predict_sentiment",
    "generate_shap_plot",
]


def analyze_text(input_text: str, *, explain: bool = True) -> dict[str, Any]:
    """Analyze non-empty financial text and return sentiment information."""
    return import_module("sentiment_predictor").analyze_text(input_text, explain=explain)


def predict_sentiment(input_text: str, *, explain: bool = True) -> dict[str, Any]:
    """Predict sentiment and produce model explanations for financial text."""
    return import_module("sentiment_predictor").predict_sentiment(
        input_text, explain=explain
    )


def analyze_article(url: str) -> dict[str, Any]:
    """Extract and analyze the readable text from an article URL."""
    text = import_module("fin_sentiment.sources").read_article(url)
    return analyze_text(text)


def analyze_file(path_or_url: str) -> dict[str, Any]:
    """Extract and analyze text from a local path or remote TXT/PDF file."""
    text = import_module("fin_sentiment.sources").read_file(path_or_url)
    return analyze_text(text)


def generate_shap_plot(shap_values: Any) -> Any:
    """Create a text plot from SHAP values."""
    return import_module("shap_analysis").generate_shap_plot(shap_values)
