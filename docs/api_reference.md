# API Reference - Financial Sentiment Analysis

This document provides a comprehensive API reference for all modules and functions in the project.

## Table of Contents
1. [dashboard.py](#dashboardpy)
2. [sentiment_predictor.py](#sentiment_predictorpy)
3. [financial_sentiment_analysis.py](#financial_sentiment_analysispy)
4. [attention_visualization.py](#attention_visualizationpy)
5. [shap_analysis.py](#shap_analysispy)

---

## dashboard.py

Streamlit-based web interface for financial sentiment analysis.

### Functions

#### `read_pdf(file: BinaryIO) -> str`

Extracts text content from a PDF file.

**Parameters:**
- `file` (BinaryIO): Uploaded PDF file object from Streamlit

**Returns:**
- `str`: Extracted text content from all pages

**Example:**
```python
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    text = read_pdf(uploaded_file)
```

---

#### `explain_shap_values(shap_values: Explanation, class_names: list) -> str`

Generates human-readable explanation of SHAP values.

**Parameters:**
- `shap_values` (Explanation): SHAP explanation object
- `class_names` (list): List of class names ['Bullish', 'Neutral', 'Bearish']

**Returns:**
- `str`: Formatted explanation text

**Example:**
```python
explanation = explain_shap_values(shap_values, ['Bullish', 'Neutral', 'Bearish'])
st.text(explanation)
```

---

## sentiment_predictor.py

Core prediction module for sentiment analysis.

### Functions

#### `model_predict(texts: list) -> np.ndarray`

Performs batch prediction on a list of texts.

**Parameters:**
- `texts` (list): List of text strings to analyze

**Returns:**
- `np.ndarray`: Logits array of shape (batch_size, 3)

**Example:**
```python
logits = model_predict(["Stock prices are rising", "Market is uncertain"])
```

---

#### `predict_sentiment(input_text: str) -> dict`

Predicts sentiment for a single text input with full analysis.

**Parameters:**
- `input_text` (str): Raw text to analyze

**Returns:**
- `dict`: Dictionary containing:
  - `sentiment` (str): Predicted class ('bullish', 'neutral', 'bearish')
  - `score` (float): VADER sentiment score
  - `direction` (str): Direction based on score
  - `preprocessed_text` (str): Cleaned text
  - `shap_values` (Explanation): SHAP explanation object
  - `attention_plot` (Figure): Matplotlib figure with attention heatmap

**Example:**
```python
result = predict_sentiment("Apple's revenue exceeded expectations")
print(f"Sentiment: {result['sentiment']}")
print(f"Score: {result['score']}")
```

---

#### `analyze_text(input_text: str) -> dict`

Wrapper function for text analysis with input validation.

**Parameters:**
- `input_text` (str): Text to analyze

**Returns:**
- `dict`: Same as `predict_sentiment()`

**Raises:**
- `ValueError`: If input is empty or not a string

**Example:**
```python
try:
    result = analyze_text(user_input)
except ValueError as e:
    print(f"Invalid input: {e}")
```

---

## financial_sentiment_analysis.py

Core module for data processing, model training, and evaluation.

### Constants

```python
MODEL_NAME = 'huawei-noah/TinyBERT_General_4L_312D'
```

### Functions

#### `read_pdf_sentences(file_path: str) -> list`

Extracts sentences from a PDF file.

**Parameters:**
- `file_path` (str): Path to PDF file

**Returns:**
- `list`: List of extracted sentences

**Example:**
```python
sentences = read_pdf_sentences("data/lloyds/report.pdf")
```

---

#### `extract_and_merge(pdf_path: str, csv_path: str) -> pd.DataFrame`

Extracts data from PDFs and merges with CSV data.

**Parameters:**
- `pdf_path` (str): Directory containing PDF files
- `csv_path` (str): Path to CSV file with news data

**Returns:**
- `pd.DataFrame`: Merged dataframe with 'content' column

**Example:**
```python
data = extract_and_merge("data/lloyds", "data/lloyds/lloyds_news.csv")
```

---

#### `assign_sentiment_scores(text: str) -> float`

Calculates VADER sentiment score for text.

**Parameters:**
- `text` (str): Input text

**Returns:**
- `float`: Compound sentiment score (-1 to 1)

**Example:**
```python
score = assign_sentiment_scores("The company reported strong earnings")
# score might be 0.6834
```

---

#### `assign_scores(data: pd.DataFrame) -> pd.DataFrame`

Assigns sentiment scores to all rows in dataframe.

**Parameters:**
- `data` (pd.DataFrame): Dataframe with 'content' column

**Returns:**
- `pd.DataFrame`: Dataframe with added 'sentiment' column

**Example:**
```python
data = assign_scores(data)
```

---

#### `assign_directions(data: pd.DataFrame) -> pd.DataFrame`

Assigns direction labels based on sentiment scores.

**Parameters:**
- `data` (pd.DataFrame): Dataframe with 'sentiment' column

**Returns:**
- `pd.DataFrame`: Dataframe with added 'direction' column

**Direction Mapping:**
- `sentiment < -0.05`: 'bearish'
- `-0.05 <= sentiment <= 0.05`: 'neutral'
- `sentiment > 0.05`: 'bullish'

**Example:**
```python
data = assign_directions(data)
```

---

#### `preprocess_text(text: str) -> str`

Preprocesses a single text string.

**Parameters:**
- `text` (str): Raw text

**Returns:**
- `str`: Cleaned and preprocessed text

**Processing Steps:**
1. Convert to lowercase
2. Remove URLs
3. Remove HTML tags
4. Remove punctuation
5. Remove numbers
6. Remove emojis
7. Tokenize
8. Lemmatize
9. Remove stopwords

**Example:**
```python
cleaned = preprocess_text("Apple's Q4 2023 revenue was $89.5B! 🚀")
# cleaned might be "apple q revenue b"
```

---

#### `preprocess_data(df: pd.DataFrame) -> pd.DataFrame`

Preprocesses all text in dataframe.

**Parameters:**
- `df` (pd.DataFrame): Dataframe with 'content' column

**Returns:**
- `pd.DataFrame`: Dataframe with preprocessed 'content'

**Example:**
```python
df = preprocess_data(df)
```

---

#### `sentiment_counts(data: pd.DataFrame) -> dict`

Counts sentiment distribution.

**Parameters:**
- `data` (pd.DataFrame): Dataframe with 'direction' column

**Returns:**
- `dict`: Dictionary with counts for each sentiment

**Example:**
```python
counts = sentiment_counts(data)
# {'bearish': 150, 'neutral': 200, 'bullish': 180}
```

---

#### `prepare_dataset(data: pd.DataFrame, tokenizer: AutoTokenizer, sample_frac: float = 0.1, random_state: int = 42, chunk_size: int = 100) -> tuple`

Prepares training and test datasets with augmentation and balancing.

**Parameters:**
- `data` (pd.DataFrame): Input dataframe
- `tokenizer` (AutoTokenizer): Hugging Face tokenizer
- `sample_frac` (float): Fraction of data to use (default: 0.1)
- `random_state` (int): Random seed (default: 42)
- `chunk_size` (int): Processing chunk size (default: 100)

**Returns:**
- `tuple`: (train_dataset, test_dataset) as Hugging Face Dataset objects

**Processing Steps:**
1. Sample data
2. Augment with synonyms
3. Apply SMOTE oversampling
4. Apply random undersampling
5. Tokenize
6. Split into train/test (80/20)

**Example:**
```python
train_ds, test_ds = prepare_dataset(data, tokenizer, sample_frac=0.2)
```

---

#### `train_and_evaluate(model_name: str, train_dataset: Dataset, test_dataset: Dataset) -> dict`

Trains and evaluates the TinyBERT model.

**Parameters:**
- `model_name` (str): Hugging Face model identifier
- `train_dataset` (Dataset): Training dataset
- `test_dataset` (Dataset): Test dataset

**Returns:**
- `dict`: Dictionary containing:
  - `accuracy` (float): Overall accuracy
  - `precision` (float): Macro-averaged precision
  - `recall` (float): Macro-averaged recall
  - `f1` (float): Macro-averaged F1 score
  - `confusion_matrix` (np.ndarray): Confusion matrix
  - `classification_report` (dict): Detailed metrics per class
  - `all_labels` (list): True labels
  - `all_preds` (list): Predicted labels

**Training Configuration:**
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Scheduler: CosineAnnealingWarmRestarts (T_0=10)
- Epochs: 3
- Batch size: 16
- Gradient accumulation: 2 steps

**Example:**
```python
metrics = train_and_evaluate(MODEL_NAME, train_ds, test_ds)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

---

#### `create_comprehensive_report(company_name: str, metrics: dict, report_df: pd.DataFrame, all_labels: list, all_preds: list) -> pd.DataFrame`

Creates a comprehensive evaluation report.

**Parameters:**
- `company_name` (str): Name of the company
- `metrics` (dict): Metrics from `train_and_evaluate()`
- `report_df` (pd.DataFrame): Classification report as dataframe
- `all_labels` (list): True labels
- `all_preds` (list): Predicted labels

**Returns:**
- `pd.DataFrame`: Comprehensive report dataframe

**Example:**
```python
report = create_comprehensive_report("Lloyds", metrics, report_df, labels, preds)
```

---

#### `main(company_name: str, pdf_path: str, csv_path: str) -> dict`

Main pipeline for a single company.

**Parameters:**
- `company_name` (str): Company name
- `pdf_path` (str): Path to PDF directory
- `csv_path` (str): Path to CSV file

**Returns:**
- `dict`: Dictionary with company name and comprehensive report

**Example:**
```python
result = main("Lloyds", "data/lloyds", "data/lloyds/lloyds_news.csv")
```

---

## attention_visualization.py

Module for visualizing attention weights from transformer models.

### Functions

#### `visualize_attention(tokens: list, attention_weights: tuple, layer: int = -1, head: int = 0) -> plt.Figure`

Creates a heatmap visualization of attention weights.

**Parameters:**
- `tokens` (list): List of token strings
- `attention_weights` (tuple): Attention weights from model output
- `layer` (int): Which layer to visualize (default: -1 for last layer)
- `head` (int): Which attention head to visualize (default: 0)

**Returns:**
- `plt.Figure`: Matplotlib figure object

**Visualization Details:**
- Colormap: 'YlOrRd' (Yellow-Orange-Red)
- Padding tokens are masked
- Shows token-to-token attention scores

**Example:**
```python
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
attention_plot = visualize_attention(tokens, attention_weights, layer=-1, head=0)
plt.show()
```

---

## shap_analysis.py

Module for generating SHAP explanations (placeholder).

### Functions

#### `generate_shap_values(text: str) -> Explanation`

Generates SHAP values for model predictions.

**Parameters:**
- `text` (str): Input text to explain

**Returns:**
- `Explanation`: SHAP explanation object

**Note:** This is a placeholder. Actual implementation is in `sentiment_predictor.py`.

---

## Data Structures

### SentimentResult

Dictionary structure returned by `predict_sentiment()`:

```python
{
    'sentiment': str,           # 'bullish', 'neutral', or 'bearish'
    'score': float,             # VADER score (-1 to 1)
    'direction': str,           # 'bullish', 'neutral', or 'bearish'
    'preprocessed_text': str,   # Cleaned text
    'shap_values': Explanation, # SHAP explanation object
    'attention_plot': Figure    # Matplotlib figure
}
```

### ModelMetrics

Dictionary structure returned by `train_and_evaluate()`:

```python
{
    'accuracy': float,                    # Overall accuracy
    'precision': float,                   # Macro precision
    'recall': float,                      # Macro recall
    'f1': float,                          # Macro F1
    'confusion_matrix': np.ndarray,       # Confusion matrix
    'classification_report': dict,        # Per-class metrics
    'all_labels': list,                   # True labels
    'all_preds': list                     # Predictions
}
```

---

## Usage Examples

### Complete Workflow Example

```python
# 1. Import modules
from sentiment_predictor import analyze_text
from financial_sentiment_analysis import main

# 2. Analyze single text
text = "Apple reported record-breaking revenue in Q4 2023"
result = analyze_text(text)

print(f"Sentiment: {result['sentiment']}")
print(f"Score: {result['score']:.4f}")
print(f"Direction: {result['direction']}")

# 3. Display attention visualization
import matplotlib.pyplot as plt
plt.figure(result['attention_plot'])
plt.show()

# 4. Train model on company data
metrics = main(
    company_name="Apple",
    pdf_path="data/apple",
    csv_path="data/apple/apple_news.csv"
)

print(f"Model Accuracy: {metrics['accuracy']:.4f}")
```

### Streamlit Dashboard Example

```python
import streamlit as st
from sentiment_predictor import analyze_text

st.title("Financial Sentiment Analysis")

# User input
text = st.text_area("Enter financial text:")

if st.button("Analyze"):
    result = analyze_text(text)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Sentiment", result['sentiment'])
    col2.metric("Score", f"{result['score']:.2f}")
    col3.metric("Direction", result['direction'])
    
    # Display attention plot
    st.pyplot(result['attention_plot'])
```

---

## Error Handling

### Common Exceptions

#### `ValueError`
- Raised when input text is empty or invalid
- Raised when dataset has insufficient samples

#### `FileNotFoundError`
- Raised when PDF or CSV files are not found

#### `RuntimeError`
- Raised when model loading fails
- Raised when CUDA is unavailable but required

### Error Handling Example

```python
try:
    result = analyze_text(user_input)
except ValueError as e:
    print(f"Invalid input: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Performance Considerations

### Memory Management

- **Model Loading**: Model is loaded once at module import
- **Batch Processing**: Use `model_predict()` for multiple texts
- **GPU Acceleration**: Automatically uses CUDA if available

### Optimization Tips

1. **Batch Predictions**: Process multiple texts together
```python
texts = ["text1", "text2", "text3"]
logits = model_predict(texts)
```

2. **Disable Gradients**: For inference only
```python
with torch.no_grad():
    outputs = model(**inputs)
```

3. **Reduce Max Length**: For shorter texts
```python
inputs = tokenizer(text, max_length=256, truncation=True)
```

---

## Configuration

### Model Configuration

```python
MODEL_NAME = 'huawei-noah/TinyBERT_General_4L_312D'
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
```

### Preprocessing Configuration

```python
SENTIMENT_THRESHOLD_POSITIVE = 0.05
SENTIMENT_THRESHOLD_NEGATIVE = -0.05
STOPWORDS = set(stopwords.words('english'))
```

---

## Testing

### Running Unit Tests

```bash
pytest unit_test.py -v
```

### Test Coverage

- PDF extraction
- Data merging
- Sentiment scoring
- Direction assignment
- Text preprocessing
- Dataset preparation

---

## Conclusion

This API reference provides comprehensive documentation for all modules and functions in the Financial Sentiment Analysis project. For more details on architecture and design, see [architecture.md](architecture.md).
