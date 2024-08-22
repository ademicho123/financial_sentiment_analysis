# Financial Sentiment Analysis Dashboard

This project provides a comprehensive financial sentiment analysis tool using natural language processing and machine learning techniques. It includes a web-based dashboard for real-time analysis of financial texts and reports.

## Features

- Upload PDF files or paste text for analysis
- Sentiment prediction (Bullish, Neutral, Bearish)
- Sentiment scoring
- SHAP (SHapley Additive exPlanations) analysis for model interpretability
- Attention visualization for understanding model focus
- Interactive visualizations

## Installation

1. Clone the repository:
git clone https://github.com/Birkbeck/msc-projects-2023-4-ademicho123.git
cd financial-sentiment-analysis
2. Install the required packages:
pip install -r requirements.txt

## Usage

1. Run the Streamlit dashboard:
streamlit run dashboard.py
2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`)

3. Upload a PDF file or paste text into the input area

4. Click "Analyze Text" to see the results

## Project Structure

- `dashboard.py`: Main Streamlit application for the web interface
- `financial_sentiment_analysis.py`: Core sentiment analysis functionality
- `sentiment_predictor.py`: Sentiment prediction model and utilities
- `shap_analysis.py`: SHAP value generation for model interpretability
- `requirements.txt`: List of required Python packages

## Model

The project evaluates multiple machine learning models for sentiment analysis:

- TinyBERT (huawei-noah/TinyBERT_General_4L_312D)
- AdaBoost
- Support Vector Machine (SVM)
- Random Forest
- Naive Bayes
- Logistic Regression
- Neural Network

TinyBERT consistently outperformed other models and was chosen for the final implementation in the Streamlit dashboard.

## Data Preprocessing

- Text cleaning (removing punctuation, numbers, emojis)
- Tokenization
- Lemmatization
- Stop word removal

## Training

The models are trained using:

- Data augmentation techniques (SynonymAug)
- SMOTE and RandomUnderSampler for handling class imbalance
- For TinyBERT: AdamW optimizer with CosineAnnealingWarmRestarts learning rate scheduler

## Evaluation

The model's performance is evaluated using:
- Accuracy
- Precision, Recall, and F1-score for each class
- Confusion matrix
- Cross-validation scores

## SHAP Analysis

- SHAP (SHapley Additive exPlanations) values are used to interpret the model's predictions, showing how each feature contributes to the output.
- Attention visualization helps users understand which parts of the input text the model focuses on when making predictions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.