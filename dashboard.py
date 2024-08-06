import streamlit as st
import pandas as pd
from sentiment_predictor import analyze_text
import PyPDF2
import io
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Financial Sentiment Analysis Dashboard", layout="wide")

st.title("Financial Sentiment Analysis Dashboard")

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def explain_shap_values(shap_values):
    feature_names = shap_values.feature_names
    values = shap_values.values[0]
    
    sorted_idx = np.argsort(np.abs(values))
    top_features = [feature_names[i] for i in sorted_idx[-5:]]
    top_values = [values[i] for i in sorted_idx[-5:]]
    
    explanation = "SHAP Analysis Explanation:\n\n"
    explanation += "The SHAP (SHapley Additive exPlanations) graph shows how different words or features contribute to the model's prediction.\n\n"
    explanation += "Top 5 influencing features:\n"
    
    for feature, value in zip(reversed(top_features), reversed(top_values)):
        impact = "positively" if value > 0 else "negatively"
        explanation += f"- '{feature}' impacts the prediction {impact} (SHAP value: {value:.4f})\n"
    
    explanation += "\nPositive SHAP values push the prediction towards a more positive sentiment, while negative values push towards a more negative sentiment."
    
    return explanation

input_type = st.radio("Select input type:", ("Upload PDF", "Paste Text"))

input_text = ""

if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        input_text = read_pdf(uploaded_file)
else:
    input_text = st.text_area("Enter financial news or report text:")

if input_text:
    st.header("Analysis Results")
    
    try:
        result = analyze_text(input_text)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment", result['sentiment'])
        col2.metric("Sentiment Score", f"{result['score']:.2f}")
        col3.metric("Direction", result['direction'])
        
        # Display SHAP plot
        st.subheader("SHAP Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(result['shap_values'][0][:, 0], max_display=10)
        st.pyplot(fig)
        
        # Add SHAP explanation
        st.subheader("SHAP Explanation")
        explanation = explain_shap_values(result['shap_values'])
        st.text(explanation)
        
        st.subheader("Input Text")
        st.text(input_text[:1000] + "..." if len(input_text) > 1000 else input_text)
    except ValueError as ve:
        st.error(f"Input Error: {str(ve)}")
    except Exception as e:
        st.error(f"An unexpected error occurred during analysis: {str(e)}")
        st.error("Please try again with different input or contact support if the problem persists.")