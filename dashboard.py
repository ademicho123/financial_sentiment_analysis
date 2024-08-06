import streamlit as st
import pandas as pd
import numpy as np
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
    try:
        # Assuming shap_values is a 2D array with shape (n_features, n_classes)
        values = shap_values.values
        feature_names = shap_values.feature_names

        # Sum absolute SHAP values across classes
        total_impact = np.abs(values).sum(axis=1)
        
        # Get indices of top 5 features
        top_indices = np.argsort(total_impact)[-5:][::-1]
        
        explanation = "SHAP Analysis Explanation:\n\n"
        explanation += "The SHAP (SHapley Additive exPlanations) values show how different words or features contribute to the model's prediction for each class.\n\n"
        explanation += "Top 5 influencing features:\n"
        
        for idx in top_indices:
            feature = feature_names[idx]
            impacts = values[idx]
            explanation += f"- '{feature}':\n"
            for class_idx, impact in enumerate(impacts):
                class_name = ['bullish', 'neutral', 'bearish'][class_idx]
                direction = "positively" if impact > 0 else "negatively"
                explanation += f"    {class_name}: impacts {direction} (SHAP value: {impact:.4f})\n"
        
        explanation += "\nPositive SHAP values push the prediction towards the respective class, while negative values push away from it."
        
        return explanation
    except Exception as e:
        return f"An error occurred while generating SHAP explanation: {str(e)}"

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
        
        # Display SHAP plots
        st.subheader("SHAP Analysis")
        shap_values = result['shap_values']
        for i, class_name in enumerate(['Bullish', 'Neutral', 'Bearish']):
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0, :, i], max_display=10, show=False)
            plt.title(f"SHAP Values for {class_name} Class")
            st.pyplot(fig)
            plt.close(fig)
        
        # Add SHAP explanation
        st.subheader("SHAP Explanation")
        explanation = explain_shap_values(shap_values)
        st.text(explanation)
        
        st.subheader("Input Text")
        st.text(input_text[:1000] + "..." if len(input_text) > 1000 else input_text)
    except ValueError as ve:
        st.error(f"Input Error: {str(ve)}")
    except Exception as e:
        st.error(f"An unexpected error occurred during analysis: {str(e)}")
        st.error("Please try again with different input or contact support if the problem persists.")