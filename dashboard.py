import streamlit as st
import pandas as pd
from sentiment_predictor import analyze_text
from shap_analysis import generate_shap_plot
import PyPDF2
import io

st.set_page_config(page_title="Financial Sentiment Analysis Dashboard", layout="wide")

st.title("Financial Sentiment Analysis Dashboard")

input_type = st.radio("Select input type:", ("Upload PDF", "Paste Text"))

if input_text:
    st.header("Analysis Results")
    
    results = analyze_text(input_text)
    
    for company, result in results.items():
        st.subheader(f"{company} Analysis")
        
        col1, col2 = st.columns(2)
        col1.metric("Sentiment", result['sentiment'])
        col2.metric("Sentiment Score", f"{result['score']:.2f}")
        
        # Generate and display SHAP plot
        st.subheader("SHAP Analysis")
        shap_plot = shap.plots.text(result['shap_values'][0], display=False)
        st.pyplot(shap_plot)
        
    st.subheader("Input Text")
    st.text(input_text[:1000] + "..." if len(input_text) > 1000 else input_text)
