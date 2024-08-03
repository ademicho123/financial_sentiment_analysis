import streamlit as st
import pandas as pd
import plotly.express as px
from financial_sentiment_analysis import process_input
from shap_analysis import generate_shap_plot

st.set_page_config(page_title="Financial Sentiment Analysis Dashboard", layout="wide")

st.title("Financial Sentiment Analysis Dashboard")

input_type = st.radio("Select input type:", ("Upload PDF", "Paste Text"))

if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        result = process_input(uploaded_file, is_pdf=True)
else:
    text_input = st.text_area("Enter financial news or report text:")
    if text_input:
        result = process_input(text_input, is_pdf=False)

if 'result' in locals():
    st.header("Analysis Results")
    
    col1, col2 = st.columns(2)
    col1.metric("Sentiment", result['sentiment'])
    col2.metric("Sentiment Score", f"{result['score']:.2f}")
    
    st.subheader("SHAP Analysis")
    generate_shap_plot(result['shap_values'], "shap_plot.html")
    st.components.v1.html(open("shap_plot.html").read(), height=600)
    
    st.subheader("Input Text")
    st.text(result['text'])