import streamlit as st
import pandas as pd
from sentiment_predictor import analyze_text
from shap_analysis import generate_shap_plot
import PyPDF2
import io

st.set_page_config(page_title="Financial Sentiment Analysis Dashboard", layout="wide")

st.title("Financial Sentiment Analysis Dashboard")

# Function to read PDF content
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

input_type = st.radio("Select input type:", ("Upload PDF", "Paste Text"))

input_text = ""  # Initialize input_text

if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        input_text = read_pdf(uploaded_file)
else:
    input_text = st.text_area("Enter financial news or report text:")

if input_text:
    st.header("Analysis Results")
    
    try:
        results = analyze_text(input_text)
        
        for company, result in results.items():
            st.subheader(f"{company} Analysis")
            
            col1, col2 = st.columns(2)
            col1.metric("Sentiment", result['sentiment'])
            col2.metric("Sentiment Score", f"{result['score']:.2f}")
            
            # Generate and display SHAP plot
            st.subheader("SHAP Analysis")
            shap_plot = generate_shap_plot(result['shap_values'])
            st.pyplot(shap_plot)
        
        st.subheader("Input Text")
        st.text(input_text[:1000] + "..." if len(input_text) > 1000 else input_text)
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")