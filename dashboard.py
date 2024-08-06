import streamlit as st
import pandas as pd
from sentiment_predictor import analyze_text
import PyPDF2
import io
import shap
import matplotlib.pyplot as plt

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
        result = analyze_text(input_text)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment", result['sentiment'])
        col2.metric("Sentiment Score", f"{result['score']:.2f}")
        col3.metric("Direction", result['direction'])
        
        # Debug information
        st.subheader("Debug Information")
        st.write("Preprocessed Text:", result['preprocessed_text'])
        st.write("SHAP Values Shape:", result['shap_values'].shape)
        st.write("SHAP Values Type:", type(result['shap_values']))
        
        # Display SHAP plot
        st.subheader("SHAP Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.text(result['shap_values'], matplotlib=True)
        st.pyplot(fig)
        
        st.subheader("Input Text")
        st.text(input_text[:1000] + "..." if len(input_text) > 1000 else input_text)
    except ValueError as ve:
        st.error(f"Input Error: {str(ve)}")
    except Exception as e:
        st.error(f"An unexpected error occurred during analysis: {str(e)}")
        st.error("Please try again with different input or contact support if the problem persists.")