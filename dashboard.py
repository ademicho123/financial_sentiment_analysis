import streamlit as st
from financial_sentiment_analysis import process_input, assign_scores, assign_directions, preprocess_text, prepare_dataset, train_and_evaluate, create_comprehensive_report
from shap_analysis import generate_shap_plot
import pandas as pd
import PyPDF2

st.set_page_config(page_title="Financial Sentiment Analysis Dashboard", layout="wide")

st.title("Financial Sentiment Analysis Dashboard")

input_type = st.radio("Select input type:", ("Upload PDF", "Paste Text"))

if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        # Extract text from PDF
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        result = process_input(text, is_pdf=True)
else:
    text_input = st.text_area("Enter financial news or report text:")
    if text_input:
        result = process_input(text_input, is_pdf=False)

if 'result' in locals():
    st.header("Analysis Results")

    # Assuming the result contains sentiment for three companies: 'Company1', 'Company2', 'Company3'
    companies = ['Company1', 'Company2', 'Company3']
    for company in companies:
        st.subheader(f"Sentiment for {company}")
        company_result = result[company]
        
        col1, col2 = st.columns(2)
        col1.metric("Sentiment", company_result['sentiment'])
        col2.metric("Sentiment Score", f"{company_result['score']:.2f}")
        
        # SHAP Analysis
        st.subheader("SHAP Analysis")
        generate_shap_plot(company_result['shap_values'], f"shap_plot_{company}.html")
        st.components.v1.html(open(f"shap_plot_{company}.html").read(), height=600)

        st.subheader("Input Text")
        st.text(company_result['text'])
