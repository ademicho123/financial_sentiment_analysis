import streamlit as st
import pandas as pd
import numpy as np
from sentiment_predictor import analyze_text
import PyPDF2
import io
import shap
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Financial Sentiment Analysis Dashboard", layout="wide")

# Title of the dashboard
st.title("Financial Sentiment Analysis Dashboard")

# Function to read PDF file
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to explain SHAP values
def explain_shap_values(shap_values, class_names):
    try:
        explanation = "SHAP Analysis Explanation:\n\n"
        explanation += "The SHAP (SHapley Additive exPlanations) values show how different features contribute to the model's prediction.\n\n"

        # Process each class separately
        for class_idx, class_name in enumerate(class_names):
            values = shap_values.values[0, :, class_idx]
            feature_names = shap_values.feature_names if shap_values.feature_names is not None else [f"Feature {i}" for i in range(len(values))]

            total_impact = np.abs(values).sum(axis=0)
            top_indices = np.argsort(total_impact)[-5:][::-1]

            explanation += f"\nTop 5 influencing features for {class_name}:\n"
            for idx in top_indices:
                feature = feature_names[idx]
                impact = values[idx]
                direction = "positively" if impact > 0 else "negatively"
                explanation += f"- '{feature}' impacts the prediction {direction} (SHAP value: {impact:.4f})\n"

        explanation += "\nPositive SHAP values push the prediction towards the respective class, while negative values push away from it."

        return explanation
    except Exception as e:
        return f"An error occurred while generating SHAP explanation: {str(e)}"

# Initialize input_text
input_text = ""

# Input type selection
input_type = st.radio("Select input type:", ("Upload PDF", "Paste Text"))

if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        input_text = read_pdf(uploaded_file)
else:
    input_text = st.text_area("Enter financial news or report text:")

# Add a button to trigger the analysis
if st.button("Analyze Text"):
    if input_text:
        st.header("Analysis Results")
        
        try:
            # Analyze the input text
            result = analyze_text(input_text)
            result = predict_sentiment(input_text)
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment", result['sentiment'])
            col2.metric("Sentiment Score", f"{result['score']:.2f}")
            col3.metric("Direction", result['direction'])

            # Display Attention Visualization
            st.subheader("Attention Visualization")
            st.pyplot(result['attention_plot'])
            
            # Display SHAP plot
            st.subheader("SHAP Analysis")
            shap_values = result['shap_values']
            class_names = ['Bullish', 'Neutral', 'Bearish']
            
            if isinstance(shap_values, shap.Explanation):
                for i, class_name in enumerate(class_names):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.plots.waterfall(shap_values[0, :, i], max_display=10, show=False)
                    plt.title(f"SHAP Values for {class_name} Class")
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Add SHAP explanation
            st.subheader("SHAP Explanation")
            explanation = explain_shap_values(shap_values, class_names)
            st.text(explanation)
            
            # Display the input text
            st.subheader("Input Text")
            st.text(input_text[:1000] + "..." if len(input_text) > 1000 else input_text)
        except ValueError as ve:
            st.error(f"Input Error: {str(ve)}")
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {str(e)}")
            st.error("Please try again with different input or contact support if the problem persists.")
    else:
        st.error("Please enter some text or upload a PDF before analyzing.")