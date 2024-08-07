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
        # Handle different possible formats of SHAP values
        if isinstance(shap_values, shap.Explanation):
            values = shap_values.values
            feature_names = shap_values.feature_names
        elif isinstance(shap_values, np.ndarray):
            values = shap_values
            feature_names = [f"Feature {i}" for i in range(values.shape[0])]
        else:
            return "Unable to interpret SHAP values due to unexpected format."

        # Ensure values is 2D (features, classes)
        if values.ndim == 3:
            values = values[0]  # Take the first sample if we have multiple

        # Sum absolute SHAP values across classes
        total_impact = np.abs(values).sum(axis=1)
        
        # Get indices of top 5 features
        top_indices = np.argsort(total_impact)[-5:][::-1]
        
        explanation = "SHAP Analysis Explanation:\n\n"
        explanation += "The SHAP (SHapley Additive exPlanations) values show how different features contribute to the model's prediction for each class.\n\n"
        explanation += "Top 5 influencing features:\n"
        
        for idx in top_indices:
            feature = feature_names[idx]
            impacts = values[idx]
            explanation += f"- '{feature}':\n"
            for class_idx, impact in enumerate(['Bullish', 'Neutral', 'Bearish']):
                direction = "positively" if impacts[class_idx] > 0 else "negatively"
                explanation += f"    {impact}: impacts {direction} (SHAP value: {impacts[class_idx]:.4f})\n"
        
        explanation += "\nPositive SHAP values push the prediction towards the respective class, while negative values push away from it."
        
        return explanation
    except Exception as e:
        return f"An error occurred while generating SHAP explanation: {str(e)}"

# Initialize input_text
input_text = ""

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
            result = analyze_text(input_text)
            
            # Debug: Print the entire result dictionary
            st.subheader("Debug: Result Dictionary")
            st.write(result)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment", result.get('sentiment', 'N/A'))
            col2.metric("Sentiment Score", f"{result.get('score', 0):.2f}")
            col3.metric("Direction", result.get('direction', 'N/A'))
            
            # Display SHAP plot
            st.subheader("SHAP Analysis")
            shap_values = result.get('shap_values')
            
            if shap_values is None:
                st.error("No SHAP values found in the result.")
            else:
                # Debug information
                st.subheader("Debug Information")
                st.write("SHAP Values Type:", type(shap_values))
                st.write("SHAP Values Shape:", shap_values.shape if hasattr(shap_values, 'shape') else "No shape attribute")
                if isinstance(shap_values, shap.Explanation):
                    st.write("SHAP Values Features:", shap_values.feature_names)
                
                # SHAP Plotting
                direction = result.get('direction', '').lower()
                direction_index = {'bullish': 0, 'neutral': 1, 'bearish': 2}.get(direction)
                
                if direction_index is None:
                    st.error(f"Unknown direction: {direction}")
                else:
                    try:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        shap.plots.waterfall(shap_values[0, :, direction_index], max_display=10, show=False)
                        plt.title(f"SHAP Values for {direction.capitalize()} Direction")
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error creating SHAP plot: {str(e)}")
                        
                        # Fallback: create a simple bar plot
                        values = shap_values.values[0, :, direction_index]
                        top_features = np.argsort(np.abs(values))[-10:]  # Top 10 features
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.barh(range(10), values[top_features])
                        ax.set_yticks(range(10))
                        ax.set_yticklabels([f"Feature {i}" for i in top_features])
                        ax.set_title(f"Top 10 Feature Impacts for {direction.capitalize()} Direction")
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # SHAP Explanation
                    st.subheader("SHAP Explanation")
                    try:
                        values = shap_values.values[0, :, direction_index]
                        feature_names = shap_values.feature_names if shap_values.feature_names else [f"Feature {i}" for i in range(len(values))]
                        
                        top_features = np.argsort(np.abs(values))[-5:][::-1]  # Top 5 features
                        
                        explanation = f"Top 5 features influencing the {direction.capitalize()} direction:\n\n"
                        for i, feature in enumerate(top_features, 1):
                            impact = "positive" if values[feature] > 0 else "negative"
                            explanation += f"{i}. {feature_names[feature]}: {impact} impact (value: {values[feature]:.4f})\n"
                        
                        st.text(explanation)
                    except Exception as e:
                        st.error(f"An error occurred while generating SHAP explanation: {str(e)}")
            
            st.subheader("Input Text")
            st.text(input_text[:1000] + "..." if len(input_text) > 1000 else input_text)
        except ValueError as ve:
            st.error(f"Input Error: {str(ve)}")
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {str(e)}")
            st.error("Please try again with different input or contact support if the problem persists.")
    else:
        st.error("Please enter some text or upload a PDF before analyzing.")