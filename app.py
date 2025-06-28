import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction import predict_sentiment

st.title("Financial Sentiment Analysis")

text = st.text_area("Enter a tweet to analyze:")

if st.button("Analyze"):
    if text:
        with st.spinner("Analyzing..."):
            prediction, probabilities = predict_sentiment(text)
            probability_value = probabilities[prediction]
            st.write(f"Prediction: {prediction.lower()}")
            st.write(f"Probability: {probability_value:.0%}")

    else:
        st.warning("Please enter some text to analyze.")
