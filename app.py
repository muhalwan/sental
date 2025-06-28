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
            st.write(f"Prediction: **{prediction}**")
            st.write("Probabilities:")
            st.write(f"- Bullish: {probabilities['Bullish']:.4f}")
            st.write(f"- Bearish: {probabilities['Bearish']:.4f}")
            st.write(f"- Neutral: {probabilities['Neutral']:.4f}")

            st.subheader("Sentiment Distribution")
            st.bar_chart(probabilities)

    else:
        st.warning("Please enter some text to analyze.")
