import streamlit as st
import pandas as pd
import joblib

# If you haven't saved your models/vectorizer, you can train them here as before.
# For a real app, it's better to save and load them with joblib or pickle.

# Example: Load your trained models and vectorizer
# vectorizer = joblib.load('vectorizer.joblib')
# models = {
#     'Logistic Regression': joblib.load('logreg.joblib'),
#     'Naive Bayes': joblib.load('nb.joblib'),
#     'Decision Tree': joblib.load('dt.joblib')
# }

# For demonstration, let's assume you have these already in memory:
# vectorizer, models = ...

def predict_sentiment(text, model_choice='Logistic Regression'):
    X_new = vectorizer.transform([text])
    model = models[model_choice]
    pred = model.predict(X_new)[0]
    sentiment_map = {0: 'Negative', 1: 'Positive'}
    return sentiment_map[pred]

st.title("Sentiment Analysis Web App")
st.write("Enter a movie review and select a model to predict its sentiment.")

user_text = st.text_area("Enter your review here:")
model_choice = st.selectbox("Choose model:", ['Logistic Regression', 'Naive Bayes', 'Decision Tree'])

if st.button("Predict Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        result = predict_sentiment(user_text, model_choice)
        st.success(f"Predicted Sentiment: {result}")