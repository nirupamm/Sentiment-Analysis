import streamlit as st
import pandas as pd
import re
import string 
import joblib
from nltk.corpus import stopwords

@st.cache_resource
def load_models():
    """Load the trained models and vectorizer"""
    vectorizer = joblib.load('vectorizer.joblib')
    models = {
        'Logistic Regression': joblib.load('logreg.joblib'),
        'Naive Bayes': joblib.load('nb.joblib'),
        'Decision Tree': joblib.load('dt.joblib')
    }
    return vectorizer, models

# Text cleaning function (same as training)
stop_words = set(stopwords.words('english'))
negations = {"no", "not", "nor", "never"}
stop_words = stop_words - negations

def clean_text(text):
    text = str(text).lower() 
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def predict_sentiment(text, model_choice):
    """Predict sentiment of given text"""
    # Load models (cached)
    vectorizer, models = load_models()
    
    # Clean and transform text
    cleaned_text = clean_text(text)
    X_new = vectorizer.transform([cleaned_text])
    
    # Predict
    model = models[model_choice]
    pred = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0]
    
    sentiment_map = {0: 'Negative', 1: 'Positive'}
    return sentiment_map[pred], prob

# Streamlit App
st.title("🎬 Sentiment Analysis App")
st.write("Enter a movie review to predict its sentiment!")

# Load models at startup
try:
    vectorizer, models = load_models()
    
    # User Interface
    user_text = st.text_area("Enter your movie review:", height=150)
    
    model_choice = st.selectbox(
        "Choose a model:", 
        ['Logistic Regression', 'Naive Bayes', 'Decision Tree']
    )
    
    if st.button("🔮 Analyze Sentiment"):
        if user_text.strip() == "":
            st.warning("Please enter a review first!")
        else:
            with st.spinner("Analyzing..."):
                result, probabilities = predict_sentiment(user_text, model_choice)
                
                # Display results
                if result == 'Positive':
                    st.success(f"**{result}** 😊")
                else:
                    st.error(f"**{result}** 😞")
                
                # Show confidence
                confidence = max(probabilities)
                st.info(f"Confidence: {confidence:.1%}")
    
    # Sample reviews for testing
    with st.expander("💡 Try these sample reviews"):
        if st.button("Positive Review"):
            st.session_state.sample_text = "This movie was absolutely fantastic! Great acting and storyline."
        if st.button("Negative Review"):
            st.session_state.sample_text = "Terrible movie, waste of time. Poor acting and boring plot."
        
        if 'sample_text' in st.session_state:
            st.text_area("Sample:", st.session_state.sample_text, key="sample_display")

except FileNotFoundError:
    st.error("Model files not found!")
    st.info("""
    Please make sure these files are in your project directory:
    - vectorizer.joblib
    - logreg.joblib
    - nb.joblib  
    - dt.joblib
    """)
    
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    

