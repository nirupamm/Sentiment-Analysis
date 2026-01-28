{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "def98fac-f8b1-42b5-a863-ebd631a3aa28",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-07-24 19:32:39.802 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\Acer\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-07-24 19:32:39.805 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "# If you haven't saved your models/vectorizer, you can train them here as before.\n",
    "# For a real app, it's better to save and load them with joblib or pickle.\n",
    "\n",
    "# Example: Load your trained models and vectorizer\n",
    "# vectorizer = joblib.load('vectorizer.joblib')\n",
    "# models = {\n",
    "#     'Logistic Regression': joblib.load('logreg.joblib'),\n",
    "#     'Naive Bayes': joblib.load('nb.joblib'),\n",
    "#     'Decision Tree': joblib.load('dt.joblib')\n",
    "# }\n",
    "\n",
    "# For demonstration, let's assume you have these already in memory:\n",
    "# vectorizer, models = ...\n",
    "\n",
    "def predict_sentiment(text, model_choice='Logistic Regression'):\n",
    "    X_new = vectorizer.transform([text])\n",
    "    model = models[model_choice]\n",
    "    pred = model.predict(X_new)[0]\n",
    "    sentiment_map = {0: 'Negative', 1: 'Positive'}\n",
    "    return sentiment_map[pred]\n",
    "\n",
    "st.title(\"Sentiment Analysis Web App\")\n",
    "st.write(\"Enter a movie review and select a model to predict its sentiment.\")\n",
    "\n",
    "user_text = st.text_area(\"Enter your review here:\")\n",
    "model_choice = st.selectbox(\"Choose model:\", ['Logistic Regression', 'Naive Bayes', 'Decision Tree'])\n",
    "\n",
    "if st.button(\"Predict Sentiment\"):\n",
    "    if user_text.strip() == \"\":\n",
    "        st.warning(\"Please enter a review.\")\n",
    "    else:\n",
    "        result = predict_sentiment(user_text, model_choice)\n",
    "        st.success(f\"Predicted Sentiment: {result}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "39166531-570f-4a06-984e-115af8e7d0e2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'streamlit run'"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
