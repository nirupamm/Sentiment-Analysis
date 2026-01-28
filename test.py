# Import necessary libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Step 1: Load the Dataset
df = pd.read_csv('imdb_sentiment_train.csv')  # Replace with your actual file path

# Step 2: Clean the Text Data
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['text'] = df['text'].apply(clean_text)

# Step 3: Vectorize the Text Data
vectorizer = CountVectorizer(max_features=5000)  # Use top 5000 words
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label']  # Target variable

# Step 4: Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = logreg.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
