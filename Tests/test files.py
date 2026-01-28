# test_prediction.py
import joblib

# Test if models load
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('logreg.joblib')

# Test prediction
test_text = ["This is a great movie!"]
X_test = vectorizer.transform(test_text)
prediction = model.predict(X_test)
print(f"Test prediction: {prediction[0]}")
print("✅ All models working correctly!")
