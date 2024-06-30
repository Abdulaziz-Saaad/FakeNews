import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv('liar_dataset/train.tsv', delimiter='\t', header=None)
df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
              'state', 'party', 'barely_true_counts', 'false_counts', 
              'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']

# Select necessary columns
df = df[['label', 'statement']]

# Map labels to binary classification
df['label'] = df['label'].map({
    'true': 1, 
    'mostly-true': 1, 
    'half-true': 1, 
    'barely-true': 0, 
    'false': 0, 
    'pants-fire': 0
})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['statement'], df['label'], test_size=0.2, random_state=42)

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = vectorizer.transform(X_test)

# Initialize Logistic Regression
model = LogisticRegression()

# Train the model
model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Print Accuracy and Classification Report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
