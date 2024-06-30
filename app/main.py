
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)
CORS(app)

# Load initial dataset
data_path = '../train/liar_dataset/train.tsv'
df = pd.read_csv(data_path, delimiter='\t', header=None)
df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
              'state', 'party', 'barely_true_counts', 'false_counts', 
              'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']
df = df[['label', 'statement']]
df['label'] = df['label'].map({
    'true': 1, 
    'mostly-true': 1, 
    'half-true': 1, 
    'barely-true': 0, 
    'false': 0, 
    'pants-fire': 0
})
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# Function to train and save the model
def train_and_save_model(df):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(df['statement'], df['label'], test_size=0.2, random_state=42)

    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Initialize and train Logistic Regression
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the model and vectorizer
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return accuracy

# Initial training
accuracy = train_and_save_model(df)
print(f"Initial Model Accuracy: {accuracy}")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    statement = data['statement']
    
    # Load the model and vectorizer
    model = joblib.load('../train/model.pkl')
    vectorizer = joblib.load('../train/vectorizer.pkl')

    X = vectorizer.transform([statement])
    prediction = model.predict(X)
    print(f"Prediction: {prediction}")

    return jsonify({
        'statement': statement,
        'prediction': 'True' if prediction == 1 else 'False',
        'accuracy': accuracy
    })

@app.route('/add_statement', methods=['POST'])
def add_statement():
    data = request.json
    statement = data['statement']
    label = data['label']  # 1 for True, 0 for False
    
    # Validate input
    if label not in [0, 1]:
        return jsonify({"error": "Invalid label. Use 1 for True, 0 for False."}), 400
    
    # Add the new data to the dataframe
    global df
    new_data = pd.DataFrame({'label': [label], 'statement': [statement]})
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Retrain the model with updated data
    new_accuracy = train_and_save_model(df)
    print(f"Updated Model Accuracy: {new_accuracy}")
    
    return jsonify({
        'message': 'New statement added and model updated successfully.',
        'new_accuracy': new_accuracy
    })

if __name__ == '__main__':
    app.run(debug=True)
