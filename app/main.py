
from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('../train/model.pkl')
vectorizer = joblib.load('../train/vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    statement = data['statement']
    X = vectorizer.transform([statement])
    prediction = model.predict(X)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
