from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load and preprocess dataset
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
df['statement'] = df['statement'].apply(preprocess_text)

# Function to train and save the model
def train_and_save_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['statement'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    for model_name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy}")

    y_pred = best_model.predict(X_test_tfidf)
    print("Confusion Matrix:\n", classification_report(y_test, y_pred))

    joblib.dump(best_model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return best_accuracy

accuracy = train_and_save_model(df)
print(f"Initial Model Accuracy: {accuracy}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(data);
    statement = data['statement']

    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    statement_processed = preprocess_text(statement)
    print(statement_processed)
    X = vectorizer.transform([statement_processed])
    print(X)
    prediction = model.predict(X)[0]
    print(prediction)

    return jsonify({
        'statement': statement,
        'prediction': 'True' if prediction == 1 else 'False',
        'accuracy': accuracy
    })

@app.route('/add_statement', methods=['POST'])
def add_statement():
    global df
    data = request.json
    statement = data['statement']
    label = data['label']  # 1 for True, 0 for False

    if label not in [0, 1]:
        return jsonify({"error": "Invalid label. Use 1 for True, 0 for False."}), 400

    statement_processed = preprocess_text(statement)

    # Check if the statement or any part of it already exists
    existing_statements = df['statement'].tolist()
    if statement_processed in existing_statements:
        return jsonify({"error": "Statement already exists in the dataset."}), 400

    for existing_statement in existing_statements:
        if statement_processed in existing_statement or existing_statement in statement_processed:
            return jsonify({"error": "A similar statement already exists in the dataset."}), 400

    
    new_data = pd.DataFrame({'label': [label], 'statement': [statement_processed]})
    df = pd.concat([df, new_data], ignore_index=True)

    new_accuracy = train_and_save_model(df)
    print(f"Updated Model Accuracy: {new_accuracy}")

    return jsonify({
        'message': 'New statement added and model updated successfully.',
        'new_accuracy': new_accuracy
    })

if __name__ == '__main__':
    app.run(debug=True)
