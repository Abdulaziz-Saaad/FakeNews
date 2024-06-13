import joblib

# Load the model and vectorizer
model = joblib.load('./train/model.pkl')
vectorizer = joblib.load('./train/vectorizer.pkl')

# Example test news
test_statements = [
    "NASA’s Perseverance rover has successfully collected its first samples from the surface of Mars.",
    "Scientists have discovered a hidden city on the dark side of the moon, populated by ancient aliens.",
    "The World Health Organization has declared the COVID-19 pandemic to be under control due to widespread vaccination efforts.",
    "Drinking bleach can cure COVID-19, according to a viral social media post.",
    "Apple has announced the release of its latest iPhone model, featuring advanced AI capabilities and a new design.",
    "The government is secretly replacing citizens' mobile phones with devices that track their every move."
]

# Transform test statements using the trained TF-IDF vectorizer
test_tfidf = vectorizer.transform(test_statements)

# Predict using the trained model
predictions = model.predict(test_tfidf)

# Print results
for statement, prediction in zip(test_statements, predictions):
    print(f"Statement: {statement}\nPredicted Label: {'True' if prediction == 1 else 'False'}\n")
