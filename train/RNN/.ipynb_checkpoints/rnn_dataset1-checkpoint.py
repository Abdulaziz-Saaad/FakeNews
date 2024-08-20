import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
def load_data():
    df_fake=pd.read_csv('../datasets/dataset1/archive/Fake.csv')
    df_real=pd.read_csv('../datasets/dataset1/archive/True.csv')
    
    # choose the necessary columns 
    df_fake=df_fake.iloc[:,0:2]
    df_real=df_real.iloc[:,0:2]

    # label the class values
    df_fake['class']=0
    df_real['class']=1

    # concatenate 2 dataframes
    df=pd.concat([df_fake, df_real], ignore_index=True, sort=False )

    print(df.shape)
    print(len(df))
    #merge "title" and "text" values in same column
    df.insert(0,column="title_text", value=df['title'] + " " + df['text']) 
    #remove previous columns that are merged
    df.drop (['title', 'text'], inplace=True, axis=1)


    return df

def preprocess_data(data):
    X = df.title_text.values
    y = df['class'].values
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Tokenize text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    
    # Pad sequences
    max_length = 100
    X = pad_sequences(X, maxlen=max_length)
    
    return X, y, tokenizer, le

# Build and train model
def build_model(vocab_size, max_length):
    model = Sequential([
        Embedding(vocab_size, 100, input_length=max_length),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    model.build(input_shape=(None, max_length))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64)
    return model

# Prediction function
def predict_fake_news(text, model, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded)[0][0]
    return prediction

# Main function to be called from Flask app
def setup_and_train():
    data = load_data()
    X, y, tokenizer, le = preprocess_data(data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vocab_size = len(tokenizer.word_index) + 1
    max_length = X.shape[1]
    
    model = build_model(vocab_size, max_length)
    trained_model = train_model(model, X_train, y_train, X_val, y_val)
    
    return trained_model, tokenizer, max_length

# Function to make predictions (to be called from Flask app)
def make_prediction(text, model, tokenizer, max_length):
    prediction = predict_fake_news(text, model, tokenizer, max_length)
    return "Fake" if prediction > 0.5 else "Real"