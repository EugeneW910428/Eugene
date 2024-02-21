import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text
    sequence = tokenizer.texts_to_sequences([text])
    # Pad sequences
    sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    return sequence

# Load the dataset
dataset_path = "drugLibTrain_raw.tsv"
data = pd.read_csv(dataset_path, sep='\t')
data.dropna(subset=['benefitsReview'], inplace=True)

# Split dataset into features and labels
X = data['benefitsReview'].values
y = data['rating'].apply(lambda x: 1 if x > 5 else 0).values  # Binary classification based on rating

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
max_sequence_length = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test))

# Save the trained model
model.save('sentiment_model.h5')

# Streamlit UI
st.title('Drug Review Sentiment Analysis')

# Load the trained model
model = tf.keras.models.load_model('sentiment_model.h5')

# File upload option for the user to upload a TSV file containing drug reviews
uploaded_file = st.file_uploader("Upload TSV file", type=['tsv'])

# Perform sentiment analysis when the user uploads a file
if uploaded_file is not None:
    # Read the uploaded TSV file into a DataFrame
    data = pd.read_csv(uploaded_file, sep='\t')
    st.write('Original Data:')
    st.write(data)
    data.dropna(subset=['benefitsReview'], inplace=True)

    # Allow user to select a medicine
    selected_drug = st.selectbox('Select a drug', sorted(data['urlDrugName'].unique()))
    
    # Filter data for the selected drug
    selected_drug_data = data[data['urlDrugName'] == selected_drug]
    
    st.write('Processed Data:')
    st.write(selected_drug_data[['urlDrugName', 'benefitsReview']])
    
    # Perform sentiment analysis on the selected drug's reviews
    for review in selected_drug_data['benefitsReview']:
        preprocessed_review = preprocess_text(review)
        prediction = model.predict(preprocessed_review)
        sentiment = 'Positive' if prediction > 0.5 else 'Negative'
        st.write(f"Review: {review}")
        st.write(f"Sentiment: {sentiment}")

