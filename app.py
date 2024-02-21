import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess your dataset
@st.cache
def load_data():
    # Load your dataset
    data = pd.read_csv("your_dataset.csv")

    # Preprocess your data (handle missing values, encoding, etc.)
    # For simplicity, let's assume the dataset is already preprocessed

    return data

# Train your machine learning model
def train_model(data):
    # Split your dataset into features (X) and target (y)
    X = data.drop(columns=["target_column"])
    y = data["target_column"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train your machine learning model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

# Streamlit UI
def main():
    st.title("Your Machine Learning App")

    # Load data
    data = load_data()

    # Display dataset if needed
    if st.checkbox("Show Dataset"):
        st.write(data)

    # Train the machine learning model
    st.write("Training the model...")
    model, accuracy = train_model(data)
    st.write("Model trained with accuracy:", accuracy)

    # User input
    st.write("Enter your data below:")
    user_input = st.text_input("Input Data")

    # Prediction
    if st.button("Predict"):
        # Perform preprocessing on user input if needed
        # For simplicity, let's assume the input is preprocessed

        # Convert user input to DataFrame
        input_data = pd.DataFrame([user_input], columns=data.columns[:-1])  # Assuming last column is target

        # Make prediction
        prediction = model.predict(input_data)

        # Display prediction
        st.write("Prediction:", prediction[0])

# Run the app
if __name__ == "__main__":
    main()
