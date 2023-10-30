import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load your trained machine learning model
with open('svm.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the StandardScaler (if you used it during training)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Create a function to make predictions
def preprocess_and_predict(data):
    # Preprocess the input data
    # Example: preprocess the input data using the same preprocessing steps used during training
    # You can use the 'scaler' to transform your input data
    data = scaler.transform(data)  # Assuming 'data' is a NumPy array or DataFrame

    # Make predictions
    prediction = model.predict(data)
    return prediction

# Streamlit app
st.title("Machine Learning App")

# Input form
st.write("Enter data to get predictions:")
input_data = st.text_input("Input data here (comma-separated):")

# Make predictions when a user clicks the "Predict" button
if st.button("Predict"):
    if input_data:
        try:
            input_data = [list(map(float, input_data.split(',')))]  # Convert input to a list of floats
            prediction = preprocess_and_predict(input_data)
            st.write("Prediction:", prediction)
        except ValueError:
            st.write("Invalid input. Please enter comma-separated numerical values.")
    else:
        st.write("Please enter data for prediction.")

# Provide instructions or information to the user
st.write("Instructions:")
st.write("- Enter data in the input box as comma-separated values.")
st.write("- Click the 'Predict' button to get predictions.")
