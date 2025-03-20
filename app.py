import os
import streamlit as st
import pickle  # Use pickle instead of joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model using pickle
with open("asthma_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Asthma Disease Prediction")

# Example input fields
age = st.number_input("Enter Age", min_value=1, max_value=100)
bmi = st.number_input("Enter BMI", min_value=10.0, max_value=50.0)
smoking = st.selectbox("Do you smoke?", ["No", "Yes"])

if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[age, bmi, 1 if smoking == "Yes" else 0]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("High risk of asthma detected!")
    else:
        st.success("No asthma detected.")
