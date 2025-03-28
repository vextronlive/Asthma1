import os
import streamlit as st

# Force reinstall scikit-learn and numpy
os.system("pip uninstall -y scikit-learn numpy")
os.system("pip install --no-cache-dir scikit-learn==1.2.2 numpy==1.23.5 pandas joblib")

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler  # Should now work!

# Load the model
model_path = "asthma_model.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
else:
    st.error("❌ Model file not found! Please check if 'asthma_model.pkl' is in the project folder.")

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
