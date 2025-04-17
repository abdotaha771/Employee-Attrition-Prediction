# employee_attrition_prediction.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the model
model = joblib.load("best_model.pkl")

# Page setup
st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")
st.title("🔮 Employee Attrition Prediction")
st.markdown("**Enter employee details to predict attrition risk.**")

# Initialize LabelEncoders
gender_encoder = LabelEncoder().fit(['Male', 'Female'])
marital_encoder = LabelEncoder().fit(['Single', 'Married', 'Divorced'])
remote_encoder = LabelEncoder().fit(['Yes', 'No'])

# Input fields
age = st.number_input("Age", min_value=18, max_value=65, value=30)
gender = st.selectbox("Gender", gender_encoder.classes_)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
wlb = st.number_input("Work-Life Balance (1-5)", min_value=1, max_value=5, value=3)
satisfaction = st.number_input("Job Satisfaction (1-5)", min_value=1, max_value=5, value=3)
distance = st.number_input("Distance from Home (km)", min_value=0, max_value=50, value=10)
marital = st.selectbox("Marital Status", marital_encoder.classes_)
job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2)
remote = st.selectbox("Remote Work?", remote_encoder.classes_)

# Prediction button
if st.button("🔍 Predict"):
    # Transform inputs
    gender_encoded = gender_encoder.transform([gender])[0]
    marital_encoded = marital_encoder.transform([marital])[0]
    remote_encoded = remote_encoder.transform([remote])[0]
    
    # Create DataFrame
    user_input = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_encoded],
        'Years at Company': [years_at_company],
        'Work-Life Balance': [wlb],
        'Job Satisfaction': [satisfaction],
        'Distance from Home': [distance],
        'Marital Status': [marital_encoded],
        'Job Level': [job_level],
        'Remote Work': [remote_encoded]
    })

    # Make prediction (assuming model returns 1=stay, 0=leave)
    pred = model.predict(user_input)[0]
    proba = model.predict_proba(user_input)[0]  # [P(leave), P(stay)]
    
    # Get stay probability (index 1 if 1=stay in model)
    stay_prob = proba[1] if model.classes_[1] == 1 else proba[0]

    # Display results
    if pred == 1:
        st.success(f"✅ Employee is **likely to stay** (Probability: {stay_prob:.2%})")
    else:
        st.error(f"🚨 Employee is **likely to leave** (Probability: {1-stay_prob:.2%})")