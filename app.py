import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("heart_disease_model.pkl")

# Page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# Title
st.title("❤🩺 Heart Disease Prediction App❤🩺")
st.write("Enter patient details to predict the risk of heart disease")

st.markdown("---")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)
resting_bp = st.number_input("Resting Blood Pressure", value=120)
cholesterol = st.number_input("Cholesterol", value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar (1 = High, 0 = Normal)", [0, 1])
resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)
max_hr = st.number_input("Max Heart Rate", value=150)
exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", value=1.0)
st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# Predict Button
if st.button("🔍 Predict"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    prediction = model.predict(input_data)[0]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
