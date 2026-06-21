
import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('diabetes_model.sav','rb'))
scaler = pickle.load(open('scaler.sav','rb'))

st.title("🩺 Diabetes Prediction System")

st.write("Enter patient details below")

Pregnancies = st.number_input("Pregnancies", min_value=0)
Glucose = st.number_input("Glucose", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0)
Age = st.number_input("Age", min_value=0)

if st.button("Predict"):

    data = np.asarray([
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DPF,
        Age
    ]).reshape(1,-1)

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 0:
        st.success("✅ The person is NOT diabetic")
    else:
        st.error("⚠️ The person IS diabetic")
