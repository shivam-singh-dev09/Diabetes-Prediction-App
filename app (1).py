import streamlit as st
import numpy as np
import pickle

# Load Model and Scaler
model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Page Configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# Sidebar
st.sidebar.title("📋 Project Information")
st.sidebar.info("""
**Diabetes Prediction System**

Developed by:
**Shivam Kumar Singh**

B.Tech CSE (Data Science)

Machine Learning based healthcare prediction system.
""")

st.sidebar.markdown("---")
st.sidebar.write("Technology Stack")
st.sidebar.write("✅ Python")
st.sidebar.write("✅ Scikit-Learn")
st.sidebar.write("✅ Streamlit")
st.sidebar.write("✅ Machine Learning")

# Main Title
st.title("🩺 Diabetes Prediction System")
st.markdown("### Predict Diabetes Risk Using Machine Learning")

st.write("Enter patient details below:")

# Input Fields
Pregnancies = st.number_input("Pregnancies", min_value=0)
Glucose = st.number_input("Glucose", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0)
Age = st.number_input("Age", min_value=0)

# BMI Category
if BMI < 18.5:
    bmi_status = "Underweight"
elif BMI < 25:
    bmi_status = "Normal Weight"
elif BMI < 30:
    bmi_status = "Overweight"
else:
    bmi_status = "Obese"

st.info(f"📊 BMI Category: **{bmi_status}**")

# Prediction Button
if st.button("🔍 Predict"):

    input_data = np.asarray([
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DPF,
        Age
    ]).reshape(1, -1)

    std_data = scaler.transform(input_data)

    prediction = model.predict(std_data)

    if prediction[0] == 0:

        st.success("✅ The person is NOT diabetic")

        st.subheader("💡 Health Tips")
        st.write("""
        - Maintain a balanced diet
        - Exercise regularly
        - Stay hydrated
        - Get regular health checkups
        """)

    else:

        st.error("⚠️ The person IS diabetic")

        st.subheader("🚨 Diabetes Prevention Tips")
        st.write("""
        - Reduce sugar intake
        - Maintain healthy weight
        - Exercise daily
        - Monitor blood glucose regularly
        - Consult a healthcare professional
        """)

# Footer
st.markdown("---")
st.markdown(
    "<center>Developed by Shivam Kumar Singh | B.Tech CSE Data Science</center>",
    unsafe_allow_html=True
)
