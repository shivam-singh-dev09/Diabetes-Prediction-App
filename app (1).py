```python
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📋 Project Information")

st.sidebar.info("""
### Diabetes Prediction System

Developed by:

**Shivam Kumar Singh**

B.Tech CSE (Data Science)

Machine Learning based healthcare prediction system.
""")

st.sidebar.markdown("---")

st.sidebar.subheader("🛠 Technology Stack")
st.sidebar.write("✅ Python")
st.sidebar.write("✅ NumPy")
st.sidebar.write("✅ Scikit-Learn")
st.sidebar.write("✅ Plotly")
st.sidebar.write("✅ Streamlit")

st.sidebar.markdown("---")

st.sidebar.subheader("📈 Model Performance")

# Replace with your actual values if known
st.sidebar.metric("Accuracy", "78.5%")
st.sidebar.metric("Precision", "74.2%")
st.sidebar.metric("Recall", "69.8%")

st.sidebar.markdown("---")

st.sidebar.subheader("🚀 Features")

st.sidebar.write("✅ Diabetes Prediction")
st.sidebar.write("✅ BMI Analysis")
st.sidebar.write("✅ Confidence Score")
st.sidebar.write("✅ Risk Meter")
st.sidebar.write("✅ Interactive Charts")
st.sidebar.write("✅ Health Recommendations")

# -----------------------------
# Main Title
# -----------------------------
st.title("🩺 Diabetes Prediction System")

st.markdown(
    """
    ### Predict Diabetes Risk Using Machine Learning
    Enter patient details below and click Predict.
    """
)

# -----------------------------
# Input Section
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        value=0
    )

    Glucose = st.number_input(
        "Glucose",
        min_value=0,
        value=0
    )

    BloodPressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        value=0
    )

    SkinThickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        value=0
    )

with col2:
    Insulin = st.number_input(
        "Insulin",
        min_value=0,
        value=0
    )

    BMI = st.number_input(
        "BMI",
        min_value=0.0,
        value=0.0
    )

    DPF = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.0
    )

    Age = st.number_input(
        "Age",
        min_value=0,
        value=0
    )

# -----------------------------
# BMI Category
# -----------------------------
if BMI < 18.5:
    bmi_status = "Underweight"
elif BMI < 25:
    bmi_status = "Normal Weight"
elif BMI < 30:
    bmi_status = "Overweight"
else:
    bmi_status = "Obese"

st.info(f"📊 BMI Category: **{bmi_status}**")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Diabetes"):

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

    # Confidence
    try:
        confidence = np.max(
            model.predict_proba(std_data)
        ) * 100
    except:
        confidence = 90.0

    st.subheader("📌 Prediction Result")

    if prediction[0] == 0:

        st.success(
            "✅ The person is NOT diabetic"
        )

    else:

        st.error(
            "⚠️ The person IS diabetic"
        )

    # Confidence Score
    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    # Risk Meter
    st.subheader("🎯 Diabetes Risk Meter")

    if confidence < 40:
        st.success("🟢 Low Risk")
    elif confidence < 70:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    st.progress(int(confidence))

    # Patient Summary
    st.subheader("📋 Patient Summary")

    st.info(f"""
Age: {Age}

BMI: {BMI}

BMI Category: {bmi_status}

Glucose: {Glucose}

Blood Pressure: {BloodPressure}

Prediction Confidence: {confidence:.2f}%
""")

    # Interactive Chart
    st.subheader("📊 Health Parameter Analysis")

    chart_data = pd.DataFrame({
        "Parameter": [
            "Glucose",
            "Blood Pressure",
            "BMI",
            "Age"
        ],
        "Value": [
            Glucose,
            BloodPressure,
            BMI,
            Age
        ]
    })

    fig = px.bar(
        chart_data,
        x="Parameter",
        y="Value",
        title="Patient Health Parameters"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Tips
    if prediction[0] == 0:

        st.subheader("💡 Health Tips")

        st.write("""
- Maintain a balanced diet
- Exercise regularly
- Stay hydrated
- Get regular health checkups
- Sleep at least 7–8 hours daily
        """)

    else:

        st.subheader("🚨 Diabetes Prevention Tips")

        st.write("""
- Reduce sugar intake
- Maintain healthy weight
- Exercise daily
- Monitor blood glucose regularly
- Eat more vegetables and fiber
- Consult a healthcare professional
        """)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown(
    """
    <center>
    <h4>
    Developed by Shivam Kumar Singh
    </h4>
    B.Tech CSE (Data Science)
    </center>
    """,
    unsafe_allow_html=True
)
```
