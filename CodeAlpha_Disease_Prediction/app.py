import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL AND PREPROCESSING FILES
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("heart_disease_model.pkl")
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, imputer, scaler


model, imputer, scaler = load_model()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("❤️ Heart Disease Prediction System")

st.write(
    "Enter the patient's information below to generate a machine-learning prediction."
)

st.divider()


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("Patient Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=55
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=140
    )


with col2:

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=250
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    restecg = st.selectbox(
        "Resting ECG",
        options=[0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )


with col3:

    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope",
        options=[1, 2, 3]
    )

    ca = st.selectbox(
        "Number of Major Vessels (CA)",
        options=[0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thal",
        options=[3, 6, 7]
    )


st.divider()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔍 Predict Heart Disease", use_container_width=True):

    # Create DataFrame with the SAME feature names
    # used during model training

    patient_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # Apply the same preprocessing used during training

    patient_imputed = imputer.transform(patient_data)

    # Convert the imputed data back to a DataFrame
    # so the scaler receives the original feature names

    patient_imputed = pd.DataFrame(
        patient_imputed,
        columns=patient_data.columns
    )

    patient_scaled = scaler.transform(patient_imputed)

    # Make prediction

    prediction = model.predict(patient_scaled)[0]

    # Display result

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ The model predicts a possibility of heart disease."
        )

    else:

        st.success(
            "✅ The model predicts no heart disease."
        )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------


st.divider()

st.subheader("📊 Model Performance")

st.write(
    "Four machine-learning algorithms were evaluated on the test dataset."
)

performance_data = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "SVM",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        86.89,
        85.25,
        90.16,
        83.61
    ]
})

st.dataframe(
    performance_data,
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    performance_data.set_index("Model")["Accuracy"]
)

st.success(
    "🏆 Random Forest achieved the highest test accuracy of 90.16%."
)


# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.divider()

st.caption(
    "⚠️ This application is an educational machine-learning project "
    "and is not a medical diagnosis. Please consult a qualified healthcare "
    "professional for medical advice."
)