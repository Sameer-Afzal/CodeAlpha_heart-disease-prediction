import joblib
import pandas as pd

# Load trained model
model = joblib.load("heart_disease_model.pkl")

# Load preprocessing objects
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names used during training
features = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

# Example patient
patient = pd.DataFrame([[
    55,   # age
    1,    # sex
    2,    # cp
    140,  # trestbps
    250,  # chol
    0,    # fbs
    1,    # restecg
    150,  # thalach
    0,    # exang
    1.0,  # oldpeak
    2,    # slope
    0,    # ca
    3     # thal
]], columns=features)

# Preprocess
patient_imputed = imputer.transform(patient)
patient_scaled = scaler.transform(patient_imputed)

# Prediction
prediction = model.predict(patient_scaled)

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("Result: Heart disease detected")
else:
    print("Result: No heart disease detected")