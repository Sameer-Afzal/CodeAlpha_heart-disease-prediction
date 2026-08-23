# ❤️ Heart Disease Prediction System

A machine learning web app that predicts the likelihood of heart disease from a patient's clinical data, built with **Streamlit** and **scikit-learn**. This is Project 2 of my Machine Learning internship with **CodeAlpha**.

🔗 **Live Demo:** _(add your deployed Streamlit link here once you deploy it)_

---

## 📋 Overview

The app takes 13 clinical features (age, sex, chest pain type, blood pressure, cholesterol, ECG results, etc.) as input and predicts whether a patient is likely to have heart disease, using a model trained on the **UCI Heart Disease dataset**.

## 🧠 Model Performance

Four algorithms were trained and compared on the held-out test set:

| Model                | Accuracy |
|-----------------------|----------|
| Logistic Regression    | 86.89%   |
| SVM                    | 85.25%   |
| **Random Forest**      | **90.16%** 🏆 |
| XGBoost                | 83.61%   |

**Random Forest** was selected as the final model based on test accuracy.

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — web app / UI
- **scikit-learn** — Logistic Regression, SVM, Random Forest, preprocessing (imputation + scaling)
- **XGBoost** — gradient boosting classifier
- **pandas / numpy** — data handling
- **joblib** — model persistence

## 📂 Project Structure

```
heart-disease-prediction/
├── app.py                    # Streamlit web app
├── load_dataset.py           # Loads and inspects the UCI dataset
├── preprocess_data.py        # Cleans, imputes, scales, and splits the data
├── train_model.py            # Trains & compares 4 models, saves the best one
├── test_model.py             # Quick script to test the saved model on a sample patient
├── requirements.txt
├── heart_disease_model.pkl   # Trained model (Random Forest)
├── imputer.pkl                # Fitted SimpleImputer
├── scaler.pkl                  # Fitted StandardScaler
└── README.md
```

## ⚙️ How It Works

1. **`load_dataset.py`** — fetches the UCI Heart Disease dataset (`ucimlrepo`, id=45) and inspects its shape, features, and missing values.
2. **`preprocess_data.py`** — converts the target to binary (0 = no disease, 1 = disease), imputes missing values with the median, splits into train/test (80/20, stratified), and standardizes features.
3. **`train_model.py`** — trains Logistic Regression, SVM, Random Forest, and XGBoost inside `sklearn` pipelines, evaluates each on the test set, and saves the best-performing model.
4. **`app.py`** — loads the saved model, imputer, and scaler, collects patient inputs through a Streamlit form, and returns a live prediction.

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 Dataset

[UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) — 13 clinical features used to predict the presence of heart disease.

## ⚠️ Disclaimer

This project is built for **educational purposes** as part of a machine learning internship. It is **not** a medical diagnostic tool. Always consult a qualified healthcare professional for medical advice.

## 🙌 Acknowledgements

Built as part of my Machine Learning internship at **[CodeAlpha](https://www.linkedin.com/company/codealpha-technologies/)**.

---

📌 **Internship Progress:**
- ✅ Project 1 — Credit Score Prediction
- ✅ Project 2 — Heart Disease Prediction (this repo)
- ⏳ Project 3 — Coming soon
