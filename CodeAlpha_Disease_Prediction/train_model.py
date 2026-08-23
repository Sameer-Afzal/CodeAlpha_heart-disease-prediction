import pandas as pd
import numpy as np
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


print("=" * 60)
print("DISEASE PREDICTION - HEART DISEASE")
print("=" * 60)


# --------------------------------------------------
# 1. Load UCI Heart Disease Dataset
# --------------------------------------------------

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets


# --------------------------------------------------
# 2. Convert target into binary classification
# --------------------------------------------------
# UCI target:
# 0 = no heart disease
# 1-4 = presence of heart disease

y = y["num"].apply(lambda value: 0 if value == 0 else 1)


print("\nOriginal dataset shape:", X.shape)
print("Target distribution:")
print(y.value_counts())


# --------------------------------------------------
# 3. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# --------------------------------------------------
# 4. Create models
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]),

    "XGBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        ))
    ])
}


# --------------------------------------------------
# 5. Train and evaluate models
# --------------------------------------------------

results = {}

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"{name} Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# --------------------------------------------------
# 6. Find best model
# --------------------------------------------------

best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

best_model = models[best_model_name]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")


# --------------------------------------------------
# 7. Save best model
# --------------------------------------------------

joblib.dump(best_model, "heart_disease_model.pkl")

print("\nBest model saved as:")
print("heart_disease_model.pkl")

print("\nTraining completed successfully!")