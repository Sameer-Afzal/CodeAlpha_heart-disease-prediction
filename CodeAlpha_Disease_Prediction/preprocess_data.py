import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Load the UCI Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

# Convert target into binary classification
# 0 = No heart disease
# 1 = Heart disease
y = (y["num"] > 0).astype(int)

print("\nOriginal dataset shape:")
print(X.shape)

print("\nOriginal target distribution:")
print(y.value_counts())

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Convert back to DataFrame
X = pd.DataFrame(X_imputed, columns=X.columns)

print("\nMissing values after preprocessing:")
print(X.isnull().sum())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessing objects
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save processed datasets
joblib.dump(X_train_scaled, "X_train.pkl")
joblib.dump(X_test_scaled, "X_test.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")

print("\nPreprocessing completed successfully!")

print("\nFinal feature shape:")
print(X.shape)

print("\nTarget distribution after conversion:")
print(y.value_counts())

print("\nSaved files:")
print("- imputer.pkl")
print("- scaler.pkl")
print("- X_train.pkl")
print("- X_test.pkl")
print("- y_train.pkl")
print("- y_test.pkl")

print("\n" + "=" * 60)