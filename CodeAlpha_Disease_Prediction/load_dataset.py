from ucimlrepo import fetch_ucirepo

# Load UCI Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

# Features
X = heart_disease.data.features

# Target
y = heart_disease.data.targets

print("=" * 60)
print("HEART DISEASE DATASET")
print("=" * 60)

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.columns.tolist())

print("\nFirst 5 Rows:")
print(X.head())

print("\nTarget Values:")
print(y.head())

print("\nMissing Values:")
print(X.isnull().sum())

print("\nDataset Information:")
print(heart_disease.metadata)