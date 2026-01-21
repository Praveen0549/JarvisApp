# task1_data_preparation.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
df = pd.read_csv("churn.csv")

# 2. Check first few rows
print("🔹 First 5 rows of dataset:")
print(df.head())

# 3. Check missing values
print("\n🔹 Missing values in each column:")
print(df.isnull().sum())

# 4. Handle missing values
# Example: Fill numeric columns with mean, categorical with mode
for col in df.columns:
    if df[col].dtype == "object":  # categorical
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:  # numeric
        df[col].fillna(df[col].mean(), inplace=True)

# 5. Drop customerID (not useful for prediction)
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# 6. Encode categorical variables
label_encoder = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = label_encoder.fit_transform(df[col])

# 7. Show cleaned dataset
print("\n🔹 Dataset after cleaning and encoding:")
print(df.head())

# 8. Save cleaned dataset for next tasks
df.to_csv("cleaned_churn_data.csv", index=False)
print("\n✅ Cleaned dataset saved as 'cleaned_churn_data.csv'")
