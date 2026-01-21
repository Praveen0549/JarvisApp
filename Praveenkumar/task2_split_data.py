# task2_split_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the cleaned dataset from Task 1
df = pd.read_csv("cleaned_churn_data.csv")

print("🔹 First 5 rows of cleaned dataset:")
print(df.head())

# 2. Define features (X) and target (y)
X = df.drop("Churn", axis=1)   # all columns except Churn
y = df["Churn"]                # target column

# 3. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Print dataset shapes
print("\n🔹 Shape of datasets:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# 5. Save split datasets for next tasks
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n✅ Data successfully split and saved as CSV files.")
