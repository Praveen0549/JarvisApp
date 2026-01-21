# task4_model_selection.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 1. Load the cleaned dataset
df = pd.read_csv("cleaned_churn_data.csv")

# 2. Define features (X) and target (y)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 3. Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Define models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# 5. Train & evaluate each model
print("🔹 Model Selection Results:\n")
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name}: Accuracy = {acc:.4f}")

# 6. Show best model
best_model = max(results, key=results.get)
print(f"\n✅ Best Model: {best_model} with Accuracy = {results[best_model]:.4f}")
