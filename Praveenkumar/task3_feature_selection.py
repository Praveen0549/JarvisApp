# task3_feature_selection.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Load the cleaned dataset
df = pd.read_csv("cleaned_churn_data.csv")

# 2. Define features (X) and target (y)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 3. Use RandomForest to find feature importance
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 4. Get importance scores
importances = model.feature_importances_

# 5. Create a DataFrame for readability
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# 6. Print top features
print("🔹 Feature Importance Ranking:")
print(feature_importance)

# 7. Save feature importance
feature_importance.to_csv("feature_importance.csv", index=False)
print("\n✅ Feature importance saved as 'feature_importance.csv'")
