import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Dataset ачаалах
data = pd.read_csv("dzud_ai_dataset.csv")

X = data.drop("dzud_risk", axis=1)
y = data["dzud_risk"]

# Train / Test хуваах
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# AI сургах (ГОЛ МӨЧ)
model.fit(X_train, y_train)

# Үр дүн шалгах
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("🎯 Accuracy:", round(acc, 3))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
print("🧠 Feature Importance:")
for name, importance in zip(X.columns, model.feature_importances_):
    print(f"{name}: {round(importance, 3)}")

# Model хадгалах
out_model = "dzud_risk_model.pkl"
joblib.dump(model, out_model)

print(f"\n✅ Model saved: {out_model}")
print("📍 Folder:", os.getcwd())

