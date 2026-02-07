import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA = "dzud_ai_dataset_yearly.csv"
OUT_MODEL = "dzud_risk_model_yearly.pkl"

df = pd.read_csv(DATA)

def to_risk(x: float) -> int:
    if x < 1.0:
        return 0
    if x < 2.0:
        return 1
    if x < 3.5:
        return 2
    return 3

df["risk"] = df["loss_value"].apply(to_risk)

feature_cols = [
    "avg_temp_mean",
    "min_temp_min",
    "wind_speed_mean",
    "snowfall_sum",
    "precip_sum",
    "livestock_region_value",
]

X = df[feature_cols]
y = df["risk"]

pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

loo = LeaveOneOut()
pred = cross_val_predict(pipe, X, y, cv=loo)

print("=== Confusion Matrix ===")
print(confusion_matrix(y, pred))

print("\n=== Classification Report ===")
print(classification_report(y, pred, digits=3))

pipe.fit(X, y)

joblib.dump(
    {
        "model": pipe,
        "feature_cols": feature_cols,
        "label_map": {0: "Бага", 1: "Дунд", 2: "Өндөр", 3: "Маш өндөр"},
    },
    OUT_MODEL
)

print(f"\n✅ Saved model: {OUT_MODEL}")

