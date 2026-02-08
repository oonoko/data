#!/usr/bin/env python3
"""
Сар тутмын dataset дээр AI model сургах
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json

# 1. Dataset уншиx
print("Loading dataset...")
df = pd.read_csv('final_dzud_ai_dataset_soum_monthly.csv')
print(f"Dataset: {len(df)} rows")
print(f"\nColumns: {df.columns.tolist()}")

# 2. Features and target
feature_cols = [
    'avg_temp', 'min_temp', 'wind_speed', 
    'snowfall_sum', 'precip_sum', 'livestock_count'
]

X = df[feature_cols].copy()
y = df['risk_level'].copy()

# Handle missing values
X = X.fillna(X.mean())

print(f"\nFeatures: {feature_cols}")
print(f"Target distribution:")
print(y.value_counts().sort_index())

# 3. Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# 4. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train models
print("\n" + "="*60)
print("Training models...")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n{name}:")
    
    # Train
    if 'Logistic' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.3f}")
    
    # Cross-validation
    if 'Logistic' in name:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }

# 6. Select best model (Random Forest)
print("\n" + "="*60)
print("Best Model: Random Forest")
print("="*60)

best_model = models['Random Forest']
y_pred_best = results['Random Forest']['predictions']

# 7. Detailed evaluation
print("\nClassification Report:")
print(classification_report(
    y_test, 
    y_pred_best,
    target_names=['Бага', 'Дунд', 'Өндөр', 'Маш өндөр']
))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# 8. Feature importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.to_string(index=False))

# 9. Save model
model_file = 'dzud_risk_model_monthly.pkl'
joblib.dump(best_model, model_file)
print(f"\n✅ Model saved: {model_file}")

# 10. Save scaler
scaler_file = 'scaler_monthly.pkl'
joblib.dump(scaler, scaler_file)
print(f"✅ Scaler saved: {scaler_file}")

# 11. Save metadata
metadata = {
    'model_type': 'RandomForestClassifier',
    'features': feature_cols,
    'label_map': {
        0: 'Бага',
        1: 'Дунд',
        2: 'Өндөр',
        3: 'Маш өндөр'
    },
    'accuracy': float(results['Random Forest']['accuracy']),
    'cv_score': float(results['Random Forest']['cv_mean']),
    'feature_importance': feature_importance.to_dict('records'),
    'note': 'Experimental model - livestock data is synthetic/proxy'
}

metadata_file = 'model_metadata_monthly.json'
with open(metadata_file, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"✅ Metadata saved: {metadata_file}")

# 12. Test prediction example
print("\n" + "="*60)
print("Example Prediction:")
print("="*60)

sample = X_test.iloc[0:1]
prediction = best_model.predict(sample)
proba = best_model.predict_proba(sample)

print(f"\nInput features:")
print(sample.to_string())
print(f"\nPredicted risk level: {prediction[0]} ({metadata['label_map'][prediction[0]]})")
print(f"Probabilities: {proba[0]}")

print("\n" + "="*60)
print("Training complete!")
print("="*60)
