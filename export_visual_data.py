# scripts/export_visual_data.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# loading processed training data
df = pd.read_csv("data/processed/cleaned_training.csv")

# splitting features and target
X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]

# scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# loading trained model
model = XGBClassifier()
model.fit(X_scaled, y)

# exporting applicant risk distribution
risk_scores = model.predict_proba(X_scaled)[:, 1]
risk_df = pd.DataFrame({
    "ApplicantID": df.index,
    "PredictedRisk": risk_scores,
    "ActualDefault": y
})
risk_df.to_csv("data/processed/applicant_risk_distribution.csv", index=False)

# exporting feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)
feature_importance.to_csv("data/processed/feature_importance.csv", index=False)

# plotting and saving ROC curve
fpr, tpr, thresholds = roc_curve(y, risk_scores)
roc_auc = roc_auc_score(y, risk_scores)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("reports/roc_curve.png")
plt.close()
