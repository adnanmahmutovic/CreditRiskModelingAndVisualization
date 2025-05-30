import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier
import os

# loading cleaned dataset
df = pd.read_csv("data/processed/cleaned_training.csv")

# splitting into features (X) and target (y)
X = df.drop(columns=['SeriousDlqin2yrs'])
y = df['SeriousDlqin2yrs']

# training XGBoost model
model = XGBClassifier(eval_metric="logloss", verbosity=0)
model.fit(X, y)

# predicting probabilities
y_proba = model.predict_proba(X)[:, 1]

# calculating ROC curve and AUC
fpr, tpr, _ = roc_curve(y, y_proba)
roc_auc = roc_auc_score(y, y_proba)

# plotting ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)

# save to file with a clear name
os.makedirs("reports/figures", exist_ok=True)
plt.savefig("reports/figures/roc_curve_xgboost.png")
plt.close()

print("✅ ROC curve saved")
