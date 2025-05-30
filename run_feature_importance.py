#!/usr/bin/env python3
import sys, os
# allow imports from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import pandas as pd
from src.preprocessing import prepare_features_and_target
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def main():
    # 1) Load the cleaned & feature-engineered data
    df = pd.read_csv("data/processed/cleaned_training.csv")

    # 2) Split into X/y
    X, y = prepare_features_and_target(df)

    # 3) Train an XGBoost model (default settings)
    model = XGBClassifier(eval_metric="logloss")
    model.fit(X, y)

    # 4) Extract and sort feature importances
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns)
    feat_imp = feat_imp.sort_values(ascending=False)

    # 5) Plot a bar chart
    plt.figure(figsize=(10,6))
    feat_imp.plot.bar()
    plt.title("XGBoost Feature Importances")
    plt.ylabel("Importance Score")
    plt.tight_layout()

    # 6) Save and show
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/feature_importances.png")
    plt.show()

if __name__ == "__main__":
    main()
