import sys, os
# ensure that the parent directory of this script is on Python's import search path
# we can import our custom modules from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import pandas as pd  # pandas for data loading and manipulation
from sklearn.model_selection import train_test_split  # to split data into training and test sets
from sklearn.linear_model import LogisticRegression  # our first model choice
from sklearn.metrics import (  # performance metrics
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt  # for plotting the ROC curve
from src.preprocessing import prepare_features_and_target  # helper to separate features (X) from target (y)

def main():
    # load the cleaned dataset from CSV into a DataFrame
    df = pd.read_csv("data/processed/cleaned_training.csv")

    # split the DataFrame into feature matrix X and target vector y
    X, y = prepare_features_and_target(df)

    # further split X and y into training and test sets
    #    - test_size=0.2 reserves 20% of data for evaluation
    #    - random_state ensures reproducibility
    #    - stratify=y keeps the same class balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # initialize and train a Logistic Regression model
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    # use the trained model to make predictions on the test set
    #    - y_pred gives the hard class labels (0 or 1)
    #    - y_proba gives the probability of the positive class (default)
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]

    # print key classification metrics
    print("=== Logistic Regression Results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # compute ROC curve data: false positive rate and true positive rate
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    # calculate the Area Under the Curve (AUC) metric
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.3f}")

    # plot the ROC curve to visualize model discrimination
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"LogReg (AUC = {roc_auc:.3f})")
    plt.plot([0,1], [0,1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()

    # save instead of showing
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/roc_curve_model_eval.png")

    # train an XGBoost classifier for comparison
    from xgboost import XGBClassifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X_train, y_train)

    # get XGBoost predicted probabilities on the test set
    y_proba2 = xgb.predict_proba(X_test)[:, 1]
    fpr2, tpr2, _ = roc_curve(y_test, y_proba2)
    # print the AUC for the XGBoost model
    print(f"XGBoos