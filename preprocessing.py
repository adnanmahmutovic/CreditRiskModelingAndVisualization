import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Fill missing values for key columns
    2) Create derived features: debt_to_income, credit_utilization, loan_to_income_ratio
    3) Scale all numerical features except the target
    """
    # 1) Fill missing values
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    df['NumberOfDependents'].fillna(0, inplace=True)

    # 2) Derive new features
    #    a) debt_to_income: use the existing DebtRatio column
    df['debt_to_income'] = df['DebtRatio']

    #    b) credit_utilization: use the existing RevolvingUtilizationOfUnsecuredLines column
    df['credit_utilization'] = df['RevolvingUtilizationOfUnsecuredLines']

    #    c) loan_to_income_ratio: number of open credit lines divided by income (avoid div by zero)
    df['loan_to_income_ratio'] = df['NumberOfOpenCreditLinesAndLoans'] / (df['MonthlyIncome'] + 1e-6)

    # 3) Scale numerical features (exclude the target column)
    scaler = MinMaxScaler()
    features_to_scale = df.drop(columns=['SeriousDlqin2yrs']).columns
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df

def prepare_features_and_target(df: pd.DataFrame):
    """
    Split the cleaned DataFrame into:
      - X: feature matrix (all columns except the target)
      - y: target vector (SeriousDlqin2yrs)
    """
    X = df.drop(columns=['SeriousDlqin2yrs'])
    y = df['SeriousDlqin2yrs']
    return X, y
