import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # fill missing values
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    df['NumberOfDependents'].fillna(0, inplace=True)

    # scale numerical features (leave target alone)
    scaler = MinMaxScaler()
    features_to_scale = df.drop(columns=['SeriousDlqin2yrs']).columns
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df

def prepare_features_and_target(df: pd.DataFrame):
    X = df.drop(columns=['SeriousDlqin2yrs'])
    y = df['SeriousDlqin2yrs']
    return X, y
