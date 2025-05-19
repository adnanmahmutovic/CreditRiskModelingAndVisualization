import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # filling nulls
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace = True)
    df['NumberOfDependents'].fillna(0, inplace = True)

    # scaling numerical features
    scaler = MinMaxScaler()
    features_to_scale = df.drop(columns = ['SeriousDlqin2yrs']).columns
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    return df

def prepare_features_and_target(df: pd.DataFrame):
    # this function separates features and target variable
    X = df.drop(columns = ['SeriousDlqin2yrs'])
    y = df['SeriousDlqin2yrs']

    return X, y

