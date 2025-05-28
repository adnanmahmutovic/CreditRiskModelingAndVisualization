import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, index_col=0)
        print(f"Data loaded successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found. Please check path: {path}")
        raise
