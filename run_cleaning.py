import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from src.data_loader import load_raw_data
from src.preprocessing import clean_data, prepare_features_and_target

def main():
    # loading raw data
    df = load_raw_data("data/raw/cs-training.csv")

    # clean & scale
    df_clean = clean_data(df)

    # split into X/y just to verify shapes
    X, y = prepare_features_and_target(df_clean)
    print("Cleaned data shape:", df_clean.shape)
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    # save cleaned data - no index column
    df_clean.to_csv("data/processed/cleaned_training.csv", index=False)
    print("✅ Cleaned data saved to data/processed/cleaned_training.csv")

if __name__ == "__main__":
    main()
