from src.data_loader import load_raw_data
from src.preprocessing import clean_data, prepare_features_and_target

# Load data
df = load_raw_data("data/raw/cs-training.csv")

# Clean data
df_clean = clean_data(df)

# Separate features and target
X, y = prepare_features_and_target(df_clean)

# Print result shapes
print("Cleaned data shape:", df_clean.shape)
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Optional: Save cleaned data
df_clean.to_csv("data/processed/cleaned_training.csv", index=False)
print("✅ Cleaned data saved to data/processed/cleaned_training.csv")
