import pandas as pd
import os

# loading cleaned data
df = pd.read_csv("data/processed/cleaned_training.csv")

# renaming target column for clarity
df.rename(columns={'SeriousDlqin2yrs': 'default'}, inplace=True)

# applicant risk distribution
df['default'].to_csv('data/processed/applicant_risk_distribution.csv', index=False)

# default rates by age group
df['age_group'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 60, 70, 80], right=False)
default_by_age_group = df.groupby('age_group')['default'].mean().reset_index()
default_by_age_group.to_csv('data/processed/default_by_age_group.csv', index=False)

# default rates by number of dependents
default_by_dependents = df.groupby('NumberOfDependents')['default'].mean().reset_index()
default_by_dependents.to_csv('data/processed/default_by_dependents.csv', index=Fa