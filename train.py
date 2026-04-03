import pandas as pd

training_features = pd.read_csv("data/training_set_features.csv")
training_labels = pd.read_csv("data/training_set_labels.csv")

print(training_features.isna().sum())