import pandas as pd
import numpy as np
from pathlib import Path

data_path = Path("/Users/felipemediavillalevinson/Documents/Kaggle_Fraud/data")
train = pd.read_csv(data_path / "train_transaction.csv", usecols=[f'V{i}' for i in range(1, 340)])

nan_groups = {}
for col in train.columns:
    nan_sum = train[col].isnull().sum()
    if nan_sum not in nan_groups:
        nan_groups[nan_sum] = []
    nan_groups[nan_sum].append(col)

for nan_sum, cols in sorted(nan_groups.items()):
    print(f"NaN count: {nan_sum}, Columns: {cols}")
