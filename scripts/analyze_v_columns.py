import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Use the script's location to determine paths
script_dir = Path(__file__).resolve().parent
data_path = script_dir.parent / "data"

print("Loading training data...")
train_transaction = pd.read_csv(data_path / "train_transaction.csv")

# Get all V columns
v_cols = [c for c in train_transaction.columns if c.startswith('V')]
print(f"\nTotal V columns: {len(v_cols)}")
print(f"V columns range: {min(v_cols)} to {max(v_cols)}")

# Calculate correlation matrix for V columns
print("\nCalculating correlation matrix...")
v_data = train_transaction[v_cols]

# Check missing values
print(f"\nMissing value percentages:")
missing_pct = (v_data.isnull().sum() / len(v_data) * 100).sort_values(ascending=False)
print(missing_pct.head(20))

# Calculate correlation matrix (only on available data)
corr_matrix = v_data.corr()

# Find highly correlated pairs (threshold > 0.9)
print("\n" + "="*80)
print("HIGHLY CORRELATED V COLUMN PAIRS (correlation > 0.9)")
print("="*80)

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.9:
            high_corr_pairs.append({
                'col1': corr_matrix.columns[i],
                'col2': corr_matrix.columns[j],
                'correlation': corr_val
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False, key=abs)
print(f"\nFound {len(high_corr_df)} pairs with |correlation| > 0.9")
print(high_corr_df.to_string())

# Save correlation matrix for reference
corr_output_path = data_path.parent / "v_columns_correlation.csv"
corr_matrix.to_csv(corr_output_path)
print(f"\nFull correlation matrix saved to: {corr_output_path}")

# Identify clusters of highly correlated features
print("\n" + "="*80)
print("CORRELATION ANALYSIS SUMMARY")
print("="*80)

# Find columns that have many high correlations
high_corr_counts = {}
for _, row in high_corr_df.iterrows():
    high_corr_counts[row['col1']] = high_corr_counts.get(row['col1'], 0) + 1
    high_corr_counts[row['col2']] = high_corr_counts.get(row['col2'], 0) + 1

if high_corr_counts:
    print("\nColumns with most high correlations (candidates for reduction):")
    sorted_counts = sorted(high_corr_counts.items(), key=lambda x: x[1], reverse=True)
    for col, count in sorted_counts[:20]:
        print(f"  {col}: {count} high correlations")

# Calculate variance for each V column (to identify low variance features)
print("\n" + "="*80)
print("LOW VARIANCE V COLUMNS (variance < 0.01)")
print("="*80)
variances = v_data.var().sort_values()
low_var = variances[variances < 0.01]
print(f"\nFound {len(low_var)} low variance columns:")
print(low_var.to_string())

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"""
Based on the analysis:
1. Total V columns: {len(v_cols)}
2. Highly correlated pairs (>0.9): {len(high_corr_df)}
3. Low variance columns (<0.01): {len(low_var)}

Dimensionality reduction strategies:
- Remove low variance features: {len(low_var)} columns
- Remove one column from each highly correlated pair: ~{len(high_corr_df)} columns
- OR use PCA to reduce all V columns to principal components explaining 95-99% variance
""")
