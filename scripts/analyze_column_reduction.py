"""
Standalone script to analyze and reduce correlated columns in the fraud detection dataset.

This script demonstrates the column reduction algorithm:
1. Groups columns (V, M, C, D) by their NaN structure
2. Finds correlated subsets within each group
3. Keeps only one representative from each highly correlated subset

Usage:
    python analyze_column_reduction.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from reduce_correlated_columns import (
    analyze_all_column_groups,
    apply_column_reduction,
    reduce_columns_by_correlation
)
from config import DATA_PATH


def load_sample_data(data_path, sample_size=50000):
    """
    Load a sample of the training data for analysis.

    Parameters
    ----------
    data_path : Path
        Path to data directory
    sample_size : int, default=50000
        Number of rows to sample for faster analysis

    Returns
    -------
    pd.DataFrame
        Sample of training data
    """
    print("Loading training data...")
    try:
        train_transaction = pd.read_csv(data_path / "train_transaction.csv")
        train_identity = pd.read_csv(data_path / "train_identity.csv")

        # Merge
        train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

        # Sample for faster analysis
        if sample_size and len(train) > sample_size:
            print(f"Sampling {sample_size} rows for analysis...")
            train = train.sample(n=sample_size, random_state=42)

        print(f"Loaded data shape: {train.shape}")
        return train

    except FileNotFoundError as e:
        print(f"Error: Required data file not found - {e}")
        print(f"Expected files in: {data_path}")
        sys.exit(1)


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("COLUMN REDUCTION ANALYSIS")
    print("="*60 + "\n")

    # Load sample data
    train = load_sample_data(DATA_PATH, sample_size=50000)

    # Analyze V columns specifically (as mentioned in the user's request)
    print("\n" + "="*60)
    print("DETAILED V COLUMN ANALYSIS")
    print("="*60)

    v_cols_to_keep, v_analysis = reduce_columns_by_correlation(
        train,
        column_prefix='V',
        correlation_threshold=0.75,
        nan_tolerance=100,  # Larger tolerance for V columns
        verbose=True
    )

    # Print detailed analysis for V columns
    print("\n" + "="*60)
    print("V COLUMN REDUCTION SUMMARY")
    print("="*60)
    print(f"\nOriginal V columns: {len([c for c in train.columns if c.startswith('V')])}")
    print(f"Reduced V columns: {len(v_cols_to_keep)}")
    print(f"\nColumns to keep: {sorted([int(c[1:]) for c in v_cols_to_keep])}")

    # Show example of one NaN group in detail
    if v_analysis:
        print("\n" + "="*60)
        print("EXAMPLE: First NaN Group Details")
        print("="*60)
        first_group_key = list(v_analysis.keys())[0]
        first_group = v_analysis[first_group_key]
        print(f"\nNaN count: ~{first_group_key}")
        print(f"Original columns: {first_group['original_columns']}")
        print(f"\nCorrelated subsets found:")
        for i, subset in enumerate(first_group['correlated_subsets'], 1):
            print(f"  Subset {i}: {subset}")
            if len(subset) > 1:
                print(f"    → Keeping: {subset[0]}")
        print(f"\nReduction: {first_group['reduction']}")

    # Analyze all column groups
    print("\n\n" + "="*60)
    print("ANALYZING ALL COLUMN GROUPS (V, M, C, D)")
    print("="*60)

    results = analyze_all_column_groups(
        train,
        prefixes=['V', 'M', 'C', 'D'],
        correlation_threshold=0.75,
        nan_tolerance=100,
        verbose=True
    )

    # Summary table
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\n{'Prefix':<10} {'Original':<12} {'Reduced':<12} {'Removed':<12} {'% Reduction':<12}")
    print("-" * 60)

    for prefix in ['V', 'M', 'C', 'D']:
        if prefix in results:
            cols_to_keep, _ = results[prefix]
            original_count = len([c for c in train.columns if c.startswith(prefix)])
            reduced_count = len(cols_to_keep)
            removed_count = original_count - reduced_count
            pct_reduction = (removed_count / original_count * 100) if original_count > 0 else 0

            print(f"{prefix:<10} {original_count:<12} {reduced_count:<12} {removed_count:<12} {pct_reduction:<12.1f}%")

    # Export results
    print("\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)

    # Create a summary file
    output_file = DATA_PATH.parent / "column_reduction_summary.txt"
    with open(output_file, 'w') as f:
        f.write("COLUMN REDUCTION ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")

        for prefix in ['V', 'M', 'C', 'D']:
            if prefix in results:
                cols_to_keep, analysis = results[prefix]
                f.write(f"\n{prefix} COLUMNS\n")
                f.write("-" * 60 + "\n")
                f.write(f"Original count: {len([c for c in train.columns if c.startswith(prefix)])}\n")
                f.write(f"Reduced count: {len(cols_to_keep)}\n")
                f.write(f"\nColumns to keep:\n")

                # Convert to numeric indices for V, M, C, D columns
                indices = sorted([int(c[1:]) for c in cols_to_keep if c[1:].isdigit()])
                f.write(f"{indices}\n")

                f.write(f"\nAs Python list:\n")
                f.write(f"{prefix}_COLS_TO_KEEP = {indices}\n")

    print(f"Summary saved to: {output_file}")

    # Also create a Python config snippet
    config_snippet_file = DATA_PATH.parent / "reduced_columns_config.py"
    with open(config_snippet_file, 'w') as f:
        f.write("# Reduced column indices based on NaN structure and correlation analysis\n")
        f.write("# Correlation threshold: 0.75\n\n")

        for prefix in ['V', 'M', 'C', 'D']:
            if prefix in results:
                cols_to_keep, _ = results[prefix]
                indices = sorted([int(c[1:]) for c in cols_to_keep if c[1:].isdigit()])
                f.write(f"{prefix}_COLS_TO_KEEP = {indices}\n\n")

    print(f"Config snippet saved to: {config_snippet_file}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
