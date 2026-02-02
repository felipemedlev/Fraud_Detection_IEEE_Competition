"""
Example: How to use the column reduction in your fraud detection pipeline.

This script demonstrates integrating the correlation-based column reduction
into your existing main_model.py workflow.
"""

import pandas as pd
from reduce_correlated_columns import (
    reduce_columns_by_correlation,
    analyze_all_column_groups,
    apply_column_reduction
)


def example_1_analyze_single_prefix():
    """
    Example 1: Analyze just V columns
    """
    print("="*60)
    print("EXAMPLE 1: Analyze V columns only")
    print("="*60)

    # Load your data (replace with actual loading)
    # train = pd.read_csv("data/train_transaction.csv")

    # Analyze V columns
    v_cols_to_keep, v_analysis = reduce_columns_by_correlation(
        train,
        column_prefix='V',
        correlation_threshold=0.75,
        nan_tolerance=100,
        verbose=True
    )

    print(f"\nRecommended V columns: {len(v_cols_to_keep)}")
    print(f"Column indices: {sorted([int(c[1:]) for c in v_cols_to_keep])}")

    return v_cols_to_keep


def example_2_analyze_all_columns():
    """
    Example 2: Analyze all V, M, C, D columns
    """
    print("="*60)
    print("EXAMPLE 2: Analyze all column groups")
    print("="*60)

    # Load your data
    # train = pd.read_csv("data/train_transaction.csv")

    # Analyze all column groups
    results = analyze_all_column_groups(
        train,
        prefixes=['V', 'M', 'C', 'D'],
        correlation_threshold=0.75,
        nan_tolerance=100,
        verbose=True
    )

    # Extract just the column names (not the analysis details)
    columns_to_keep = {}
    for prefix, (cols, details) in results.items():
        columns_to_keep[prefix] = cols
        print(f"\n{prefix}: {len(cols)} columns to keep")

    return columns_to_keep


def example_3_apply_reduction():
    """
    Example 3: Apply reduction to train and test dataframes
    """
    print("="*60)
    print("EXAMPLE 3: Apply column reduction")
    print("="*60)

    # Load your data
    # train = pd.read_csv("data/train_transaction.csv")
    # test = pd.read_csv("data/test_transaction.csv")

    # First, analyze to get columns to keep
    results = analyze_all_column_groups(
        train,
        prefixes=['V', 'M', 'C', 'D'],
        correlation_threshold=0.75,
        nan_tolerance=100,
        verbose=False  # Set to False for cleaner output
    )

    # Extract columns to keep
    columns_to_keep = {prefix: cols for prefix, (cols, _) in results.items()}

    # Apply reduction to both train and test
    train_reduced, test_reduced = apply_column_reduction(
        train, test, columns_to_keep, verbose=True
    )

    print(f"\nOriginal train shape: {train.shape}")
    print(f"Reduced train shape: {train_reduced.shape}")
    print(f"Columns removed: {train.shape[1] - train_reduced.shape[1]}")

    return train_reduced, test_reduced


def example_4_custom_threshold():
    """
    Example 4: Use different correlation threshold
    """
    print("="*60)
    print("EXAMPLE 4: Custom correlation threshold")
    print("="*60)

    # Load your data
    # train = pd.read_csv("data/train_transaction.csv")

    # Try different thresholds
    for threshold in [0.70, 0.75, 0.80, 0.85]:
        v_cols, _ = reduce_columns_by_correlation(
            train,
            column_prefix='V',
            correlation_threshold=threshold,
            nan_tolerance=100,
            verbose=False
        )
        print(f"Threshold {threshold}: {len(v_cols)} V columns kept")


def example_5_integrate_into_pipeline():
    """
    Example 5: Full integration into your modeling pipeline

    This shows how to add it to main_model.py
    """
    print("="*60)
    print("EXAMPLE 5: Integration into main_model.py")
    print("="*60)

    print("""
    # In main_model.py, add this after loading data:

    from reduce_correlated_columns import analyze_all_column_groups, apply_column_reduction

    def main():
        # 1. Load data
        train, test = load_data(DATA_PATH)

        # 2. Reduce memory usage
        train = reduce_mem_usage(train)
        test = reduce_mem_usage(test)

        # 3. NEW: Analyze and reduce correlated columns
        print("\\nAnalyzing column correlations...")
        results = analyze_all_column_groups(
            train,
            prefixes=['V', 'C'],  # Only V and C, keep M and D as-is
            correlation_threshold=0.75,
            nan_tolerance=100,
            verbose=True
        )

        # Extract columns to keep
        columns_to_keep = {prefix: cols for prefix, (cols, _) in results.items()}

        # Apply reduction
        train, test = apply_column_reduction(train, test, columns_to_keep)

        # 4. Continue with feature engineering
        train, test = engineer_features(train, test)

        # ... rest of your pipeline
    """)


def example_6_export_to_config():
    """
    Example 6: Export results to update config.py
    """
    print("="*60)
    print("EXAMPLE 6: Export to config format")
    print("="*60)

    # Load and analyze
    # train = pd.read_csv("data/train_transaction.csv")

    results = analyze_all_column_groups(
        train,
        prefixes=['V', 'M', 'C', 'D'],
        correlation_threshold=0.75,
        nan_tolerance=100,
        verbose=False
    )

    # Generate config.py format
    print("\n# Add these to your config.py:\n")
    for prefix in ['V', 'M', 'C', 'D']:
        if prefix in results:
            cols, _ = results[prefix]
            indices = sorted([int(c[1:]) for c in cols if c[1:].isdigit()])
            print(f"{prefix}_COLS_TO_KEEP = {indices}\n")


# Main execution
if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  Column Reduction Examples                                 ║
    ║  Demonstrates how to use the correlation-based reduction   ║
    ╚════════════════════════════════════════════════════════════╝

    This file contains 6 examples showing different use cases.

    To run a specific example, uncomment it below.
    Note: You'll need to load your actual data first.
    """)

    # Uncomment the example you want to run:

    # example_1_analyze_single_prefix()
    # example_2_analyze_all_columns()
    # example_3_apply_reduction()
    # example_4_custom_threshold()
    # example_5_integrate_into_pipeline()
    # example_6_export_to_config()
