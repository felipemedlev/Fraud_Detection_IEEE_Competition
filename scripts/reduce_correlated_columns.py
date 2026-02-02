"""
Reduce correlated columns based on NaN structure and correlation analysis.

This module provides functionality to:
1. Group columns by their NaN patterns
2. Find highly correlated subsets within each group
3. Keep only one representative from each correlated subset
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def get_nan_pattern(df: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
    """
    Calculate the number of NaN values for each column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns
    columns : List[str]
        List of column names to analyze

    Returns
    -------
    Dict[str, int]
        Dictionary mapping column name to NaN count
    """
    nan_counts = {}
    for col in columns:
        if col in df.columns:
            nan_counts[col] = df[col].isna().sum()
    return nan_counts


def group_by_nan_structure(df: pd.DataFrame, columns: List[str], tolerance: int = 10) -> Dict[int, List[str]]:
    """
    Group columns by similar NaN counts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns
    columns : List[str]
        List of column names to group
    tolerance : int, default=10
        Tolerance for grouping (columns with NaN counts within this range are grouped together)

    Returns
    -------
    Dict[int, List[str]]
        Dictionary mapping NaN count to list of columns with that count
    """
    nan_counts = get_nan_pattern(df, columns)

    # Group by exact NaN count (or within tolerance)
    groups = {}
    for col, nan_count in nan_counts.items():
        # Round to nearest tolerance to create groups
        group_key = round(nan_count / tolerance) * tolerance
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(col)

    return groups


def find_correlated_subsets(df: pd.DataFrame, columns: List[str], threshold: float = 0.75) -> List[List[str]]:
    """
    Find subsets of highly correlated columns.

    Uses a greedy algorithm to group columns:
    - Start with first column as a new subset
    - For each subsequent column, check if it's highly correlated with any existing subset
    - If yes, add to that subset; if no, create a new subset

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns
    columns : List[str]
        List of column names to analyze
    threshold : float, default=0.75
        Correlation threshold (columns with |correlation| > threshold are grouped)

    Returns
    -------
    List[List[str]]
        List of subsets, where each subset contains highly correlated columns
    """
    if len(columns) == 0:
        return []

    # Calculate correlation matrix for these columns
    valid_cols = [c for c in columns if c in df.columns]
    if len(valid_cols) == 0:
        return []

    # Only use numeric columns and drop rows with all NaN
    df_subset = df[valid_cols].select_dtypes(include=[np.number])

    # If no valid numeric columns, return empty
    if df_subset.shape[1] == 0:
        return []

    # Calculate correlation matrix
    corr_matrix = df_subset.corr().abs()

    # Greedy grouping algorithm
    subsets = []
    assigned = set()

    for col in valid_cols:
        if col in assigned:
            continue

        # Start a new subset with this column
        current_subset = [col]
        assigned.add(col)

        # Find all columns highly correlated with this one
        for other_col in valid_cols:
            if other_col in assigned:
                continue

            # Check correlation with all members of current subset
            if col in corr_matrix.index and other_col in corr_matrix.columns:
                if corr_matrix.loc[col, other_col] > threshold:
                    current_subset.append(other_col)
                    assigned.add(other_col)

        subsets.append(current_subset)

    return subsets


def reduce_columns_by_correlation(
    df: pd.DataFrame,
    column_prefix: str,
    correlation_threshold: float = 0.75,
    nan_tolerance: int = 10,
    verbose: bool = True
) -> Tuple[List[str], Dict]:
    """
    Reduce columns with a given prefix by grouping by NaN structure and correlation.

    Algorithm:
    1. Find all columns starting with the given prefix
    2. Group them by similar NaN counts
    3. Within each NaN group, find correlated subsets
    4. Keep only the first column from each correlated subset

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns
    column_prefix : str
        Prefix of columns to analyze (e.g., 'V', 'M', 'C', 'D')
    correlation_threshold : float, default=0.75
        Correlation threshold for grouping
    nan_tolerance : int, default=10
        Tolerance for NaN count grouping
    verbose : bool, default=True
        Whether to print detailed information

    Returns
    -------
    Tuple[List[str], Dict]
        - List of column names to keep
        - Dictionary with analysis details for each NaN group
    """
    # Find all columns with this prefix
    all_cols = [c for c in df.columns if c.startswith(column_prefix)]

    if len(all_cols) == 0:
        if verbose:
            print(f"No columns found with prefix '{column_prefix}'")
        return [], {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"Analyzing {column_prefix} columns: {len(all_cols)} total")
        print(f"{'='*60}")

    # Group by NaN structure
    nan_groups = group_by_nan_structure(df, all_cols, tolerance=nan_tolerance)

    columns_to_keep = []
    analysis_details = {}

    # Process each NaN group
    for nan_count, cols_in_group in sorted(nan_groups.items()):
        if verbose:
            print(f"\nNaN Group (≈{nan_count} NaNs): {len(cols_in_group)} columns")
            print(f"  Columns: {cols_in_group}")

        # Find correlated subsets within this group
        correlated_subsets = find_correlated_subsets(df, cols_in_group, threshold=correlation_threshold)

        # Keep first column from each subset
        kept_from_group = []
        for subset in correlated_subsets:
            if len(subset) > 0:
                kept_col = subset[0]
                kept_from_group.append(kept_col)

                if verbose and len(subset) > 1:
                    print(f"  Correlated subset: {subset}")
                    print(f"    → Keeping: {kept_col}")

        columns_to_keep.extend(kept_from_group)

        analysis_details[nan_count] = {
            'original_columns': cols_in_group,
            'correlated_subsets': correlated_subsets,
            'kept_columns': kept_from_group,
            'reduction': f"{len(cols_in_group)} → {len(kept_from_group)}"
        }

        if verbose:
            print(f"  Reduction: {len(cols_in_group)} → {len(kept_from_group)} columns")

    if verbose:
        print(f"Total {column_prefix} columns: {len(all_cols)} → {len(columns_to_keep)}")
        print(f"Reduction: {len(all_cols) - len(columns_to_keep)} columns removed")

    return columns_to_keep, analysis_details


def analyze_all_column_groups(
    df: pd.DataFrame,
    prefixes: List[str] = ['V', 'M', 'C', 'D'],
    correlation_threshold: float = 0.75,
    nan_tolerance: int = 10,
    verbose: bool = True
) -> Dict[str, Tuple[List[str], Dict]]:
    """
    Analyze and reduce all column groups (V, M, C, D).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns
    prefixes : List[str], default=['V', 'M', 'C', 'D']
        List of column prefixes to analyze
    correlation_threshold : float, default=0.75
        Correlation threshold for grouping
    nan_tolerance : int, default=10
        Tolerance for NaN count grouping
    verbose : bool, default=True
        Whether to print detailed information

    Returns
    -------
    Dict[str, Tuple[List[str], Dict]]
        Dictionary mapping prefix to (columns_to_keep, analysis_details)
    """
    results = {}

    for prefix in prefixes:
        cols_to_keep, details = reduce_columns_by_correlation(
            df, prefix, correlation_threshold, nan_tolerance, verbose
        )
        results[prefix] = (cols_to_keep, details)

    return results


def apply_column_reduction(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns_to_keep: Dict[str, List[str]],
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply column reduction to both train and test dataframes.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    columns_to_keep : Dict[str, List[str]]
        Dictionary mapping prefix to list of columns to keep
    verbose : bool, default=True
        Whether to print information

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Reduced train and test dataframes
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Applying column reduction")
        print(f"{'='*60}")

    for prefix, cols in columns_to_keep.items():
        # Find all columns with this prefix
        all_cols = [c for c in train_df.columns if c.startswith(prefix)]
        cols_to_drop = [c for c in all_cols if c not in cols]

        if len(cols_to_drop) > 0:
            train_df = train_df.drop(cols_to_drop, axis=1)
            test_df = test_df.drop([c for c in cols_to_drop if c in test_df.columns], axis=1)

            if verbose:
                print(f"{prefix} columns: {len(all_cols)} → {len(cols)} (dropped {len(cols_to_drop)})")

    if verbose:
        print(f"\nFinal shapes:")
        print(f"  Train: {train_df.shape}")
        print(f"  Test: {test_df.shape}")
        print(f"{'='*60}\n")

    return train_df, test_df


# Example usage
if __name__ == "__main__":
    # This is an example of how to use the functions
    print("This module provides functions for reducing correlated columns.")
    print("\nExample usage:")
    print("""
    from reduce_correlated_columns import analyze_all_column_groups, apply_column_reduction

    # Analyze all column groups
    results = analyze_all_column_groups(
        train_df,
        prefixes=['V', 'M', 'C', 'D'],
        correlation_threshold=0.75,
        nan_tolerance=10
    )

    # Extract columns to keep
    columns_to_keep = {prefix: cols for prefix, (cols, _) in results.items()}

    # Apply reduction
    train_reduced, test_reduced = apply_column_reduction(
        train_df, test_df, columns_to_keep
    )
    """)
