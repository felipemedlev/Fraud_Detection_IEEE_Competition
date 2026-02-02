import sys
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# Import configuration
from config import (
    DATA_PATH, V_COLS_TO_KEEP, TRAIN_VAL_SPLIT,
    LGBM_PARAMS, XGB_PARAMS, CATBOOST_PARAMS, EARLY_STOPPING_ROUNDS, LOG_EVAL_PERIOD,
    TRANSACTION_CATEGORICAL, IDENTITY_CATEGORICAL, COLS_TO_DROP_FOR_MODELING
)

# Pandas display options
pd.set_option('display.max_columns', 500)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def reduce_mem_usage(df, verbose=True):
    """
    Reduce memory usage by downcasting numeric columns to smallest suitable dtype.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage decreased to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')

    return df


def reduce_v_columns(train_df, test_df, v_cols_to_keep):
    """
    Keep only specified V columns or reduce them by correlation.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    v_cols_to_keep : list of int
        Specific V-column indices to keep. If None, uses correlation reduction.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Training and test dataframes with selected V columns
    """
    v_cols = [c for c in train_df.columns if c.startswith('V')]
    print(f"Found {len(v_cols)} V columns initially")

    # Convert numeric indices to column names (e.g., 1 -> 'V1')
    keep_names = [f'V{i}' for i in v_cols_to_keep]
    # Only keep columns that actually exist in the dataframe
    keep_names = [c for c in keep_names if c in v_cols]
    print(f"Keeping {len(keep_names)} specified V columns")

    v_cols_to_drop = [c for c in v_cols if c not in keep_names]

    train_df = train_df.drop(v_cols_to_drop, axis=1)
    test_df = test_df.drop(v_cols_to_drop, axis=1)

    return train_df, test_df


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def create_normalized_date_features(df):
    """
    Create normalized date features D1n, D4n, D10n, D15n.

    These represent specific calendar dates rather than time differences,
    which helps detect temporal patterns in fraud.

    D1 = days since client's first transaction (client age)
    D1n = current_day - D1 = actual calendar date of client's first transaction

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with TransactionDT and D columns

    Returns
    -------
    pd.DataFrame
        DataFrame with new D#n columns added
    """
    # Calculate current transaction day (day index)
    day_index = np.floor(df['TransactionDT'] / (24 * 60 * 60))

    # Create normalized dates (subtract time difference to get calendar date)
    df['D1n'] = day_index - df['D1']

    # Optional D columns (may not exist in all datasets)
    for d_col in ['D4', 'D10', 'D15']:
        if d_col in df.columns:
            df[f'{d_col}n'] = day_index - df[d_col]

    return df


def create_uid(df):
    """
    Create unique user identifier (UID) from card1, addr1, and D1n.

    This is a critical feature for fraud detection:
    - card1: Primary card identifier
    - addr1: Billing address
    - D1n: Client start date (calendar date)

    Same UID = same cardholder. Multiple transactions per UID is normal.
    BUT: if UID has inconsistent patterns (varying amounts, dates, etc.),
    it may indicate the card was stolen or shared.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with card1, addr1, and D1n columns

    Returns
    -------
    pd.Series
        UID strings for each transaction

    """

    # Combine card1, addr1, D1n into single identifier
    uid = (df['card1'].fillna(-999).astype(str) + '_' +
           df['addr1'].fillna(-999).astype(str) + '_' +
           df['D1n'].fillna(-999).astype(str))

    return uid


def encode_aggregations(df, agg_cols, group_cols, agg_funcs, prefix=''):
    """
    Create aggregated features by grouping data.

    This creates features like "mean transaction amount per UID" or
    "std of D4n per UID" which help detect fraud patterns:
    - std=0 for date features → single consistent client
    - std>0 for date features → possible card sharing/fraud
    - High std in amounts → inconsistent spending behavior

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to create features on
    agg_cols : list of str
        Column names to aggregate
    group_cols : list of str
        Column names to group by
    agg_funcs : list of str
        Aggregation functions ('mean', 'std', 'nunique', etc.)
    prefix : str, default=''
        Optional prefix for feature names

    Returns
    -------
    pd.DataFrame
        DataFrame with new aggregated features added
    """
    for col in agg_cols:
        # Skip if column doesn't exist
        if col not in df.columns:
            print(f"Warning: Column {col} not found, skipping aggregation...")
            continue

        for func in agg_funcs:
            # Compute aggregation
            grouped = df.groupby(group_cols)[col].agg(func)

            # Create descriptive feature name
            group_str = '_'.join(group_cols)
            if prefix:
                feature_name = f'{prefix}_{group_str}_{col}_{func}'
            else:
                feature_name = f'{group_str}_{col}_{func}'

            # Map aggregation back to original dataframe
            if len(group_cols) == 1:
                df[feature_name] = df[group_cols[0]].map(grouped)
            else:
                df[feature_name] = df.set_index(group_cols).index.map(grouped)

    return df


def create_time_features(df):
    """
    Create hour and day of week features from TransactionDT.

    Fraud patterns often vary by time of day and day of week.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with TransactionDT column

    Returns
    -------
    pd.DataFrame
        DataFrame with 'hour' and 'day' features added
    """
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['day'] = (df['TransactionDT'] // (3600 * 24)) % 7
    return df


def create_cents_feature(df):
    """
    Create 'cents' feature - the decimal portion of TransactionAmt.

    Fraud transactions may have distinctive patterns in decimal amounts.
    For example, round numbers might be more common in certain fraud types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with TransactionAmt column

    Returns
    -------
    pd.DataFrame
        DataFrame with 'cents' feature added
    """
    df['cents'] = (df['TransactionAmt'] - np.floor(df['TransactionAmt'])).astype('float32')
    return df


def encode_FE(df1, df2, cols):
    """
    Frequency encoding - encode features by their frequency in combined train+test.

    This captures how common/rare a value is, which is informative for fraud:
    - Very rare card numbers might be more suspicious
    - Very common addresses might be less suspicious

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe (typically train)
    df2 : pd.DataFrame
        Second dataframe (typically test)
    cols : list of str
        Column names to frequency encode

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Both dataframes with _FE features added
    """
    for col in cols:
        if col not in df1.columns or col not in df2.columns:
            print(f"Warning: {col} not found in both dataframes, skipping...")
            continue

        # Combine train and test to get overall frequency
        df_combined = pd.concat([df1[col], df2[col]])
        vc = df_combined.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1  # Handle missing values

        feature_name = col + '_FE'
        df1[feature_name] = df1[col].map(vc).astype('float32')
        df2[feature_name] = df2[col].map(vc).astype('float32')

        # Fill NaN with -1 for missing values
        df1[feature_name].fillna(-1, inplace=True)
        df2[feature_name].fillna(-1, inplace=True)

        print(f"  Created: {feature_name}")

    return df1, df2


def encode_CB(df1, df2, col1, col2):
    """
    Combine two features into a single feature, then label encode it.

    Example: card1='1234', addr1='567' -> card1_addr1='1234_567'

    This creates interaction features that capture joint patterns.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe (typically train)
    df2 : pd.DataFrame
        Second dataframe (typically test)
    col1 : str
        First column name
    col2 : str
        Second column name

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Both dataframes with combined feature added
    """
    if col1 not in df1.columns or col2 not in df1.columns:
        print(f"Warning: {col1} or {col2} not found, skipping combination...")
        return df1, df2

    feature_name = f'{col1}_{col2}'

    # Create combined string feature
    df1[feature_name] = df1[col1].astype(str) + '_' + df1[col2].astype(str)
    df2[feature_name] = df2[col1].astype(str) + '_' + df2[col2].astype(str)

    # Label encode the combined feature
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    combined_values = list(df1[feature_name].values) + list(df2[feature_name].values)
    le.fit(combined_values)

    df1[feature_name] = le.transform(df1[feature_name])
    df2[feature_name] = le.transform(df2[feature_name])

    print(f"  Created: {feature_name}")

    return df1, df2


def encode_AG2(df1, df2, main_columns, group_columns):
    """
    Create nunique (count unique) aggregation features.

    For each UID, count how many unique values of a feature exist.
    Example: How many unique email domains does this UID have?
    - If nunique=1, it's consistent behavior (same domain always)
    - If nunique>1, might indicate card sharing or fraud

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe (typically train)
    df2 : pd.DataFrame
        Second dataframe (typically test)
    main_columns : list of str
        Columns to count unique values for
    group_columns : list of str
        Columns to group by (typically ['UID_encoded'])

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Both dataframes with nunique features added
    """
    for main_col in main_columns:
        if main_col not in df1.columns or main_col not in df2.columns:
            print(f"Warning: {main_col} not found, skipping nunique aggregation...")
            continue

        for group_col in group_columns:
            if group_col not in df1.columns or group_col not in df2.columns:
                print(f"Warning: {group_col} not found, skipping nunique aggregation...")
                continue

            # Combine train and test for aggregation
            comb = pd.concat([
                df1[[group_col, main_col]],
                df2[[group_col, main_col]]
            ], axis=0)

            # Count unique values per group
            nunique_map = comb.groupby(group_col)[main_col].nunique().to_dict()

            feature_name = f'{group_col}_{main_col}_ct'
            df1[feature_name] = df1[group_col].map(nunique_map).astype('float32')
            df2[feature_name] = df2[group_col].map(nunique_map).astype('float32')

            # Fill NaN with -1
            df1[feature_name].fillna(-1, inplace=True)
            df2[feature_name].fillna(-1, inplace=True)

            print(f"  Created: {feature_name}")

    return df1, df2


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(data_path):
    """
    Load and merge transaction and identity data.

    Parameters
    ----------
    data_path : Path
        Path to data directory

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Training and test dataframes

    Raises
    ------
    FileNotFoundError
        If required data files are missing
    """
    print("Loading data...")

    try:
        train_identity = pd.read_csv(data_path / "train_identity.csv")
        train_transaction = pd.read_csv(data_path / "train_transaction.csv")
        test_identity = pd.read_csv(data_path / "test_identity.csv")
        test_transaction = pd.read_csv(data_path / "test_transaction.csv")
    except FileNotFoundError as e:
        print(f"Error: Required data file not found - {e}")
        print(f"Expected files in: {data_path}")
        sys.exit(1)

    # Fix column naming inconsistencies in test data
    test_identity.columns = test_identity.columns.str.replace('-', '_')

    print("Merging transaction and identity data...")
    train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

    # Clean up memory
    del train_identity, train_transaction, test_identity, test_transaction
    gc.collect()

    return train, test


def engineer_features(train, test):
    """
    This implements the critical strategies:
    1. D column normalization (transform time deltas to calendar dates)
    2. Cents feature (decimal portion of TransactionAmt)
    3. Frequency encoding for key features
    4. Combined features (card1_addr1, card1_addr1_P_emaildomain)
    5. Card-based aggregations (TransactionAmt, D9, D11)
    6. Magic UID features (47+ features)
    7. Time-based features

    Parameters
    ----------
    train : pd.DataFrame
        Training dataframe
    test : pd.DataFrame
        Test dataframe

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        Training and test dataframes with engineered features
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING (MAGIC VERSION)")
    print("="*60)

    # ========================================================================
    # STEP 1: NORMALIZE D COLUMNS
    # ========================================================================
    print("\n1. Normalizing D columns (subtract TransactionDT to get calendar dates)...")
    # Skip D1, D2, D3, D5, D9 as per notebook
    d_cols_to_normalize = [4, 6, 7, 8, 10, 11, 12, 13, 14, 15]
    for i in d_cols_to_normalize:
        col_name = f'D{i}'
        if col_name in train.columns:
            train[col_name] = train[col_name] - train['TransactionDT'] / np.float32(24*60*60)
            test[col_name] = test[col_name] - test['TransactionDT'] / np.float32(24*60*60)
            print(f"  Normalized: {col_name}")

    # ========================================================================
    # STEP 2: CREATE CENTS FEATURE
    # ========================================================================
    print("\n2. Creating cents feature (decimal portion of TransactionAmt)...")
    train = create_cents_feature(train)
    test = create_cents_feature(test)
    print("  Created: cents")

    # ========================================================================
    # STEP 3: FREQUENCY ENCODING
    # ========================================================================
    print("\n3. Frequency encoding key features...")
    freq_cols = ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain']
    train, test = encode_FE(train, test, freq_cols)

    # ========================================================================
    # STEP 4: COMBINED FEATURES
    # ========================================================================
    print("\n4. Creating combined features...")
    train, test = encode_CB(train, test, 'card1', 'addr1')
    train, test = encode_CB(train, test, 'card1_addr1', 'P_emaildomain')

    # ========================================================================
    # STEP 5: FREQUENCY ENCODE COMBINED FEATURES
    # ========================================================================
    print("\n5. Frequency encoding combined features...")
    combined_freq_cols = ['card1_addr1', 'card1_addr1_P_emaildomain']
    train, test = encode_FE(train, test, combined_freq_cols)

    # ========================================================================
    # STEP 6: CARD-BASED AGGREGATIONS (NON-UID)
    # ========================================================================
    print("\n6. Creating card-based aggregations...")
    # Group by card1, card1_addr1, card1_addr1_P_emaildomain
    group_features = ['card1', 'card1_addr1', 'card1_addr1_P_emaildomain']
    agg_features = ['TransactionAmt', 'D9', 'D11']

    for group_col in group_features:
        if group_col not in train.columns:
            continue
        for agg_col in agg_features:
            if agg_col not in train.columns:
                continue
            # Mean
            grouped_mean = pd.concat([train, test]).groupby(group_col)[agg_col].mean()
            train[f'{agg_col}_{group_col}_mean'] = train[group_col].map(grouped_mean).astype('float32')
            test[f'{agg_col}_{group_col}_mean'] = test[group_col].map(grouped_mean).astype('float32')
            # Std
            grouped_std = pd.concat([train, test]).groupby(group_col)[agg_col].std()
            train[f'{agg_col}_{group_col}_std'] = train[group_col].map(grouped_std).astype('float32')
            test[f'{agg_col}_{group_col}_std'] = test[group_col].map(grouped_std).astype('float32')
            print(f"  Created: {agg_col}_{group_col}_mean, {agg_col}_{group_col}_std")

    # ========================================================================
    # STEP 7: CREATE D1 for UID (not normalized)
    # ========================================================================
    print("\n7. D1 remains unchanged for UID creation...")
    # D1 is NOT normalized - it's used as-is in the UID calculation
    if 'D1' not in train.columns:
        print("  Warning: D1 not found in data")

    # ========================================================================
    # STEP 8: CREATE UID (MAGIC BEGINS HERE!)
    # ========================================================================
    print("\n8. Creating UID (card1_addr1 + floor(day - D1))...")
    # Calculate day index
    train['day'] = train['TransactionDT'] / (24 * 60 * 60)
    test['day'] = test['TransactionDT'] / (24 * 60 * 60)

    # Create UID
    train['UID_encoded'] = (train['card1_addr1'].astype(str) + '_' +
                            np.floor(train['day'] - train['D1']).fillna(-999).astype(str))
    test['UID_encoded'] = (test['card1_addr1'].astype(str) + '_' +
                           np.floor(test['day'] - test['D1']).fillna(-999).astype(str))
    print("  Created: UID_encoded")

    # ========================================================================
    # STEP 9: MAGIC UID FEATURES (47 FEATURES!)
    # ========================================================================
    print("\n9. Creating MAGIC UID-based features (this is the secret sauce!)...")

    # 9a. Frequency encode UID
    print("  9a. Frequency encoding UID...")
    train, test = encode_FE(train, test, ['UID_encoded'])

    # 9b. Aggregate TransactionAmt, D4, D9, D10, D15 by UID (mean, std)
    print("  9b. Aggregating amounts and D columns by UID...")
    uid_agg_cols = ['TransactionAmt', 'D4', 'D9', 'D10', 'D15']
    for col in uid_agg_cols:
        if col not in train.columns:
            continue
        # Mean
        grouped_mean = pd.concat([train, test]).groupby('UID_encoded')[col].mean()
        train[f'{col}_UID_encoded_mean'] = train['UID_encoded'].map(grouped_mean).astype('float32')
        test[f'{col}_UID_encoded_mean'] = test['UID_encoded'].map(grouped_mean).astype('float32')
        # Std
        grouped_std = pd.concat([train, test]).groupby('UID_encoded')[col].std()
        train[f'{col}_UID_encoded_std'] = train['UID_encoded'].map(grouped_std).astype('float32')
        test[f'{col}_UID_encoded_std'] = test['UID_encoded'].map(grouped_std).astype('float32')
        print(f"    Created: {col}_UID_encoded_mean, {col}_UID_encoded_std")

    # 9c. Aggregate all C columns (except C3) by UID (mean)
    print("  9c. Aggregating C columns by UID...")
    c_cols = [f'C{i}' for i in range(1, 15) if i != 3]
    for col in c_cols:
        if col not in train.columns:
            continue
        grouped_mean = pd.concat([train, test]).groupby('UID_encoded')[col].mean()
        train[f'{col}_UID_encoded_mean'] = train['UID_encoded'].map(grouped_mean).astype('float32')
        test[f'{col}_UID_encoded_mean'] = test['UID_encoded'].map(grouped_mean).astype('float32')
        print(f"    Created: {col}_UID_encoded_mean")

    # 9d. Aggregate all M columns by UID (mean) - but skip M5 as it was removed
    print("  9d. Aggregating M columns by UID...")
    m_cols = [f'M{i}' for i in range(1, 10) if i != 5]  # Skip M5

    # Pre-process M columns to numeric
    m_mapping = {'T': 1, 'F': 0, 'M0': 0, 'M1': 1, 'M2': 2}
    for col in m_cols:
        if col in train.columns:
            # Map values if they are strings
            if train[col].dtype == 'object':
                train[col] = train[col].map(m_mapping)
                print(f"    Mapped {col} to numeric")
        if col in test.columns:
            if test[col].dtype == 'object':
                test[col] = test[col].map(m_mapping)

    for col in m_cols:
        if col not in train.columns:
            continue
        grouped_mean = pd.concat([train, test]).groupby('UID_encoded')[col].mean()
        train[f'{col}_UID_encoded_mean'] = train['UID_encoded'].map(grouped_mean).astype('float32')
        test[f'{col}_UID_encoded_mean'] = test['UID_encoded'].map(grouped_mean).astype('float32')
        print(f"    Created: {col}_UID_encoded_mean")

    # 9e. Count unique values by UID
    print("  9e. Counting unique values by UID...")
    # Create DT_M (month feature) first
    import datetime
    START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
    train['DT_M'] = train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)).month)
    test['DT_M'] = test['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)).month)

    nunique_cols = ['P_emaildomain', 'dist1', 'DT_M', 'id_02', 'cents']
    train, test = encode_AG2(train, test, nunique_cols, ['UID_encoded'])

    # 9f. C14 std by UID
    print("  9f. C14 std by UID...")
    if 'C14' in train.columns:
        grouped_std = pd.concat([train, test]).groupby('UID_encoded')['C14'].std()
        train['C14_UID_encoded_std'] = train['UID_encoded'].map(grouped_std).astype('float32')
        test['C14_UID_encoded_std'] = test['UID_encoded'].map(grouped_std).astype('float32')
        print("    Created: C14_UID_encoded_std")

    # 9g. More nunique features
    print("  9g. More nunique counts by UID...")
    more_nunique_cols = ['C13', 'V314']
    train, test = encode_AG2(train, test, more_nunique_cols, ['UID_encoded'])

    v_nunique_cols = ['V127', 'V136', 'V309', 'V307', 'V320']
    train, test = encode_AG2(train, test, v_nunique_cols, ['UID_encoded'])

    # 9h. Outsider15 feature
    print("  9h. Creating outsider15 feature...")
    if 'D1' in train.columns and 'D15' in train.columns:
        train['outsider15'] = (np.abs(train['D1'] - train['D15']) > 3).astype('int8')
        test['outsider15'] = (np.abs(test['D1'] - test['D15']) > 3).astype('int8')
        print("    Created: outsider15")

    # ========================================================================
    # STEP 10: TIME FEATURES
    # ========================================================================
    print("\n10. Creating time-based features (hour, day of week)...")
    train = create_time_features(train)
    test = create_time_features(test)

    print("\n" + "="*60)
    print("Feature engineering complete! Added 50+ magic features.")
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    print("="*60 + "\n")

    return train, test



def encode_categorical_features(train, test):
    """
    Encode categorical features using LabelEncoder.

    Note: For tree-based models like LightGBM, we could also use categorical
    feature support directly. This approach uses label encoding for simplicity.

    Parameters
    ----------
    train : pd.DataFrame
        Training dataframe
    test : pd.DataFrame
        Test dataframe

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, list)
        Training and test dataframes with encoded features, and list of
        categorical column names that were successfully encoded
    """
    print("Encoding categorical features...")

    categorical_features = TRANSACTION_CATEGORICAL + IDENTITY_CATEGORICAL + ['UID_encoded']
    encoded_categorical = []

    for col in categorical_features:
        if col in train.columns and col in test.columns:
            le = LabelEncoder()
            # Fit on combined train+test to handle unseen labels gracefully
            combined_values = (
                list(train[col].astype(str).values) +
                list(test[col].astype(str).values)
            )
            le.fit(combined_values)

            train[col] = le.transform(list(train[col].astype(str).values))
            test[col] = le.transform(list(test[col].astype(str).values))
            encoded_categorical.append(col)
        else:
            print(f'  Skipping {col} (not found in both train and test)')

    print(f"Encoded {len(encoded_categorical)} categorical features")
    return train, test, encoded_categorical


def prepare_model_data(train, test):
    """
    Prepare final datasets for modeling.

    Steps:
    1. Drop temporary/identifier columns
    2. Sort by time for proper train/val split
    3. Create train/validation split (time-based)

    Parameters
    ----------
    train : pd.DataFrame
        Training dataframe
    test : pd.DataFrame
        Test dataframe

    Returns
    -------
    tuple of (X_train, X_val, y_train, y_val, X_test)
        Train/validation/test splits ready for modeling
    """
    print("\n" + "="*60)
    print("PREPARING DATA FOR MODELING")
    print("="*60)

    # Calculate temporary day index for sorting (then drop it)
    train['TransactionDay'] = train['TransactionDT'] / (24 * 60 * 60)

    # Drop identification columns to prevent overfitting
    print("\nDropping identifier columns to prevent overfitting...")
    for col in COLS_TO_DROP_FOR_MODELING:
        if col in train.columns:
            train.drop(col, axis=1, inplace=True)
            print(f"  Dropped: {col}")
        if col in test.columns:
            test.drop(col, axis=1, inplace=True)

    # Sort by time (critical for time-based split)
    print("\nSorting by TransactionDT for time-based validation...")
    train = train.sort_values('TransactionDT')

    # Create features (X) and target (y)
    drop_cols = ['isFraud', 'TransactionDT', 'TransactionID']
    X = train.drop(drop_cols, axis=1)
    y = train['isFraud']

    # Prepare test data (only drop columns that exist)
    X_test = test.drop([c for c in drop_cols if c in test.columns], axis=1)

    # Ensure train and test have same columns
    X_test = X_test[X.columns]

    # Time-based train/validation split (80/20)
    split_idx = int(TRAIN_VAL_SPLIT * len(X))
    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_val = y.iloc[split_idx:]

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"\nFraud rate in train: {y_train.mean():.4f}")
    print(f"Fraud rate in validation: {y_val.mean():.4f}")

    print("="*60 + "\n")

    return X_train, X_val, y_train, y_val, X_test


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_lightgbm(X_train, y_train, X_val, y_val):
    """
    Train LightGBM classifier with early stopping.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation labels

    Returns
    -------
    lgb.LGBMClassifier
        Trained LightGBM model
    """
    print("="*60)
    print("TRAINING LIGHTGBM MODEL")
    print("="*60)
    print(f"\nHyperparameters:")
    for key, value in LGBM_PARAMS.items():
        print(f"  {key}: {value}")
    print(f"  early_stopping_rounds: {EARLY_STOPPING_ROUNDS}")
    print()

    # Initialize model
    clf = lgb.LGBMClassifier(**LGBM_PARAMS)

    # Train with early stopping
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(LOG_EVAL_PERIOD)
        ]
    )

    # Calculate and display validation metrics
    val_preds = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)

    print("\n" + "="*60)
    print(f"VALIDATION AUC: {val_auc:.6f}")
    print(f"Best iteration: {clf.best_iteration_}")
    print("="*60 + "\n")

    return clf


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost classifier with early stopping.

    Parameters
    ----------
    X_train : pd.DataFrame
       Training features
    y_train : pd.Series
       Training labels
    X_val : pd.DataFrame
       Validation features
    y_val : pd.Series
       Validation labels

    Returns
    -------
    xgb.XGBClassifier
       Trained XGBoost model
    """
    print("="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60)
    print(f"\nHyperparameters:")
    for key, value in XGB_PARAMS.items():
        print(f"  {key}: {value}")

    # Initialize model
    clf = xgb.XGBClassifier(**XGB_PARAMS)
    # Note: Using XGB_PARAMS from config

    # Train with early stopping
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        # eval_metric='auc', # Already in XGB_PARAMS
        verbose=LOG_EVAL_PERIOD
    )

    # Calculate and display validation metrics
    val_preds = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)

    print("\n" + "="*60)
    print(f"VALIDATION AUC (XGB): {val_auc:.6f}")
    if hasattr(clf, 'best_iteration'):
        print(f"Best iteration: {clf.best_iteration}")
    print("="*60 + "\n")

    return clf


def train_catboost(X_train, y_train, X_val, y_val, categorical_features):
    """
    Train CatBoost classifier with early stopping.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation labels
    categorical_features : list
        List of categorical feature names

    Returns
    -------
    cb.CatBoostClassifier
        Trained CatBoost model
    """
    print("="*60)
    print("TRAINING CATBOOST MODEL")
    print("="*60)
    print(f"\nHyperparameters:")
    for key, value in CATBOOST_PARAMS.items():
        print(f"  {key}: {value}")

    # Initialize model
    clf = cb.CatBoostClassifier(**CATBOOST_PARAMS)

    # Train with early stopping
    clf.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=[c for c in categorical_features if c in X_train.columns],
        use_best_model=True,
        verbose=LOG_EVAL_PERIOD
    )

    # Calculate and display validation metrics
    val_preds = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)

    print("\n" + "="*60)
    print(f"VALIDATION AUC (CatBoost): {val_auc:.6f}")
    print(f"Best iteration: {clf.get_best_iteration()}")
    print("="*60 + "\n")

    return clf



def create_submission(predictions, data_path, filename="submission.csv"):
    """
    Create submission file with fraud predictions.

    Parameters
    ----------
    predictions : np.array
        Probability predictions
    data_path : Path
        Path to data directory
    filename : str, default="submission.csv"
        Name of the output file

    Returns
    -------
    None
        Saves submission.csv to parent directory
    """
    print(f"Creating submission file: {filename}...")

    # Load sample submission and update with predictions
    submission_df = pd.read_csv(data_path / "sample_submission.csv")
    submission_df['isFraud'] = predictions

    # Save submission
    submission_path = data_path.parent / filename
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path}")
    print(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Mean prediction: {predictions.mean():.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("IEEE-CIS FRAUD DETECTION MODEL")
    print("="*60 + "\n")

    # 1. Load data
    train, test = load_data(DATA_PATH)

    # 2. Reduce memory usage
    print("\nOptimizing memory usage...")
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    # 3. Reduce V columns (user specified list)
    print(f"\nSelecting V columns from specified list...")
    train, test = reduce_v_columns(train, test, v_cols_to_keep=V_COLS_TO_KEEP)

    # 4. Feature engineering
    train, test = engineer_features(train, test)

    # 5. Encode categorical features
    train, test, _ = encode_categorical_features(train, test)

    # 6. Prepare modeling data
    X_train, X_val, y_train, y_val, X_test = prepare_model_data(train, test)

    # Clean up memory
    del train, test
    gc.collect()

    # 7. Train models
    print("\nTraining Ensemble Models...")
    cat_feats = [c for c in TRANSACTION_CATEGORICAL + IDENTITY_CATEGORICAL + ['UID_encoded'] if c in X_train.columns]
    catboost_model = train_catboost(X_train, y_train, X_val, y_val, cat_feats)
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

    # 8. Evaluation and Ensembling
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION")
    print("="*60)

    cat_val_preds = catboost_model.predict_proba(X_val)[:, 1]
    lgb_val_preds = lgb_model.predict_proba(X_val)[:, 1]
    xgb_val_preds = xgb_model.predict_proba(X_val)[:, 1]

    # Simple Average Ensemble
    ensemble_val_preds = (lgb_val_preds + xgb_val_preds + cat_val_preds) / 3

    cat_auc = roc_auc_score(y_val, cat_val_preds)
    lgb_auc = roc_auc_score(y_val, lgb_val_preds)
    xgb_auc = roc_auc_score(y_val, xgb_val_preds)
    ensemble_auc = roc_auc_score(y_val, ensemble_val_preds)

    print(f"CatBoost Validation AUC: {cat_auc:.6f}")
    print(f"LightGBM Validation AUC: {lgb_auc:.6f}")
    print(f"XGBoost  Validation AUC: {xgb_auc:.6f}")
    print(f"Ensemble Validation AUC: {ensemble_auc:.6f}")
    print("="*60 + "\n")

    # 9. Create final predictions and submission
    print("Generating test predictions...")
    cat_test_preds = catboost_model.predict_proba(X_test)[:, 1]
    lgb_test_preds = lgb_model.predict_proba(X_test)[:, 1]
    xgb_test_preds = xgb_model.predict_proba(X_test)[:, 1]

    ensemble_test_preds = (lgb_test_preds + xgb_test_preds + cat_test_preds) / 3

    # Save ensemble submission
    create_submission(ensemble_test_preds, DATA_PATH, filename="submission_ensemble.csv")

    # Also save individual for comparison
    create_submission(lgb_test_preds, DATA_PATH, filename="submission_lgb.csv")
    create_submission(xgb_test_preds, DATA_PATH, filename="submission_xgb.csv")
    create_submission(cat_test_preds, DATA_PATH, filename="submission_cat.csv")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()