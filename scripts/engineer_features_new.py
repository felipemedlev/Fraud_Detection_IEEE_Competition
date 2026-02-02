def engineer_features(train, test):
    """
    Apply all feature engineering steps from the winning notebook.

    This implements the critical winning strategies:
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

    # Create UID as per winning notebook
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
        test[f'{col}_UID_encoded_mean'] = test['UID_encoded'].map(grouped_mean).astype('float32')\n        print(f"    Created: {col}_UID_encoded_mean")

    # 9d. Aggregate all M columns by UID (mean) - but skip M5 as it was removed
    print("  9d. Aggregating M columns by UID...")
    m_cols = [f'M{i}' for i in range(1, 10) if i != 5]  # Skip M5
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
