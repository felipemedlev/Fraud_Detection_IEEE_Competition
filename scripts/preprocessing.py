import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import gc
from sklearn.preprocessing import LabelEncoder

# Embedded configuration (to avoid external dependencies in SageMaker)
TRANSACTION_CATEGORICAL = [
    'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
]

IDENTITY_CATEGORICAL = [
    'DeviceType', 'DeviceInfo',
    'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18',
    'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25',
    'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
    'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38'
]

class PreProcessor:
    """
    Stateful PreProcessor for Fraud Detection Model.

    Learns statistics (frequency encodings, aggregations) from training data (fit)
    and applies them to new data (transform).
    """
    def __init__(self):
        self.freq_encodings = {}
        self.aggregations = {}
        self.combined_cols_stats = {}
        self.uid_stats = {}
        self.categorical_encoders = {}
        self.global_means = {}  # Fallback for unseen values

    def fit(self, train_df):
        """
        Learn statistics from training data.
        """
        print("Fitting PreProcessor...")
        df = train_df.copy()

        # 0. Basic Preprocessing needed for fit
        # Normalized D columns
        d_cols_to_normalize = [4, 6, 7, 8, 10, 11, 12, 13, 14, 15]
        for i in d_cols_to_normalize:
            col_name = f'D{i}'
            if col_name in df.columns:
                df[f'{col_name}n'] = df[col_name] - df['TransactionDT'] / np.float32(24*60*60)

        # Cents
        df['cents'] = (df['TransactionAmt'] - np.floor(df['TransactionAmt'])).astype('float32')

        # 1. Frequency Encoding (Learn counts)
        freq_cols = ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain']
        for col in freq_cols:
            if col in df.columns:
                vc = df[col].value_counts(normalize=True).to_dict()
                self.freq_encodings[col] = vc
                # Save global mean for fallback
                self.global_means[f'{col}_FE'] = df[col].map(vc).mean()

        # 2. Combined Features (Learn counts)
        # Create temporary combined cols
        df['card1_addr1'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
        df['card1_addr1_P_emaildomain'] = df['card1_addr1'] + '_' + df['P_emaildomain'].astype(str)

        combined_freq_cols = ['card1_addr1', 'card1_addr1_P_emaildomain']
        for col in combined_freq_cols:
             vc = df[col].value_counts(normalize=True).to_dict()
             self.freq_encodings[col] = vc
             self.global_means[f'{col}_FE'] = df[col].map(vc).mean()

        # 3. Card-based Aggregations (Learn mean/std)
        # Group by card1, card1_addr1, card1_addr1_P_emaildomain
        group_features = ['card1', 'card1_addr1', 'card1_addr1_P_emaildomain']
        agg_features = ['TransactionAmt', 'D9', 'D11']

        for group_col in group_features:
            if group_col not in df.columns: continue

            self.aggregations[group_col] = {}
            for agg_col in agg_features:
                if agg_col not in df.columns: continue

                # Mean
                grouped_mean = df.groupby(group_col)[agg_col].mean().to_dict()
                self.aggregations[group_col][f'{agg_col}_mean'] = grouped_mean
                self.global_means[f'{agg_col}_{group_col}_mean'] = df[agg_col].mean()

                # Std
                grouped_std = df.groupby(group_col)[agg_col].std().to_dict()
                self.aggregations[group_col][f'{agg_col}_std'] = grouped_std
                self.global_means[f'{agg_col}_{group_col}_std'] = df[agg_col].std()

        # 4. UID Creation (needed for next steps but strict logic usually doesn't need "fitting" UID itself unless we use it for grouping)
        df['day'] = df['TransactionDT'] / (24 * 60 * 60)
        df['UID_encoded'] = (df['card1_addr1'].astype(str) + '_' +
                                np.floor(df['day'] - df['D1']).fillna(-999).astype(str))

        # 5. UID Based Aggregations (Learn mean/std per UID)
        self.uid_stats = {}

        # 5a. Frequency encode UID
        vc = df['UID_encoded'].value_counts(normalize=True).to_dict()
        self.freq_encodings['UID_encoded'] = vc
        self.global_means['UID_encoded_FE'] = df['UID_encoded'].map(vc).mean()

        # 5b. Aggregating amounts and D columns by UID
        uid_agg_cols = ['TransactionAmt', 'D4', 'D9', 'D10', 'D15']
        for col in uid_agg_cols:
            if col in df.columns:
                self.uid_stats[f'{col}_mean'] = df.groupby('UID_encoded')[col].mean().to_dict()
                self.uid_stats[f'{col}_std'] = df.groupby('UID_encoded')[col].std().to_dict()
                self.global_means[f'{col}_UID_encoded_mean'] = df[col].mean()
                self.global_means[f'{col}_UID_encoded_std'] = df[col].std()

        # 5c. C columns mean by UID
        c_cols = [f'C{i}' for i in range(1, 15) if i != 3]
        for col in c_cols:
            if col in df.columns:
                self.uid_stats[f'{col}_mean'] = df.groupby('UID_encoded')[col].mean().to_dict()
                self.global_means[f'{col}_UID_encoded_mean'] = df[col].mean()

        # 5d. M columns (mapped to numeric) mean by UID
        m_cols = [f'M{i}' for i in range(1, 10) if i != 5]
        m_mapping = {'T': 1, 'F': 0, 'M0': 0, 'M1': 1, 'M2': 2}
        for col in m_cols:
            if col in df.columns:
                # Apply mapping locally for calculation
                s = df[col].map(m_mapping) if df[col].dtype == 'object' else df[col]
                self.uid_stats[f'{col}_mean'] = df.groupby('UID_encoded')[col].apply(lambda x: x.map(m_mapping).mean() if x.dtype=='object' else x.mean()).to_dict()
                # Note: previous line is complex, simplifying:
                # We should map M columns first in a real pipeline, but here we do it ad-hoc
                # Let's trust the logic in transform to map them, so we just aggregate numeric values here
                if df[col].dtype == 'object':
                     df[col] = df[col].map(m_mapping)

                self.uid_stats[f'{col}_mean'] = df.groupby('UID_encoded')[col].mean().to_dict()
                self.global_means[f'{col}_UID_encoded_mean'] = df[col].mean()

        # 5e. Count unique values (nunique) by UID
        # We learn the count of unique values for a feature associated with a UID in the training set
        self.nunique_encodings = {}

        # Define the pairs to compute nunique for
        # format: (group_col, target_col)
        nunique_pairs = [
            ('UID_encoded', 'P_emaildomain'),
            ('UID_encoded', 'dist1'),
            ('UID_encoded', 'id_02'),
            ('UID_encoded', 'cents'),
            ('UID_encoded', 'C13'),
            ('UID_encoded', 'V314'),
            ('UID_encoded', 'V127'),
            ('UID_encoded', 'V136'),
            ('UID_encoded', 'V309'),
            ('UID_encoded', 'V307'),
            ('UID_encoded', 'V320')
        ]

        # Features that might need Month based aggregation (DT_M)
        # We skip DT_M for simplicity in this version unless strict checking needed

        for group_col, target_col in nunique_pairs:
             if group_col in df.columns and target_col in df.columns:
                 # Count unique keys
                 mapping = df.groupby(group_col)[target_col].nunique().to_dict()
                 feature_name = f'{group_col}_{target_col}_ct'
                 self.nunique_encodings[feature_name] = mapping
                 self.global_means[feature_name] = df[feature_name].mean() if feature_name in df.columns else 0 # approximate

                 self.global_means[feature_name] = df[feature_name].mean() if feature_name in df.columns else 0 # approximate

        # 6. Categorical Label Encoding
        # We need to encode categorical features statefully
        from sklearn.preprocessing import LabelEncoder

        # Combine lists
        # We also need to encode the Combined Features which are strings!
        cat_cols = TRANSACTION_CATEGORICAL + IDENTITY_CATEGORICAL + ['UID_encoded', 'card1_addr1', 'card1_addr1_P_emaildomain']

        for col in cat_cols:
            if col in df.columns:
                # Convert to string to ensure consistency
                df[col] = df[col].astype(str)

                # We use a custom dictionary approach for safety with unseen values
                # or we can use LabelEncoder and handle exceptions in transform
                # Let's use value_counts keys as the vocabulary
                unique_vals = df[col].unique()
                mapping = {val: i for i, val in enumerate(unique_vals)}
                self.categorical_encoders[col] = mapping

        print("PreProcessor fit complete.")
        return self

    def transform(self, df):
        """
        Apply learned statistics to new data.
        """
        print("Transforming data...")
        df = df.copy()

        # ========================================================================
        # STEP 1: NORMALIZE D COLUMNS
        # ========================================================================
        d_cols_to_normalize = [4, 6, 7, 8, 10, 11, 12, 13, 14, 15]
        for i in d_cols_to_normalize:
            col_name = f'D{i}'
            if col_name in df.columns:
                df[f'{col_name}n'] = df[col_name] - df['TransactionDT'] / np.float32(24*60*60)

        # ========================================================================
        # STEP 2: CREATE CENTS FEATURE
        # ========================================================================
        if 'TransactionAmt' in df.columns:
            df['cents'] = (df['TransactionAmt'] - np.floor(df['TransactionAmt'])).astype('float32')

        # ========================================================================
        # STEP 3: FREQUENCY ENCODING
        # ========================================================================
        freq_cols = ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain']
        for col in freq_cols:
            if col in df.columns and col in self.freq_encodings:
                # Map using learned frequencies, fill unkonwn with -1 or global mean?
                # Using -1 for consistency with training script
                # However, for API, unknown values might better be -1 or 0 prob
                df[f'{col}_FE'] = df[col].map(self.freq_encodings[col]).fillna(-1).astype('float32')

        # ========================================================================
        # STEP 4: COMBINED FEATURES
        # ========================================================================
        if 'card1' in df.columns and 'addr1' in df.columns:
            df['card1_addr1'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)

        if 'card1_addr1' in df.columns and 'P_emaildomain' in df.columns:
            df['card1_addr1_P_emaildomain'] = df['card1_addr1'] + '_' + df['P_emaildomain'].astype(str)

        # ========================================================================
        # STEP 5: FREQUENCY ENCODE COMBINED FEATURES
        # ========================================================================
        combined_freq_cols = ['card1_addr1', 'card1_addr1_P_emaildomain']
        for col in combined_freq_cols:
            if col in df.columns and col in self.freq_encodings:
                df[f'{col}_FE'] = df[col].map(self.freq_encodings[col]).fillna(-1).astype('float32')

        # ========================================================================
        # STEP 6: CARD-BASED AGGREGATIONS (NON-UID)
        # ========================================================================
        group_features = ['card1', 'card1_addr1', 'card1_addr1_P_emaildomain']
        agg_features = ['TransactionAmt', 'D9', 'D11']

        for group_col in group_features:
            if group_col not in df.columns or group_col not in self.aggregations:
                continue

            for agg_col in agg_features:
                # Mean
                if f'{agg_col}_mean' in self.aggregations[group_col]:
                    mean_map = self.aggregations[group_col][f'{agg_col}_mean']
                    default_val = self.global_means.get(f'{agg_col}_{group_col}_mean', -1)
                    df[f'{agg_col}_{group_col}_mean'] = df[group_col].map(mean_map).fillna(default_val).astype('float32')

                # Std
                if f'{agg_col}_std' in self.aggregations[group_col]:
                    std_map = self.aggregations[group_col][f'{agg_col}_std']
                    default_val = self.global_means.get(f'{agg_col}_{group_col}_std', -1)
                    df[f'{agg_col}_{group_col}_std'] = df[group_col].map(std_map).fillna(default_val).astype('float32')

        # ========================================================================
        # STEP 8: CREATE UID
        # ========================================================================
        if 'TransactionDT' in df.columns:
            df['day'] = df['TransactionDT'] / (24 * 60 * 60)

        # Create UID
        # We need D1
        if 'D1' in df.columns and 'card1_addr1' in df.columns:
             df['UID_encoded'] = (df['card1_addr1'].astype(str) + '_' +
                                    np.floor(df['day'] - df['D1']).fillna(-999).astype(str))

        # ========================================================================
        # STEP 9: MAGIC UID FEATURES
        # ========================================================================

        # 9a. Frequency Data
        if 'UID_encoded' in self.freq_encodings:
             df['UID_encoded_FE'] = df['UID_encoded'].map(self.freq_encodings['UID_encoded']).fillna(-1).astype('float32')

        # 9b, 9c, 9d. UID Aggregations
        # All stored in self.uid_stats with keys like 'TransactionAmt_mean', 'C1_mean', etc.

        # Helper list of all columns we aggregated
        all_uid_aggs = []
        uid_agg_cols = ['TransactionAmt', 'D4', 'D9', 'D10', 'D15']
        all_uid_aggs.extend([(c, 'mean') for c in uid_agg_cols])
        all_uid_aggs.extend([(c, 'std') for c in uid_agg_cols])

        c_cols = [f'C{i}' for i in range(1, 15) if i != 3]
        all_uid_aggs.extend([(c, 'mean') for c in c_cols])

        m_cols = [f'M{i}' for i in range(1, 10) if i != 5]
        all_uid_aggs.extend([(c, 'mean') for c in m_cols])

        # Pre-process M columns for mapping if they exist in df
        m_mapping = {'T': 1, 'F': 0, 'M0': 0, 'M1': 1, 'M2': 2}
        for col in m_cols:
            if col in df.columns and df[col].dtype == 'object':
                 df[col] = df[col].map(m_mapping)

        for col, stat in all_uid_aggs:
            key = f'{col}_{stat}'
            feature_name = f'{col}_UID_encoded_{stat}'

            if key in self.uid_stats:
                mapping = self.uid_stats[key]
                default_val = self.global_means.get(feature_name, -1)
                df[feature_name] = df['UID_encoded'].map(mapping).fillna(default_val).astype('float32')

        # 9e. Count unique values by UID
        nunique_pairs = [
            ('UID_encoded', 'P_emaildomain'),
            ('UID_encoded', 'dist1'),
            ('UID_encoded', 'id_02'),
            ('UID_encoded', 'cents'),
            ('UID_encoded', 'C13'),
            ('UID_encoded', 'V314'),
            ('UID_encoded', 'V127'),
            ('UID_encoded', 'V136'),
            ('UID_encoded', 'V309'),
            ('UID_encoded', 'V307'),
            ('UID_encoded', 'V320')
        ]

        for group_col, target_col in nunique_pairs:
             feature_name = f'{group_col}_{target_col}_ct'
             if feature_name in self.nunique_encodings:
                  mapping = self.nunique_encodings[feature_name]
                  # Use 0 or -1 for unseen UIDs? Usually 0 unique items is impossible but reasonable as logic "unknown"
                  # Original code used -1 for NaNs.
                  default_val = -1
                  df[feature_name] = df[group_col].map(mapping).fillna(default_val).astype('float32')

        # 9h. Outsider15 feature
        if 'D1' in df.columns and 'D15' in df.columns:
            df['outsider15'] = (np.abs(df['D1'] - df['D15']) > 3).astype('int8')

        # ========================================================================
        # STEP 10: TIME FEATURES
        # ========================================================================
        if 'TransactionDT' in df.columns:
            df['hour'] = (df['TransactionDT'] // 3600) % 24
            df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7

        if 'TransactionDT' in df.columns:
            df['hour'] = (df['TransactionDT'] // 3600) % 24
            df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7

        # ========================================================================
        # STEP 11: CATEGORICAL LABEL ENCODING
        # ========================================================================
        cat_cols = TRANSACTION_CATEGORICAL + IDENTITY_CATEGORICAL + ['UID_encoded', 'card1_addr1', 'card1_addr1_P_emaildomain']

        for col in cat_cols:
            if col in df.columns and col in self.categorical_encoders:
                df[col] = df[col].astype(str)
                mapping = self.categorical_encoders[col]
                # Map and fill unseen with -1
                df[col] = df[col].map(mapping).fillna(-1).astype('int32')
            elif col in df.columns:
                # If column exists but we didn't learn it (e.g. wasn't in train), fill -1
                df[col] = -1

        # ========================================================================
        # STEP 12: ENSURE ALL REMAINING OBJECT COLUMNS ARE NUMERIC
        # ========================================================================
        # Convert any remaining object dtype columns to numeric or fill them
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric, fill NaNs with -1
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype('float32')
                    print(f"Warning: Converted object column '{col}' to numeric")
                except:
                    # If conversion fails, label encode it
                    df[col] = df[col].astype(str).astype('category').cat.codes.astype('int32')
                    print(f"Warning: Label encoded unexpected object column '{col}'")

        return df

    def save(self, filepath):
        joblib.dump(self, filepath)
        print(f"PreProcessor saved to {filepath}")

    @staticmethod
    def load(filepath):
        return joblib.load(filepath)
