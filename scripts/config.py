"""
Configuration file for IEEE-CIS Fraud Detection model.

Contains all hyperparameters, paths, and constants used in the modeling pipeline.
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent / "data"
RANDOM_STATE = 42

# Train/validation split ratio (time-based split)
TRAIN_VAL_SPLIT = 0.9

LGBM_PARAMS = {
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'num_leaves': 256,
    'objective': 'binary',
    'metric': 'auc',
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'random_state': RANDOM_STATE,
    'n_jobs': 4,
    'verbose': -1
}

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.4,
    'tree_method': 'hist',
    'random_state': RANDOM_STATE,
    'n_jobs': 4,
    'early_stopping_rounds': 100
}


CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'eval_metric': 'AUC',
    'random_seed': RANDOM_STATE,
    'bagging_temperature': 0.5,
    'od_type': 'Iter',
    'metric_period': 100,
    'od_wait': 100,
    'allow_writing_files': False,
    'task_type': 'CPU'
}

# Early stopping rounds
EARLY_STOPPING_ROUNDS = 100

# Logging frequency during training
LOG_EVAL_PERIOD = 100

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

# Columns to drop before modeling (high cardinality or temporary features)
COLS_TO_DROP_FOR_MODELING = [
    'TransactionDay',  # Temporary feature
    'D4n', 'D10n', 'D15n',  # Raw features (we keep aggregations)
    'UID_encoded',  # Raw UID (we keep UID-based aggregations)
    'DeviceInfo'  # Very high cardinality
]
