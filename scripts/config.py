"""
Configuration file for IEEE-CIS Fraud Detection model.

Contains all hyperparameters, paths, and constants used in the modeling pipeline.
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent / "data"
RANDOM_STATE = 42

# Specific V-columns to keep (based on manual selection/analysis)
V_COLS_TO_KEEP = [
    1, 3, 4, 6, 8, 11, 13, 14, 17, 20, 23, 26, 27, 30, 36, 37, 40, 41, 44, 47, 48,
    54, 56, 59, 62, 65, 67, 68, 70, 76, 78, 80, 82, 86, 88, 89, 91, 96, 98, 99, 104,
    107, 108, 111, 115, 117, 120, 121, 123, 124, 127, 129, 130, 136, 138, 139, 142,
    147, 156, 162, 165, 160, 166, 178, 176, 173, 182, 187, 203, 205, 207, 215, 169,
    171, 175, 180, 185, 188, 198, 210, 209, 218, 223, 224, 226, 228, 229, 235, 240,
    258, 257, 253, 252, 260, 261, 264, 266, 267, 274, 277, 220, 221, 234, 238, 250,
    271, 294, 284, 285, 286, 291, 297, 303, 305, 307, 309, 310, 320, 281, 283, 289,
    296, 301, 314, 332, 325, 335, 338
]

# Train/validation split ratio (time-based split)
TRAIN_VAL_SPLIT = 0.8

LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 2500,
    'learning_rate': 0.02,
    'num_leaves': 256,
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
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'max_depth': 12,
    'subsample': 0.8,
    'colsample_bytree': 0.4,
    'tree_method': 'hist',
    'random_state': RANDOM_STATE,
    'n_jobs': 4,
    'early_stopping_rounds': 100
}


CATBOOST_PARAMS = {
    'iterations': 3000,
    'learning_rate': 0.03,
    'depth': 8,
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
