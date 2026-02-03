import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Global variables for model artifacts
preprocessor = None
models = {}

def model_fn(model_dir):
    """
    Load all model artifacts.
    Called once when container starts.
    """
    print(f"Loading models from {model_dir}")
    global preprocessor, models

    # Load PreProcessor
    pp_path = os.path.join(model_dir, "preprocessor.joblib")
    if not os.path.exists(pp_path):
        raise FileNotFoundError(f"PreProcessor not found: {pp_path}")

    preprocessor = joblib.load(pp_path)
    print(f"✅ PreProcessor loaded from {pp_path}")

    # Load LightGBM
    lgb_path = os.path.join(model_dir, "lgb_model.pkl")
    if os.path.exists(lgb_path):
        models['lgb'] = joblib.load(lgb_path)
        print(f"✅ LightGBM loaded")

    # Load XGBoost
    xgb_path = os.path.join(model_dir, "xgb_model.pkl")
    if os.path.exists(xgb_path):
        models['xgb'] = joblib.load(xgb_path)
        print(f"✅ XGBoost loaded")

    # Load CatBoost
    cat_path = os.path.join(model_dir, "catboost_model.cbm")
    if os.path.exists(cat_path):
        model = cb.CatBoostClassifier()
        model.load_model(cat_path)
        models['cat'] = model
        print(f"✅ CatBoost loaded")

    if not models:
        raise ValueError("No models loaded!")

    print(f"Loaded {len(models)} models successfully")
    return {'preprocessor': preprocessor, 'models': models}

def input_fn(request_body, content_type='application/json'):
    """
    Parse input data payload.
    Expects JSON with transaction data.
    """
    print(f"Parsing input with content_type: {content_type}")

    if content_type == 'application/json':
        data = json.loads(request_body)

        # Handle both single dict and list of dicts
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unexpected data format: {type(data)}")

        print(f"Parsed {len(df)} transactions with {len(df.columns)} features")
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_dict):
    """
    Transform data and generate predictions.

    Args:
        input_data: DataFrame with raw transaction data
        model_dict: Dict with 'preprocessor' and 'models'

    Returns:
        Array of fraud probabilities
    """
    print(f"Making predictions for {len(input_data)} transactions")

    preprocessor = model_dict['preprocessor']
    models = model_dict['models']

    # Step 1: Transform using PreProcessor
    print("Transforming data with PreProcessor...")
    try:
        processed_data = preprocessor.transform(input_data)
        print(f"Transformed shape before dropping: {processed_data.shape}")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        raise

    # Step 1.5: Drop columns that should not be used for modeling
    # Must match config.COLS_TO_DROP_FOR_MODELING exactly
    # Plus drop identifier columns that are always dropped for training
    cols_to_drop = [
        'TransactionID', 'TransactionDT',  # Identifiers
        'TransactionDay',  # Temporary feature
        'D4n', 'D10n', 'D15n',  # Raw features (we keep aggregations)
        'UID_encoded',  # Raw UID (we keep UID-based aggregations)
        'DeviceInfo'  # Very high cardinality
    ]
    cols_dropped = []
    for col in cols_to_drop:
        if col in processed_data.columns:
            processed_data = processed_data.drop(col, axis=1)
            cols_dropped.append(col)

    if cols_dropped:
        print(f"Dropped columns: {cols_dropped}")
        print(f"Transformed shape after dropping: {processed_data.shape}")

    # Step 1.6: Reorder columns to match model expectations (CRITICAL for XGB/CatBoost)
    feature_names = None
    if 'cat' in models:
        feature_names = models['cat'].feature_names_
    elif 'lgb' in models:
        # LightGBM sklearn wrapper stores feature names in booster_
        try:
            feature_names = models['lgb'].booster_.feature_name()
        except:
            pass
    elif 'xgb' in models:
         if hasattr(models['xgb'], 'feature_names_in_'):
             feature_names = models['xgb'].feature_names_in_

    if feature_names is not None:
        # Check for missing columns
        missing = set(feature_names) - set(processed_data.columns)
        if missing:
            print(f"⚠️ Warning: Missing columns for model: {missing}")
            for c in missing:
                processed_data[c] = -1 # Fill missing with default

        # Check for extra columns
        extra = set(processed_data.columns) - set(feature_names)
        if extra:
            print(f"ℹ️ Ignoring extra columns: {extra}")

        # Reorder
        processed_data = processed_data[feature_names]
        print("✅ Reordered columns to match model.")

    # Step 2: Get predictions from all models
    predictions = []

    if 'lgb' in models:
        try:
            pred = models['lgb'].predict_proba(processed_data)[:, 1]
            predictions.append(pred)
            print(f"✅ LightGBM predictions: {pred[:3]}")
        except Exception as e:
            print(f"⚠️ LightGBM prediction failed: {e}")

    if 'xgb' in models:
        try:
            # Convert to DMatrix format for better compatibility
            import xgboost as xgb
            dmatrix = xgb.DMatrix(processed_data)
            pred = models['xgb'].predict(dmatrix)
            predictions.append(pred)
            print(f"✅ XGBoost predictions: {pred[:3]}")
        except Exception as e:
            print(f"⚠️ XGBoost prediction failed: {e}")

    if 'cat' in models:
        try:
            # CatBoost - all features are already properly encoded as numeric
            # Use direct prediction without Pool to avoid categorical feature type errors
            pred = models['cat'].predict_proba(processed_data)[:, 1]
            predictions.append(pred)
            print(f"✅ CatBoost predictions: {pred[:3]}")
        except Exception as e:
            print(f"⚠️ CatBoost prediction failed: {e}")

    if not predictions:
        raise ValueError("No model predictions available")

    # Step 3: Ensemble average
    ensemble_pred = np.mean(predictions, axis=0)
    print(f"Ensemble predictions: {ensemble_pred[:3]}")

    return ensemble_pred

def output_fn(prediction, accept='application/json'):
    """
    Format predictions for output.
    """
    if accept == 'application/json':
        # Convert to list for JSON serialization
        if isinstance(prediction, np.ndarray):
            output = prediction.tolist()
        else:
            output = list(prediction)

        return json.dumps({
            "predictions": output,
            "model": "fraud-detection-ensemble-v1"
        }), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
