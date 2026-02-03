#!/usr/bin/env python3
"""
Local test script to verify preprocessing pipeline and model predictions.
This tests the exact same flow as the SageMaker endpoint without deploying.
"""
import json
import pandas as pd
import joblib
import sys
import numpy as np
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent / 'code'))
from preprocessing import PreProcessor

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

def create_sample_transaction():
    """Create the same sample transaction as test_endpoint_fixed.py"""
    V_COLS_TO_KEEP = [279, 284, 285, 286, 290, 291, 297, 302, 305, 309, 310, 312, 319, 95, 98, 99, 104, 107, 108, 109, 111, 114, 115, 117, 118, 120, 121, 123, 124, 129, 130, 131, 135, 281, 282, 283, 288, 296, 300, 313, 314, 12, 14, 15, 19, 23, 25, 27, 29, 53, 55, 56, 57, 61, 65, 66, 68, 69, 75, 77, 79, 80, 82, 86, 88, 89, 90, 35, 37, 39, 41, 44, 46, 48, 52, 1, 2, 4, 6, 8, 10, 220, 221, 234, 238, 250, 256, 270, 169, 170, 174, 175, 180, 184, 188, 194, 198, 208, 209, 167, 172, 173, 176, 181, 187, 205, 214, 215, 217, 223, 224, 226, 228, 235, 240, 241, 242, 247, 258, 260, 262, 265, 266, 276, 322, 325, 326, 328, 334, 335, 337, 338, 138, 139, 141, 143, 144, 146, 148, 161, 166]

    sample = {
        "TransactionID": 3663549,
        "TransactionDT": 18403224,
        "TransactionAmt": 31.95,
        "ProductCD": "W",
        "card1": 10409,
        "card2": 111.0,
        "card3": 150.0,
        "card4": "discover",
        "card5": 226.0,
        "card6": "debit",
        "addr1": 170.0,
        "addr2": 87.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": None,
        "D1": 14.0,
        "D2": None,
        "D3": 0.0,
        "D4": 0.0,
        "D5": None,
        "D6": None,
        "D7": None,
        "D8": None,
        "D9": None,
        "D10": 14.0,
        "D11": None,
        "D12": None,
        "D13": None,
        "D14": None,
        "D15": 0.0,
        "C1": 1.0,
        "C2": 1.0,
        "C3": None,
        "C4": None,
        "C5": None,
        "C6": None,
        "C7": None,
        "C8": None,
        "C9": None,
        "C10": None,
        "C11": None,
        "C12": None,
        "C13": 1.0,
        "C14": 1.0,
        "M1": "T",
        "M2": "T",
        "M3": "T",
        "M4": "M0",
        "M5": None,
        "M6": "F",
        "M7": None,
        "M8": None,
        "M9": None,
        "dist1": 19.0,
        "dist2": None,
    }

    # Add V columns
    for v_num in V_COLS_TO_KEEP:
        sample[f"V{v_num}"] = None

    # Add identity columns
    identity_cols = {
        "DeviceType": None,
        "DeviceInfo": None,
        "id_01": None,
        "id_02": None,
        "id_03": None,
        "id_04": None,
        "id_05": None,
        "id_06": None,
        "id_07": None,
        "id_08": None,
        "id_09": None,
        "id_10": None,
        "id_11": None,
        "id_12": None,
        "id_13": None,
        "id_14": None,
        "id_15": None,
        "id_16": None,
        "id_17": None,
        "id_18": None,
        "id_19": None,
        "id_20": None,
        "id_21": None,
        "id_22": None,
        "id_23": None,
        "id_24": None,
        "id_25": None,
        "id_26": None,
        "id_27": None,
        "id_28": None,
        "id_29": None,
        "id_30": None,
        "id_31": None,
        "id_32": None,
        "id_33": None,
        "id_34": None,
        "id_35": None,
        "id_36": None,
        "id_37": None,
        "id_38": None,
    }

    sample.update(identity_cols)
    return sample

def test_preprocessing_and_prediction():
    """Test the preprocessing pipeline and model predictions locally"""
    print("="*60)
    print("LOCAL INFERENCE TEST (End-to-End)")
    print("="*60)

    # 1. PREPROCESSING
    # ========================================================================
    preprocessor_path = Path("models/preprocessor.joblib")
    if not preprocessor_path.exists():
        print(f"❌ ERROR: Preprocessor not found at {preprocessor_path}")
        return False

    print(f"\n1. Loading preprocessor from {preprocessor_path}...")
    preprocessor = joblib.load(preprocessor_path)
    print("✅ Preprocessor loaded")

    # Create sample transaction
    print("\n2. Creating sample transaction...")
    sample = create_sample_transaction()
    df = pd.DataFrame([sample])
    print(f"✅ Created sample with {len(df.columns)} features")

    # Transform
    print("\n3. Transforming data...")
    try:
        processed = preprocessor.transform(df)
        print(f"✅ Transformed successfully")
    except Exception as e:
        print(f"❌ Transformation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check dtypes
    print("\n4. Checking data types...")
    object_cols = processed.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"⚠️  WARNING: Found {len(object_cols)} object columns:")
        for col in object_cols[:5]:
            print(f"     - {col}: {processed[col].dtype}")
        return False
    else:
        print("✅ All columns are numeric")

    # Drop columns as inference.py does (UPDATED LOGIC)
    print("\n5. Dropping columns (updated logic)...")
    cols_to_drop = [
        'TransactionID',       # Always dropped for training
        'TransactionDT',       # Always dropped for training
        'TransactionDay',      # In config drop list
        'D4n', 'D10n', 'D15n', # In config drop list
        'UID_encoded',         # In config drop list
        'DeviceInfo',          # In config drop list
    ]

    cols_dropped = []
    for col in cols_to_drop:
        if col in processed.columns:
            processed = processed.drop(col, axis=1)
            cols_dropped.append(col)

    print(f"   Dropped {len(cols_dropped)} columns: {cols_dropped}")

    # Check feature count
    print("\n6. Verifying feature count...")
    if processed.shape[1] == 310:
        print("✅ Feature count is correct: 310")
    else:
        print(f"⚠️  WARNING: Feature count is {processed.shape[1]}, expected 310")
        print(f"   Difference: {processed.shape[1] - 310}")
        if processed.shape[1] != 310:
            return False

    # 2. MODEL PREDICTION
    # ========================================================================
    print("\n" + "="*60)
    print("PART 2: LOCAL PREDICTION TEST")
    print("="*60)

    models = {}

    # Load LightGBM
    lgb_path = Path("models/lgb_model.pkl")
    if lgb_path.exists():
        print(f"Loading LightGBM from {lgb_path}...")
        try:
            models['lgb'] = joblib.load(lgb_path)
            print("✅ LightGBM loaded")
        except Exception as e:
            print(f"⚠️ Failed to load LightGBM: {e}")

    # Load XGBoost
    xgb_path = Path("models/xgb_model.pkl")
    if xgb_path.exists():
        print(f"Loading XGBoost from {xgb_path}...")
        try:
            models['xgb'] = joblib.load(xgb_path)
            print("✅ XGBoost loaded")
        except Exception as e:
            print(f"⚠️ Failed to load XGBoost: {e}")

    # Load CatBoost
    cat_path = Path("models/catboost_model.cbm")
    if cat_path.exists():
        print(f"Loading CatBoost from {cat_path}...")
        try:
            import catboost as cb
            model = cb.CatBoostClassifier()
            model.load_model(str(cat_path))
            models['cat'] = model
            print("✅ CatBoost loaded")
        except Exception as e:
            print(f"⚠️ Failed to load CatBoost: {e}")

    if not models:
        print("❌ No models loaded! Cannot test prediction.")
        return False

    # CRITICAL FIX: Reorder columns to match model expectations
    # All models should have same features, using CatBoost's list as reference
    if 'cat' in models:
        print("\nReordering columns to match model feature order...")
        model_features = models['cat'].feature_names_

        # Check if we have all features
        missing = set(model_features) - set(processed.columns)
        if missing:
             print(f"❌ Critical Error: Missing features cannot be reordered: {missing}")
             return False

        # Reorder
        processed = processed[model_features]
        print("✅ Reordered columns successfully")

    # Make predictions
    print("\nMaking predictions...")
    predictions = []

    if 'lgb' in models:
        try:
            pred = models['lgb'].predict_proba(processed)[:, 1]
            predictions.append(pred)
            print(f"✅ LightGBM prediction: {pred[0]:.4f}")
        except Exception as e:
            print(f"❌ LightGBM prediction failed: {e}")

    if 'xgb' in models:
        try:
            print(f"   XGBoost model type: {type(models['xgb'])}")

            # Check feature names
            model_features = None
            if hasattr(models['xgb'], 'feature_names_in_'):
                model_features = models['xgb'].feature_names_in_
            elif hasattr(models['xgb'], 'get_booster'):
                model_features = models['xgb'].get_booster().feature_names

            if model_features is not None:
                # find mismatches
                missing = set(model_features) - set(processed.columns)
                extra = set(processed.columns) - set(model_features)
                if missing: print(f"   ⚠️ MISSING features in input: {missing}")
                if extra: print(f"   ⚠️ EXTRA features in input: {extra}")

            # Try direct dataframe prediction first (if it's sklearn wrapper)
            try:
                pred = models['xgb'].predict_proba(processed)[:, 1]
                print(f"✅ XGBoost prediction (predict_proba): {pred[0]:.4f}")
                predictions.append(pred)
            except Exception as e:
                print(f"   XGBoost predict_proba failed: {e}")
                # Fallback to DMatrix
                import xgboost as xgb
                print("   Falling back to DMatrix...")
                dmatrix = xgb.DMatrix(processed)
                pred = models['xgb'].predict(dmatrix)
                if isinstance(pred, np.ndarray) and pred.ndim > 1:
                     pred = pred[:, 1]
                predictions.append(pred)
                print(f"✅ XGBoost prediction (DMatrix): {pred[0]:.4f}")
        except Exception as e:
            print(f"❌ XGBoost debug failed: {e}")

    if 'cat' in models:
        try:
            # Direct prediction WITHOUT Pool (let CatBoost handle it)
            pred = models['cat'].predict_proba(processed)[:, 1]
            predictions.append(pred)
            print(f"✅ CatBoost prediction: {pred[0]:.4f}")
        except Exception as e:
            print(f"❌ CatBoost prediction failed: {e}")

    if not predictions:
        print("❌ No successful predictions")
        return False

    # Ensemble
    ensemble_pred = np.mean(predictions, axis=0)
    print(f"\n🔮 Ensemble prediction: {ensemble_pred[0]:.4f}")

    if ensemble_pred[0] > 0.5:
        print("⚠️  HIGH RISK - Transaction flagged as potential fraud")
    elif ensemble_pred[0] > 0.2:
        print("⚡ MEDIUM RISK - Transaction requires review")
    else:
        print("✅ LOW RISK - Transaction appears legitimate")

    # Final summary
    print("\n" + "="*60)
    print("✅ LOCAL TEST COMPLETED SUCCESSFULLY!")
    print("="*60)

    return True

if __name__ == "__main__":
    success = test_preprocessing_and_prediction()
    sys.exit(0 if success else 1)
