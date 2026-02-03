"""
Test script for SageMaker endpoint with complete feature set
Usage: python test_endpoint_fixed.py
"""
import boto3
import json
import pandas as pd

ENDPOINT_NAME = "fraud-detection-endpoint"
REGION = "eu-north-1"

V_COLS_TO_KEEP = [279, 284, 285, 286, 290, 291, 297, 302, 305, 309, 310, 312, 319, 95, 98, 99, 104, 107, 108, 109, 111, 114, 115, 117, 118, 120, 121, 123, 124, 129, 130, 131, 135, 281, 282, 283, 288, 296, 300, 313, 314, 12, 14, 15, 19, 23, 25, 27, 29, 53, 55, 56, 57, 61, 65, 66, 68, 69, 75, 77, 79, 80, 82, 86, 88, 89, 90, 35, 37, 39, 41, 44, 46, 48, 52, 1, 2, 4, 6, 8, 10, 220, 221, 234, 238, 250, 256, 270, 169, 170, 174, 175, 180, 184, 188, 194, 198, 208, 209, 167, 172, 173, 176, 181, 187, 205, 214, 215, 217, 223, 224, 226, 228, 235, 240, 241, 242, 247, 258, 260, 262, 265, 266, 276, 322, 325, 326, 328, 334, 335, 337, 338, 138, 139, 141, 143, 144, 146, 148, 161, 166]

def create_sample_transaction():
    """Create a complete sample transaction with all required features"""

    # Base transaction features
    sample = {
        "TransactionID": 3663549,
        "TransactionDT": 18403224,
        "TransactionAmt": 31.95,
        "ProductCD": "W",

        # Card features
        "card1": 10409,
        "card2": 111.0,
        "card3": 150.0,
        "card4": "discover",
        "card5": 226.0,
        "card6": "debit",

        # Address features
        "addr1": 170.0,
        "addr2": 87.0,

        # Email domains
        "P_emaildomain": "gmail.com",
        "R_emaildomain": None,

        # D columns (time deltas)
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

        # C columns (counts)
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

        # M columns (match indicators)
        "M1": "T",
        "M2": "T",
        "M3": "T",
        "M4": "M0",
        "M5": None,
        "M6": "F",
        "M7": None,
        "M8": None,
        "M9": None,

        # Distance
        "dist1": 19.0,
        "dist2": None,
    }

    # Add V columns (Vesta engineered features)
    for v_num in V_COLS_TO_KEEP:
        sample[f"V{v_num}"] = None  # Most V columns will be None for a simple test

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

def test_endpoint():
    """Test the SageMaker endpoint"""
    print("=" * 60)
    print("Testing SageMaker Fraud Detection Endpoint")
    print("=" * 60)
    print(f"\nEndpoint: {ENDPOINT_NAME}")
    print(f"Region: {REGION}")
    print()

    # Create SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime', region_name=REGION)

    # Create sample transaction
    sample_transaction = create_sample_transaction()

    print(f"Sample transaction created with {len(sample_transaction)} features")
    print("\nKey features:")
    print(f"  TransactionAmt: ${sample_transaction['TransactionAmt']}")
    print(f"  ProductCD: {sample_transaction['ProductCD']}")
    print(f"  card4: {sample_transaction['card4']}")
    print(f"  card6: {sample_transaction['card6']}")
    print(f"  P_emaildomain: {sample_transaction['P_emaildomain']}")
    print()

    # Prepare payload (send as list)
    payload = json.dumps([sample_transaction])

    # Invoke endpoint
    print("Invoking endpoint...")
    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=payload
        )

        # Parse response
        result = json.loads(response['Body'].read().decode())

        print("Success!")
        print()
        print("Response:")
        print(json.dumps(result, indent=2))
        print()

        # Interpret prediction
        if 'predictions' in result and len(result['predictions']) > 0:
            fraud_prob = result['predictions'][0]
            print(f"Fraud Probability: {fraud_prob:.4f}")

            if fraud_prob > 0.5:
                print("HIGH RISK - Transaction flagged as potential fraud")
            elif fraud_prob > 0.2:
                print("MEDIUM RISK - Transaction requires review")
            else:
                print("LOW RISK - Transaction appears legitimate")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_endpoint()
