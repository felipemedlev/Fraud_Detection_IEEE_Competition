import requests
import json
import time

# Sample transaction (same as test_endpoint_fixed.py)
sample_transaction = {
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
    "C1": 1.0, "C2": 1.0, "C3": 0.0, "C4": 0.0, "C5": 0.0, "C6": 1.0, "C7": 0.0, "C8": 0.0, "C9": 1.0, "C10": 0.0, "C11": 1.0, "C12": 0.0, "C13": 1.0, "C14": 1.0,
    "D1": 14.0, "D2": None, "D3": 0.0, "D4": 0.0, "D5": None, "D6": None, "D7": None, "D8": None, "D9": None, "D10": 14.0, "D11": None, "D12": None, "D13": None, "D14": None, "D15": 0.0,
    "M1": "T", "M2": "T", "M3": "T", "M4": "M0", "M5": "F", "M6": "T", "M7": "F", "M8": "F", "M9": "T",
    "dist1": 19.0, "dist2": None
}

# Add V columns (just a few key ones to keep it valid)
for i in range(1, 340):
    sample_transaction[f"V{i}"] = None

print("Testing Flask App at http://localhost:5001/predict")
try:
    response = requests.post(
        "http://localhost:5001/predict",
        json=sample_transaction,
        headers={"Content-Type": "application/json"}
    )

    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))

    if response.status_code == 200:
        print("✅ Flask App Test Passed!")
    else:
        print("❌ Flask App Test Failed")

except Exception as e:
    print(f"❌ Error connecting to Flask app: {e}")
