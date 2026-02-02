# IEEE-CIS Fraud Detection

A comprehensive fraud detection solution for the IEEE-CIS Fraud Detection Kaggle Competition. This project achieves a **public score AUC of 0.945100** and **private score AUC of 0.924285** using ensemble machine learning techniques and advanced feature engineering.

## Overview

This fraud detection system identifies fraudulent transactions from a real-world e-commerce dataset. The solution leverages:

- **Advanced Feature Engineering**: Client-level aggregations and behavioral patterns
- **Ensemble Learning**: Combining LightGBM, XGBoost, and CatBoost models
- **User Identification (UID)**: Novel approach to identify unique clients from transaction patterns
- **Correlation-Based Feature Reduction**: Intelligent dimensionality reduction preserving predictive power

## Competition Results

- **Public Leaderboard**: 0.945100 AUC
- **Private Leaderboard**: 0.924285 AUC
- **Validation Score**: 0.9483 AUC (Rank Average Ensemble)

## Methodology

For a complete walkthrough of the fraud detection methodology, see [fraud_detection_modeling.ipynb](fraud_detection_modeling.ipynb).

### Key Approach: Client-Level Fraud Detection

The core insight is that **fraud detection is about flagging clients, not just individual transactions**. The key innovation is constructing a unique client identifier (UID) from transaction features:

**UID = `card1` + `addr1` + `D1n`**

Where:
- `card1`: Primary card identifier
- `addr1`: Billing address
- `D1n`: Client start date (normalized from transaction timestamp)

This UID allows the model to:
- Track behavioral patterns per client
- Detect anomalies (e.g., unusual spending amounts)
- Identify suspicious patterns (e.g., multiple IP addresses per client)

### Feature Engineering Pipeline

The feature engineering process creates **50+ features** that dramatically improve model performance (from ~0.90 AUC to ~0.95+ AUC):

#### 1. **Normalized Date Features**
Converts time-delta features (D1, D4, D10, D15) into calendar dates by subtracting from `TransactionDT`:
```python
D1n = TransactionDT - D1  # Client start date
```

#### 2. **Transaction Pattern Features**
- `cents`: Decimal portion of transaction amount (e.g., `49.99 → 0.99`)
  - Human transactions often end in `.99` or `.00`
  - Automated fraud may have random decimals
- `hour`: Hour of day (fraud patterns vary by time)
- `day`: Day of week

#### 3. **Frequency Encoding**
Encodes categorical rarity for features like:
- Card numbers (rare cards = more suspicious)
- Addresses, email domains
- Combined features (`card1_addr1`, `card1_addr1_P_emaildomain`)

#### 4. **UID-Based Aggregations (47+ features)**
For each unique client (UID), calculate:

- **Transaction statistics**: Mean/std of amounts, time deltas
- **Consistency metrics**:
  - `std(D4n, D10n, D15n)` by UID:
    - `std=0` → Single consistent client ✓
    - `std>0` → Multiple clients merged (model learns to split)
- **Behavioral counts**:
  - Unique email domains per UID (1 = normal, >1 = suspicious)
  - Unique IP addresses per UID
  - Unique transaction months
- **C/M column aggregations**: Mean values of count and match features
- **Outlier detection**: `outsider15` flag when `|D1 - D15| > 3`

#### 5. **V Column Reduction**
Reduces 339 engineered V-columns to 141 by:
1. Grouping by NaN structure (columns with similar missing patterns)
2. Finding correlations within each group (threshold: 0.75)
3. Keeping only representative columns from each correlated subset

This reduces multicollinearity while preserving predictive information.

### Model Architecture

#### Individual Models

Three gradient boosting models trained on time-split data (75% train / 25% validation):

1. **LightGBM**
   - 5000 estimators, learning rate 0.01
   - 256 leaves, early stopping at 200 rounds
   - Validation AUC: **0.9466**

2. **XGBoost**
   - 2500 estimators, learning rate 0.02
   - Max depth 12, histogram method
   - Validation AUC: **0.9475**

3. **CatBoost**
   - 2500 iterations, depth 8
   - Native categorical feature handling
   - Validation AUC: **0.9367**

#### Ensemble Strategy

Four ensemble methods compared:

1. **Rank Average** ← **Best Method (0.9483 AUC)**
   - Converts predictions to ranks
   - Averages ranks across models
   - Robust to scale differences

2. **Weighted Average** (0.9482 AUC)
   - Weights based on individual AUC scores

3. **Simple Average** (0.9482 AUC)
   - Equal weights for all models

4. **Stacking with Logistic Regression** (0.9470 AUC)
   - Meta-model learns optimal combination

### Why Time-Based Validation?

**Critical**: Random K-Fold splitting would leak future fraud patterns into training, giving falsely high scores. The model must:
- **Train on the past** (first 75% of transactions by time)
- **Predict the future** (last 25% of transactions)

This simulates real-world deployment where the model sees only historical data.

## Project Structure

```
Kaggle_Fraud/
├── fraud_detection_modeling.ipynb    # Main notebook with complete methodology
├── data/                              # Dataset files (not included)
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   └── test_identity.csv
├── scripts/
│   └── reduce_correlated_columns.py  # V-column correlation reduction logic
└── README.md                          # This file
```

## Dataset

The dataset consists of transaction and identity data from IEEE-CIS:

### Transaction Data
- **TransactionID**: Unique identifier
- **TransactionDT**: Time delta from reference point
- **TransactionAmt**: Amount in USD
- **ProductCD**: Product code (categorical)
- **card1-card6**: Card attributes (categorical)
- **addr1, addr2**: Address information (categorical)
- **dist1, dist2**: Distance features
- **P_emaildomain / R_emaildomain**: Email domains (categorical)
- **C1-C14**: Count features (meaning masked)
- **D1-D15**: Time-delta features
- **M1-M9**: Match indicators (categorical)
- **V1-V339**: Engineered Vesta features

### Identity Data
- **DeviceType, DeviceInfo**: Device information (categorical)
- **id_1-id_38**: Network/browser features (12-38 are categorical)

> **Note**: Column semantics are intentionally masked for security, as this contains real transaction data.

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn scipy
```

### Running the Model

1. Download the competition data from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
2. Place data files in the `data/` directory
3. Open and run [fraud_detection_modeling.ipynb](fraud_detection_modeling.ipynb)

The notebook executes the full pipeline:
- Data loading and memory optimization
- Feature engineering (50+ features)
- V-column reduction (339 → 141)
- Model training (LightGBM, XGBoost, CatBoost)
- Ensemble evaluation
- Submission file generation

## Key Insights

### What Makes This Solution Work?

1. **UID is the Secret Sauce**: Identifying unique clients enables behavioral profiling
2. **Aggregations Reveal Patterns**: Client-level statistics expose anomalies
3. **Time-Based Validation**: Prevents data leakage and tests real-world performance
4. **Smart Feature Reduction**: Correlation-based V-column reduction maintains signal while reducing noise
5. **Ensemble Diversity**: Different models capture different fraud patterns

### Example: Detecting Fraud Through UID

Consider a client with `UID = "card123_addr456_day100"`:
- **Normal**: All transactions ~$50, same email domain, single IP
- **Fraud**: Suddenly $5000 transaction, different email, new IP address

The model detects this through:
- `TransactionAmt_UID_encoded_mean/std`: Unusual amount
- `UID_encoded_P_emaildomain_ct`: Multiple email domains
- `UID_encoded_id_02_ct`: Multiple devices/IPs

## Performance Metrics

| Model | Validation AUC |
|-------|----------------|
| LightGBM | 0.9466 |
| XGBoost | 0.9475 |
| CatBoost | 0.9367 |
| **Ensemble (Rank Avg)** | **0.9483** |

---

**For detailed implementation and code walkthrough, see [fraud_detection_modeling.ipynb](fraud_detection_modeling.ipynb)**
