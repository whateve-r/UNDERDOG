# ML Preprocessing Pipeline - Visual Guide

## 🎯 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML PREPROCESSING PIPELINE                    │
│                                                                   │
│  Raw OHLCV  →  Log Returns  →  ADF Test  →  Technical  →  Lag  │
│   Data          (Stationary)   (Validate)   Features      Shift  │
│                                                                   │
│  →  Standardize  →  Drop NaN  →  Ready for ML Training           │
│     (μ=0, σ=1)      (Clean)                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Step-by-Step Transformation

### Input: Raw OHLCV Data

```
timestamp            open     high     low      close    volume
2020-01-01 00:00    1.1000   1.1005   1.0995   1.1002    150
2020-01-01 01:00    1.1002   1.1010   1.1000   1.1008    200
2020-01-01 02:00    1.1008   1.1012   1.1005   1.1010    180
```

**Issues:**
- ❌ Non-stationary (mean/variance change over time)
- ❌ Price levels vary (1.1000 vs 1.5000)
- ❌ No features (just raw prices)

---

### Step 1: Log Returns (Stationarity)

```python
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
```

**Formula:** `R_t = ln(P_t / P_{t-1})`

```
timestamp            close    log_return
2020-01-01 00:00    1.1000   NaN          ← No previous price
2020-01-01 01:00    1.1002   0.000182     ← ln(1.1002/1.1000)
2020-01-01 02:00    1.1010   0.000727     ← ln(1.1010/1.1002)
```

**Benefits:**
- ✅ Stationary (mean ≈ 0, constant variance)
- ✅ Normalized (relative changes, not absolute prices)
- ✅ Time-additive (can sum returns)

---

### Step 2: ADF Test (Validation)

```python
result = adfuller(df['log_return'].dropna())
p_value = result[1]

if p_value < 0.05:
    print("✓ Series IS stationary")
else:
    print("✗ Series NOT stationary - cannot train ML model")
```

**Decision Rule:**
- p-value < 0.05 → **Reject null hypothesis** → Series **IS stationary** ✓
- p-value ≥ 0.05 → **Fail to reject** → Series **NOT stationary** ✗

**Example Output:**
```
ADF Statistic: -15.234
p-value: 0.0001
Critical Values: {'1%': -3.43, '5%': -2.86, '10%': -2.57}

✓ Series IS stationary (p-value = 0.0001)
```

---

### Step 3: Technical Features

```python
df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
df['rsi'] = talib.RSI(df['close'], 14)
df['sma_20'] = talib.SMA(df['close'], 20)
```

**CRITICAL:** All indicators are LAGGED by 1 period:

```python
df['atr'] = df['atr'].shift(1)    # Prevent look-ahead bias
df['rsi'] = df['rsi'].shift(1)
df['sma_20'] = df['sma_20'].shift(1)
```

```
timestamp            close    log_return   atr      rsi      sma_20
2020-01-01 00:00    1.1000   NaN          NaN      NaN      NaN
2020-01-01 01:00    1.1002   0.000182     NaN      NaN      NaN
...
2020-01-01 14:00    1.1050   0.000320     0.0012   65.3     1.1025
2020-01-01 15:00    1.1055   0.000453     0.0012   67.8     1.1028
```

**Features Added:**
- ATR (volatility measure)
- RSI (momentum indicator)
- SMA (trend indicator)
- Volume ratio
- Seasonality (day_sin, day_cos, hour_sin, hour_cos)

---

### Step 4: Lagged Features (Causality)

```python
for lag in [1, 2, 3, 5, 10]:
    df[f'log_return_lag_{lag}'] = df['log_return'].shift(lag)
    df[f'atr_lag_{lag}'] = df['atr'].shift(lag)
    df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
```

**Causality Principle:**
> Signal at time `t` can ONLY use data from `t-1` or earlier.

```
timestamp            log_return   log_return_lag_1   log_return_lag_5
2020-01-01 00:00    NaN          NaN                NaN
2020-01-01 01:00    0.000182     NaN                NaN
2020-01-01 02:00    0.000727     0.000182           NaN
2020-01-01 03:00    -0.000091    0.000727           NaN
...
2020-01-01 06:00    0.000234     0.000156           0.000182
```

**Why This Matters:**
```python
# ❌ WRONG (look-ahead bias)
signal_t = f(return_t, atr_t, rsi_t)  # Uses FUTURE data (not available at t)

# ✅ CORRECT (causality)
signal_t = f(return_{t-1}, atr_{t-1}, rsi_{t-1})  # Uses PAST data only
```

---

### Step 5: Standardization (μ=0, σ=1)

```python
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
```

**Before Standardization:**
```
log_return    atr        rsi
0.000182      0.001200   65.3
0.000727      0.001180   67.8
-0.000091     0.001250   58.2
```

**After Standardization:**
```
log_return    atr        rsi
0.123         0.456      0.789   ← Mean ≈ 0, Std ≈ 1
1.234         0.234      1.345
-0.567        0.678      -0.789
```

**Why Standardize:**
- ✅ Neural Networks: Gradient descent stability
- ✅ SVM/KNN: Distance metrics require same scale
- ❌ Tree models: NOT needed (Random Forest, XGBoost)

**CRITICAL Pattern:**
```python
# Fit on TRAIN, transform on TEST (no data leakage)
scaler.fit(X_train)          # Learn μ and σ from train
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use TRAIN μ and σ
```

---

### Step 6: Drop NaN Rows

```python
df = df.dropna()
```

**Why NaN Exists:**
- First `n` rows from lagging (e.g., lag_10 → 10 NaN rows)
- First `m` rows from indicators (e.g., ATR(14) → 14 NaN rows)
- Log returns first row (no previous price)

**Example:**
```
Total rows: 10,000
Dropped rows: 250 (from lagging/indicators)
Final rows: 9,750
```

---

### Output: ML-Ready Data

```
timestamp            log_return   atr      rsi      log_return_lag_1   ...   (50+ features)
2020-01-01 20:00    0.123        0.456    0.789    0.234              ...
2020-01-01 21:00    1.234        0.234    1.345    0.123              ...
2020-01-01 22:00    -0.567       0.678    -0.789   1.234              ...
```

**Characteristics:**
- ✅ Stationary (validated with ADF test)
- ✅ Lagged (no look-ahead bias)
- ✅ Standardized (μ=0, σ=1)
- ✅ Clean (no NaN values)
- ✅ Ready for training

---

## 🔄 Train/Test Split Pattern

```
┌──────────────────────────────────────────────────────────┐
│                    COMPLETE DATASET                       │
│                   (2020-01-01 to 2024-12-31)              │
└──────────────────────────────────────────────────────────┘
         │                                       │
         ▼                                       ▼
┌────────────────────────┐        ┌──────────────────────────┐
│   TRAIN (70%)          │        │   TEST (30%)             │
│   2020-01-01           │        │   2023-04-01             │
│   to                   │        │   to                     │
│   2023-03-31           │        │   2024-12-31             │
│                        │        │                          │
│   FIT SCALER HERE ✓    │        │   TRANSFORM ONLY ✓       │
│   (Learn μ and σ)      │        │   (Use train μ and σ)    │
└────────────────────────┘        └──────────────────────────┘
```

**CRITICAL:**
```python
# Step 1: Split data FIRST (before preprocessing)
split_idx = int(len(df) * 0.7)
df_train = df.iloc[:split_idx]
df_test = df.iloc[split_idx:]

# Step 2: Preprocess train (fit=True)
preprocessor = MLPreprocessor()
df_train_processed, _, _ = preprocessor.preprocess(df_train, fit=True)

# Step 3: Preprocess test (fit=False - use train scaler)
df_test_processed, _, _ = preprocessor.preprocess(df_test, fit=False)

# Step 4: Create features/target
X_train, y_train = create_features_target(df_train_processed)
X_test, y_test = create_features_target(df_test_processed)

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Evaluate (OOS performance)
y_pred = model.predict(X_test)
```

---

## 🧠 Key Concepts

### 1. Stationarity

**Non-Stationary Series (BAD):**
```
Price Series: 1.10 → 1.15 → 1.12 → 1.20 → 1.18 → ...
Mean: Changes over time (trending)
Variance: Changes over time (volatility clusters)
```

**Stationary Series (GOOD):**
```
Log Returns: 0.0005 → -0.0003 → 0.0008 → -0.0002 → ...
Mean: Constant (≈ 0)
Variance: Constant (≈ 0.001)
```

**Why It Matters:**
- ML models assume **stationarity** (constant statistics)
- Non-stationary data → **spurious regression** (false patterns)
- Example: "Stock prices correlated with my cat's mood" (both trending)

---

### 2. Look-Ahead Bias

**Wrong (Look-Ahead Bias):**
```python
# Signal at time t uses data from time t (not available yet)
if close[t] > sma[t]:  # ❌ SMA[t] uses close[t]!
    signal = BUY
```

**Correct (Causality):**
```python
# Signal at time t uses data from time t-1 (available)
if close[t-1] > sma[t-1]:  # ✓ Uses past data only
    signal = BUY
```

**Real-World Example:**
```
Time: 10:00 AM
Available data: Up to 09:59 AM
Decision: Can ONLY use 09:59 AM data

❌ Using 10:00 AM close price → Look-ahead bias
✓ Using 09:59 AM close price → Correct
```

---

### 3. Data Leakage

**Wrong (Data Leakage):**
```python
# Fit scaler on ALL data (train + test)
scaler.fit(df_all)  # ❌ Test data leaks into train!

X_train = scaler.transform(df_train)
X_test = scaler.transform(df_test)
```

**Correct (No Leakage):**
```python
# Fit scaler ONLY on train data
scaler.fit(df_train)  # ✓ Test data isolated

X_train = scaler.transform(df_train)
X_test = scaler.transform(df_test)  # Uses train μ and σ
```

**Why It Matters:**
- Data leakage → **Overly optimistic** backtest results
- Real trading → Model sees **unseen data patterns** → Fails

---

## 📈 Expected Results

### ADF Test (Stationarity Validation)

```
✓ Series IS stationary (p-value = 0.0001)

ADF Statistic: -15.234
p-value: 0.0001
Critical Values:
  1%: -3.43
  5%: -2.86
  10%: -2.57
```

**Interpretation:**
- ADF statistic (-15.234) << Critical value (-2.86 at 5%)
- p-value (0.0001) < Significance (0.05)
- **Conclusion**: Series IS stationary ✓

---

### Feature Count

```
Original columns: 5 (OHLCV)
After preprocessing: 50+

New features:
- log_return (1)
- Technical indicators (10+): ATR, RSI, SMA_20, SMA_50, SMA_200, volume_ratio, etc.
- Lagged features (30+): log_return_lag_1...10, atr_lag_1...10, rsi_lag_1...10
- Seasonality (4): day_sin, day_cos, hour_sin, hour_cos
```

---

### Model Performance

```
Train accuracy: 55.2%
Test accuracy: 53.8%

Baseline (always predict UP): 50.5%
Model improvement: +6.5%

✅ Model shows meaningful improvement over baseline
```

**Acceptable Results:**
- Accuracy > 52% (forex is noisy)
- Test accuracy ≈ Train accuracy (no overfitting)
- Calmar Ratio > 2.0 (after backtesting)

---

## ⚠️ Common Mistakes

### 1. Using Raw Prices (Non-Stationary)
```python
# ❌ WRONG
X = df[['close', 'volume']]  # Non-stationary
model.fit(X, y)  # Will fail OOS

# ✅ CORRECT
X = df[['log_return', 'atr', 'rsi']]  # Stationary
model.fit(X, y)  # Robust to OOS
```

### 2. Not Lagging Features
```python
# ❌ WRONG (look-ahead bias)
df['rsi'] = talib.RSI(df['close'], 14)  # Uses current price
signal_t = f(rsi_t)  # Look-ahead bias!

# ✅ CORRECT (causality)
df['rsi'] = talib.RSI(df['close'], 14).shift(1)  # Lagged
signal_t = f(rsi_{t-1})  # No look-ahead
```

### 3. Fitting Scaler on All Data
```python
# ❌ WRONG (data leakage)
scaler.fit(df_all)
X_train = scaler.transform(df_train)
X_test = scaler.transform(df_test)

# ✅ CORRECT (no leakage)
scaler.fit(df_train)
X_train = scaler.transform(df_train)
X_test = scaler.transform(df_test)
```

### 4. Skipping ADF Test
```python
# ❌ WRONG (no validation)
df = log_returns(df)
model.fit(X, y)  # Hope it's stationary?

# ✅ CORRECT (validate)
df = log_returns(df)
adf_result = adf_test(df['log_return'])
if adf_result['is_stationary']:
    model.fit(X, y)  # Confirmed stationary
else:
    raise ValueError("Series not stationary!")
```

---

## 🚀 Usage Example

```python
from underdog.ml.preprocessing import MLPreprocessor

# Load data
df = pd.read_parquet('data/parquet/EURUSD/5min/EURUSD_2020_5min.parquet')
df.set_index('timestamp', inplace=True)

# Preprocess
preprocessor = MLPreprocessor()
df_processed, validation = preprocessor.preprocess(
    df,
    target_col='close',
    lags=[1, 2, 3, 5, 10, 20],
    validate_stationarity=True,
    fit=True
)

# Check stationarity
if validation['adf_test']['is_stationary']:
    print("✓ Data ready for ML training")
    
    # Create features/target
    feature_cols = [col for col in df_processed.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'log_return']]
    
    X = df_processed[feature_cols]
    y = (df_processed['log_return'].shift(-1) > 0).astype(int)
    
    # Train model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
else:
    print("✗ Data NOT stationary - cannot train model")
```

---

## 📚 Further Reading

- **Lopez de Prado (2018)**: *Advances in Financial Machine Learning*
  - Chapter 2: Financial Data Structures (stationarity)
  - Chapter 5: Fractional Differentiation (log returns)
  - Chapter 11: The Dangers of Backtesting (look-ahead bias)

- **statsmodels Documentation**: Augmented Dickey-Fuller test
- **sklearn Documentation**: StandardScaler, train_test_split

---

## ✅ Checklist

Before training ML model:

- [ ] Data is stationary (ADF p-value < 0.05)
- [ ] All features are lagged (no look-ahead bias)
- [ ] Scaler fitted on train only (no data leakage)
- [ ] NaN rows dropped (clean data)
- [ ] Features standardized if using NN/SVM (μ=0, σ=1)
- [ ] Train/test split is temporal (not random shuffle)
- [ ] Test set is truly OOS (no overlap with train)

---

**Last Updated**: 2025-01-XX  
**Author**: UNDERDOG Project  
**Status**: Production-Ready ✅
