
# **Project Report — Deep Learning Bitcoin LOB**

*Submitted by: Aayush Gupta, Prisha Gupta, Punnag Choudhury, Sehajpreet Kaur*


---

# **Deep Learning for Bitcoin Price Movement Prediction Using Limit Order Book Data**

This project explores deep learning models to predict short-term Bitcoin price movements using high-frequency Limit Order Book (LOB) data. It focuses on extracting meaningful signals from noisy crypto microstructure data and proving whether deep learning can produce profitable trading strategies.

---

# **1. Introduction and Motivation**

## **Financial Context**

Cryptocurrency exchanges produce enormous microstructure data each millisecond.
A Limit Order Book (LOB):

* Contains all buy/sell orders at different price levels
* Reflects *real-time market intentions*
* Offers more information than typical OHLCV charts

Bitcoin is ideal for LOB prediction due to:

* 24/7 trading
* High volatility
* Deep liquidity
* Transparent order flow

However, this also means **extreme noise and complexity**.

---

# **2. Core Challenges**

* **Low signal-to-noise ratio** — Most updates reflect random fluctuations.
* **Temporal dependencies** — Price movements depend on events many steps before.
* **Class imbalance** — "Flat" movement dominates.
* **Non-stationarity** — Market regimes shift rapidly.
* **Overfitting risk** — Deep models may memorize noise.

---

# **3. Research Objectives**

* Build deep learning systems for high-frequency LOB data
* Engineer robust labels that reduce noise
* Compare classical ML vs deep learning
* Test strategies using realistic trading simulation
* Identify key determinants of predictive success

---

# **4. Literature Review**

## **DeepLOB (2018)**

* CNN + LSTM hybrid
* First deep learning architecture specifically for LOBs
* Strong results but sensitive to hyperparameters and volatile markets

## **Better Inputs Matter More (2025)**

* Feature engineering > model depth
* Even simple models succeed with clean inputs
* Drawback: highly dependent on preprocessing quality

## **Digital Asset Limit Order Books (2020)**

* Crypto-focused
* Temporal CNNs on Coinbase
* ~47% accuracy (still meaningful for HFT)

### **Research Gaps Identified**

1. Labels in prior research captured noise, not true movement
2. Lack of ternary ("flat") classification
3. Many models were never tested in trading simulations
4. Underuse of microstructure features (imbalance, depth, spreads)

---

# **5. Dataset and Data Engineering**

## **Data Source: Binance BTC/USDT**

Update frequency: **250 ms**

Each LOB snapshot contains:

* 10 bid prices + volumes
* 10 ask prices + volumes
* Timestamp
  → **40 features** per timestep

---

## **Temporal Windowing**

Each input = **20 consecutive LOB snapshots** (~5 seconds)

### Formal:

```
X_t = [LOB_{t−19}, ..., LOB_t]
```

---

## **Feature Engineering**

* **Mid-Price:**
  ((best_bid + best_ask)/2)
* **Bid-Ask Spread**
* **Order Imbalance:**
  ((bid_vol − ask_vol)/(bid_vol + ask_vol))
* **Depth Ratios**
* **Weighted Mid-Price**

---

## **Savitzky–Golay Filtering**

* Smooths microstructure noise
* Preserves trend shape
* Critical for stable label generation

---

# **6. Label Engineering (Most Important Innovation)**

## **Binary Labels**

* **1 = Up**, **0 = Down**

## **Ternary Labels**

* **2 = Up**
* **1 = Flat**
* **0 = Down**

### Return formula:

[
Return = \frac{Mid_{t+H} - Mid_t}{Mid_t}
]

### Best configuration found:

* **H = 10 timesteps (~2.5 sec)**
* **Threshold = 0.0005 (0.05%)**

This setup removed noise and balanced classes.

---

# **7. Class Imbalance Solutions**

* Threshold tuning (worked well)
* Class weights (reduced performance)
* Better evaluation metrics (ROC-AUC, PR curves)

---

# **8. Models Used**

## **Baselines**

* **Logistic Regression** — captures linear relationships
* **XGBoost** — strong performance on tabular data

## **DeepLOB Architecture**

### CNN Layers:

* Extract spatial structure from LOB
* Causal 1D CNN
* BatchNorm + ReLU

### LSTM Layer:

* Captures temporal evolution
* 128 units + dropout

### Dense Head:

* Dropout (0.3)
* Softmax

## **Training Setup**

* Optimizer: **Adam (0.001)**
* Loss: **Cross-entropy with label smoothing**
* Regularization: BN, dropout, early stopping, LR scheduler

## **Transformer Architecture** (Experimental)

* Multi-head attention
* Positional embeddings
* But: **overfitted noise** → underperformed DeepLOB

---

# **9. Experimental Results**

## **Binary DeepLOB**

* **Accuracy: 75.8%**
* **ROC-AUC: 0.84**

## **Ternary DeepLOB**

* **97.4% accuracy** (with optimal labels)
* As low as *58%* with bad labels → highlights the importance of label engineering

---

## **Sensitivity Analysis**

### **Horizon (H)**

* Too small (<10): captures noise
* Too large (>100): patterns diffuse
* **Best: H = 10**

### **Threshold (τ)**

* Too small: noise becomes signal
* Too large: discard useful data
* **Best: τ = 0.0005**

---

# **10. Model Comparison Table**

| Model               | Accuracy | ROC-AUC | Notes                     |
| ------------------- | -------- | ------- | ------------------------- |
| Logistic Regression | ~68%     | 0.72    | Strong baseline           |
| XGBoost             | ~72%     | 0.78    | Best ML baseline          |
| Binary DeepLOB      | ~76%     | 0.84    | Best overall              |
| Ternary DeepLOB     | 58–97%   | N/A     | Extremely label-dependent |
| Transformer         | ~65%     | 0.75    | Overfits noise            |

---

# **11. Trading Simulation**

## **Setup**

* Starting capital: **$10,000**
* Transaction cost: **0.1% per trade**
* Slippage: **0.05%**

### **Position Logic**

* Predict "Up" → Long
* Predict "Down" → Short
* Predict "Flat" → Stay out

---

## **Raw Predictions (No Filtering)**

→ **Poor performance**

* Too many trades
* High cost
* Low-quality signals

---

## **Confidence-Based Filtering (Major Breakthrough)**

If model confidence > **0.6**:

```
execute_trade()
```

Else:

```
hold()
```

### Result:

* Trade frequency ↓ 70%
* Costs drop dramatically
* Smooth equity curve
* **Final return: ~12× initial capital**

---

# **12. Key Learnings**

* **Label engineering > model choice**
* CNN-LSTM outperforms Transformers for noisy financial data
* Confidence filtering is essential for real trading
* Class weighting can make performance worse
* Even simple models do well if features are engineered properly

---

# **13. Limitations**

* Only tested on Bitcoin
* Market-state sensitivity
* Simple execution model
* Possible subtle data leakage
* Deep learning needs high compute

---

# **14. Future Work**

* Improved regularized Transformers
* Reinforcement learning for policy optimization
* Multi-asset prediction
* Confidence calibration (Bayesian DL, ensembles)
* Integration of news, sentiment, on-chain metrics
* Real-time production pipeline optimization

---

# **15. Conclusion**

The study demonstrates that profitable trading strategies can be built using:

1. **Savitzky–Golay noise filtering**
2. **Carefully engineered labels (H + threshold)**
3. **Confidence-based prediction filtering**

The model achieved **97.4% accuracy** and **12× ROI** in simulation.

It shows that deep learning **can** extract meaningful signals from noisy financial markets — but success depends more on **data engineering** than on model complexity.



---
[📄 Download DL_report.pdf](sandbox:/Users/prishagupta/Downloads/DL_report.pdf)

