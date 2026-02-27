# Walmart Sales Forecasting

> Can a machine learn the rhythm of retail well enough to see next week before it happens?

**Problem:** Inaccurate weekly sales forecasts lead to inventory mismatches, staffing inefficiencies, and lost revenue — especially during holidays when demand spikes sharply and unpredictably.

**Solution:** A time-series forecasting pipeline trained on 143 weeks of data across 45 stores, using lag features, rolling momentum, seasonality encoding, and store-level interactions. Three models were benchmarked — Random Forest, XGBoost, and LightGBM — with XGBoost achieving the best result: R² of 83.12% and MAE of $495,233.

---

## Dataset

- **Source:** [Walmart Sales Forecast via KaggleHub](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast)
- **Coverage:** 143 weeks of sales data across 45 stores and multiple departments
- **Files Used:**
  - `train.csv` — Historical weekly sales by store and department
  - `features.csv` — Supplementary features (temperature, fuel price, markdowns, CPI, unemployment, holiday flag)
  - `stores.csv` — Store type and size

---

## Project Structure

```
Walmart_Sales_Forecasting.ipynb
│
├── Libraries
├── Dataset
├── Exploratory Data Analysis
├── Feature Engineering
│   ├── Time-Based Features
│   ├── Seasonality
│   ├── Lag Features
│   ├── Rolling Statistics
│   ├── Year-over-Year Difference
│   ├── Momentum Features
│   └── Interaction Features
├── Data Splitting
├── Regression Models
│   ├── Random Forest
│   ├── XGBoost
│   └── LightGBM
└── Comparison Analysis
```

---

## Exploratory Data Analysis

- Mean weekly sales ≈ $47.1M; median ≈ $46.2M — relatively symmetric distribution.
- Sales range from $39.6M (min) to $80.9M (max), with notable holiday spikes.
- Top sales weeks correspond to Thanksgiving and Christmas holiday periods.
- Seasonal decomposition reveals a stable upward trend with clear annual seasonality and irregular residuals.

---

## Feature Engineering

| Feature Group | Examples |
|---|---|
| Log Transformation | Log of weekly sales to stabilize variance |
| Time-Based | Week, month, quarter, year, day of year |
| Seasonality | Is holiday, holiday proximity |
| Lag Features | Sales from previous 1, 2, 4, 52 weeks |
| Rolling Statistics | 4-week and 12-week rolling mean and std |
| Year-over-Year Difference | Sales change from same week last year |
| Momentum | Short vs long-term sales trend |
| Interaction | Store x department, size x type |

---

## Data Splitting

A **chronological (time-aware) split** was used to prevent data leakage:

- **Train:** 2010-02-05 → 2012-04-06 (first 80% of timeline)
- **Test:** 2012-04-13 → 2012-10-26 (last 20%)

---

## Models & Results

### Random Forest

Random Forest builds an ensemble of decision trees, each trained on a random subset of features and data, and averages their predictions. It is robust to overfitting and handles non-linear relationships well, making it a strong baseline for tabular time-series data. Here, momentum features — the ratio of short-term to long-term rolling sales — were the most important predictors, suggesting the model learned that recent sales trajectory matters more than raw calendar position.

| MAE | RMSE | R² |
|---|---|---|
| 795,579 | 1,095,377 | 59.95% |

### LightGBM

LightGBM is a gradient boosting framework that uses histogram-based splitting and leaf-wise tree growth, making it faster and more memory-efficient than traditional implementations. It is particularly effective on large datasets with many features. On the full feature set, the model overfit — training performance was strong but test performance lagged. This was resolved by restricting training to the top 10 features ranked by feature importance, which improved generalization significantly.

| MAE | RMSE | R² |
|---|---|---|
| 999,726.35 | 1,277,574.34 | 45.51% |

### XGBoost

XGBoost is a regularized gradient boosting algorithm that sequentially builds trees to correct the residual errors of prior trees. Its built-in L1/L2 regularization makes it more resistant to overfitting than standard boosting, and it tends to perform well on structured data with engineered features. Given the rich feature set constructed here — particularly lag and rolling statistics — XGBoost had the information it needed to model both short-term momentum and longer seasonal cycles.

| MAE | RMSE | R² |
|---|---|---|
| **495,233** | **711,039** | **83.12%** |

---

## Conclusion

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 795,579 | 1,095,377 | 59.95% |
| LightGBM (feature-selected) | 999,726.35 | 1,277,574.34 | 45.51% |
| **XGBoost** | **495,233** | **711,039** | **83.12%** |

XGBoost was the clear winner across every metric. Random Forest provided a solid baseline at R² of 59.95%. LightGBM, despite feature selection to address overfitting, actually underperformed Random Forest — its R² of 45.51% and MAE of ~$1M suggest the feature reduction may have gone too far, stripping out signal alongside noise. XGBoost's built-in regularization made it the best fit for this problem, handling the high-dimensional engineered feature space without sacrificing predictive power. The 83.12% R² and MAE of ~$495K per week represent a substantial improvement over both baselines and confirm the model has successfully learned the seasonal and momentum patterns driving Walmart's weekly performance.

---

## Libraries Used

```python
numpy, pandas, matplotlib, seaborn
statsmodels (seasonal_decompose)
scikit-learn (RandomForestRegressor, train_test_split, TimeSeriesSplit, GridSearchCV)
xgboost (XGBRegressor)
lightgbm (LGBMRegressor)
kagglehub
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Moamen-Elgohary/Walmart-Sales-Forecast
   ```

2. Install dependencies:
   ```bash
   pip install kagglehub xgboost lightgbm scikit-learn statsmodels pandas numpy matplotlib seaborn
   ```

3. Open the notebook:
   ```bash
   jupyter notebook Walmart_Sales_Forecasting.ipynb
   ```

---

## License

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast) and made available under the **CC0: Public Domain** license. You are free to copy, modify, distribute, and use the data for any purpose without permission.

This project was completed as part of the Elevvo Pathways internship program.