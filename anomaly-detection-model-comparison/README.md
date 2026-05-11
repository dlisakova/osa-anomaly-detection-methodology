# Anomaly Detection Model Comparison

Comparison of three models for detecting anomalous drops in hourly sales:

- **LightGBM** (gradient boosting)
- **Temporal Convolutional Network (TCN)**
- **Isolation Forest**

All models were trained and tested on **the same** dataset (top-500 products by sales volume in August + 13 products with synthetic anomalies).

## Folder Contents

- `01_lightgbm_sales_drop.ipynb` — LightGBM + z-score threshold
- `02_tcn_sales_drop.ipynb` — Temporal Convolutional Network
- `03_isolation_forest_sales_drop.ipynb` — Isolation Forest
- `04_model_comparison_diagnostics.ipynb` — Statistical comparison of models and selection of the best one
