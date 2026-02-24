# FUTURE_ML_01 - Sales Forecasting for Retail Business

> A complete end-to-end time-series sales forecasting project built as part of the **Future ML** program.  
> Uses historical retail transaction data and the **Prophet** forecasting model to predict future sales.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Model Details](#model-details)
- [Results](#results)
- [Visualizations](#visualizations)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Quick Reuse Guide](#quick-reuse-guide)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

Retail teams need forward-looking sales estimates to support planning, inventory management, and business decisions. This project builds a baseline forecasting workflow — from raw order records all the way to future sales predictions.

The notebook covers:

- Data loading, cleaning, and preprocessing
- Time-series preparation from transaction-level records
- Exploratory Data Analysis (EDA) and trend visualization
- Forecasting with **Prophet**
- Model evaluation using **MAE** and **RMSE**
- Export of forecasted values for downstream reporting

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| pandas | Data loading, cleaning, and transformation |
| matplotlib | Visualizations and trend plots |
| seaborn | Advanced plots and heatmaps |
| prophet | Time-series forecasting model |
| scikit-learn | Evaluation metrics (MAE, RMSE) |
| Jupyter Notebook | Interactive development environment |

---

## Dataset

- **Source:** [Sample Sales Data — Kaggle](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data)
- **File in repo:** `sales_data_sample.csv`
- **Description:** Historical retail sales transaction records including order numbers, sales figures, product lines, dates, and more.
- **Key fields used:**
  - `ORDERDATE` — transaction date
  - `SALES` — revenue per transaction

---

## Project Workflow

### 1. Data Loading
Load the sales CSV into a pandas DataFrame for inspection and processing.

### 2. Data Preprocessing
- Parse `ORDERDATE` to datetime format
- Handle missing values and duplicates
- Feature engineering from date columns (Month, Year, Quarter)
- Aggregate transaction-level data into daily sales totals
- Rename columns to Prophet schema: `ds` (date) and `y` (sales)

### 3. Exploratory Data Analysis (EDA)
- Plot sales trends over time
- Generate a correlation matrix heatmap
- Visualize moving averages to capture trends and seasonality

### 4. Train / Test Split
- Chronological (80/20) split to preserve time ordering:
  ```python
  split_index = int(len(sales_data) * 0.8)
  train = sales_data.iloc[:split_index]
  test  = sales_data.iloc[split_index:]
  ```

### 5. Model Building
- Fit **Prophet** on the training set
- Generate predictions for test dates and future dates

### 6. Model Evaluation
- Compute **MAE** and **RMSE** on test set predictions
- Plot **Actual vs Predicted Sales**
- Analyze **Residuals Distribution** and **Residuals vs Predicted**

### 7. Forecast Export
- Export forecast output to `forecasted_sales.csv`
- Columns: `ds`, `yhat`, `yhat_lower`, `yhat_upper`

---

## Model Details

The project uses **Prophet** in two modes:

**Future Forecasting Mode**
```python
future   = model.make_future_dataframe(periods=180)
forecast = model.predict(future)
```

**Evaluation Mode (Train/Test)**
```python
# Fit only on training data, predict on test dates
model.fit(train)
forecast = model.predict(test[['ds']])
```

---

## Results

| Metric | Value |
|---|---|
| MAE (Mean Absolute Error) | **19,815.74** |
| RMSE (Root Mean Square Error) | **23,827.22** |

The notebook demonstrates a working end-to-end forecasting pipeline and exports predictions to `forecasted_sales.csv`.

---

## Visualizations

### Sales Over Time
![Sales Over Time](images/sales_over_time.png)

### Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)

### Moving Average Plot
![Moving Average](images/moving_average_plot.png)

### Sales Forecast
![Sales Forecast](images/sales_forecast.png)

### Seasonality Components
![Seasonality Components](images/seasonality_components.png)

### Actual vs Predicted (Test Set)
![Actual vs Predicted](images/actual_vs_predicted.png)

### Residuals Distribution
![Residuals Distribution](images/residuals_distribution.png)

> All images are saved in the `images/` folder.

---

## Repository Structure

```
FUTURE_ML_01/
├── sales_forecasting.ipynb     # Main Jupyter Notebook (all code + outputs)
├── sales_data_sample.csv       # Raw dataset
├── forecasted_sales.csv        # Exported forecast output
├── images/                     # Saved visualization plots
└── README.md
```

---

## How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/FUTURE_ML_01.git
cd FUTURE_ML_01
```

**2. Install dependencies:**
```bash
pip install pandas matplotlib seaborn prophet scikit-learn jupyter
```

**3. Launch Jupyter and open the notebook:**
```bash
jupyter notebook sales_forecasting.ipynb
```

**4.** Run cells in order to reproduce preprocessing, EDA, training, evaluation, and forecasts.

---

## Quick Reuse Guide

To apply this workflow to updated or new data:

1. Keep the same schema prep (`ORDERDATE` → `ds`, `SALES` → `y`)
2. Re-run the training cells in `sales_forecasting.ipynb`
3. Adjust the forecast horizon by changing `periods` in `make_future_dataframe()`
4. Re-export `forecasted_sales.csv` for downstream reporting or dashboarding

---

## Limitations

- Uses a single primary target signal (`SALES`) with limited exogenous variables
- No holiday or promotional event regressors in the current version
- Baseline model only — no extensive hyperparameter search
- Data quality and seasonality assumptions depend on the source records

---

## Future Improvements

- Add holiday/event regressors for better seasonality modeling
- Compare Prophet with ARIMA, XGBoost, and LSTM baselines
- Implement time-series cross-validation for robustness
- Package the notebook workflow into reproducible Python scripts
- Hyperparameter tuning for Prophet (changepoint scale, seasonality mode, etc.)

---

## Author

**Adnan Rahman Sayeem**  
Connect on [LinkedIn](https://www.linkedin.com/in/adnan-rahman-sayeem/)

---

## Acknowledgements

- [Kaggle](https://www.kaggle.com/) for the sample sales dataset
- The **Future ML** program for the project framework
- Prophet (Meta) for the open-source forecasting library
