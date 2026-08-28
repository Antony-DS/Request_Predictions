# Municipal Garbage Report Forecasting – Patras

A statistical forecasting project for predicting the daily number of garbage-related citizen reports in the Municipality of Patras, Greece.

The project analyzes the temporal behaviour of historical municipal request data, develops and evaluates count-based forecasting models, and extends the analysis from city-wide forecasts to geographic area-level predictions.

The main objective is to estimate future reporting workload while accounting for **long-term trend, weekly and monthly seasonality, recent activity, holidays, and temporal dependence**.

---

## Project Overview

Citizen-report counts are discrete, highly variable, and influenced by recurring temporal patterns.

The project therefore follows a forecasting workflow designed specifically for **count data**:

* exploratory time-series analysis
* trend and seasonality analysis
* Poisson overdispersion diagnostics
* baseline forecasting models
* Negative Binomial regression
* lag and rolling-history features
* chronological out-of-sample validation
* recursive multi-step forecasting
* prediction-interval calibration
* geographic area-level forecasting
* forecast visualization

The analysis focuses primarily on **garbage-related reports**.

---

## Why Negative Binomial Regression?

A Poisson model is a natural starting point for count data, but it assumes that the conditional variance is approximately equal to the conditional mean.

The garbage-report data shows substantial **overdispersion**.

For the analyzed daily series:

* Mean daily reports: approximately **19.4**
* Variance: approximately **168.9**
* Variance / Mean ratio: approximately **8.7**

Conditional Poisson diagnostics also indicated substantial remaining dispersion.

Because of this, **Negative Binomial regression** was preferred for the main city-wide forecasting model.

---

## Exploratory Analysis

The first stage examines the temporal structure of the report series.

The analysis investigates:

* long-term changes in reporting volume
* weekday effects
* month-of-year seasonality
* autocorrelation
* recurring weekly patterns
* behaviour after removing trend effects

Special attention is given to distinguishing genuine temporal structure from changes that may also reflect increased usage of the reporting platform over time.

Notebook:

```text
01_eda_seasonality.ipynb
```

---

## City-Wide Forecasting Model

The selected city-wide model is a **Negative Binomial regression** using a combination of calendar, trend, and recent-history features.

Conceptually:

```text
Daily Reports
    ~ Log Time Trend
    + Weekday
    + Month
    + Greek Holiday Indicator
    + Recent 3-Day Average
    + 7-Day Lag
```

### Features

| Feature                  | Purpose                                                                          |
| ------------------------ | -------------------------------------------------------------------------------- |
| **Log trend**            | Captures long-term growth while allowing the rate of growth to flatten over time |
| **Weekday**              | Captures recurring weekly behaviour                                              |
| **Month**                | Captures broader seasonal differences                                            |
| **Holiday indicator**    | Accounts for Greek public holidays                                               |
| **Recent 3-day average** | Represents short-term reporting activity                                         |
| **Lag 7**                | Captures weekly temporal dependence                                              |

The log trend is defined using a transformed time variable rather than assuming indefinitely constant linear growth.

---

## Model Selection

Multiple alternatives are evaluated using chronological out-of-sample testing.

These include simple baselines such as:

* historical mean
* weekday mean
* previous week's value
* rolling 7-day mean
* average of the same weekday over the previous four weeks

Several Negative Binomial specifications are also compared using different combinations of:

* lag 1
* lag 7
* recent 3-day activity
* previous same-weekday activity
* linear or logarithmic trend
* weekday and month effects
* holiday effects

Models are evaluated primarily using:

* **MAE**
* **RMSE**
* estimated Negative Binomial dispersion

The chronological evaluation uses later years as unseen test periods rather than randomly splitting the time series.

For the tested 2023–2025 periods, the `recent3 + lag7` Negative Binomial specification with holiday effects achieved an average:

| Metric   |                Result |
| -------- | --------------------: |
| **MAE**  | **6.642 reports/day** |
| **RMSE** | **8.995 reports/day** |

Notebook:

```text
02_forecasting_models.ipynb
```

---

## Forecasting Strategy

Two forecasting modes are considered.

### Daily-Updated Forecasting

This represents the realistic operational scenario.

As new real observations become available, lag-based predictors are updated using the actual report counts.

```text
Actual observations
      ↓
Update recent-history features
      ↓
Predict next day
      ↓
Receive new actual value
      ↓
Update again
```

This mode is appropriate for continuously updated short-term forecasts.

---

### Fully Recursive Forecasting

For longer-range planning, future actual observations are unavailable.

Predicted values must therefore be fed back into the model to generate future lag and rolling-history features.

```text
Historical Data
      ↓
Predict Day 1
      ↓
Use prediction in future features
      ↓
Predict Day 2
      ↓
...
```

This creates a more difficult forecasting problem because prediction errors may propagate forward through the recursive features.

---

## 2026 Out-of-Sample Evaluation

Data from 2026 is kept separate from the historical model-development period and used to evaluate the forecasting workflow on genuinely later observations.

Using available 2026 observations:

| Forecasting Mode    |       MAE |       RMSE |
| ------------------- | --------: | ---------: |
| **Daily Updated**   | **6.314** |  **8.919** |
| **Fully Recursive** | **7.035** | **10.098** |

As expected, the daily-updated model performs better because it can incorporate newly observed report counts.

The fully recursive version represents the harder long-range planning scenario.

---

## Prediction Intervals

Point forecasts alone do not communicate the uncertainty associated with future daily report volumes.

The project therefore also explores **Negative Binomial prediction intervals** and **conformal calibration**.

The calibration workflow uses a chronological split:

```text
Training
2020–2023
    ↓
Calibration
2024
    ↓
Final Evaluation
2025
```

A target interval level of **95%** is used.

Conformal calibration evaluates how often observed values fall inside the raw Negative Binomial intervals and learns an additional widening adjustment from the calibration period.

This allows forecast uncertainty to be evaluated separately from point-prediction accuracy.

Notebook:

```text
03_prediction_intervals_conformal.ipynb
```

---

## Geographic Area-Level Forecasting

The project also extends the city-wide model to predict report activity across individual municipal areas.

Requests containing geographic coordinates are assigned to municipal polygons using a spatial join.

```text
Request Coordinates
        ↓
Municipal Area Polygons
        ↓
Spatial Mapping
        ↓
Daily Report Count per Area
```

The retained area-level model is a shared **Poisson Generalized Linear Model** containing:

* area-specific baseline effects
* linear time trend
* weekday
* month
* holiday indicator
* recent 7-day activity within each area

Unlike the city-wide model, the area-level data contains a large number of **zero-count area-days**, making it a different modeling problem.

### 2026 Area-Level Evaluation

Evaluation on the available 2026 observations produced:

| Metric               |                        Result |
| -------------------- | ----------------------------: |
| **MAE**              | **0.3632 reports / area-day** |
| **RMSE**             |                    **0.6900** |
| **Poisson Deviance** |                    **0.7558** |

Notebook:

```text
04_area_forecasting.ipynb
```

---

## Recursive Backtesting

A dedicated backtesting notebook evaluates what would have happened if a future period had been completely unknown at prediction time.

The model is not allowed to use actual observations from the backtest period when constructing future lag features.

This helps prevent **temporal leakage** and provides a more realistic estimate of long-range forecasting performance.

The notebook also compares:

```text
Fully Recursive Forecast
vs.
Daily-Updated Forecast
```

Notebook:

```text
05_citywide_backtesting.ipynb
```

---

## Forecast Visualization

The final stage produces presentation-oriented visualizations of:

* historical daily report volumes
* observed 2026 reports
* daily-updated forecasts
* fully recursive future forecasts
* smoothed historical and forecast trends

Notebook:

```text
06_forecast_visualizations.ipynb
```

---

## Repository Structure

```text
Request_Predictions/
│
├── data/
│   ├── requests.csv
│   ├── 2026requests.csv
│   └── area_boundaries.csv
│
├── src/
│   ├── citywide_forecasting.py
│   ├── area_forecasting.py
│   ├── area_mapping.py
│   ├── calendar_features.py
│   └── __init__.py
│
├── 01_eda_seasonality.ipynb
│   └── Trend, seasonality and autocorrelation analysis
│
├── 02_forecasting_models.ipynb
│   └── Baseline and Negative Binomial model comparison
│
├── 03_prediction_intervals_conformal.ipynb
│   └── Prediction intervals and conformal calibration
│
├── 04_area_forecasting.ipynb
│   └── Geographic area-level forecasting
│
├── 05_citywide_backtesting.ipynb
│   └── Recursive and daily-updated out-of-sample testing
│
└── 06_forecast_visualizations.ipynb
    └── Final forecasting visualizations
```

---

## Technologies

* Python
* pandas
* NumPy
* statsmodels
* SciPy
* Matplotlib
* GeoPandas
* scikit-learn
* python-holidays
* Jupyter Notebook

### Main Techniques

* Time-Series Analysis
* Count Regression
* Negative Binomial Regression
* Poisson Regression
* Generalized Linear Models
* Lag Features
* Rolling Features
* Seasonal Features
* Recursive Forecasting
* Chronological Backtesting
* Conformal Prediction
* Geospatial Mapping

---

## Workflow

```text
Historical Municipal Reports
          ↓
Data Preparation
          ↓
Exploratory Time-Series Analysis
          ↓
Trend & Seasonality Detection
          ↓
Poisson Overdispersion Check
          ↓
Baseline Model Comparison
          ↓
Negative Binomial Regression
          ↓
Chronological Validation
          ↓
Prediction-Interval Calibration
          ↓
2026 Out-of-Sample Testing
          ↓
City-Wide / Area-Level Forecasts
          ↓
Operational Forecast Visualizations
```

---

## Context

This project was developed using real-world municipal citizen-report data from Patras, Greece.

The aim is to demonstrate how statistical forecasting methods can be applied to operational municipal data while accounting for the practical characteristics of the problem: count-valued observations, overdispersion, seasonality, temporal dependence, geographic variation, and the difference between short-term updated forecasts and fully recursive long-range predictions.

The forecasts should be interpreted as estimates of **reported demand**, rather than direct estimates of the underlying number of real-world garbage incidents, since reporting behaviour and platform usage can also change over time.
