"""
Reusable utilities for daily city-wide garbage-report forecasting.

Selected model:
    Negative Binomial regression

Formula:
    reports ~ trend_years
            + C(weekday)
            + C(month)
            + recent3_avg
            + C(is_holiday)

recent3_avg uses only the previous 3 actual daily counts, so evaluation
with this feature is a rolling one-day-ahead forecasting setup.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import nbinom

from .calendar_features import add_greek_holiday_features


CITYWIDE_FORMULA = """
reports ~ trend_years
        + C(weekday)
        + C(month)
        + recent3_avg
        + C(is_holiday)
"""


def prepare_daily_data(
    df: pd.DataFrame,
    issue_type: str = "garbage",
    start_year: int | None = 2020,
    end_year: int | None = 2025,
) -> pd.DataFrame:
    """Convert request-level data into a continuous daily count series."""

    data = df.copy()
    data["reported"] = pd.to_datetime(data["reported"], errors="coerce")
    data = data.loc[data["reported"].notna() & data["issue"].eq(issue_type)].copy()

    if start_year is not None:
        data = data.loc[data["reported"].dt.year >= start_year]

    if end_year is not None:
        data = data.loc[data["reported"].dt.year <= end_year]

    daily = (
        data.set_index("reported")
        .resample("D")
        .size()
        .rename("reports")
        .reset_index()
    )

    daily["weekday"] = daily["reported"].dt.day_name()
    daily["month"] = daily["reported"].dt.month

    daily = add_greek_holiday_features(daily, "reported")

    origin = daily["reported"].min()
    daily["trend_years"] = (daily["reported"] - origin).dt.days / 365.25

    daily["recent3_avg"] = (
        daily["reports"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    return daily


def fit_citywide_model(train_df: pd.DataFrame):
    """Fit the selected city-wide Negative Binomial model."""

    train = train_df.dropna(subset=["recent3_avg", "is_holiday"]).copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        return smf.negativebinomial(
            CITYWIDE_FORMULA,
            data=train,
        ).fit(maxiter=300, disp=False)


def predict_counts(model, data: pd.DataFrame) -> np.ndarray:
    """Return expected report counts."""

    return np.asarray(model.predict(data), dtype=float)


def prediction_interval(
    model,
    data: pd.DataFrame,
    level: float = 0.95,
):
    """Return expected count and NB2 predictive interval."""

    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")

    mu = predict_counts(model, data)
    alpha = float(model.params["alpha"])

    n = 1.0 / alpha
    p = n / (n + mu)
    tail = (1.0 - level) / 2.0

    lower = np.maximum(nbinom.ppf(tail, n, p), 0)
    upper = nbinom.ppf(1.0 - tail, n, p)

    return mu, lower.astype(int), upper.astype(int)


def classify_activity(actual, lower, upper) -> np.ndarray:
    """Classify observations relative to predictive bounds."""

    actual = np.asarray(actual)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    return np.select(
        [actual < lower, actual > upper],
        ["Unusually low", "Unusually high"],
        default="Normal",
    )


# Backward-compatible names used by the existing interval notebook.
fit_nb_recent3_holiday = fit_citywide_model
predict_nb_recent3 = predict_counts
nb_prediction_interval = prediction_interval
