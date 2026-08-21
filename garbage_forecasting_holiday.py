"""
Reusable utilities for the garbage-report forecasting project.

Current selected model:
    nb_recent3

Formula:
    reports ~ trend_years + C(weekday) + C(month) + recent3_avg

where recent3_avg is the mean report count over the previous 3 days.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import holidays
import statsmodels.formula.api as smf
from scipy.stats import nbinom


NB_RECENT3_FORMULA = (
    "reports ~ trend_years + C(weekday) + C(month) + recent3_avg"
)

NB_RECENT3_HOLIDAY_FORMULA = (
    "reports ~ trend_years + C(weekday) + C(month) "
    "+ recent3_avg + C(is_holiday)"
)


def prepare_daily_data(
    df: pd.DataFrame,
    issue_type: str = "garbage",
    start_year: int | None = 2020,
    end_year: int | None = 2025,
) -> pd.DataFrame:
    """
    Convert raw report-level data into a continuous daily count series
    with the features required by nb_recent3.

    Required raw columns:
        - reported
        - issue
    """
    data = df.copy()
    data["reported"] = pd.to_datetime(data["reported"], errors="coerce")
    data = data.dropna(subset=["reported"])

    mask = data["issue"].eq(issue_type)

    if start_year is not None:
        mask &= data["reported"].dt.year >= start_year

    if end_year is not None:
        mask &= data["reported"].dt.year <= end_year

    data = data.loc[mask].sort_values("reported").copy()

    daily_counts = (
        data.set_index("reported")
            .resample("D")
            .size()
    )

    daily_df = daily_counts.reset_index(name="reports")

    daily_df["weekday"] = daily_df["reported"].dt.day_name()
    daily_df["month"] = daily_df["reported"].dt.month

    # Greek public-holiday features.
    # The boolean is used by the model; holiday_name is only for inspection.
    min_year = int(daily_df["reported"].dt.year.min())
    max_year = int(daily_df["reported"].dt.year.max())

    greek_holidays = holidays.country_holidays(
        "GR",
        years=range(min_year, max_year + 1),
    )

    report_dates = daily_df["reported"].dt.date

    daily_df["is_holiday"] = report_dates.isin(greek_holidays)
    daily_df["holiday_name"] = report_dates.apply(
        lambda date: greek_holidays.get(date, "")
    )

    trend_origin = daily_df["reported"].min()
    daily_df["trend_years"] = (
        (daily_df["reported"] - trend_origin).dt.days / 365.25
    )

    # Previous 3 days only; current day's target is excluded.
    daily_df["recent3_avg"] = (
        daily_df["reports"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    return daily_df


def fit_nb_recent3(train_df: pd.DataFrame):
    """
    Fit the currently selected Negative Binomial model.
    Rows without recent3_avg are removed automatically.
    """
    train = train_df.dropna(subset=["recent3_avg"]).copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model = smf.negativebinomial(
            NB_RECENT3_FORMULA,
            data=train,
        ).fit(maxiter=300, disp=False)

    return model


def fit_nb_recent3_holiday(train_df: pd.DataFrame):
    """
    Fit nb_recent3 with an additional Greek public-holiday indicator.
    The base nb_recent3 model remains unchanged for fair comparison.
    """
    required = ["recent3_avg", "is_holiday"]
    train = train_df.dropna(subset=required).copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model = smf.negativebinomial(
            NB_RECENT3_HOLIDAY_FORMULA,
            data=train,
        ).fit(maxiter=300, disp=False)

    return model


def predict_nb_recent3(model, data: pd.DataFrame) -> np.ndarray:
    """Return expected counts from a fitted nb_recent3 model."""
    return np.asarray(model.predict(data), dtype=float)


def nb_prediction_interval(
    model,
    data: pd.DataFrame,
    level: float = 0.95,
):
    """
    Return expected count, lower bound, and upper bound for a Negative
    Binomial predictive interval.

    Uses the NB2 variance parameterization:
        Var(Y | X) = mu + alpha * mu^2
    """
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")

    mu = predict_nb_recent3(model, data)
    alpha = float(model.params["alpha"])

    n_param = 1.0 / alpha
    p_param = n_param / (n_param + mu)

    tail = (1.0 - level) / 2.0

    lower = nbinom.ppf(tail, n_param, p_param)
    upper = nbinom.ppf(1.0 - tail, n_param, p_param)

    lower = np.maximum(lower, 0)

    return mu, lower.astype(int), upper.astype(int)


def classify_activity(
    actual,
    lower,
    upper,
):
    """
    Classify actual counts relative to predictive bounds.
    """
    actual = np.asarray(actual)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    return np.select(
        [actual < lower, actual > upper],
        ["Unusually low", "Unusually high"],
        default="Normal",
    )
