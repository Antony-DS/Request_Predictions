"""
Reusable utilities for daily garbage-report forecasting by geographic area.

Selected area-level model:
    Poisson GLM

Formula:
    reports ~ C(area_name)
            + trend_years
            + C(weekday)
            + C(month)
            + C(is_holiday)
            + recent7_avg

recent7_avg uses only the previous 7 actual days for the same area,
so evaluation with it is a rolling one-day-ahead setup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_poisson_deviance,
)

from .calendar_features import add_greek_holiday_features

AREA_FORMULA = """
reports ~ C(area_name)
        + trend_years
        + C(weekday)
        + C(month)
        + C(is_holiday)
        + recent7_avg
"""


def prepare_area_panel(
    df: pd.DataFrame,
    issue_type: str = "garbage",
    start_year: int = 2020,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Convert mapped request-level data into a complete area x day panel."""

    data = df.copy()
    data["reported"] = pd.to_datetime(data["reported"], errors="coerce")

    data = data.loc[
        data["reported"].notna()
        & data["area_name"].notna()
        & data["issue"].eq(issue_type)
        & data["reported"].dt.year.between(start_year, end_year)
    ].copy()

    data["date"] = data["reported"].dt.normalize()

    counts = (
        data.groupby(["area_name", "date"])
        .size()
        .rename("reports")
    )

    dates = pd.date_range(data["date"].min(), data["date"].max(), freq="D")
    areas = sorted(data["area_name"].unique())

    index = pd.MultiIndex.from_product(
        [areas, dates],
        names=["area_name", "date"],
    )

    panel = (
        counts.reindex(index, fill_value=0)
        .rename("reports")
        .reset_index()
    )

    panel["reports"] = panel["reports"].astype(int)
    panel["weekday"] = panel["date"].dt.day_name()
    panel["month"] = panel["date"].dt.month

    panel = add_greek_holiday_features(panel, "date")

    origin = panel["date"].min()
    panel["trend_years"] = (panel["date"] - origin).dt.days / 365.25

    panel = panel.sort_values(["area_name", "date"]).reset_index(drop=True)
    panel["recent7_avg"] = (
        panel.groupby("area_name")["reports"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )

    return panel


def area_sparsity_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Return one row per area with simple activity/sparsity statistics."""

    return (
        panel.groupby("area_name")
        .agg(
            total_reports=("reports", "sum"),
            avg_reports_per_day=("reports", "mean"),
            zero_day_pct=("reports", lambda x: 100 * (x == 0).mean()),
            active_day_pct=("reports", lambda x: 100 * (x > 0).mean()),
        )
        .reset_index()
        .sort_values("total_reports", ascending=False)
    )


def fit_area_poisson(
    train_df: pd.DataFrame,
    formula: str = AREA_FORMULA,
):
    """Fit the selected shared area-level Poisson GLM."""

    train = train_df.dropna(subset=["recent7_avg"]).copy()

    return smf.glm(
        formula=formula,
        data=train,
        family=sm.families.Poisson(),
    ).fit()


def evaluate_area_model(
    panel: pd.DataFrame,
    formula: str = AREA_FORMULA,
    test_years=(2023, 2024, 2025),
):
    """Expanding-window, rolling one-day-ahead evaluation."""

    metrics = []
    predictions = []

    for test_year in test_years:
        train = panel.loc[
            panel["date"].dt.year < test_year
        ].dropna(subset=["recent7_avg"]).copy()

        test = panel.loc[
            panel["date"].dt.year == test_year
        ].dropna(subset=["recent7_avg"]).copy()

        model = fit_area_poisson(train, formula=formula)

        pred = np.asarray(model.predict(test), dtype=float)
        pred_safe = np.maximum(pred, 1e-12)
        actual = test["reports"].to_numpy()

        zero = actual == 0
        active = actual > 0

        metrics.append({
            "year": test_year,
            "MAE": mean_absolute_error(actual, pred),
            "RMSE": np.sqrt(mean_squared_error(actual, pred)),
            "Poisson_deviance": mean_poisson_deviance(actual, pred_safe),
            "zero_day_MAE": mean_absolute_error(actual[zero], pred[zero]),
            "active_day_MAE": mean_absolute_error(actual[active], pred[active]),
            "mean_error": np.mean(actual - pred),
            "mean_actual": np.mean(actual),
            "mean_predicted": np.mean(pred),
            "observed_zero_pct": 100 * zero.mean(),
            "expected_zero_pct": 100 * np.exp(-pred).mean(),
        })

        out_test = test.copy()
        out_test["predicted"] = pred
        out_test["residual"] = actual - pred
        out_test["pearson_residual"] = (
            (actual - pred) / np.sqrt(pred_safe)
        )
        out_test["test_year"] = test_year

        predictions.append(out_test)

    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def poisson_dispersion(model) -> pd.Series:
    """Return Pearson and deviance dispersion summaries."""

    return pd.Series({
        "pearson_dispersion": (
            np.sum(model.resid_pearson ** 2) / model.df_resid
        ),
        "deviance_per_df": model.deviance / model.df_resid,
    })


def zero_calibration(predictions: pd.DataFrame) -> pd.Series:
    """Compare observed and Poisson-expected zero frequencies."""

    observed = 100 * (predictions["reports"] == 0).mean()
    expected = 100 * np.exp(-predictions["predicted"]).mean()

    return pd.Series({
        "observed_zero_pct": observed,
        "expected_zero_pct": expected,
        "difference_pp": observed - expected,
    })


def residual_autocorrelation(
    predictions: pd.DataFrame,
    lags=(1, 2, 7, 14),
) -> pd.DataFrame:
    """Summarize Pearson-residual autocorrelation across areas."""

    rows = []
    ordered = predictions.sort_values(["area_name", "date"])

    for lag in lags:
        correlations = []

        for _, group in ordered.groupby("area_name"):
            r = group["pearson_residual"]
            corr = r.corr(r.shift(lag))

            if pd.notna(corr):
                correlations.append(corr)

        rows.append({
            "lag": lag,
            "mean_area_correlation": np.mean(correlations),
            "median_area_correlation": np.median(correlations),
            "min_area_correlation": np.min(correlations),
            "max_area_correlation": np.max(correlations),
        })

    return pd.DataFrame(rows)


def calibration_table(
    predictions: pd.DataFrame,
    bins: int = 10,
) -> pd.DataFrame:
    """Compare average predicted and observed counts by prediction quantile."""

    data = predictions.copy()
    data["prediction_bin"] = pd.qcut(
        data["predicted"],
        q=bins,
        duplicates="drop",
    )

    return (
        data.groupby("prediction_bin", observed=False)
        .agg(
            observations=("reports", "size"),
            mean_predicted=("predicted", "mean"),
            mean_actual=("reports", "mean"),
        )
        .reset_index()
    )
