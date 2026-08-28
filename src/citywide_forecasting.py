"""
City-wide garbage-report forecasting utilities.

Selected model:
    Negative Binomial
    reports ~ log_trend + C(weekday) + C(month)
            + recent3_avg + lag7 + C(is_holiday)

where log_trend = log1p(trend_years / 3).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

import holidays
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASELINES = ["mean", "weekday_mean", "lag7", "rolling7", "same_weekday_4wk"]
SELECTED_MODEL = "nb_recent3_lag7"
SELECTED_TREND_MODE = "log"
SELECTED_LOG_TREND_DIVISOR = 3
SELECTED_INCLUDE_HOLIDAYS = True


def add_features(
    daily: pd.DataFrame,
    *,
    origin: pd.Timestamp | None = None,
    log_divisor: int = SELECTED_LOG_TREND_DIVISOR,
) -> pd.DataFrame:
    """Add calendar, holiday, trend and leakage-safe lag features."""
    if log_divisor <= 0:
        raise ValueError("log_divisor must be positive.")

    out = daily.copy()
    out["reported"] = pd.to_datetime(out["reported"], errors="coerce")
    out = out.sort_values("reported").reset_index(drop=True)

    out["weekday"] = out["reported"].dt.day_name()
    out["month"] = out["reported"].dt.month

    years = sorted(out["reported"].dropna().dt.year.unique())
    gr = holidays.country_holidays("GR", years=years)
    dates = out["reported"].dt.date
    out["is_holiday"] = dates.isin(gr)
    out["holiday_name"] = dates.map(lambda d: gr.get(d, "") if pd.notna(d) else "")

    origin = pd.Timestamp(origin if origin is not None else out["reported"].min())
    out["trend_years"] = (out["reported"] - origin).dt.days / 365.25
    out["log_trend"] = np.log1p(out["trend_years"] / log_divisor)

    r = out["reports"]
    out["lag1"] = r.shift(1)
    out["lag7"] = r.shift(7)
    out["recent3_avg"] = r.shift(1).rolling(3).mean()
    out["rolling7_avg"] = r.shift(1).rolling(7).mean()
    out["same_weekday_4wk_avg"] = (
        r.shift(7) + r.shift(14) + r.shift(21) + r.shift(28)
    ) / 4
    return out


def prepare_daily_data(
    raw_df: pd.DataFrame,
    *,
    issue_type: str = "garbage",
    start_year: int = 2020,
    end_year: int = 2025,
    log_divisor: int = SELECTED_LOG_TREND_DIVISOR,
) -> pd.DataFrame:
    """Convert request rows to daily counts and add model features."""
    data = raw_df.copy()
    data["reported"] = pd.to_datetime(data["reported"], errors="coerce")

    mask = data["reported"].notna() & data["reported"].dt.year.between(start_year, end_year)
    if "issue" in data.columns:
        mask &= data["issue"].eq(issue_type)
    data = data.loc[mask].sort_values("reported")

    if data.empty:
        raise ValueError("No rows remain after filtering.")

    daily = (
        data.set_index("reported")
        .resample("D")
        .size()
        .rename("reports")
        .reset_index()
    )
    return add_features(daily, origin=daily["reported"].min(), log_divisor=log_divisor)


def prepare_actual_year(
    raw_df: pd.DataFrame,
    year: int,
    *,
    issue_type: str = "garbage",
) -> pd.DataFrame:
    """Daily actual counts from Jan 1 through the last available date."""
    data = raw_df.copy()
    data["reported"] = pd.to_datetime(data["reported"], errors="coerce")

    mask = data["reported"].notna() & data["reported"].dt.year.eq(year)
    if "issue" in data.columns:
        mask &= data["issue"].eq(issue_type)
    data = data.loc[mask].sort_values("reported")

    if data.empty:
        raise ValueError(f"No rows found for {year}.")

    dates = pd.date_range(f"{year}-01-01", data["reported"].max().normalize(), freq="D")
    counts = (
        data.set_index("reported")
        .resample("D")
        .size()
        .reindex(dates, fill_value=0)
    )
    return counts.rename("reports").rename_axis("reported").reset_index()


def _trend_name(mode: str) -> str | None:
    mode = mode.lower()
    if mode == "linear":
        return "trend_years"
    if mode == "log":
        return "log_trend"
    if mode in {"none", "no_trend"}:
        return None
    raise ValueError("trend_mode must be 'linear', 'log', or 'none'.")


def get_nb_formulas(trend_mode: str = SELECTED_TREND_MODE) -> dict[str, str]:
    """Candidate NB models used by notebook 02."""
    trend = _trend_name(trend_mode)
    base = "reports ~ C(weekday) + C(month)"
    no_weekday = "reports ~ C(month) + recent3_avg + lag7"

    if trend:
        base = f"reports ~ {trend} + C(weekday) + C(month)"
        no_weekday = f"reports ~ {trend} + C(month) + recent3_avg + lag7"

    formulas = {
        "nb_base": base,
        "nb_lag1": base + " + lag1",
        "nb_recent3": base + " + recent3_avg",
        "nb_recent3_lag7": base + " + recent3_avg + lag7",
        "nb_recent3_lag7_no_weekday": no_weekday,
        "nb_recent3_lag7_no_weekday_no_trend": "reports ~ C(month) + recent3_avg + lag7",
        "nb_lag1_lag7": base + " + lag1 + lag7",
        "nb_4wk": base + " + same_weekday_4wk_avg",
    }
    if trend:
        formulas["nb_quadratic"] = base + f" + I({trend} ** 2)"
    return formulas


def model_formula(
    model_name: str,
    *,
    trend_mode: str = SELECTED_TREND_MODE,
    include_holidays: bool = True,
) -> str:
    """Return one complete Negative Binomial formula."""
    formulas = get_nb_formulas(trend_mode)
    if model_name not in formulas:
        raise ValueError(f"Unknown NB model {model_name!r}.")
    formula = formulas[model_name]
    return formula + " + C(is_holiday)" if include_holidays else formula


def _required(formula: str) -> list[str]:
    cols = ["lag1", "lag7", "recent3_avg", "rolling7_avg", "same_weekday_4wk_avg"]
    return [c for c in cols if c in formula]


def fit_nb(
    daily: pd.DataFrame,
    *,
    model_name: str = SELECTED_MODEL,
    trend_mode: str = SELECTED_TREND_MODE,
    include_holidays: bool = True,
):
    """Fit one Negative Binomial model."""
    formula = model_formula(
        model_name,
        trend_mode=trend_mode,
        include_holidays=include_holidays,
    )
    fit_data = daily.dropna(subset=_required(formula)).copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return smf.negativebinomial(formula, data=fit_data).fit(maxiter=300, disp=False)


def poisson_overdispersion(
    daily: pd.DataFrame,
    *,
    end_year: int = 2024,
    trend_mode: str = "linear",
) -> pd.Series:
    """Conditional Poisson dispersion after trend + weekday + month."""
    trend = _trend_name(trend_mode)
    formula = "reports ~ C(weekday) + C(month)"
    if trend:
        formula = f"reports ~ {trend} + C(weekday) + C(month)"

    train = daily[daily["reported"].dt.year <= end_year]
    model = smf.glm(formula, data=train, family=sm.families.Poisson()).fit()

    mean, var = train["reports"].mean(), train["reports"].var()
    return pd.Series({
        "raw_mean": mean,
        "raw_variance": var,
        "variance_mean_ratio": var / mean,
        "pearson_dispersion": model.pearson_chi2 / model.df_resid,
        "deviance_per_df": model.deviance / model.df_resid,
        "pearson_p_value": chi2.sf(model.pearson_chi2, model.df_resid),
    })


def _expand_models(requested, formulas):
    requested = [requested] if isinstance(requested, str) else list(requested)
    if not requested:
        raise ValueError("models_to_test cannot be empty.")

    valid = BASELINES + list(formulas)
    out = []

    for name in requested:
        if name == "all":
            if len(requested) != 1:
                raise ValueError("Use 'all' by itself.")
            out.extend(valid)
        elif name == "baselines":
            out.extend(BASELINES)
        elif name in valid:
            out.append(name)
        else:
            raise ValueError(f"Unknown model {name!r}.")

    return list(dict.fromkeys(out))


def evaluate_models(
    daily: pd.DataFrame,
    *,
    models_to_test,
    holiday_mode: str = "with",
    test_years: Sequence[int] = (2023, 2024, 2025),
    trend_mode: str = SELECTED_TREND_MODE,
):
    """Expanding-window validation. Returns yearly, average and OOS predictions."""
    formulas = get_nb_formulas(trend_mode)
    models = _expand_models(models_to_test, formulas)

    holiday_options = {"without": [False], "with": [True], "both": [False, True]}
    if holiday_mode not in holiday_options:
        raise ValueError("holiday_mode must be 'without', 'with', or 'both'.")

    specs = []
    for name in models:
        if name in formulas:
            for use_holiday in holiday_options[holiday_mode]:
                version = "with" if use_holiday else "without"
                specs.append((
                    name,
                    f"{name} | {version} holidays",
                    version,
                    model_formula(name, trend_mode=trend_mode, include_holidays=use_holiday),
                ))
        else:
            specs.append((name, name, "not applicable", None))

    results, predictions = [], []

    for year in test_years:
        train = daily[daily["reported"].dt.year < year].copy()
        test = daily[daily["reported"].dt.year == year].copy()

        for base_name, result_name, holiday_version, formula in specs:
            alpha = np.nan

            if base_name == "mean":
                pred = pd.Series(train["reports"].mean(), index=test.index)
            elif base_name == "weekday_mean":
                pred = test["weekday"].map(train.groupby("weekday")["reports"].mean())
            elif base_name == "lag7":
                pred = test["lag7"]
            elif base_name == "rolling7":
                pred = test["rolling7_avg"]
            elif base_name == "same_weekday_4wk":
                pred = test["same_weekday_4wk_avg"]
            else:
                required = _required(formula)
                tr = train.dropna(subset=required)
                te = test.dropna(subset=required)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    fitted = smf.negativebinomial(
                        formula, data=tr
                    ).fit(maxiter=300, disp=False)

                pred = pd.Series(fitted.predict(te), index=te.index)
                alpha = fitted.params.get("alpha", np.nan)

            idx = test.index.intersection(pred.dropna().index)
            actual, predicted = test.loc[idx, "reports"], pred.loc[idx]

            results.append({
                "Test Year": year,
                "Model": result_name,
                "Base Model": base_name,
                "Holiday Version": holiday_version,
                "MAE": mean_absolute_error(actual, predicted),
                "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
                "Alpha": alpha,
                "N": len(idx),
            })

            predictions.extend({
                "reported": test.loc[i, "reported"],
                "reports": test.loc[i, "reports"],
                "weekday": test.loc[i, "weekday"],
                "month": test.loc[i, "month"],
                "Model": result_name,
                "Base Model": base_name,
                "Holiday Version": holiday_version,
                "prediction": float(pred.loc[i]),
                "Test Year": year,
            } for i in idx)

    yearly = pd.DataFrame(results)
    average = (
        yearly.groupby(["Model", "Base Model", "Holiday Version"], as_index=False)
        .agg(
            Mean_MAE=("MAE", "mean"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_Alpha=("Alpha", "mean"),
        )
        .sort_values("Mean_MAE")
        .reset_index(drop=True)
    )
    return yearly, average, pd.DataFrame(predictions)


def evaluate_log_divisors(
    daily: pd.DataFrame,
    divisors: Iterable[int],
    *,
    model_name: str = SELECTED_MODEL,
    test_years: Sequence[int] = (2023, 2024, 2025),
    include_holidays: bool = True,
):
    """Compare integer k in log1p(trend_years / k) chronologically."""
    rows = []

    for k in map(int, divisors):
        if k <= 0:
            raise ValueError("Divisors must be positive integers.")

        data = daily.copy()
        data["log_trend"] = np.log1p(data["trend_years"] / k)

        yearly, _, _ = evaluate_models(
            data,
            models_to_test=[model_name],
            holiday_mode="with" if include_holidays else "without",
            test_years=test_years,
            trend_mode="log",
        )

        for r in yearly.to_dict("records"):
            rows.append({
                "k": k,
                "year": r["Test Year"],
                "MAE": r["MAE"],
                "RMSE": r["RMSE"],
                "alpha": r["Alpha"],
            })

    by_year = pd.DataFrame(rows)
    summary = (
        by_year.groupby("k", as_index=False)
        .agg(
            Mean_MAE=("MAE", "mean"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_Alpha=("alpha", "mean"),
        )
    )
    summary["MAE_rank"] = summary["Mean_MAE"].rank(method="min")
    summary["RMSE_rank"] = summary["Mean_RMSE"].rank(method="min")
    summary["combined_rank"] = summary["MAE_rank"] + summary["RMSE_rank"]

    return (
        summary.sort_values(["combined_rank", "Mean_MAE"]).reset_index(drop=True),
        by_year,
    )


def forecast_metrics(actual_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.Series:
    """MAE/RMSE/bias and totals after date alignment."""
    merged = actual_df[["reported", "reports"]].merge(
        pred_df[["reported", "predicted"]],
        on="reported",
        how="inner",
    ).dropna()

    actual = merged["reports"].to_numpy(float)
    pred = merged["predicted"].to_numpy(float)
    error = actual - pred

    return pd.Series({
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "Mean error": np.mean(error),
        "Actual total": actual.sum(),
        "Predicted total": pred.sum(),
        "Total difference": pred.sum() - actual.sum(),
    })


def _future_row(date, history, origin, log_divisor):
    if len(history) < 28:
        raise ValueError("Recursive forecasting needs at least 28 history days.")

    trend = (date - origin).days / 365.25
    gr = holidays.country_holidays("GR", years=[date.year])

    return {
        "reported": date,
        "weekday": date.day_name(),
        "month": date.month,
        "is_holiday": date.date() in gr,
        "trend_years": trend,
        "log_trend": np.log1p(trend / log_divisor),
        "lag1": history[-1],
        "lag7": history[-7],
        "recent3_avg": np.mean(history[-3:]),
        "rolling7_avg": np.mean(history[-7:]),
        "same_weekday_4wk_avg": np.mean(
            [history[-7], history[-14], history[-21], history[-28]]
        ),
    }


def recursive_forecast(
    model,
    history_daily: pd.DataFrame,
    forecast_dates,
    *,
    origin: pd.Timestamp | None = None,
    log_divisor: int = SELECTED_LOG_TREND_DIVISOR,
) -> pd.DataFrame:
    """Recursive mean forecast: predicted counts become future lag inputs."""
    history = (
        history_daily.sort_values("reported")["reports"]
        .astype(float)
        .tolist()
    )
    origin = pd.Timestamp(
        origin if origin is not None else history_daily["reported"].min()
    )

    rows = []
    for date in pd.DatetimeIndex(forecast_dates):
        date = pd.Timestamp(date)
        row = pd.DataFrame([_future_row(date, history, origin, log_divisor)])
        pred = float(model.predict(row).iloc[0])
        rows.append({"reported": date, "predicted": pred})
        history.append(pred)

    return pd.DataFrame(rows)


def daily_updated_predictions(
    model,
    history_daily: pd.DataFrame,
    actual_future_daily: pd.DataFrame,
    *,
    origin: pd.Timestamp | None = None,
    log_divisor: int = SELECTED_LOG_TREND_DIVISOR,
) -> pd.DataFrame:
    """One-day-ahead predictions; lag features use real preceding counts."""
    origin = pd.Timestamp(
        origin if origin is not None else history_daily["reported"].min()
    )

    counts = pd.concat([
        history_daily[["reported", "reports"]],
        actual_future_daily[["reported", "reports"]],
    ], ignore_index=True).sort_values("reported")

    featured = add_features(counts, origin=origin, log_divisor=log_divisor)
    start = actual_future_daily["reported"].min()
    end = actual_future_daily["reported"].max()
    test = featured[featured["reported"].between(start, end)].copy()

    out = test[["reported", "reports"]].copy()
    out["predicted"] = np.asarray(model.predict(test), float)
    return out


def hybrid_forecast(
    model,
    history_daily: pd.DataFrame,
    actual_future_daily: pd.DataFrame,
    forecast_end,
    *,
    origin: pd.Timestamp | None = None,
    log_divisor: int = SELECTED_LOG_TREND_DIVISOR,
):
    """Use real updates while available, then recurse after the last actual."""
    origin = pd.Timestamp(
        origin if origin is not None else history_daily["reported"].min()
    )

    updated = daily_updated_predictions(
        model,
        history_daily,
        actual_future_daily,
        origin=origin,
        log_divisor=log_divisor,
    )

    last_actual = actual_future_daily["reported"].max()
    end = pd.Timestamp(forecast_end)

    if end <= last_actual:
        empty = pd.DataFrame(columns=["reported", "predicted"])
        return updated[["reported", "predicted"]], updated, empty

    combined = pd.concat([
        history_daily[["reported", "reports"]],
        actual_future_daily[["reported", "reports"]],
    ], ignore_index=True).sort_values("reported")

    future_dates = pd.date_range(
        last_actual + pd.Timedelta(days=1),
        end,
        freq="D",
    )
    future = recursive_forecast(
        model,
        combined,
        future_dates,
        origin=origin,
        log_divisor=log_divisor,
    )

    full = pd.concat(
        [updated[["reported", "predicted"]], future],
        ignore_index=True,
    )
    return full, updated, future


def recursive_backtest(
    daily: pd.DataFrame,
    forecast_year: int,
    *,
    model_name: str = SELECTED_MODEL,
    trend_mode: str = SELECTED_TREND_MODE,
    include_holidays: bool = True,
    log_divisor: int = SELECTED_LOG_TREND_DIVISOR,
):
    """Train through year-1, recursively forecast the full year, then score."""
    data = daily.copy()
    data["log_trend"] = np.log1p(data["trend_years"] / log_divisor)

    train = data[data["reported"].dt.year < forecast_year].copy()
    actual = data[
        data["reported"].dt.year == forecast_year
    ][["reported", "reports"]].copy()

    model = fit_nb(
        train,
        model_name=model_name,
        trend_mode=trend_mode,
        include_holidays=include_holidays,
    )
    dates = pd.date_range(
        f"{forecast_year}-01-01",
        f"{forecast_year}-12-31",
        freq="D",
    )
    pred = recursive_forecast(
        model,
        train[["reported", "reports"]],
        dates,
        origin=data["reported"].min(),
        log_divisor=log_divisor,
    )
    return forecast_metrics(actual, pred), pred, model
