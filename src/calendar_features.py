"""Reusable calendar features for municipal-report forecasting."""

from __future__ import annotations

import holidays
import pandas as pd


def add_greek_holiday_features(
    df: pd.DataFrame,
    date_col: str,
    *,
    is_holiday_col: str = "is_holiday",
    holiday_name_col: str = "holiday_name",
) -> pd.DataFrame:
    """Return a copy with Greek public-holiday indicator and name columns.

    Invalid or missing dates are treated as non-holidays and receive an empty
    holiday name. The holiday calendar is limited to years present in the data.
    """

    if date_col not in df.columns:
        raise ValueError(f"Data is missing date column: {date_col!r}")

    result = df.copy()
    dates = pd.to_datetime(result[date_col], errors="coerce")
    valid_dates = dates.dropna()

    if valid_dates.empty:
        result[is_holiday_col] = False
        result[holiday_name_col] = ""
        return result

    years = sorted(valid_dates.dt.year.unique().tolist())
    greek_holidays = holidays.country_holidays("GR", years=years)
    calendar_dates = dates.dt.date

    result[is_holiday_col] = calendar_dates.isin(greek_holidays)
    result[holiday_name_col] = calendar_dates.map(
        lambda date: greek_holidays.get(date, "") if pd.notna(date) else ""
    )

    return result
