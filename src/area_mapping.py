"""
Map municipal requests to geographic area polygons.

Input:
    data/requests.csv
    data/area_boundaries.csv

Output:
    data/mapped_requests.csv

Requests outside all polygons are kept with empty area fields.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

REQUESTS_PATH = DATA_DIR / "requests.csv"
AREAS_PATH = DATA_DIR / "area_boundaries.csv"
OUTPUT_PATH = DATA_DIR / "mapped_requests.csv"

CRS = "EPSG:4326"


def _prepare_areas(areas: pd.DataFrame) -> gpd.GeoDataFrame:
    """Parse WKT polygons and repair invalid geometries when possible."""

    required = {"id", "code", "name_gr", "boundary"}
    missing = required - set(areas.columns)

    if missing:
        raise ValueError(f"Area file is missing columns: {sorted(missing)}")

    areas = areas.copy()
    areas["geometry"] = gpd.GeoSeries.from_wkt(areas["boundary"])

    gdf = gpd.GeoDataFrame(areas, geometry="geometry", crs=CRS)

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    bad = gdf.geometry.isna() | gdf.geometry.is_empty
    if bad.any():
        raise ValueError(f"{bad.sum()} area geometries could not be used.")

    return gdf


def map_requests_to_areas(
    requests: pd.DataFrame,
    areas: pd.DataFrame,
) -> pd.DataFrame:
    """Return all requests with area_id, area_code and area_name added."""

    required = {"lat", "lng"}
    missing = required - set(requests.columns)

    if missing:
        raise ValueError(f"Requests file is missing columns: {sorted(missing)}")

    requests = requests.copy()
    requests["_row"] = range(len(requests))
    requests["lat"] = pd.to_numeric(requests["lat"], errors="coerce")
    requests["lng"] = pd.to_numeric(requests["lng"], errors="coerce")

    valid_mask = (
        requests["lat"].between(-90, 90)
        & requests["lng"].between(-180, 180)
    )

    valid = requests.loc[valid_mask].copy()
    invalid = requests.loc[~valid_mask].copy()

    points = gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(valid["lng"], valid["lat"]),
        crs=CRS,
    )

    areas_gdf = _prepare_areas(areas)
    area_lookup = areas_gdf[
        ["id", "code", "name_gr", "geometry"]
    ].rename(
        columns={
            "id": "area_id",
            "code": "area_code",
            "name_gr": "area_name",
        }
    )

    mapped = gpd.sjoin(
        points,
        area_lookup,
        how="left",
        predicate="within",
    )

    # A request should map to at most one municipal area.
    mapped = mapped.drop_duplicates(subset="_row", keep="first")
    mapped = mapped.drop(columns=["index_right", "geometry"], errors="ignore")

    invalid["area_id"] = pd.NA
    invalid["area_code"] = pd.NA
    invalid["area_name"] = pd.NA

    return (
        pd.concat([mapped, invalid], ignore_index=True, sort=False)
        .sort_values("_row")
        .drop(columns="_row")
        .reset_index(drop=True)
    )


def print_mapping_summary(mapped: pd.DataFrame) -> None:
    """Print a short mapping-quality summary."""

    matched = mapped["area_name"].notna()
    total = len(mapped)

    print(f"Total requests:   {total:,}")
    print(f"Mapped to area:   {matched.sum():,}")
    print(f"Unmapped:         {(~matched).sum():,}")
    print(f"Mapped percentage:{100 * matched.mean():.2f}%")


def main() -> None:
    requests = pd.read_csv(REQUESTS_PATH)
    areas = pd.read_csv(AREAS_PATH)

    mapped = map_requests_to_areas(requests, areas)
    print_mapping_summary(mapped)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
