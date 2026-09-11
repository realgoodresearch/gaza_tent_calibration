from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parents[1]
else:
    REPO_ROOT = Path.cwd()

TILE_HALF_WIDTH_METERS = 50.0

# Shared with generate_validation_data.py, which writes these as h5 attrs.
SUN_SATELLITE_FIELDS = (
    "satellite_azimuth",
    "satellite_elevation",
    "sun_azimuth",
    "sun_elevation",
)


def get_tile_polygon(
    center_x: float,
    center_y: float,
    target_crs: str = "EPSG:32636",
) -> gpd.GeoDataFrame:
    """Build a square tile polygon (in EPSG:32636) around a center point."""
    half_width = TILE_HALF_WIDTH_METERS
    polygon = box(
        center_x - half_width,
        center_y - half_width,
        center_x + half_width,
        center_y + half_width,
    )
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:32636")
    if target_crs == "EPSG:4326":
        return gdf.to_crs("EPSG:4326")
    return gdf


def parse_date_value(value: Any) -> pd.Timestamp:
    """Parse a date-like value into a pandas Timestamp (NaT if invalid/empty)."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")
