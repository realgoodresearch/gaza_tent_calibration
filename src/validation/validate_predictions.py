"""Compare validated ground-truth tent counts against actual model predictions.

This script depends on `manual_validation_results.csv` (produced by
`do_manual_validation.py`) already existing. It looks up the matching
predicted tent count for each validated tile and reports accuracy broken
down by land use, time of day, month, year, and satellite/sun geometry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import SUN_SATELLITE_FIELDS, get_tile_polygon


if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parents[2]
else:
    REPO_ROOT = Path.cwd()

VALIDATED_CSV = REPO_ROOT / "src" / "validation" / "manual_validation_results.csv"
OUTPUT_CSV = REPO_ROOT / "src" / "validation" / "prediction_accuracy_results.csv"
OUTPUT_GROUPED_SUMMARY_CSV = REPO_ROOT / "src" / "validation" / "prediction_accuracy_grouped_summary.csv"
OUTPUT_GROUPED_ZERO_BINARY_CSV = REPO_ROOT / "src" / "validation" / "prediction_accuracy_grouped_zero_binary.csv"
OUTPUT_GROUPED_POSITIVE_ONLY_CSV = REPO_ROOT / "src" / "validation" / "prediction_accuracy_grouped_positive_only.csv"
COVERAGE_DIR = REPO_ROOT / "data" / "tent_predictions" / "coverage"
PREDICTION_DIR = REPO_ROOT / "data" / "tent_predictions" / "merged"

TARGET_CRS = "EPSG:32636"
GAZA_LATITUDE = 31.5
GAZA_LONGITUDE = 34.47
GAZA_TIMEZONE = "Asia/Gaza"

# Bin width (degrees) used to group continuous azimuth/elevation values.
AZIMUTH_BIN_SIZE = 30
ELEVATION_BIN_SIZE = 15
DISTANCE_BIN_SIZE_M = 1000.0
TENT_COUNT_BIN_SIZE = 10


def _parse_time_of_day(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    parsed = pd.to_datetime(str(value), format="%H:%M:%S", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def _hours_from_sunrise(date_value: Any, time_of_day: Any) -> float:
    parsed_date = pd.to_datetime(date_value, errors="coerce")
    parsed_time = _parse_time_of_day(time_of_day)
    if pd.isna(parsed_date) or parsed_time is None:
        return np.nan

    try:
        from astral import LocationInfo
        from astral.sun import sun
    except ImportError as exc:
        raise ImportError(
            "astral is required for sunrise-based time features. "
            "Install dependencies from pyproject.toml."
        ) from exc

    location = LocationInfo(
        name="Gaza Strip",
        region="PS",
        timezone=GAZA_TIMEZONE,
        latitude=GAZA_LATITUDE,
        longitude=GAZA_LONGITUDE,
    )
    tz = ZoneInfo(GAZA_TIMEZONE)

    obs_date = parsed_date.date()
    sun_times = sun(location.observer, date=obs_date, tzinfo=tz)
    sunrise = sun_times["sunrise"]

    observation_dt = pd.Timestamp(
        year=obs_date.year,
        month=obs_date.month,
        day=obs_date.day,
        hour=parsed_time.hour,
        minute=parsed_time.minute,
        second=parsed_time.second,
        tz=tz,
    ).to_pydatetime()

    delta_hours = (observation_dt - sunrise).total_seconds() / 3600.0
    return float(delta_hours)


def _format_hour_bin(hour_value: Any) -> Any:
    if pd.isna(hour_value):
        return np.nan
    low = int(np.floor(float(hour_value)))
    high = low + 1
    return f"{low} to {high} h"


def _format_range_bin(value: Any, bin_size: float, suffix: str = "") -> Any:
    if pd.isna(value):
        return np.nan
    low = float(np.floor(float(value) / float(bin_size)) * float(bin_size))
    high = low + float(bin_size)
    if float(bin_size).is_integer() and low.is_integer() and high.is_integer():
        return f"{int(low)}-{int(high)}{suffix}"
    return f"{low:g}-{high:g}{suffix}"


def parse_coverage_range(coverage_file: Path) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    stem = coverage_file.stem.replace("_coverage", "")
    try:
        start_date = pd.to_datetime(stem[:8], format="%Y%m%d")
        end_date = pd.to_datetime(stem[9:17], format="%Y%m%d")
    except ValueError:
        return None
    return start_date, end_date


def build_coverage_index() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for coverage_file in sorted(COVERAGE_DIR.glob("*_coverage.geojson")):
        parsed = parse_coverage_range(coverage_file)
        if parsed is None:
            continue
        start_date, end_date = parsed
        rows.append(
            {
                "coverage_file": coverage_file,
                "week_key": coverage_file.stem.replace("_coverage", ""),
                "start_date": start_date,
                "end_date": end_date,
            }
        )
    return rows


def find_matching_week_key(date_value: Any, coverage_index: List[Dict[str, Any]]) -> Optional[str]:
    parsed = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(parsed):
        return None

    for row in coverage_index:
        if row["start_date"] <= parsed <= row["end_date"]:
            return str(row["week_key"])

    return None


def load_validated_data() -> pd.DataFrame:
    if not VALIDATED_CSV.exists():
        raise FileNotFoundError(
            f"Validated data not found: {VALIDATED_CSV}. "
            "Run do_manual_validation.py first."
        )

    return pd.read_csv(VALIDATED_CSV)


def load_coverage_geometry_by_week(week_key: str) -> Optional[gpd.GeoDataFrame]:
    coverage_path = COVERAGE_DIR / f"{week_key}_coverage.geojson"
    if not coverage_path.exists():
        return None

    coverage = gpd.read_file(coverage_path)
    if coverage.empty or coverage.geometry.isna().all():
        return coverage

    if coverage.crs is None:
        return None

    if coverage.crs.to_string() != TARGET_CRS:
        coverage = coverage.to_crs(TARGET_CRS)

    return coverage[["geometry"]].copy()


def load_predictions_by_week(week_key: str) -> Optional[gpd.GeoDataFrame]:
    gpkg_path = PREDICTION_DIR / f"{week_key}.gpkg"
    if not gpkg_path.exists():
        return None

    predictions = gpd.read_file(gpkg_path)
    if predictions.empty:
        return predictions

    if predictions.crs is None:
        return None

    if predictions.crs.to_string() != TARGET_CRS:
        predictions = predictions.to_crs(TARGET_CRS)

    predictions = predictions[predictions.geometry.notna()].copy()
    return predictions


def build_tile_gdf(group: pd.DataFrame) -> gpd.GeoDataFrame:
    tiles = []
    for row_idx, row in group.iterrows():
        center_x = row.get("center_x")
        center_y = row.get("center_y")

        if center_x is None or center_y is None or pd.isna(center_x) or pd.isna(center_y):
            continue

        tile = get_tile_polygon(float(center_x), float(center_y), target_crs=TARGET_CRS)
        tile["record_index"] = row_idx
        tiles.append(tile)

    if not tiles:
        return gpd.GeoDataFrame(columns=["record_index", "geometry"], geometry="geometry", crs=TARGET_CRS)

    return gpd.GeoDataFrame(pd.concat(tiles, ignore_index=True), geometry="geometry", crs=TARGET_CRS)


def attach_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Look up the predicted tent count/date/time for every validated row."""
    df = df.copy()

    df["prediction_count"] = np.nan
    df["prediction_date"] = pd.NA
    df["prediction_time"] = pd.NA
    df["prediction_available"] = False
    df["prediction_status"] = "missing_date_coverage"

    coverage_index = build_coverage_index()
    if not coverage_index:
        return df

    df["week_key"] = df["date"].apply(lambda value: find_matching_week_key(value, coverage_index))

    progress = tqdm(total=len(df), desc="Matching predictions", unit="tile")

    for week_key, group in df.groupby("week_key", dropna=False):
        idx = group.index

        if week_key is None or pd.isna(week_key):
            df.loc[idx, "prediction_status"] = "missing_date_coverage"
            progress.update(len(group))
            continue

        coverage = load_coverage_geometry_by_week(str(week_key))
        if coverage is None or coverage.empty:
            df.loc[idx, "prediction_status"] = "missing_region_coverage"
            progress.update(len(group))
            continue

        tiles = build_tile_gdf(group)
        if tiles.empty:
            df.loc[idx, "prediction_status"] = "missing_tile_coordinates"
            progress.update(len(group))
            continue

        coverage_union = coverage.geometry.union_all()
        tile_covered = tiles.geometry.intersects(coverage_union)
        covered_tiles = tiles[tile_covered].copy()
        uncovered_tiles = tiles[~tile_covered].copy()

        if not uncovered_tiles.empty:
            df.loc[uncovered_tiles["record_index"], "prediction_status"] = "missing_region_coverage"

        predictions = load_predictions_by_week(str(week_key))
        if predictions is None:
            if not covered_tiles.empty:
                df.loc[covered_tiles["record_index"], "prediction_status"] = "missing_prediction_week"
            progress.update(len(group))
            continue

        if covered_tiles.empty:
            progress.update(len(group))
            continue

        if predictions.empty:
            df.loc[covered_tiles["record_index"], "prediction_count"] = 0.0
            df.loc[covered_tiles["record_index"], "prediction_available"] = True
            df.loc[covered_tiles["record_index"], "prediction_status"] = "ok"
            progress.update(len(group))
            continue

        matches = gpd.sjoin(predictions, covered_tiles, how="inner", predicate="within")

        counts = matches.groupby("record_index").size() if not matches.empty else pd.Series(dtype=int)

        available_idx = covered_tiles["record_index"]
        df.loc[available_idx, "prediction_count"] = 0.0
        df.loc[available_idx, "prediction_available"] = True
        df.loc[available_idx, "prediction_status"] = "ok"

        if not counts.empty:
            df.loc[counts.index, "prediction_count"] = counts.astype(float)

        if "date" in matches.columns and not matches.empty:
            pred_dates = matches.groupby("record_index")["date"].agg(
                lambda values: str(pd.Series(values).dropna().mode().iloc[0])
                if not pd.Series(values).dropna().empty
                else pd.NA
            )
            df.loc[pred_dates.index, "prediction_date"] = pred_dates.astype("object")

        if "time" in matches.columns and not matches.empty:
            pred_times = matches.groupby("record_index")["time"].agg(
                lambda values: str(pd.Series(values).dropna().mode().iloc[0])
                if not pd.Series(values).dropna().empty
                else pd.NA
            )
            df.loc[pred_times.index, "prediction_time"] = pred_times.astype("object")

        progress.update(len(group))

    progress.close()

    df = df.drop(columns=["week_key"])
    return df


def compute_accuracy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add error/accuracy columns and date/geometry-derived grouping columns."""
    df = df.copy()

    pred = pd.to_numeric(df["prediction_count"], errors="coerce")
    actual = pd.to_numeric(df["tent_count"], errors="coerce")

    df["error"] = pred - actual
    df["abs_error"] = df["error"].abs()
    df["correct"] = np.where(pred.notna() & actual.notna(), pred == actual, np.nan)
    df["manual_zero"] = np.where(actual.notna(), actual == 0, np.nan)
    df["pred_zero"] = np.where(pred.notna(), pred == 0, np.nan)
    df["zero_binary_correct"] = np.where(
        pred.notna() & actual.notna(),
        (pred == 0) == (actual == 0),
        np.nan,
    )

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = parsed_dates.dt.month
    df["year"] = parsed_dates.dt.year

    if "time_of_day" in df.columns:
        df["hours_from_sunrise"] = [
            _hours_from_sunrise(date_value, time_value)
            for date_value, time_value in zip(df["date"], df["time_of_day"])
        ]
        df["sunrise_hour_bin"] = df["hours_from_sunrise"].apply(_format_hour_bin)
    else:
        df["hours_from_sunrise"] = np.nan
        df["sunrise_hour_bin"] = np.nan

    for field in SUN_SATELLITE_FIELDS:
        if field not in df.columns:
            continue
        bin_size = AZIMUTH_BIN_SIZE if "azimuth" in field else ELEVATION_BIN_SIZE
        values = pd.to_numeric(df[field], errors="coerce")
        binned = (values // bin_size) * bin_size
        df[f"{field}_bin"] = binned.apply(
            lambda low: f"{low:g}-{low + bin_size:g}" if pd.notna(low) else np.nan
        )

    for distance_col in ["distance_to_nearest_outpost", "distance_to_nearest_base", "distance_to_yellow_line"]:
        if distance_col in df.columns:
            vals = pd.to_numeric(df[distance_col], errors="coerce")
            df[f"{distance_col}_bin"] = vals.apply(lambda x: _format_range_bin(x, DISTANCE_BIN_SIZE_M, "m"))

    if "tent_count" in df.columns:
        counts = pd.to_numeric(df["tent_count"], errors="coerce")
        df["tent_count_bin"] = counts.apply(lambda x: _format_range_bin(x, TENT_COUNT_BIN_SIZE))

    for boolean_col in ["is_humanitarian_zone", "has_greenhouse", "is_agriculture", "is_destruction"]:
        if boolean_col in df.columns:
            norm = (
                df[boolean_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"1": "yes", "true": "yes", "t": "yes", "yes": "yes", "0": "no", "false": "no", "f": "no", "no": "no"})
            )
            df[boolean_col] = norm.fillna("unknown")

    return df


def summarize_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Aggregate accuracy metrics by a grouping column."""
    if group_col not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby(group_col, dropna=False)

    summary = grouped.agg(
        num_samples=("tent_count", "size"),
        num_scored=("prediction_available", "sum"),
        mean_abs_error=("abs_error", "mean"),
        mean_error=("error", "mean"),
        accuracy=("correct", "mean"),
        mean_actual_count=("tent_count", "mean"),
        mean_predicted_count=("prediction_count", "mean"),
    ).reset_index()

    return summary


def build_grouped_summary_file(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Build a single tidy table of grouped accuracy metrics across grouping dimensions."""
    rows: List[pd.DataFrame] = []
    for group_col in group_cols:
        summary = summarize_by(df, group_col)
        if summary.empty:
            continue
        summary = summary.rename(columns={group_col: "group_value"}).copy()
        summary.insert(0, "group_type", group_col)
        rows.append(summary)

    if not rows:
        return pd.DataFrame(
            columns=[
                "group_type",
                "group_value",
                "num_samples",
                "num_scored",
                "mean_abs_error",
                "mean_error",
                "accuracy",
                "mean_actual_count",
                "mean_predicted_count",
            ]
        )

    out = pd.concat(rows, ignore_index=True)
    out["group_value"] = out["group_value"].astype("object")
    return out


def build_grouped_zero_binary_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Grouped binary performance for predicting zero-vs-nonzero tents."""
    rows: List[pd.DataFrame] = []
    for group_col in group_cols:
        if group_col not in df.columns:
            continue
        grouped = df.groupby(group_col, dropna=False)
        summary = grouped.agg(
            num_samples=("tent_count", "size"),
            num_scored=("prediction_available", "sum"),
            manual_zero_rate=("manual_zero", "mean"),
            predicted_zero_rate=("pred_zero", "mean"),
            zero_binary_accuracy=("zero_binary_correct", "mean"),
        ).reset_index()
        summary = summary.rename(columns={group_col: "group_value"}).copy()
        summary.insert(0, "group_type", group_col)
        rows.append(summary)

    if not rows:
        return pd.DataFrame(
            columns=[
                "group_type",
                "group_value",
                "num_samples",
                "num_scored",
                "manual_zero_rate",
                "predicted_zero_rate",
                "zero_binary_accuracy",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    out["group_value"] = out["group_value"].astype("object")
    return out


def build_grouped_positive_only_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Grouped count performance restricted to tiles where manual tent_count > 0."""
    positive_df = df[pd.to_numeric(df["tent_count"], errors="coerce") > 0].copy()
    rows: List[pd.DataFrame] = []
    for group_col in group_cols:
        if group_col not in positive_df.columns:
            continue
        grouped = positive_df.groupby(group_col, dropna=False)
        summary = grouped.agg(
            num_samples=("tent_count", "size"),
            num_scored=("prediction_available", "sum"),
            mean_abs_error=("abs_error", "mean"),
            mean_error=("error", "mean"),
            accuracy=("correct", "mean"),
            mean_actual_count=("tent_count", "mean"),
            mean_predicted_count=("prediction_count", "mean"),
        ).reset_index()
        summary = summary.rename(columns={group_col: "group_value"}).copy()
        summary.insert(0, "group_type", group_col)
        rows.append(summary)

    if not rows:
        return pd.DataFrame(
            columns=[
                "group_type",
                "group_value",
                "num_samples",
                "num_scored",
                "mean_abs_error",
                "mean_error",
                "accuracy",
                "mean_actual_count",
                "mean_predicted_count",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    out["group_value"] = out["group_value"].astype("object")
    return out


def main() -> None:
    print(f"Loading validated ground truth from {VALIDATED_CSV}...")
    validated = load_validated_data()

    print("Looking up matching predictions for each validated tile...")
    with_predictions = attach_predictions(validated)

    print("Computing accuracy metrics...")
    results = compute_accuracy_columns(with_predictions)

    results.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved per-tile accuracy results to {OUTPUT_CSV}")

    group_cols = [
        "land_use_label",
        "municipality",
        "sunrise_hour_bin",
        "month",
        "year",
        "is_humanitarian_zone",
        "has_greenhouse",
        "is_agriculture",
        "is_destruction",
        "distance_to_nearest_outpost_bin",
        "distance_to_nearest_base_bin",
        "distance_to_yellow_line_bin",
        "tent_count_bin",
        *(f"{field}_bin" for field in SUN_SATELLITE_FIELDS),
    ]
    grouped_summary = build_grouped_summary_file(results, group_cols)
    grouped_summary.to_csv(OUTPUT_GROUPED_SUMMARY_CSV, index=False)
    print(f"Saved grouped accuracy summary to {OUTPUT_GROUPED_SUMMARY_CSV}")

    grouped_zero_binary = build_grouped_zero_binary_summary(results, group_cols)
    grouped_zero_binary.to_csv(OUTPUT_GROUPED_ZERO_BINARY_CSV, index=False)
    print(f"Saved grouped zero-binary summary to {OUTPUT_GROUPED_ZERO_BINARY_CSV}")

    grouped_positive_only = build_grouped_positive_only_summary(results, group_cols)
    grouped_positive_only.to_csv(OUTPUT_GROUPED_POSITIVE_ONLY_CSV, index=False)
    print(f"Saved grouped positive-only summary to {OUTPUT_GROUPED_POSITIVE_ONLY_CSV}")

    num_total = int(len(results))
    num_scored = int(results["prediction_available"].sum())
    print(f"Scored rows: {num_scored}/{num_total}")

    overall_accuracy = float(results["correct"].mean()) if not results.empty else np.nan
    overall_mae = float(results["abs_error"].mean()) if not results.empty else np.nan
    print(f"\nOverall accuracy: {overall_accuracy:.3f}")
    print(f"Overall mean absolute error: {overall_mae:.3f}")

    print("\nPrediction status counts:")
    print(results["prediction_status"].value_counts(dropna=False).to_string())

    for group_col in group_cols:
        print(f"\nAccuracy by {group_col}:")
        print(summarize_by(results, group_col).to_string(index=False))


if __name__ == "__main__":
    main()
