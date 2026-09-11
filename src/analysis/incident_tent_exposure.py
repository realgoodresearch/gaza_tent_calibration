"""Analyze airstrike/civilian-attack incidents relative to tents and evacuations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.strtree import STRtree

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src/calibration"))
from workflow.postprocessing import apply_weekly_postprocessing, load_yellow_line_context  # noqa: E402

INCIDENTS_PATH = REPO_ROOT / "data/incidents/gaza_incidents.csv"
TENTS_DIR = REPO_ROOT / "data/tent_predictions/merged"
COVERAGE_DIR = REPO_ROOT / "data/tent_predictions/coverage"
COUNTS_PATH = REPO_ROOT / "outputs/tent_counts/block_counts_by_week.csv"
BLOCKS_PATH = REPO_ROOT / "data/gaza_boundaries/dash_population_blocks.geojson"
ZONE_PATH = REPO_ROOT / "data/spatial_context/humanitarian_zone.json"
ORDERS_PATH = REPO_ROOT / "data/military_activity/evac-orders-grid.geojson"
ARROWS_PATH = REPO_ROOT / "data/military_activity/arrows.geojson"
YELLOW_LINE_PATH = REPO_ROOT / "data/spatial_context/yellow-line-zone.json"
OUTPUT_PATH = REPO_ROOT / "outputs/analysis/incident_tent_exposure.json"
TARGET_CRS = "EPSG:32636"
RADII_M = [10, 20, 50, 100, 200]
FILTER_PATTERN = r"air[\s_-]*strike|attack[\s_-]*on[\s_-]*civilians?"
DIRECTION_VECTORS = {
    "north": (0.0, 1.0), "south": (0.0, -1.0), "west": (-1.0, 0.0),
    "north-west": (-1.0, 1.0), "south-west": (-1.0, -1.0),
}


def _week_table() -> pd.DataFrame:
    rows = []
    for path in TENTS_DIR.glob("*.gpkg"):
        try:
            start_text, end_text = path.stem.split("-")
            rows.append({
                "week_key": path.stem,
                "week_start": pd.to_datetime(start_text, format="%Y%m%d"),
                "week_end": pd.to_datetime(end_text, format="%Y%m%d"),
                "path": path,
            })
        except ValueError:
            continue
    return pd.DataFrame(rows).sort_values("week_start")


def _assign_week(dates: pd.Series, weeks: pd.DataFrame) -> pd.Series:
    result = pd.Series(pd.NA, index=dates.index, dtype="string")
    for row in weeks.itertuples(index=False):
        result.loc[dates.between(row.week_start, row.week_end)] = row.week_key
    return result


def _read_arrows() -> gpd.GeoDataFrame:
    payload = json.loads(ARROWS_PATH.read_text())
    rows = [{**f.get("properties", {}), "geometry": shape(f["geometry"])} for f in payload["features"]]
    arrows = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(TARGET_CRS)
    arrows["date"] = pd.to_datetime(arrows.date, errors="coerce")
    arrows["direction"] = arrows.direction.str.lower()
    return arrows


def _distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "iqr": [float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))],
    }


def _incident_count_summary(observations: list[dict], periods: list[str]) -> dict:
    indicated = sum(
        row[period]["indicated_direction"]["incidents"]
        for row in observations for period in periods
    )
    not_indicated = sum(
        row[period]["not_indicated_direction"]["incidents"]
        for row in observations for period in periods
    )
    total = indicated + not_indicated
    return {
        "counting_unit": "incident-order-direction association in an adjacent block",
        "total_incidents_in_adjacent_blocks": int(total),
        "indicated_direction": {
            "incidents": int(indicated),
            "percent_of_adjacent_block_incidents": float(indicated / total * 100) if total else None,
        },
        "not_indicated_direction": {
            "incidents": int(not_indicated),
            "percent_of_adjacent_block_incidents": float(not_indicated / total * 100) if total else None,
        },
    }


def _incident_group_counts(event: pd.DataFrame, column: str) -> dict:
    indicated = int(event.loc[event.in_indicated_direction, column].sum())
    other = int(event.loc[~event.in_indicated_direction, column].sum())
    total = indicated + other
    return {
        "total_incidents_in_adjacent_blocks": total,
        "indicated_direction": {
            "n_adjacent_blocks": int(event.in_indicated_direction.sum()),
            "incidents": indicated,
            "percent_of_adjacent_block_incidents": indicated / total * 100 if total else None,
        },
        "not_indicated_direction": {
            "n_adjacent_blocks": int((~event.in_indicated_direction).sum()),
            "incidents": other,
            "percent_of_adjacent_block_incidents": other / total * 100 if total else None,
        },
    }


def main() -> None:
    weeks = _week_table()
    raw = pd.read_csv(INCIDENTS_PATH)
    text = raw["Incident Tags"].fillna("").astype(str) + " " + raw["Incident Type"].fillna("").astype(str)
    filtered = raw[text.str.contains(FILTER_PATTERN, case=False, regex=True)].copy()
    filtered["date"] = pd.to_datetime(filtered.Date, format="%Y-%m-%d", errors="coerce")
    filtered["lat"] = pd.to_numeric(filtered.Lat, errors="coerce")
    filtered["lon"] = pd.to_numeric(filtered.Lon, errors="coerce")
    filtered = filtered.dropna(subset=["date", "lat", "lon"])
    incidents = gpd.GeoDataFrame(
        filtered, geometry=gpd.points_from_xy(filtered.lon, filtered.lat), crs="EPSG:4326"
    ).to_crs(TARGET_CRS)
    incidents["week_key"] = _assign_week(incidents.date, weeks)
    matched = incidents.dropna(subset=["week_key"]).copy()

    # Proximity to detected tent points in the incident's observation week.
    proximity_rows = []
    postprocessing_log = []
    yellow, yellow_windows = load_yellow_line_context(YELLOW_LINE_PATH)
    for week_key, week_incidents in matched.groupby("week_key"):
        tent_path = TENTS_DIR / f"{week_key}.gpkg"
        coverage_path = COVERAGE_DIR / f"{week_key}_coverage.geojson"
        tents = gpd.read_file(tent_path).to_crs(TARGET_CRS)
        tents, postprocess_result = apply_weekly_postprocessing(
            tents, week_key, yellow, yellow_windows, eps_m=40.0, min_samples=3
        )
        postprocessing_log.append({
            "week_key": week_key,
            "n_input": postprocess_result.n_input,
            "n_after_dbscan": postprocess_result.n_after_dbscan,
            "n_removed_yellow": postprocess_result.n_removed_yellow,
            "n_output": postprocess_result.n_output,
        })
        coverage = gpd.read_file(coverage_path).to_crs(TARGET_CRS).geometry.union_all()
        tree = STRtree(tents.geometry.to_numpy())
        for idx, point in week_incidents.geometry.items():
            if not coverage.covers(point):
                continue
            row = {"incident_index": int(idx), "week_key": week_key}
            for radius in RADII_M:
                row[f"tents_within_{radius}m"] = int(len(tree.query(point, predicate="dwithin", distance=radius)))
            proximity_rows.append(row)
    proximity = pd.DataFrame(proximity_rows).set_index("incident_index")
    matched = matched.join(proximity.drop(columns="week_key"), how="left")
    proximity_incidents = matched[matched[f"tents_within_{RADII_M[0]}m"].notna()].copy()

    blocks = gpd.read_file(BLOCKS_PATH).to_crs(TARGET_CRS).rename(columns={"Name": "block_id"})
    blocks["block_id"] = pd.to_numeric(blocks.block_id, errors="coerce").astype("Int64")
    blocks = blocks.dropna(subset=["block_id"])[["block_id", "geometry"]]
    blocks["block_area_km2"] = blocks.geometry.area / 1_000_000
    joined = gpd.sjoin(matched, blocks, how="left", predicate="within").drop(columns="index_right")

    counts = pd.read_csv(COUNTS_PATH)
    counts["block_id"] = pd.to_numeric(counts.block, errors="coerce").astype("Int64")
    for col in ("corrected_count_mean", "covered_area_m2", "raw_points"):
        counts[col] = pd.to_numeric(counts[col], errors="coerce")
    counts = counts.dropna(subset=[
        "block_id", "week_key", "corrected_count_mean", "covered_area_m2", "raw_points"
    ])
    counts = counts.groupby(["block_id", "week_key"], as_index=False).agg(
        corrected_tents=("corrected_count_mean", "sum"),
        covered_area_m2=("covered_area_m2", "sum"),
        raw_tents=("raw_points", "sum"),
    )
    counts["covered_area_km2"] = counts.covered_area_m2 / 1_000_000
    counts["tent_density_per_km2"] = counts.corrected_tents / counts.covered_area_km2.replace(0, np.nan)
    joined = joined.merge(
        counts[["block_id", "week_key", "tent_density_per_km2"]],
        on=["block_id", "week_key"], how="left",
    )

    zones = gpd.read_file(ZONE_PATH).to_crs(TARGET_CRS)
    zones["safe_date"] = pd.to_datetime(zones.safe_date, errors="coerce")
    zones["end_date"] = pd.to_datetime(zones.end_date, errors="coerce")
    joined["inside_active_humanitarian_zone"] = False
    for date, date_rows in joined.groupby("date"):
        active = zones[(zones.safe_date <= date) & (zones.end_date >= date)]
        if not active.empty:
            joined.loc[date_rows.index, "inside_active_humanitarian_zone"] = date_rows.geometry.within(
                active.geometry.union_all()
            ).to_numpy()
    joined["inside_active_zone_and_density_over_1000"] = (
        joined.inside_active_humanitarian_zone & (joined.tent_density_per_km2 > 1000)
    )

    # Global block-week density quintiles and incident rates per covered km2-week.
    exposure = counts.merge(
        weeks[["week_key", "week_start", "week_end"]], on="week_key", how="left"
    ).dropna(subset=["tent_density_per_km2", "week_start", "week_end"])
    incident_period_start = incidents.date.min()
    incident_period_end = incidents.date.max()
    exposure = exposure[
        (exposure.week_end >= incident_period_start) & (exposure.week_start <= incident_period_end)
    ].copy()
    exposure["density_category"] = "no_tents"
    has_tents = exposure.raw_tents > 0
    exposure.loc[has_tents, "density_category"] = pd.qcut(
        exposure.loc[has_tents, "tent_density_per_km2"].rank(method="first"),
        5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    ).astype(str)
    incident_counts = joined.dropna(subset=["block_id", "week_key"]).groupby(
        ["block_id", "week_key"]
    ).size().rename("incidents")
    exposure = exposure.join(incident_counts, on=["block_id", "week_key"])
    exposure["incidents"] = exposure.incidents.fillna(0)
    quintiles = []
    category_order = ["no_tents", "Q1", "Q2", "Q3", "Q4", "Q5"]
    for category in category_order:
        group = exposure[exposure.density_category == category]
        n_incidents = int(group.incidents.sum())
        area_exposure = float(group.covered_area_km2.sum())
        quintiles.append({
            "category": category,
            "density_range_tents_per_km2": [
                float(group.tent_density_per_km2.min()), float(group.tent_density_per_km2.max())
            ],
            "incidents": n_incidents,
            "covered_block_area_km2_weeks": area_exposure,
            "incident_rate_per_covered_km2_week": n_incidents / area_exposure,
        })
    def rate_ratio_result(high: dict, low: dict) -> dict:
        ratio = high["incident_rate_per_covered_km2_week"] / low["incident_rate_per_covered_km2_week"]
        log_se = np.sqrt(1 / high["incidents"] + 1 / low["incidents"])
        return {
            "rate_ratio": ratio,
            "rate_ratio_95_percent_confidence_interval": [
                float(np.exp(np.log(ratio) - 1.96 * log_se)),
                float(np.exp(np.log(ratio) + 1.96 * log_se)),
            ],
        }

    no_tents, q1, high = quintiles[0], quintiles[1], quintiles[-1]

    # Adjacent-block incident occurrence by evacuation-arrow direction.
    orders = gpd.read_file(ORDERS_PATH).to_crs(TARGET_CRS)
    orders["order_date"] = pd.to_datetime(orders.order_date, errors="coerce")
    orders["block_id"] = pd.to_numeric(orders.name, errors="coerce").astype("Int64")
    arrows = _read_arrows()
    incidents_by_block_week = joined.dropna(subset=["block_id", "week_key"]).groupby(
        ["block_id", "week_key"]
    ).size().to_dict()
    week_keys = weeks.week_key.tolist()
    observations = []
    for date, date_orders in orders.dropna(subset=["order_date", "block_id"]).groupby("order_date"):
        date_arrows = arrows[(arrows.date == date) & arrows.direction.isin(DIRECTION_VECTORS)]
        week_match = weeks[(weeks.week_start <= date) & (weeks.week_end >= date)]
        if date_arrows.empty or week_match.empty:
            continue
        week_t = week_match.week_key.iloc[0]
        t_idx = week_keys.index(week_t)
        if t_idx + 1 >= len(week_keys):
            continue
        week_t_plus_1 = week_keys[t_idx + 1]
        affected_ids = set(date_orders.block_id.astype(int))
        affected = blocks[blocks.block_id.isin(affected_ids)]
        if affected.empty:
            continue
        adjacent = blocks[
            ~blocks.block_id.isin(affected_ids) & blocks.geometry.touches(affected.geometry.union_all())
        ].copy()
        centroid = affected.geometry.centroid
        origin_x, origin_y = centroid.x.mean(), centroid.y.mean()
        for direction in sorted(date_arrows.direction.unique()):
            dx, dy = DIRECTION_VECTORS[direction]
            event = adjacent.copy()
            centers = event.geometry.centroid
            event["in_indicated_direction"] = ((centers.x - origin_x) * dx + (centers.y - origin_y) * dy) > 0
            event["incident_t"] = event.block_id.map(
                lambda b: incidents_by_block_week.get((int(b), week_t), 0)
            )
            event["incident_t_plus_1"] = event.block_id.map(
                lambda b: incidents_by_block_week.get((int(b), week_t_plus_1), 0)
            )
            observations.append({
                "order_date": date.strftime("%Y-%m-%d"), "direction": direction,
                "week_t": week_t, "week_t_plus_1": week_t_plus_1,
                "affected_blocks_in_order": len(affected_ids), "adjacent_blocks": len(event),
                "t_evacuation_order_week": _incident_group_counts(event, "incident_t"),
                "t_plus_1_following_week": _incident_group_counts(event, "incident_t_plus_1"),
            })

    proximity_summary = {}
    for radius in RADII_M:
        values = proximity_incidents[f"tents_within_{radius}m"].to_numpy(float)
        proximity_summary[f"within_{radius}m"] = {
            "incidents_with_at_least_one_tent": int(np.sum(values > 0)),
            "percent_incidents_with_at_least_one_tent": float(np.mean(values > 0) * 100),
            "tent_count_per_incident": _distribution(values),
        }
    hz_count = int(joined.inside_active_humanitarian_zone.sum())
    output = {
        "filter": {
            "rule": "Case-insensitive airstrike/air strike or attack on civilian(s) in Incident Tags or Incident Type.",
            "incidents_after_text_filter": int(len(filtered)),
            "incidents_with_valid_date_and_coordinates": int(len(incidents)),
            "incidents_matched_to_available_tent_week": int(len(matched)),
            "incidents_inside_corresponding_weekly_prediction_coverage": int(len(proximity_incidents)),
        },
        "tent_proximity": {
            "denominator": "Filtered incidents with valid coordinates/date that match an available weekly tent file and fall inside that week's prediction-coverage polygon.",
            "tent_points_are": "Detected tent point geometries in the corresponding weekly merged GPKG after canonical DBSCAN noise removal and time-valid yellow-line exclusion.",
            "postprocessing": {
                "dbscan_eps_m": 40.0,
                "dbscan_min_samples": 3,
                "yellow_line_path": str(YELLOW_LINE_PATH.relative_to(REPO_ROOT)),
                "weekly_log": postprocessing_log,
            },
            "buffers": proximity_summary,
        },
        "humanitarian_zone_and_high_density": {
            "incidents_inside_humanitarian_zone_when_officially_active": hz_count,
            "percent_of_matched_incidents_inside_active_humanitarian_zone": float(hz_count / len(joined) * 100),
            "incidents_inside_active_zone_and_block_density_over_1000_per_km2": int(joined.inside_active_zone_and_density_over_1000.sum()),
            "percent_of_all_matched_incidents_inside_active_zone_and_density_over_1000_per_km2": float(joined.inside_active_zone_and_density_over_1000.mean() * 100),
            "percent_of_active_zone_incidents_in_blocks_over_1000_tents_per_km2": (
                float(joined.inside_active_zone_and_density_over_1000.sum() / hz_count * 100) if hz_count else None
            ),
        },
        "block_density_quintile_incident_rates": {
            "method": "Block-weeks with zero raw detected tents form a separate no_tents category. Corrected tent density among tent-present block-weeks is divided into five global quintiles. Rates use summed covered block km2-weeks as exposure.",
            "incident_and_exposure_period": [
                incident_period_start.strftime("%Y-%m-%d"), incident_period_end.strftime("%Y-%m-%d")
            ],
            "incidents_assigned_to_a_block_week_with_density": int(exposure.incidents.sum()),
            "quintiles": quintiles,
            "Q5_vs_no_tents": rate_ratio_result(high, no_tents),
            "Q5_vs_Q1": rate_ratio_result(high, q1),
        },
        "evacuation_direction_adjacent_block_incidents": {
            "method": "Incident counts are pooled across evacuation-order-direction events and classified by whether their adjacent block lies in the indicated half-plane. An incident can contribute to more than one association when orders/directions overlap.",
            "t_evacuation_order_week": _incident_count_summary(
                observations, ["t_evacuation_order_week"]
            ),
            "t_plus_1_following_week": _incident_count_summary(
                observations, ["t_plus_1_following_week"]
            ),
            "t_and_t_plus_1_combined": _incident_count_summary(
                observations, ["t_evacuation_order_week", "t_plus_1_following_week"]
            ),
            "observations": observations,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
