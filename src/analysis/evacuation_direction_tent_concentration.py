"""Compare tent growth in adjacent blocks by evacuation-arrow direction."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, ttest_ind
from shapely.geometry import shape

REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTS_PATH = REPO_ROOT / "outputs/tent_counts/block_counts_by_week.csv"
ORDERS_PATH = REPO_ROOT / "data/military_activity/evac-orders-grid.geojson"
ARROWS_PATH = REPO_ROOT / "data/military_activity/arrows.geojson"
BLOCKS_PATH = REPO_ROOT / "data/gaza_boundaries/dash_population_blocks.geojson"
OUTPUT_PATH = REPO_ROOT / "outputs/analysis/evacuation_direction_tent_concentration.json"
TARGET_CRS = "EPSG:32636"
MAX_ORDER_DATE = pd.Timestamp("2024-05-31")

DIRECTION_VECTORS = {
    "north": (0.0, 1.0), "south": (0.0, -1.0), "west": (-1.0, 0.0),
    "north-west": (-1.0, 1.0), "south-west": (-1.0, -1.0),
}


def _read_geojson_arrows() -> gpd.GeoDataFrame:
    """Read manually because this source includes invalid calendar values."""
    payload = json.loads(ARROWS_PATH.read_text())
    rows = [{**f.get("properties", {}), "geometry": shape(f["geometry"])} for f in payload["features"]]
    return gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(TARGET_CRS)


def _week_for_date(week_keys: list[str], date: pd.Timestamp) -> str | None:
    for key in week_keys:
        start_text, end_text = key.split("-")
        start = pd.to_datetime(start_text, format="%Y%m%d")
        end = pd.to_datetime(end_text, format="%Y%m%d")
        if start <= date <= end:
            return key
    return None


def _aggregate_block_weeks(counts: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple image/tile rows to one tent estimate per block-week."""
    return counts.groupby(["block_id", "week_key"], as_index=False).agg(
        corrected_count_mean=("corrected_count_mean", "sum")
    )


def _comparison(frame: pd.DataFrame, value_column: str) -> dict:
    indicated = frame.loc[frame.in_indicated_direction, value_column].dropna().to_numpy(float)
    not_indicated = frame.loc[~frame.in_indicated_direction, value_column].dropna().to_numpy(float)
    indicated_mean = float(np.mean(indicated)) if len(indicated) else None
    not_indicated_mean = float(np.mean(not_indicated)) if len(not_indicated) else None
    difference = (
        indicated_mean - not_indicated_mean
        if indicated_mean is not None and not_indicated_mean is not None else None
    )
    if len(indicated) >= 2 and len(not_indicated) >= 2:
        test = ttest_ind(indicated, not_indicated, equal_var=False)
        test_result = {
            "t_statistic": float(test.statistic),
            "p_value_two_sided": float(test.pvalue),
            "degrees_of_freedom": float(test.df),
        }
    else:
        test_result = None
    return {
        "indicated_direction": {
            "n_block_order_direction_pairs": int(len(indicated)),
            "average_percent_change_from_t_minus_1": indicated_mean,
        },
        "not_indicated_direction": {
            "n_block_order_direction_pairs": int(len(not_indicated)),
            "average_percent_change_from_t_minus_1": not_indicated_mean,
        },
        "difference_in_average_percent_change_percentage_points": difference,
        "welch_t_test": test_result,
    }


def _event_difference_summary(observations: list[dict], period: str) -> dict:
    """Test equally weighted order-direction mean differences against zero."""
    differences = np.asarray([
        observation[period]["difference_in_average_percent_change_percentage_points"]
        for observation in observations
        if observation[period]["difference_in_average_percent_change_percentage_points"] is not None
    ], dtype=float)
    test = ttest_1samp(differences, popmean=0.0)
    mean_difference = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
    margin = float(student_t.ppf(0.975, df=len(differences) - 1) * standard_error)
    return {
        "unit_of_analysis": "evacuation-order-direction event",
        "n_events_with_both_groups": int(len(differences)),
        "average_indicated_minus_not_indicated_percent_change_percentage_points": mean_difference,
        "average_difference_95_percent_confidence_interval": [
            mean_difference - margin,
            mean_difference + margin,
        ],
        "standard_deviation_of_event_differences": float(np.std(differences, ddof=1)),
        "one_sample_t_test_against_zero": {
            "t_statistic": float(test.statistic),
            "p_value_two_sided": float(test.pvalue),
            "degrees_of_freedom": int(len(differences) - 1),
        },
    }


def main() -> None:
    counts = pd.read_csv(COUNTS_PATH)
    counts["block_id"] = pd.to_numeric(counts["block"], errors="coerce").astype("Int64")
    counts["corrected_count_mean"] = pd.to_numeric(counts.corrected_count_mean, errors="coerce")
    counts = _aggregate_block_weeks(counts.dropna(subset=["block_id", "week_key", "corrected_count_mean"]))
    week_keys = sorted(counts.week_key.unique(), key=lambda k: pd.to_datetime(k.split("-")[0], format="%Y%m%d"))

    blocks = gpd.read_file(BLOCKS_PATH).to_crs(TARGET_CRS).rename(columns={"Name": "block_id"})
    blocks["block_id"] = pd.to_numeric(blocks.block_id, errors="coerce").astype("Int64")
    blocks = blocks.dropna(subset=["block_id"])[["block_id", "geometry"]]
    centroids = blocks.geometry.centroid
    blocks["cx"], blocks["cy"] = centroids.x, centroids.y

    orders = gpd.read_file(ORDERS_PATH).to_crs(TARGET_CRS)
    orders["order_date"] = pd.to_datetime(orders.order_date, errors="coerce")
    orders["block_id"] = pd.to_numeric(orders.name, errors="coerce").astype("Int64")
    arrows = _read_geojson_arrows()
    arrows["date"] = pd.to_datetime(arrows.date, errors="coerce")
    arrows["direction"] = arrows.direction.str.lower()

    rows: list[pd.DataFrame] = []
    observations: list[dict] = []
    event_ids: set[str] = set()
    for date, date_orders in orders.dropna(subset=["order_date", "block_id"]).groupby("order_date"):
        if date > MAX_ORDER_DATE:
            continue
        date_arrows = arrows[(arrows.date == date) & arrows.direction.isin(DIRECTION_VECTORS)]
        if date_arrows.empty:
            continue
        affected_ids = set(date_orders.block_id.astype(int))
        affected = blocks[blocks.block_id.isin(affected_ids)]
        week_t = _week_for_date(week_keys, date)
        if affected.empty or week_t is None:
            continue
        t_index = week_keys.index(week_t)
        if t_index == 0 or t_index + 1 >= len(week_keys):
            continue
        week_t_minus_1, week_t_plus_1 = week_keys[t_index - 1], week_keys[t_index + 1]

        affected_union = affected.geometry.union_all()
        adjacent = blocks[
            ~blocks.block_id.isin(affected_ids) & blocks.geometry.touches(affected_union)
        ].copy()
        if adjacent.empty:
            continue
        n_adjacent_geometry = len(adjacent)
        origin_x, origin_y = affected["cx"].mean(), affected["cy"].mean()
        values = counts[counts.week_key.isin([week_t_minus_1, week_t, week_t_plus_1])].pivot(
            index="block_id", columns="week_key", values="corrected_count_mean"
        )
        adjacent = adjacent.merge(values, left_on="block_id", right_index=True, how="inner")
        adjacent = adjacent[adjacent[week_t_minus_1] > 0].copy()
        adjacent["percent_change_t"] = (
            (adjacent[week_t] - adjacent[week_t_minus_1]) / adjacent[week_t_minus_1] * 100
        )
        adjacent["percent_change_t_plus_1"] = (
            (adjacent[week_t_plus_1] - adjacent[week_t_minus_1]) / adjacent[week_t_minus_1] * 100
        )

        for direction in sorted(date_arrows.direction.unique()):
            dx, dy = DIRECTION_VECTORS[direction]
            event_id = f"{date:%Y-%m-%d}|{direction}"
            event_ids.add(event_id)
            event = adjacent.copy()
            event["in_indicated_direction"] = (
                (event["cx"] - origin_x) * dx + (event["cy"] - origin_y) * dy
            ) > 0
            event["event_id"] = event_id
            event_values = event[[
                "event_id", "in_indicated_direction", "percent_change_t", "percent_change_t_plus_1"
            ]]
            rows.append(event_values)
            observations.append({
                "order_date": date.strftime("%Y-%m-%d"),
                "direction": direction,
                "week_t_minus_1": week_t_minus_1,
                "week_t": week_t,
                "week_t_plus_1": week_t_plus_1,
                "affected_blocks_in_order": len(affected_ids),
                "affected_blocks_matched_to_boundaries": len(affected),
                "adjacent_blocks": n_adjacent_geometry,
                "adjacent_blocks_with_nonzero_t_minus_1": len(event),
                "t_evacuation_order_week_vs_t_minus_1": _comparison(event_values, "percent_change_t"),
                "t_plus_1_vs_t_minus_1": _comparison(event_values, "percent_change_t_plus_1"),
            })

    if not rows:
        raise RuntimeError("No adjacent block/order/direction observations were available")
    output = {
        "analysis": "Average tent-count change in blocks adjacent to evacuation-order blocks, by indicated direction.",
        "adjacency_definition": "A block is adjacent when its geometry touches the union of blocks in the evacuation order.",
        "baseline": "t_minus_1 is the week before the evacuation-order week; pairs with zero baseline tent count are excluded because percent change is undefined.",
        "maximum_order_date_included": MAX_ORDER_DATE.strftime("%Y-%m-%d"),
        "summary_test": "Two-sided one-sample t-test of the equally weighted indicated-minus-not-indicated average percentage-point difference for each evacuation-order-direction event against zero.",
        "n_order_direction_events": len(event_ids),
        "t_evacuation_order_week_vs_t_minus_1": _event_difference_summary(
            observations, "t_evacuation_order_week_vs_t_minus_1"
        ),
        "t_plus_1_vs_t_minus_1": _event_difference_summary(
            observations, "t_plus_1_vs_t_minus_1"
        ),
        "observations": observations,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
