"""Estimate block-level tent changes after first observed military action."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from shapely.geometry import shape

REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTS_PATH = REPO_ROOT / "outputs/tent_counts/block_counts_by_week.csv"
BLOCKS_PATH = REPO_ROOT / "data/gaza_boundaries/dash_population_blocks.geojson"
TANKS_PATH = REPO_ROOT / "data/military_activity/tanks.geojson"
BERMS_PATH = REPO_ROOT / "data/military_activity/berms.geojson"
INCIDENTS_PATH = REPO_ROOT / "data/incidents/gaza_incidents.csv"
OUTPUT_PATH = REPO_ROOT / "outputs/analysis/military_action_tent_change.json"
TARGET_CRS = "EPSG:32636"
INCIDENT_PATTERN = r"air[\s_-]*strike|attack[\s_-]*on[\s_-]*civilians?"
MIN_BASELINE_TENTS = 100


def _read_berms() -> tuple[gpd.GeoDataFrame, int]:
    """Read berm GeoJSON while skipping malformed one-coordinate lines."""
    payload = json.loads(BERMS_PATH.read_text())
    rows = []
    skipped = 0
    for feature in payload["features"]:
        try:
            geometry = shape(feature["geometry"])
        except Exception:
            skipped += 1
            continue
        rows.append({**feature.get("properties", {}), "geometry": geometry})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(TARGET_CRS), skipped


def _assign_week(dates: pd.Series, weeks: pd.DataFrame) -> pd.Series:
    assigned = pd.Series(pd.NA, index=dates.index, dtype="string")
    for week in weeks.itertuples(index=False):
        assigned.loc[dates.between(week.week_start, week.week_end)] = week.week_key
    return assigned


def _summary(events: pd.DataFrame, column: str) -> dict:
    values = events[column].dropna().to_numpy(float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(len(values))
    margin = float(student_t.ppf(0.975, len(values) - 1) * se)
    return {
        "n_blocks": int(len(values)),
        "mean_percent_change": mean,
        "mean_percent_change_95_percent_confidence_interval": [mean - margin, mean + margin],
        "median_percent_change": float(np.median(values)),
        "interquartile_range": [float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))],
        "blocks_with_tent_decline": int(np.sum(values < 0)),
        "percent_blocks_with_tent_decline": float(np.mean(values < 0) * 100),
    }


def _event_study(
    first_actions: pd.DataFrame,
    counts: pd.DataFrame,
    weeks: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    first_actions = first_actions.copy()
    first_actions["week_t"] = _assign_week(first_actions.first_action_date, weeks)
    records = []
    week_keys = weeks.week_key.tolist()
    lookup = counts.set_index(["block_id", "week_key"]).corrected_tents.to_dict()
    for row in first_actions.dropna(subset=["week_t"]).itertuples(index=False):
        t_index = week_keys.index(row.week_t)
        if t_index == 0:
            continue
        week_t_minus_1 = week_keys[t_index - 1]
        # Require truly consecutive calendar weeks rather than bridging missing data.
        if (weeks.iloc[t_index].week_start - weeks.iloc[t_index - 1].week_start).days != 7:
            continue
        baseline = lookup.get((int(row.block_id), week_t_minus_1))
        value_t = lookup.get((int(row.block_id), row.week_t))
        week_t_plus_1 = week_keys[t_index + 1] if t_index + 1 < len(week_keys) else None
        value_t_plus_1 = None
        if week_t_plus_1 is not None and (weeks.iloc[t_index + 1].week_start - weeks.iloc[t_index].week_start).days == 7:
            value_t_plus_1 = lookup.get((int(row.block_id), week_t_plus_1))
        if baseline is None or baseline <= MIN_BASELINE_TENTS:
            continue
        records.append({
            "block_id": int(row.block_id),
            "first_action_date": row.first_action_date.strftime("%Y-%m-%d"),
            "week_t_minus_1": week_t_minus_1,
            "week_t": row.week_t,
            "week_t_plus_1": week_t_plus_1,
            "corrected_tents_t_minus_1": float(baseline),
            "corrected_tents_t": float(value_t) if value_t is not None else None,
            "corrected_tents_t_plus_1": float(value_t_plus_1) if value_t_plus_1 is not None else None,
            "percent_change_t_vs_t_minus_1": (
                float((value_t - baseline) / baseline * 100) if value_t is not None else None
            ),
            "percent_change_t_plus_1_vs_t_minus_1": (
                float((value_t_plus_1 - baseline) / baseline * 100)
                if value_t_plus_1 is not None else None
            ),
        })
    event_frame = pd.DataFrame(records)
    summary = {
        "t_action_week_vs_t_minus_1": _summary(event_frame, "percent_change_t_vs_t_minus_1"),
        "t_plus_1_vs_t_minus_1": _summary(event_frame, "percent_change_t_plus_1_vs_t_minus_1"),
    }
    return event_frame, summary


def main() -> None:
    counts = pd.read_csv(COUNTS_PATH)
    counts["block_id"] = pd.to_numeric(counts.block, errors="coerce").astype("Int64")
    counts["corrected_count_mean"] = pd.to_numeric(counts.corrected_count_mean, errors="coerce")
    counts = counts.dropna(subset=["block_id", "week_key", "corrected_count_mean"])
    counts = counts.groupby(["block_id", "week_key"], as_index=False).agg(
        corrected_tents=("corrected_count_mean", "sum")
    )
    weeks = counts[["week_key"]].drop_duplicates().copy()
    weeks["week_start"] = pd.to_datetime(weeks.week_key.str.split("-").str[0], format="%Y%m%d")
    weeks["week_end"] = pd.to_datetime(weeks.week_key.str.split("-").str[1], format="%Y%m%d")
    weeks = weeks.sort_values("week_start").reset_index(drop=True)

    blocks = gpd.read_file(BLOCKS_PATH).to_crs(TARGET_CRS).rename(columns={"Name": "block_id"})
    blocks["block_id"] = pd.to_numeric(blocks.block_id, errors="coerce").astype("Int64")
    blocks = blocks.dropna(subset=["block_id"])[["block_id", "geometry"]]

    tanks = gpd.read_file(TANKS_PATH).to_crs(TARGET_CRS)
    tanks["date"] = pd.to_datetime(tanks.date, errors="coerce")
    tanks = tanks.dropna(subset=["date", "geometry"])
    tanks_in_blocks = gpd.sjoin(tanks[["date", "geometry"]], blocks, predicate="within", how="inner")
    tanks_first = tanks_in_blocks.groupby("block_id").date.min().rename("tank_first_date")

    berms, malformed_berms_skipped = _read_berms()
    berms["date"] = pd.to_datetime(berms.date, errors="coerce")
    berms = berms.dropna(subset=["date", "geometry"])
    berms_in_blocks = gpd.sjoin(berms[["date", "geometry"]], blocks, predicate="intersects", how="inner")
    berms_first = berms_in_blocks.groupby("block_id").date.min().rename("berm_first_date")

    ground = pd.concat([tanks_first, berms_first], axis=1)
    ground["first_action_date"] = ground[["tank_first_date", "berm_first_date"]].min(axis=1)
    ground = ground.reset_index()[["block_id", "first_action_date"]]

    incident_data = pd.read_csv(INCIDENTS_PATH)
    incident_text = (
        incident_data["Incident Tags"].fillna("").astype(str) + " "
        + incident_data["Incident Type"].fillna("").astype(str)
    )
    incident_data = incident_data[
        incident_text.str.contains(INCIDENT_PATTERN, case=False, regex=True)
    ].copy()
    incident_data["date"] = pd.to_datetime(incident_data.Date, format="%Y-%m-%d", errors="coerce")
    incident_data["lat"] = pd.to_numeric(incident_data.Lat, errors="coerce")
    incident_data["lon"] = pd.to_numeric(incident_data.Lon, errors="coerce")
    incident_data = incident_data.dropna(subset=["date", "lat", "lon"])
    incidents = gpd.GeoDataFrame(
        incident_data,
        geometry=gpd.points_from_xy(incident_data.lon, incident_data.lat),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)
    incidents_in_blocks = gpd.sjoin(
        incidents[["date", "geometry"]], blocks, predicate="within", how="inner"
    )
    incident_first = incidents_in_blocks.groupby("block_id").date.min().rename("first_action_date").reset_index()

    ground_events, ground_summary = _event_study(ground, counts, weeks)
    incident_events, incident_summary = _event_study(incident_first, counts, weeks)
    output = {
        "design": (
            "For each block, identify the first observation week containing the specified military action. "
            "Calculate corrected-tent percentage change at t and t+1 relative to t-1, then summarize equally across blocks."
        ),
        "confidence_interval_method": "Two-sided 95% Student-t confidence interval for the mean block-level percentage change.",
        "baseline_filter": "Only block events with more than 100 corrected tents at t-1 are included.",
        "ground_military_evidence": {
            "definition": "First dated tank point within, or newly observed berm geometry intersecting, each block.",
            "blocks_with_first_action_date": int(len(ground)),
            "malformed_berm_features_skipped": malformed_berms_skipped,
            "summary": ground_summary,
            "block_events": json.loads(ground_events.to_json(orient="records")),
        },
        "airstrike_or_attack_on_civilians": {
            "definition": "First filtered airstrike/air strike or attack-on-civilian(s) incident point within each block.",
            "blocks_with_first_action_date": int(len(incident_first)),
            "summary": incident_summary,
            "block_events": json.loads(incident_events.to_json(orient="records")),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
