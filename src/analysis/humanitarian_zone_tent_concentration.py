"""Measure longitudinal concentration of calibrated tents in humanitarian zones."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTS_PATH = REPO_ROOT / "outputs/tent_counts/block_counts_by_week.csv"
BLOCKS_PATH = REPO_ROOT / "data/gaza_boundaries/dash_population_blocks.geojson"
GAZA_PATH = REPO_ROOT / "data/gaza_boundaries/municipal/GazaStrip_MunicipalBoundaries.shp"
ZONE_PATH = REPO_ROOT / "data/spatial_context/humanitarian_zone.json"
COVERAGE_DIR = REPO_ROOT / "data/tent_predictions/coverage"
OUTPUT_PATH = REPO_ROOT / "outputs/analysis/humanitarian_zone_tent_concentration.json"
TARGET_CRS = "EPSG:32636"
N_DRAWS = 10_000
RNG_SEED = 2026
Z_95 = 1.959963984540054
MIN_GAZA_COVERAGE = 0.80


def _date(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def _zone_for_date(zones: gpd.GeoDataFrame, date: pd.Timestamp):
    active = zones[
        zones.safe_date_parsed.notna()
        & zones.end_date_parsed.notna()
        & (zones.safe_date_parsed <= date)
        & (zones.end_date_parsed >= date)
    ]
    if not active.empty:
        return active.geometry.union_all(), True
    previous = zones[zones.safe_date_parsed.notna() & (zones.safe_date_parsed <= date)]
    if previous.empty:
        return None, False
    latest_start = previous.safe_date_parsed.max()
    return previous[previous.safe_date_parsed == latest_start].geometry.union_all(), False


def main() -> None:
    counts = pd.read_csv(COUNTS_PATH)
    counts["block_id"] = pd.to_numeric(counts.block, errors="coerce").astype("Int64")
    for column in ("corrected_count_mean", "corrected_count_p2_5", "corrected_count_p97_5"):
        counts[column] = pd.to_numeric(counts[column], errors="coerce")
    counts = counts.dropna(subset=[
        "block_id", "week_key", "corrected_count_mean",
        "corrected_count_p2_5", "corrected_count_p97_5",
    ]).copy()
    counts["variance"] = (
        (counts.corrected_count_p97_5 - counts.corrected_count_p2_5) / (2 * Z_95)
    ) ** 2
    # Block IDs can span municipality/tile rows. Their expected counts add;
    # variances are added under the documented independent-marginal approximation.
    counts = counts.groupby(["block_id", "week_key"], as_index=False).agg(
        corrected_count_mean=("corrected_count_mean", "sum"),
        variance=("variance", "sum"),
    )
    counts["sd"] = np.sqrt(counts.variance)
    counts["week_end"] = counts.week_key.map(
        lambda key: pd.to_datetime(str(key).split("-")[1], format="%Y%m%d")
    )

    blocks = gpd.read_file(BLOCKS_PATH).to_crs(TARGET_CRS).rename(columns={"Name": "block_id"})
    blocks["block_id"] = pd.to_numeric(blocks.block_id, errors="coerce").astype("Int64")
    blocks = blocks.dropna(subset=["block_id"])[["block_id", "geometry"]]
    blocks["block_area_m2"] = blocks.geometry.area

    zones = gpd.read_file(ZONE_PATH).to_crs(TARGET_CRS)
    zones["safe_date_parsed"] = zones.safe_date.map(_date)
    zones["end_date_parsed"] = zones.end_date.map(_date)
    gaza = gpd.read_file(GAZA_PATH).to_crs(TARGET_CRS).geometry.union_all()
    gaza_area_m2 = float(gaza.area)

    rng = np.random.default_rng(RNG_SEED)
    weekly = []
    for week_key, week_counts in counts.groupby("week_key"):
        week_end = week_counts.week_end.iloc[0]
        zone, official_zone_active = _zone_for_date(zones, week_end)
        if zone is None or zone.is_empty:
            continue
        zone = zone.intersection(gaza)
        coverage_path = COVERAGE_DIR / f"{week_key}_coverage.geojson"
        if not coverage_path.exists():
            continue
        coverage = gpd.read_file(coverage_path).to_crs(TARGET_CRS).geometry.union_all().intersection(gaza)
        coverage_fraction = float(coverage.area / gaza_area_m2)
        if coverage_fraction < MIN_GAZA_COVERAGE:
            continue
        spatial = blocks.copy()
        covered_block_area = spatial.geometry.intersection(coverage).area
        covered_zone_area = spatial.geometry.intersection(coverage.intersection(zone)).area
        spatial["fraction_in_zone"] = np.divide(
            covered_zone_area,
            covered_block_area,
            out=np.zeros(len(spatial), dtype=float),
            where=covered_block_area > 0,
        )
        data = week_counts.merge(spatial[["block_id", "fraction_in_zone"]], on="block_id", how="inner")
        means = data.corrected_count_mean.to_numpy(float)
        sds = data.sd.to_numpy(float)
        fractions = data.fraction_in_zone.to_numpy(float)
        draws = np.clip(rng.normal(means, sds, size=(N_DRAWS, len(data))), 0, None)
        total_draws = draws.sum(axis=1)
        zone_draws = draws @ fractions
        share_draws = np.divide(zone_draws, total_draws, out=np.zeros_like(zone_draws), where=total_draws > 0)

        total_mean = float(means.sum())
        zone_mean = float(means @ fractions)
        share = zone_mean / total_mean if total_mean > 0 else 0.0
        zone_area_fraction = float(zone.area / gaza_area_m2)
        accessible_zone_area = float(zone.intersection(coverage).area)
        accessible_zone_fraction = accessible_zone_area / float(coverage.area)
        outside_share = 1 - share
        outside_area_fraction = 1 - accessible_zone_fraction
        density_ratio = (
            (share / accessible_zone_fraction) / (outside_share / outside_area_fraction)
            if accessible_zone_fraction > 0 and outside_share > 0 and outside_area_fraction > 0 else None
        )
        weekly.append({
            "week_key": week_key,
            "week_end": week_end.strftime("%Y-%m-%d"),
            "official_humanitarian_zone_active": official_zone_active,
            "calibrated_tent_count": total_mean,
            "calibrated_tents_in_humanitarian_zone": zone_mean,
            "fraction_of_tents_in_humanitarian_zone": share,
            "percent_of_tents_in_humanitarian_zone": share * 100,
            "percent_of_tents_in_humanitarian_zone_95_cri": [
                float(np.quantile(share_draws, 0.025) * 100),
                float(np.quantile(share_draws, 0.975) * 100),
            ],
            "humanitarian_zone_area_km2": float(zone.area / 1_000_000),
            "humanitarian_zone_percent_of_gaza_land_area": zone_area_fraction * 100,
            "gaza_land_area_covered_percent": coverage_fraction * 100,
            "humanitarian_zone_percent_of_covered_gaza_area": accessible_zone_fraction * 100,
            "tent_density_ratio_zone_vs_remainder_of_covered_gaza": density_ratio,
            "blocks_with_calibrated_counts": int(len(data)),
        })

    if not weekly:
        raise RuntimeError("No calibrated weeks overlap an active humanitarian zone")
    weekly.sort(key=lambda row: row["week_end"])
    baseline = weekly[0]
    peak = max(weekly, key=lambda row: row["fraction_of_tents_in_humanitarian_zone"])
    output = {
        "primary_metric": "Fraction of all calibrated tents located within the contemporaneously designated humanitarian zone.",
        "spatial_allocation_method": "Each calibrated block-week count is allocated to the active zone in proportion to the fraction of the block polygon lying inside it.",
        "temporal_rule": "The humanitarian-zone designation active on the final day of each observation week is used. Where no designation is active, the most recently available humanitarian-zone polygon is carried forward and the week is flagged as unofficial.",
        "coverage_rule": "Only weeks with geospatial prediction coverage of at least 80% of Gaza's land area are included.",
        "uncertainty_method": (
            "10,000 Monte Carlo draws from independent non-negative normal marginal approximations reconstructed from each block-week 95% credible interval. This approximates, rather than recovers, the unavailable joint posterior draws."
        ),
        "land_area_denominator": "Zone size is reported against Gaza's full municipal-boundary land area; the density ratio compares the zone with the remainder of contemporaneously covered Gaza.",
        "gaza_land_area_km2": gaza_area_m2 / 1_000_000,
        "baseline": baseline,
        "peak": peak,
        "change_baseline_to_peak": {
            "percentage_point_change_in_zone_tent_share": (
                peak["percent_of_tents_in_humanitarian_zone"]
                - baseline["percent_of_tents_in_humanitarian_zone"]
            ),
            "fold_change_in_zone_tent_share": (
                peak["fraction_of_tents_in_humanitarian_zone"]
                / baseline["fraction_of_tents_in_humanitarian_zone"]
                if baseline["fraction_of_tents_in_humanitarian_zone"] > 0 else None
            ),
        },
        "weeks": weekly,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
