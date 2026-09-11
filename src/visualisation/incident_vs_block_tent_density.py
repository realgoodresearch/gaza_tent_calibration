"""Smoke-test plot of corrected block tent density against incident counts."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INCIDENTS_PATH = REPO_ROOT / "data/incidents/gaza_incidents.csv"
COUNTS_PATH = REPO_ROOT / "outputs/tent_counts/block_counts_by_week.csv"
BLOCKS_PATH = REPO_ROOT / "data/gaza_boundaries/dash_population_blocks.geojson"
OUTPUT_PATH = REPO_ROOT / "outputs/visualisation/incident_vs_block_tent_density.png"
TARGET_CRS = "EPSG:32636"
FILTER_PATTERN = r"air[\s_-]*strike|attack[\s_-]*on[\s_-]*civilians?"


def main() -> None:
    counts = pd.read_csv(COUNTS_PATH)
    counts["block_id"] = pd.to_numeric(counts.block, errors="coerce").astype("Int64")
    counts["density"] = pd.to_numeric(counts.density_per_km2_median, errors="coerce")
    counts["area"] = pd.to_numeric(counts.covered_area_m2, errors="coerce")
    counts = counts.dropna(subset=["block_id", "week_key", "density", "area"])
    # Block IDs split across source rows are combined with covered-area weights.
    counts["weighted_density"] = counts.density * counts.area
    block_weeks = counts.groupby(["block_id", "week_key"], as_index=False).agg(
        weighted_density=("weighted_density", "sum"),
        covered_area_m2=("area", "sum"),
    )
    block_weeks["corrected_tent_density_per_km2"] = (
        block_weeks.weighted_density / block_weeks.covered_area_m2
    )
    block_weeks["week_start"] = pd.to_datetime(
        block_weeks.week_key.str.split("-").str[0], format="%Y%m%d"
    )
    block_weeks["week_end"] = pd.to_datetime(
        block_weeks.week_key.str.split("-").str[1], format="%Y%m%d"
    )

    incidents = pd.read_csv(INCIDENTS_PATH)
    text = incidents["Incident Tags"].fillna("").astype(str) + " " + incidents["Incident Type"].fillna("").astype(str)
    incidents = incidents[text.str.contains(FILTER_PATTERN, case=False, regex=True)].copy()
    incidents["date"] = pd.to_datetime(incidents.Date, format="%Y-%m-%d", errors="coerce")
    incidents["lat"] = pd.to_numeric(incidents.Lat, errors="coerce")
    incidents["lon"] = pd.to_numeric(incidents.Lon, errors="coerce")
    incidents = incidents.dropna(subset=["date", "lat", "lon"])
    incident_points = gpd.GeoDataFrame(
        incidents,
        geometry=gpd.points_from_xy(incidents.lon, incidents.lat),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)

    blocks = gpd.read_file(BLOCKS_PATH).to_crs(TARGET_CRS).rename(columns={"Name": "block_id"})
    blocks["block_id"] = pd.to_numeric(blocks.block_id, errors="coerce").astype("Int64")
    incident_points = gpd.sjoin(
        incident_points, blocks[["block_id", "geometry"]], how="inner", predicate="within"
    ).drop(columns="index_right")
    incident_points["week_key"] = pd.NA
    for week in block_weeks[["week_key", "week_start", "week_end"]].drop_duplicates().itertuples(index=False):
        mask = incident_points.date.between(week.week_start, week.week_end)
        incident_points.loc[mask, "week_key"] = week.week_key
    incident_counts = incident_points.dropna(subset=["week_key"]).groupby(
        ["block_id", "week_key"]
    ).size().rename("incident_count")

    study_start, study_end = incidents.date.min(), incidents.date.max()
    plot_data = block_weeks[
        (block_weeks.week_end >= study_start) & (block_weeks.week_start <= study_end)
    ].copy()
    plot_data = plot_data.join(incident_counts, on=["block_id", "week_key"])
    plot_data["incident_count"] = plot_data.incident_count.fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    jitter = np.random.default_rng(2026).normal(0, 0.045, len(plot_data))
    ax.scatter(
        plot_data.corrected_tent_density_per_km2,
        plot_data.incident_count + jitter,
        s=12, alpha=0.20, color="#5f3b85", edgecolors="none",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Corrected block tent density (tents per km², posterior median)")
    ax.set_ylabel("Filtered incidents in block-week")
    ax.set_title("Block tent density and airstrike/civilian-attack incidents", loc="left", weight="bold")
    ax.grid(color="#dddddd", linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.99, 0.98,
        f"{len(plot_data):,} block-weeks\n{int(plot_data.incident_count.sum()):,} assigned incidents",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
