"""Plot the share of corrected tents in blocks exposed to military activity."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import shape

REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTS_PATH = REPO_ROOT / "outputs/tent_counts/block_counts_by_week.csv"
BLOCKS_PATH = REPO_ROOT / "data/gaza_boundaries/dash_population_blocks.geojson"
INCIDENTS_PATH = REPO_ROOT / "data/incidents/gaza_incidents.csv"
TANKS_PATH = REPO_ROOT / "data/military_activity/tanks.geojson"
BERMS_PATH = REPO_ROOT / "data/military_activity/berms.geojson"
OUTPUT_PATH = REPO_ROOT / "outputs/visualisation/tent_exposure_to_military_activity.png"
OUTPUT_DATA_PATH = REPO_ROOT / "outputs/visualisation/tent_exposure_to_military_activity.csv"
TARGET_CRS = "EPSG:32636"
N_DRAWS = 2_000
RNG_SEED = 2026
Z_95 = 1.959963984540054
INCIDENT_PATTERN = r"air[\s_-]*strike|attack[\s_-]*on[\s_-]*civilians?"
START_DATE = pd.Timestamp("2023-10-01")
END_DATE = pd.Timestamp("2026-06-30")
EVENTS = [
    ("2023-10-13", "North Gaza evacuation order"),
    ("2024-05-06", "Rafah evacuation order"),
    ("2025-01-19", "First ceasefire"),
    ("2025-03-18", "End of ceasefire"),
    ("2025-09-09", "Gaza City evacuation order"),
    ("2025-10-10", "Second ceasefire"),
]
RED = "#b23a48"
TEAL = "#087f8c"
GRID = "#dddddd"


def _read_berms() -> gpd.GeoDataFrame:
    payload = json.loads(BERMS_PATH.read_text())
    rows = []
    for feature in payload["features"]:
        try:
            geometry = shape(feature["geometry"])
        except Exception:
            continue
        rows.append({**feature.get("properties", {}), "geometry": geometry})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(TARGET_CRS)


def _assign_week(frame: gpd.GeoDataFrame, weeks: pd.DataFrame) -> gpd.GeoDataFrame:
    frame = frame.copy()
    frame["week_key"] = pd.NA
    for week in weeks.itertuples(index=False):
        frame.loc[frame.date.between(week.week_start, week.week_end), "week_key"] = week.week_key
    return frame.dropna(subset=["week_key"])


def _activity_blocks(
    features: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    weeks: pd.DataFrame,
    predicate: str,
) -> dict[str, set[int]]:
    features = _assign_week(features, weeks)
    joined = gpd.sjoin(
        features[["week_key", "geometry"]], blocks[["block_id", "geometry"]],
        how="inner", predicate=predicate,
    )
    return {
        str(week): set(group.block_id.dropna().astype(int))
        for week, group in joined.groupby("week_key")
    }


def _share(draws: np.ndarray, means: np.ndarray, exposed: np.ndarray) -> tuple[float, float, float]:
    total_draws = draws.sum(axis=1)
    exposed_draws = draws[:, exposed].sum(axis=1) if exposed.any() else np.zeros(len(draws))
    shares = np.divide(exposed_draws, total_draws, out=np.zeros_like(exposed_draws), where=total_draws > 0) * 100
    point = means[exposed].sum() / means.sum() * 100 if means.sum() > 0 else 0.0
    return point, float(np.quantile(shares, 0.025)), float(np.quantile(shares, 0.975))


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
    counts = counts.groupby(["block_id", "week_key"], as_index=False).agg(
        corrected_count_mean=("corrected_count_mean", "sum"),
        variance=("variance", "sum"),
    )
    counts["sd"] = np.sqrt(counts.variance)
    weeks = counts[["week_key"]].drop_duplicates().copy()
    weeks["week_start"] = pd.to_datetime(weeks.week_key.str.split("-").str[0], format="%Y%m%d")
    weeks["week_end"] = pd.to_datetime(weeks.week_key.str.split("-").str[1], format="%Y%m%d")
    weeks = weeks.sort_values("week_start").reset_index(drop=True)

    blocks = gpd.read_file(BLOCKS_PATH).to_crs(TARGET_CRS).rename(columns={"Name": "block_id"})
    blocks["block_id"] = pd.to_numeric(blocks.block_id, errors="coerce").astype("Int64")
    blocks = blocks.dropna(subset=["block_id"])[["block_id", "geometry"]]

    raw_incidents = pd.read_csv(INCIDENTS_PATH)
    incident_text = (
        raw_incidents["Incident Tags"].fillna("").astype(str) + " "
        + raw_incidents["Incident Type"].fillna("").astype(str)
    )
    raw_incidents = raw_incidents[
        incident_text.str.contains(INCIDENT_PATTERN, case=False, regex=True)
    ].copy()
    raw_incidents["date"] = pd.to_datetime(raw_incidents.Date, errors="coerce")
    raw_incidents["lat"] = pd.to_numeric(raw_incidents.Lat, errors="coerce")
    raw_incidents["lon"] = pd.to_numeric(raw_incidents.Lon, errors="coerce")
    raw_incidents = raw_incidents.dropna(subset=["date", "lat", "lon"])
    incidents = gpd.GeoDataFrame(
        raw_incidents,
        geometry=gpd.points_from_xy(raw_incidents.lon, raw_incidents.lat),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)
    incident_blocks = _activity_blocks(incidents, blocks, weeks, "within")

    tanks = gpd.read_file(TANKS_PATH).to_crs(TARGET_CRS)
    tanks["date"] = pd.to_datetime(tanks.date, errors="coerce")
    tanks = tanks.dropna(subset=["date", "geometry"])
    tank_blocks = _activity_blocks(tanks, blocks, weeks, "within")
    berms = _read_berms()
    berms["date"] = pd.to_datetime(berms.date, errors="coerce")
    berms = berms.dropna(subset=["date", "geometry"])
    berm_blocks = _activity_blocks(berms, blocks, weeks, "intersects")
    ground_blocks = {
        week: tank_blocks.get(week, set()) | berm_blocks.get(week, set())
        for week in set(tank_blocks) | set(berm_blocks)
    }

    rng = np.random.default_rng(RNG_SEED)
    output_rows = []
    for week in weeks.itertuples(index=False):
        block_week = counts[counts.week_key == week.week_key]
        block_ids = block_week.block_id.astype(int).to_numpy()
        means = block_week.corrected_count_mean.to_numpy(float)
        sds = block_week.sd.to_numpy(float)
        draws = np.clip(rng.normal(means, sds, size=(N_DRAWS, len(block_week))), 0, None)
        incident_result = _share(draws, means, np.isin(block_ids, list(incident_blocks.get(week.week_key, set()))))
        ground_result = _share(draws, means, np.isin(block_ids, list(ground_blocks.get(week.week_key, set()))))
        output_rows.append({
            "week_key": week.week_key,
            "week_end": week.week_end,
            "airstrike_or_civilian_attack_percent": incident_result[0],
            "airstrike_or_civilian_attack_ci_low": incident_result[1],
            "airstrike_or_civilian_attack_ci_high": incident_result[2],
            "ground_invasion_percent": ground_result[0],
            "ground_invasion_ci_low": ground_result[1],
            "ground_invasion_ci_high": ground_result[2],
        })
    data = pd.DataFrame(output_rows)
    data.to_csv(OUTPUT_DATA_PATH, index=False)

    figure_data = data[data.week_end.between(START_DATE, END_DATE)]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 14, "axes.labelsize": 11, "axes.spines.top": False,
    })
    fig, (ax_air, ax_ground) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True, facecolor="white",
        gridspec_kw={"hspace": 0.16},
    )
    panels = [
        (
            ax_air, "airstrike_or_civilian_attack", RED,
            "A   Airstrikes and attacks on civilians",
        ),
        (
            ax_ground, "ground_invasion", TEAL,
            "B   Ground invasion: newly observed tanks or berms",
        ),
    ]
    ymax = max(
        float(figure_data.airstrike_or_civilian_attack_ci_high.max()),
        float(figure_data.ground_invasion_ci_high.max()),
    ) * 1.08
    for axis, prefix, color, title in panels:
        values = figure_data[f"{prefix}_percent"].to_numpy(float)
        error = np.vstack([
            values - figure_data[f"{prefix}_ci_low"].to_numpy(float),
            figure_data[f"{prefix}_ci_high"].to_numpy(float) - values,
        ])
        axis.bar(
            figure_data.week_end, values, width=6,
            color=color, alpha=0.82, edgecolor="none",
            yerr=error, capsize=1.3,
            error_kw={"ecolor": color, "elinewidth": 0.65, "alpha": 0.55},
        )
        axis.set_ylabel("Tents in exposed blocks (%)")
        axis.set_title(title, loc="left")
        axis.grid(axis="y", color=GRID, lw=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylim(0, ymax)
        axis.set_xlim(START_DATE, END_DATE)

    ax_ground.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_ground.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    for tick_label in ax_ground.get_xticklabels():
        tick_label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 1.0})
        tick_label.set_zorder(5)

    for index, (date_text, label) in enumerate(EVENTS):
        date = pd.Timestamp(date_text)
        for axis in (ax_air, ax_ground):
            axis.axvline(date, color="#777777", lw=0.8, ls=(0, (3, 3)), alpha=0.65, zorder=0)
        ax_ground.annotate(
            label, xy=(date, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -42 if index % 2 == 0 else -66), textcoords="offset points",
            ha="center", va="top", fontsize=7.5, color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            arrowprops={
                "arrowstyle": "-", "color": "#777777", "lw": 0.8,
                "linestyle": (0, (3, 3)),
            },
            annotation_clip=False, zorder=5,
        )

    fig.suptitle(
        "Share of tents exposed to military activity in the same block-week",
        x=0.09, ha="left", fontsize=16,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.20)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")
    print(f"Saved {OUTPUT_DATA_PATH}")


if __name__ == "__main__":
    main()
