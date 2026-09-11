"""Plot six validation tiles beside pre-war imagery with corrected predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from shapely.geometry import Point


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration.workflow.postprocessing import (  # noqa: E402
    apply_weekly_postprocessing,
    load_yellow_line_context,
    parse_week_key_range,
)

VALIDATION_CSV = REPO_ROOT / "src/validation/prediction_accuracy_results.csv"
DEFAULT_H5 = REPO_ROOT / "src/validation/tile_validation_samples.h5"
HIGH_DENSITY_H5 = REPO_ROOT / "src/validation/tile_validation_sample_high_density.h5"
PREDICTIONS_DIR = REPO_ROOT / "data/tent_predictions/merged"
BLOCKS_PATH = REPO_ROOT / "data/gaza_boundaries/dash_population_blocks.geojson"
BLOCK_COUNTS_PATH = REPO_ROOT / "outputs/tent_counts/block_counts_by_week.csv"
YELLOW_LINE_PATH = REPO_ROOT / "data/spatial_context/yellow-line-zone.json"
OUTPUT = REPO_ROOT / "outputs/visualisation/example_predictions.png"
TARGET_CRS = "EPSG:32636"
TILE_HALF_WIDTH = 50.0

# Source must be explicit because h5_index is local to each HDF5 file.
EXAMPLES = [
    ("default", 667),
    ("default", 843),
    ("high_density", 8),
    ("high_density", 5),
    ("high_density", 158),
    ("high_density", 199),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--dbscan-eps-m", type=float, default=40.0)
    parser.add_argument("--dbscan-min-samples", type=int, default=3)
    return parser


def _week_for_date(date: str) -> str:
    target = pd.Timestamp(date).normalize()
    for path in sorted(PREDICTIONS_DIR.glob("*.gpkg")):
        start, end = parse_week_key_range(path.stem)
        if pd.notna(start) and pd.notna(end) and start.normalize() <= target <= end.normalize():
            return path.stem
    raise ValueError(f"No prediction week contains {target.date()}")


def _load_examples() -> pd.DataFrame:
    validation = pd.read_csv(VALIDATION_CSV)
    selected = []
    for source, index in EXAMPLES:
        match = validation[
            validation["h5_source"].astype(str).eq(source)
            & pd.to_numeric(validation["h5_index"], errors="coerce").eq(index)
        ]
        if len(match) != 1:
            raise ValueError(f"Expected one row for {source}:{index}; found {len(match)}")
        selected.append(match.iloc[0])
    return pd.DataFrame(selected).reset_index(drop=True)


def _load_images(source: str, index: int) -> tuple[np.ndarray, np.ndarray]:
    path = HIGH_DENSITY_H5 if source == "high_density" else DEFAULT_H5
    with h5py.File(path, "r") as h5:
        group = h5[f"sample_{index}"]
        return np.asarray(group["prewar_rgb"]), np.asarray(group["historic_rgb"])


def _load_blocks() -> gpd.GeoDataFrame:
    blocks = gpd.read_file(BLOCKS_PATH)
    blocks = blocks.set_crs("EPSG:4326") if blocks.crs is None else blocks
    blocks = blocks.to_crs(TARGET_CRS)[["Name", "geometry"]].copy()
    blocks["block_id"] = pd.to_numeric(blocks["Name"], errors="coerce").astype("Int64")
    return blocks.dropna(subset=["block_id"])


def _load_corrected_counts() -> pd.Series:
    counts = pd.read_csv(BLOCK_COUNTS_PATH)
    counts["block_id"] = pd.to_numeric(counts["block"], errors="coerce").astype("Int64")
    counts["corrected_count_median"] = pd.to_numeric(
        counts["corrected_count_median"], errors="coerce"
    )
    counts = counts.dropna(subset=["week_key", "block_id", "corrected_count_median"])
    # Blocks that cross municipal reporting units have multiple component rows.
    return counts.groupby(["week_key", "block_id"])["corrected_count_median"].sum()


def _containing_block(blocks: gpd.GeoDataFrame, center_x: float, center_y: float) -> pd.Series:
    point = Point(center_x, center_y)
    match = blocks[blocks.geometry.covers(point)]
    if match.empty:
        raise ValueError(f"No evacuation block covers tile centre ({center_x}, {center_y})")
    if len(match) > 1:
        match = match.assign(_area=match.geometry.area).sort_values("_area")
    return match.iloc[0]


def _postprocessed_week(
    week: str,
    cache: dict[str, gpd.GeoDataFrame],
    yellow,
    yellow_windows,
    eps_m: float,
    min_samples: int,
) -> gpd.GeoDataFrame:
    if week not in cache:
        points = gpd.read_file(PREDICTIONS_DIR / f"{week}.gpkg")
        points = points.set_crs(TARGET_CRS) if points.crs is None else points.to_crs(TARGET_CRS)
        points, _ = apply_weekly_postprocessing(
            points, week, yellow, yellow_windows, eps_m=eps_m, min_samples=min_samples
        )
        if "adjusted_peak" not in points:
            raise ValueError(f"Post-processed predictions for {week} lack adjusted_peak")
        points["adjusted_peak"] = pd.to_numeric(points["adjusted_peak"], errors="coerce")
        cache[week] = points
    return cache[week]


def _retained_tile_predictions(
    row: pd.Series,
    blocks: gpd.GeoDataFrame,
    corrected_counts: pd.Series,
    postprocessed_cache: dict[str, gpd.GeoDataFrame],
    yellow,
    yellow_windows,
    eps_m: float,
    min_samples: int,
) -> tuple[gpd.GeoDataFrame, dict[str, int | str]]:
    center_x, center_y = float(row.center_x), float(row.center_y)
    week = _week_for_date(str(row.date))
    block = _containing_block(blocks, center_x, center_y)
    block_id = int(block.block_id)
    try:
        corrected_count = max(0, int(round(float(corrected_counts.loc[(week, block_id)]))))
    except KeyError as exc:
        raise ValueError(f"No corrected count for block {block_id}, week {week}") from exc

    postprocessed = _postprocessed_week(
        week, postprocessed_cache, yellow, yellow_windows, eps_m, min_samples
    )
    block_points = postprocessed[postprocessed.geometry.intersects(block.geometry)].copy()
    retained = block_points.nlargest(min(corrected_count, len(block_points)), "adjusted_peak")
    tile = retained[
        retained.geometry.x.between(center_x - TILE_HALF_WIDTH, center_x + TILE_HALF_WIDTH)
        & retained.geometry.y.between(center_y - TILE_HALF_WIDTH, center_y + TILE_HALF_WIDTH)
    ].copy()
    return tile, {
        "week": week,
        "block_id": block_id,
        "corrected_block_count": corrected_count,
        "postprocessed_block_points": int(len(block_points)),
        "retained_block_points": int(len(retained)),
        "tile_prediction": int(len(tile)),
    }


def _spatial_to_pixel(x: float, y: float, center_x: float, center_y: float,
                      image_width: int, image_height: int) -> tuple[float, float]:
    col = (x - (center_x - TILE_HALF_WIDTH)) * image_width / (2 * TILE_HALF_WIDTH)
    row = ((center_y + TILE_HALF_WIDTH) - y) * image_height / (2 * TILE_HALF_WIDTH)
    return col, row


def main() -> None:
    args = build_parser().parse_args()
    examples = _load_examples()
    blocks = _load_blocks()
    corrected_counts = _load_corrected_counts()
    yellow, yellow_windows = load_yellow_line_context(YELLOW_LINE_PATH)
    cache: dict[str, gpd.GeoDataFrame] = {}

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.titleweight": "normal",
    })
    fig = plt.figure(figsize=(12.5, 9.6), facecolor="white")
    # A narrow spacer column separates examples while each pre-war/current
    # pair uses adjacent columns with almost no internal gap.
    grid = fig.add_gridspec(
        3, 5, width_ratios=[1, 1, 0.16, 1, 1], wspace=0.025, hspace=0.30
    )
    summaries = []
    for example_number, (_, row) in enumerate(examples.iterrows()):
        source, index = str(row.h5_source), int(row.h5_index)
        prewar, current = _load_images(source, index)
        predictions, summary = _retained_tile_predictions(
            row, blocks, corrected_counts, cache, yellow, yellow_windows,
            args.dbscan_eps_m, args.dbscan_min_samples,
        )
        observed = int(row.tent_count)
        predicted = int(summary["tile_prediction"])
        absolute_error = abs(predicted - observed)
        summary.update(source=source, sample=index, observed=observed,
                       absolute_error=absolute_error)
        summaries.append(summary)

        grid_row = example_number // 2
        start_col = 0 if example_number % 2 == 0 else 3
        ax_prewar = fig.add_subplot(grid[grid_row, start_col])
        ax_current = fig.add_subplot(grid[grid_row, start_col + 1])
        ax_prewar.imshow(prewar)
        ax_current.imshow(current)
        # Square imagery leaves spare horizontal room in these grid cells;
        # anchor that room outside the pair so the two dates read as a unit.
        ax_prewar.set_anchor("E")
        ax_current.set_anchor("W")
        date_value = pd.Timestamp(row.date)
        date_label = f"{date_value.day} {date_value:%b %Y}"
        ax_prewar.set_title(
            f"{chr(65 + example_number)}  {row.municipality} · {date_label}\nPre-war",
            loc="left", pad=4,
        )
        ax_current.set_title(
            "Current",
            loc="left", pad=4,
        )
        height, width = current.shape[:2]
        for geometry in predictions.geometry:
            col, pixel_row = _spatial_to_pixel(
                geometry.x, geometry.y, float(row.center_x), float(row.center_y), width, height
            )
            if 0 <= col < width and 0 <= pixel_row < height:
                ax_current.add_patch(Circle(
                    (col, pixel_row), radius=3.2, fill=False, edgecolor="#e31a1c",
                    linewidth=1.35, alpha=0.9,
                ))
        for axis in (ax_prewar, ax_current):
            axis.set_axis_off()

    fig.suptitle(
        "Examples of corrected tent predictions in 100 × 100 m validation tiles",
        x=0.02, y=0.995, ha="left", fontsize=15, fontweight="normal",
    )
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.01, top=0.94)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
