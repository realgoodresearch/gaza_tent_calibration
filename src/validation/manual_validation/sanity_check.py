from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parents[3]
else:
    REPO_ROOT = Path.cwd()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibration.workflow.postprocessing import (  # noqa: E402
    apply_weekly_postprocessing,
    load_yellow_line_context,
    parse_week_key_range,
)

H5_PATH = REPO_ROOT / "src" / "validation" / "tile_validation_samples.h5"
H5_HIGH_DENSITY_PATH = REPO_ROOT / "src" / "validation" / "tile_validation_sample_high_density.h5"
CSV_PATH = REPO_ROOT / "src" / "validation" / "prediction_accuracy_results.csv"
TENT_PREDICTIONS_DIR = REPO_ROOT / "data" / "tent_predictions" / "merged"
YELLOW_LINE_PATH = REPO_ROOT / "data" / "spatial_context" / "yellow-line-zone.json"
TILE_HALF_WIDTH = 50.0  # meters
IMAGE_SIZE = 201  # pixels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check manual vs predicted tent counts")
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--h5-path", type=Path, default=H5_PATH)
    parser.add_argument("--h5-high-density-path", type=Path, default=H5_HIGH_DENSITY_PATH)
    parser.add_argument("--h5-source", type=str, default=None, help="Optional source filter (e.g., default, high_density)")
    parser.add_argument("--prediction-column", type=str, default="prediction_count")
    parser.add_argument("--summary-only", action="store_true", help="Print summary + random examples without opening plots")
    parser.add_argument("--random-n", type=int, default=10, help="Number of random rows to print in summary mode")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--h5-index", type=int, default=None, help="Show one specific h5_index and exit")
    return parser.parse_args()


def load_csv_data(csv_path: Path, prediction_column: str) -> pd.DataFrame:
    """Load prediction accuracy results from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "h5_index" not in df.columns or "tent_count" not in df.columns:
        raise ValueError(f"CSV must contain 'h5_index' and 'tent_count' columns")
    if prediction_column not in df.columns:
        raise ValueError(f"CSV must contain '{prediction_column}' column")

    if prediction_column != "prediction_count":
        df = df.copy()
        df["prediction_count"] = pd.to_numeric(df[prediction_column], errors="coerce")

    return df


def load_h5_images(h5_path: Path, h5_index: int) -> Dict[str, np.ndarray]:
    """Load prewar and historic images from HDF5 file."""
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        if f"sample_{h5_index}" not in h5:
            raise IndexError(f"Sample {h5_index} not found in HDF5 file")

        group = h5[f"sample_{h5_index}"]
        return {
            "prewar_rgb": np.array(group["prewar_rgb"]),
            "historic_rgb": np.array(group["historic_rgb"]),
        }


def _resolve_h5_path_for_source(args: argparse.Namespace, h5_source: str | None) -> Path:
    src = str(h5_source or "").strip().lower()
    if src == "high_density":
        return args.h5_high_density_path
    return args.h5_path


def find_geopackage_for_date(date_str: str) -> Optional[Path]:
    """Find the geopackage file that contains the given date."""
    if not date_str or pd.isna(date_str):
        return None

    try:
        target_date = pd.to_datetime(date_str).date()
    except Exception:
        return None

    if not TENT_PREDICTIONS_DIR.exists():
        return None

    for gpkg_file in sorted(TENT_PREDICTIONS_DIR.glob("*.gpkg")):
        # Parse date range from filename like "20250630-20250706.gpkg"
        stem = gpkg_file.stem
        if len(stem) == 17 and stem[8] == "-":  # YYYYMMDD-YYYYMMDD format
            try:
                start_date = pd.to_datetime(stem[:8]).date()
                end_date = pd.to_datetime(stem[9:17]).date()
                if start_date <= target_date <= end_date:
                    return gpkg_file
            except Exception:
                continue

    return None


def week_key_for_date(date_str: str) -> Optional[str]:
    if not date_str or pd.isna(date_str):
        return None
    try:
        target_date = pd.to_datetime(date_str).normalize()
    except Exception:
        return None

    for gpkg_file in sorted(TENT_PREDICTIONS_DIR.glob("*.gpkg")):
        wk = gpkg_file.stem
        start, end = parse_week_key_range(wk)
        if pd.isna(start) or pd.isna(end):
            continue
        if start.normalize() <= target_date <= end.normalize():
            return wk
    return None


def load_tent_predictions(center_x: float, center_y: float, date_str: str) -> Optional[gpd.GeoDataFrame]:
    """Load tent predictions that fall within the tile bounds."""
    gpkg_path = find_geopackage_for_date(date_str)
    if gpkg_path is None:
        return None

    try:
        gdf = gpd.read_file(gpkg_path)
        if gdf.empty or gdf.crs is None:
            return None

        # Ensure we're in EPSG:32636
        if gdf.crs != "EPSG:32636":
            gdf = gdf.to_crs("EPSG:32636")

        # Filter to points within tile bounds
        min_x = center_x - TILE_HALF_WIDTH
        max_x = center_x + TILE_HALF_WIDTH
        min_y = center_y - TILE_HALF_WIDTH
        max_y = center_y + TILE_HALF_WIDTH

        bounds = gdf.total_bounds
        if bounds[0] > max_x or bounds[2] < min_x or bounds[1] > max_y or bounds[3] < min_y:
            # No overlap
            return None

        # Clip to tile
        mask = (
            (gdf.geometry.x >= min_x)
            & (gdf.geometry.x <= max_x)
            & (gdf.geometry.y >= min_y)
            & (gdf.geometry.y <= max_y)
        )
        return gdf[mask] if mask.any() else None

    except Exception as e:
        print(f"Warning: Could not load predictions for {date_str}: {e}")
        return None


def split_raw_and_postprocessed_predictions(
    center_x: float,
    center_y: float,
    date_str: str,
    eps_m: float = 40.0,
    min_samples: int = 3,
) -> Tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame], int, int]:
    gpkg_path = find_geopackage_for_date(date_str)
    if gpkg_path is None:
        return None, None, 0, 0

    week_key = week_key_for_date(date_str)
    if week_key is None:
        return None, None, 0, 0

    try:
        gdf = gpd.read_file(gpkg_path)
        if gdf.empty or gdf.crs is None:
            return None, None, 0, 0

        if gdf.crs != "EPSG:32636":
            gdf = gdf.to_crs("EPSG:32636")

        min_x = center_x - TILE_HALF_WIDTH
        max_x = center_x + TILE_HALF_WIDTH
        min_y = center_y - TILE_HALF_WIDTH
        max_y = center_y + TILE_HALF_WIDTH

        raw_mask = (
            (gdf.geometry.x >= min_x)
            & (gdf.geometry.x <= max_x)
            & (gdf.geometry.y >= min_y)
            & (gdf.geometry.y <= max_y)
        )
        raw_tile = gdf[raw_mask].copy() if raw_mask.any() else gdf.iloc[0:0].copy()

        yellow, windows = load_yellow_line_context(YELLOW_LINE_PATH)
        gdf_post, _ = apply_weekly_postprocessing(gdf, week_key, yellow, windows, eps_m=eps_m, min_samples=min_samples)
        post_mask = (
            (gdf_post.geometry.x >= min_x)
            & (gdf_post.geometry.x <= max_x)
            & (gdf_post.geometry.y >= min_y)
            & (gdf_post.geometry.y <= max_y)
        )
        post_tile = gdf_post[post_mask].copy() if post_mask.any() else gdf_post.iloc[0:0].copy()
        return raw_tile, post_tile, int(len(raw_tile)), int(len(post_tile))
    except Exception as e:
        print(f"Warning: Could not load raw/postprocessed predictions for {date_str}: {e}")
        return None, None, 0, 0


def spatial_to_pixel_coords(spatial_x: float, spatial_y: float, center_x: float, center_y: float) -> tuple:
    """Convert spatial coordinates (EPSG:32636) to image pixel coordinates."""
    # Image is 201x201 pixels representing a 100m x 100m area centered at (center_x, center_y)
    # Pixel (0, 0) is at the top-left
    pixel_col = (spatial_x - (center_x - TILE_HALF_WIDTH)) * IMAGE_SIZE / (2 * TILE_HALF_WIDTH)
    # Flip y-axis: spatial y increases upward, image y increases downward
    pixel_row = ((center_y + TILE_HALF_WIDTH) - spatial_y) * IMAGE_SIZE / (2 * TILE_HALF_WIDTH)
    return pixel_col, pixel_row


def show_sample(row: pd.Series, images: Dict[str, np.ndarray]) -> None:
    """Display a sample with images, tent counts, and prediction overlays."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Get spatial coordinates for prediction overlay
    center_x = row.get("center_x")
    center_y = row.get("center_y")
    date_str = row.get("date")

    # Load predictions if available
    predictions_gdf_raw = None
    predictions_gdf_post = None
    raw_count = 0
    post_count = 0
    if pd.notna(center_x) and pd.notna(center_y) and date_str:
        predictions_gdf_raw, predictions_gdf_post, raw_count, post_count = split_raw_and_postprocessed_predictions(
            float(center_x),
            float(center_y),
            date_str,
        )

    # Display images with prediction overlays
    panel_defs = [
        (axes[0], "prewar_rgb", "Prewar", None, "white"),
        (axes[1], "historic_rgb", f"Current + Raw ({raw_count})", predictions_gdf_raw, "yellow"),
        (axes[2], "historic_rgb", f"Current + Postprocessed ({post_count})", predictions_gdf_post, "cyan"),
    ]
    for ax, img_key, title, panel_preds, color in panel_defs:
        ax.imshow(images[img_key])
        ax.set_title(title)
        ax.axis("off")

        if panel_preds is not None and len(panel_preds) > 0:
            for _, pred_row in panel_preds.iterrows():
                geom = pred_row.geometry
                # Handle both Point and MultiPoint geometries
                if geom.geom_type == "Point":
                    points = [(geom.x, geom.y)]
                elif geom.geom_type == "MultiPoint":
                    points = [(pt.x, pt.y) for pt in geom.geoms]
                else:
                    continue

                for px, py in points:
                    pixel_col, pixel_row = spatial_to_pixel_coords(px, py, float(center_x), float(center_y))
                    # Only plot if within image bounds
                    if 0 <= pixel_col < IMAGE_SIZE and 0 <= pixel_row < IMAGE_SIZE:
                        circle = plt.Circle(
                            (pixel_col, pixel_row),
                            radius=3,
                            fill=False,
                            edgecolor=color,
                            linewidth=1.5,
                            alpha=0.5,
                        )
                        ax.add_patch(circle)

    # Build title with metadata
    h5_idx = int(row["h5_index"])
    municipality = row.get("municipality", "N/A")
    date = row.get("date", "N/A")
    tent_count = int(row["tent_count"]) if pd.notna(row["tent_count"]) else "N/A"
    prediction_count = int(row["prediction_count"]) if pd.notna(row["prediction_count"]) else "N/A"

    title = f"Sample {h5_idx} | {municipality} | {date}"
    fig.suptitle(title, fontsize=11)

    # Add tent count information at bottom (manual vs prediction)
    info_text = (
        f"Manual: {tent_count}  |  CSV Prediction: {prediction_count}  |  "
        f"Raw Tile Count: {raw_count}  |  Postprocessed Tile Count: {post_count}"
    )
    fig.text(
        0.5,
        0.02,
        info_text,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Add navigation buttons
    prev_ax = fig.add_axes([0.35, 0.05, 0.08, 0.04])
    next_ax = fig.add_axes([0.57, 0.05, 0.08, 0.04])

    prev_button = Button(prev_ax, "Previous")
    next_button = Button(next_ax, "Next")

    # Navigation state
    nav_state = {"action": None}

    def on_prev(_event):
        nav_state["action"] = "prev"
        plt.close(fig)

    def on_next(_event):
        nav_state["action"] = "next"
        plt.close(fig)

    def on_key(event):
        if event.key in {"left", "a"}:
            on_prev(None)
        elif event.key in {"right", "d"}:
            on_next(None)
        elif event.key in {"q", "escape"}:
            nav_state["action"] = "quit"
            plt.close(fig)

    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)
    fig.canvas.mpl_connect("key_press_event", on_key)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.12, wspace=0.04)
    plt.show(block=True)

    return nav_state.get("action")


def _print_summary(df: pd.DataFrame, random_n: int, seed: int) -> None:
    pred = pd.to_numeric(df["prediction_count"], errors="coerce")
    manual = pd.to_numeric(df["tent_count"], errors="coerce")
    err = pred - manual
    abs_err = err.abs()
    print(f"Rows: {len(df)}")
    print(f"Manual mean: {manual.mean():.2f} | Pred mean: {pred.mean():.2f}")
    print(f"MAE: {abs_err.mean():.2f} | RMSE: {np.sqrt(np.mean(np.square(err.dropna()))):.2f} | Bias: {err.mean():.2f}")
    show_cols = [c for c in ["h5_source", "h5_index", "municipality", "date", "tent_count", "prediction_count"] if c in df.columns]
    n = min(int(random_n), len(df))
    sample = df.sample(n=n, random_state=int(seed))[show_cols]
    print("\nRandom sample rows:")
    print(sample.to_string(index=False))


def main() -> None:
    """Main loop for sanity checking reviewed samples."""
    args = parse_args()
    try:
        df = load_csv_data(args.csv_path, args.prediction_column)

        if args.h5_source:
            df = df[df.get("h5_source", "").astype(str).str.lower() == str(args.h5_source).strip().lower()].copy()

        if len(df) == 0:
            print("No samples found in CSV.")
            return

        print(f"Loaded {len(df)} samples from {args.csv_path}")
        print("Comparing manual tent counts against model predictions")
        if args.summary_only:
            _print_summary(df, random_n=args.random_n, seed=args.seed)
            return

        if args.h5_index is not None:
            subset = df[pd.to_numeric(df["h5_index"], errors="coerce") == int(args.h5_index)]
            if subset.empty:
                raise ValueError(f"No row found with h5_index={args.h5_index}")
            if len(subset) > 1 and not args.h5_source:
                rows = subset[[c for c in ["h5_source", "h5_index", "date", "municipality"] if c in subset.columns]]
                raise ValueError(
                    f"h5_index={args.h5_index} has multiple rows across sources. "
                    f"Re-run with --h5-source. Matches:\n{rows.to_string(index=False)}"
                )
            row = subset.iloc[0]
            h5_source = str(row.get("h5_source", "default"))
            chosen_h5 = _resolve_h5_path_for_source(args, h5_source)
            images = load_h5_images(chosen_h5, int(row["h5_index"]))
            show_sample(row, images)
            return

        print("\nControls:")
        print("  - Left arrow / A: Previous random sample")
        print("  - Right arrow / D: Next random sample")
        print("  - Q / Escape: Quit")
        print()

        # Start with a random sample
        np.random.seed()
        current_idx = np.random.randint(0, len(df))

        while True:
            try:
                row = df.iloc[current_idx]
                h5_index = int(row["h5_index"])

                # Load and display
                h5_source = str(row.get("h5_source", "default"))
                chosen_h5 = _resolve_h5_path_for_source(args, h5_source)
                images = load_h5_images(chosen_h5, h5_index)
                action = show_sample(row, images)

                # Handle navigation
                if action == "quit" or action is None:
                    break
                elif action == "next":
                    current_idx = np.random.randint(0, len(df))
                elif action == "prev":
                    current_idx = np.random.randint(0, len(df))

            except (IndexError, KeyError) as e:
                print(f"Error loading sample at index {current_idx}: {e}")
                print("Skipping to next random sample...")
                current_idx = np.random.randint(0, len(df))
                continue

        print("Sanity check complete.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
