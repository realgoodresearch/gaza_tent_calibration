from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, TextBox
from shapely.geometry import Point

if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parents[2]
else:
    REPO_ROOT = Path.cwd()

# utils.py lives in validation/, one level up from this manual_validation/ package.
sys.path.insert(0, str(REPO_ROOT / "validation"))
from utils import SUN_SATELLITE_FIELDS, get_tile_polygon, parse_date_value

H5_PATH = REPO_ROOT / "validation" / "tile_validation_samples.h5"
H5_HIGH_DENSITY_PATH = REPO_ROOT / "validation" / "tile_validation_sample_high_density.h5"
CSV_PATH = REPO_ROOT / "validation" / "manual_validation_results.csv"
MODEL_DIR = REPO_ROOT / "validation" / "land_use_classification"
MODEL_PATH = MODEL_DIR / "land_use_kmeans.pkl"
LABELS_PATH = MODEL_DIR / "manual_land_use_labels.json"
SPATIAL_CONTEXT_DIR = REPO_ROOT / "data" / "spatial_context"


def parse_date_and_time_from_filename(source_tif: str) -> Tuple[str, str]:
    """Extract date and time from filenames like ..._20241014_120709_....tif."""
    if not source_tif:
        return "", ""

    stem = Path(source_tif).stem
    parts = stem.split("_")

    for idx in range(len(parts) - 1):
        candidate = parts[idx]
        next_candidate = parts[idx + 1]
        if len(candidate) == 8 and candidate.isdigit():
            date_value = f"{candidate[:4]}-{candidate[4:6]}-{candidate[6:8]}"
            if len(next_candidate) == 6 and next_candidate.isdigit():
                time_value = f"{next_candidate[:2]}:{next_candidate[2:4]}:{next_candidate[4:6]}"
                return date_value, time_value
            return date_value, ""

    return "", ""


def load_h5_sample(index: int, h5_path: Path) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as h5:
        if f"sample_{index}" not in h5:
            raise IndexError(f"HDF5 index {index} does not exist.")

        group = h5[f"sample_{index}"]
        source_tif = str(group.attrs.get("source_tif", ""))
        date_value = str(group.attrs.get("date", ""))
        time_value = str(group.attrs.get("time_of_day", ""))

        if not date_value or not time_value:
            date_value, time_value = parse_date_and_time_from_filename(source_tif)

        sample = {
            "h5_index": index,
            "municipality": group.attrs.get("municipality", ""),
            "source_tif": source_tif,
            "date": date_value,
            "time_of_day": time_value,
            "center_x": float(group.attrs.get("center_x", np.nan)),
            "center_y": float(group.attrs.get("center_y", np.nan)),
            "historic_rgb": np.array(group["historic_rgb"]),
            "prewar_rgb": np.array(group["prewar_rgb"]),
        }

        for field in SUN_SATELLITE_FIELDS:
            sample[field] = float(group.attrs.get(field, np.nan))

        return sample


def build_review_index_set(h5_path: Path) -> List[int]:
    if not h5_path.exists():
        raise FileNotFoundError(f"Validation HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        keys = [int(name.replace("sample_", "")) for name in h5.keys() if name.startswith("sample_")]

    return sorted(keys)


def completed_index_set(csv_path: Path, h5_source: str) -> set[int]:
    if not csv_path.exists():
        return set()

    df = load_results_csv(csv_path)
    if "h5_index" not in df.columns:
        return set()

    if "h5_source" not in df.columns:
        df = df.copy()
        df["h5_source"] = "default"

    df = df[df["h5_source"].astype(str) == str(h5_source)]

    return {int(v) for v in df["h5_index"].dropna().tolist()}


def next_unreviewed_index(h5_path: Path, h5_source: str) -> Optional[int]:
    all_indices = build_review_index_set(h5_path)
    done = completed_index_set(CSV_PATH, h5_source=h5_source)

    for idx in all_indices:
        if idx not in done:
            return idx

    return None


def load_land_use_labels() -> Dict[str, str]:
    if not LABELS_PATH.exists():
        return {}

    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError):
        return {}

    return {}


def load_land_use_model() -> Optional[Any]:
    if not MODEL_PATH.exists():
        return None

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_land_use_class(prewar_rgb: np.ndarray, model: Any) -> int:
    flat = np.asarray(prewar_rgb, dtype=np.float32).reshape(-1, 3)
    labels = model.predict(flat)
    counts = np.bincount(labels)
    return int(np.argmax(counts))


def load_spatial_context_layer(layer_name: str) -> gpd.GeoDataFrame:
    path = SPATIAL_CONTEXT_DIR / layer_name
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:32636")

    gdf = gpd.read_file(path)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:32636")


def tile_context_features(record: Dict[str, Any]) -> Dict[str, Any]:
    center_x = record.get("center_x")
    center_y = record.get("center_y")
    tile_date = parse_date_value(record.get("date"))

    result: Dict[str, Any] = {
        "is_agriculture": False,
        "distance_to_nearest_base": np.nan,
        "is_destruction": False,
        "has_greenhouse": False,
        "building_density": np.nan,
        "is_humanitarian_zone": False,
        "distance_to_nearest_outpost": np.nan,
        "distance_to_yellow_line": np.nan,
    }

    if center_x is None or center_y is None or pd.isna(center_x) or pd.isna(center_y):
        return result

    tile_geom = get_tile_polygon(float(center_x), float(center_y), target_crs="EPSG:32636")
    tile_area = float(tile_geom.geometry.iloc[0].area)
    tile_center = Point(float(center_x), float(center_y))

    agriculture = load_spatial_context_layer("agriculture.json")
    if not agriculture.empty:
        intersects = gpd.overlay(agriculture, tile_geom, how="intersection")
        if not intersects.empty:
            covered = float(intersects.geometry.area.sum())
            result["is_agriculture"] = covered > (0.5 * tile_area)

    bases = load_spatial_context_layer("bases.json")
    if not bases.empty:
        filtered = bases.copy()
        if "date_start" in filtered.columns:
            filtered["date_start_parsed"] = filtered["date_start"].apply(parse_date_value)
            filtered = filtered[filtered["date_start_parsed"].notna() & (filtered["date_start_parsed"] >= tile_date)]
        if not filtered.empty:
            nearest = filtered.geometry.distance(tile_center).min()
            result["distance_to_nearest_base"] = float(nearest) if pd.notna(nearest) else np.nan

    destruction = load_spatial_context_layer("destruction.json")
    if not destruction.empty:
        filtered = destruction.copy()
        if "date_start" in filtered.columns:
            filtered["date_start_parsed"] = filtered["date_start"].apply(parse_date_value)
            filtered = filtered[filtered["date_start_parsed"].notna() & (filtered["date_start_parsed"] >= tile_date)]
        if not filtered.empty:
            intersects = gpd.overlay(filtered, tile_geom, how="intersection")
            result["is_destruction"] = not intersects.empty

    greenhouses = load_spatial_context_layer("greenhouses-combined.json")
    if not greenhouses.empty:
        points = greenhouses[greenhouses.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
        if not points.empty:
            matches = gpd.sjoin(points, tile_geom, how="inner", predicate="within")
            result["has_greenhouse"] = not matches.empty

    density = load_spatial_context_layer("h3_density.json")
    if not density.empty:
        intersects = gpd.overlay(density, tile_geom, how="intersection")
        if not intersects.empty and "n_buildings" in intersects.columns:
            numeric = pd.to_numeric(intersects["n_buildings"], errors="coerce")
            result["building_density"] = float(numeric.mean()) if not numeric.dropna().empty else np.nan

    humanitarian = load_spatial_context_layer("humanitarian_zone.json")
    if not humanitarian.empty:
        filtered = humanitarian.copy()
        start_col = "safe_date" if "safe_date" in filtered.columns else None
        end_col = "end_date" if "end_date" in filtered.columns else None
        if start_col and end_col:
            filtered["safe_date_parsed"] = filtered[start_col].apply(parse_date_value)
            filtered["end_date_parsed"] = filtered[end_col].apply(parse_date_value)
            filtered = filtered[
                filtered["safe_date_parsed"].notna() & filtered["end_date_parsed"].notna()
                & (filtered["safe_date_parsed"] <= tile_date)
                & (filtered["end_date_parsed"] >= tile_date)
            ]
        if not filtered.empty:
            intersects = gpd.overlay(filtered, tile_geom, how="intersection")
            if not intersects.empty:
                covered = float(intersects.geometry.area.sum())
                result["is_humanitarian_zone"] = covered > (0.5 * tile_area)

    outposts = load_spatial_context_layer("outposts.json")
    if not outposts.empty:
        filtered = outposts.copy()
        if "date_start" in filtered.columns:
            filtered["date_start_parsed"] = filtered["date_start"].apply(parse_date_value)
            filtered = filtered[filtered["date_start_parsed"].notna() & (filtered["date_start_parsed"] >= tile_date)]
        if not filtered.empty:
            nearest = filtered.geometry.distance(tile_center).min()
            result["distance_to_nearest_outpost"] = float(nearest) if pd.notna(nearest) else np.nan

    yellow = load_spatial_context_layer("yellow-line-zone.json")
    if not yellow.empty:
        filtered = yellow.copy()
        if "date_start" in filtered.columns:
            filtered["date_start_parsed"] = filtered["date_start"].apply(parse_date_value)
            filtered = filtered[filtered["date_start_parsed"].notna() & (filtered["date_start_parsed"] >= tile_date)]
        if not filtered.empty:
            nearest = filtered.geometry.distance(tile_center).min()
            result["distance_to_yellow_line"] = float(nearest) if pd.notna(nearest) else np.nan

    return result


def save_result_row(record: Dict[str, Any], h5_source: str) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "h5_source",
        "h5_index",
        "municipality",
        "source_tif",
        "date",
        "time_of_day",
        "center_x",
        "center_y",
        "tent_count",
        *SUN_SATELLITE_FIELDS,
        "is_agriculture",
        "distance_to_nearest_base",
        "is_destruction",
        "has_greenhouse",
        "building_density",
        "is_humanitarian_zone",
        "distance_to_nearest_outpost",
        "distance_to_yellow_line",
        "land_use_class",
        "land_use_label",
    ]

    labels = load_land_use_labels()
    model = load_land_use_model()
    if model is not None and "prewar_rgb" in record:
        record["land_use_class"] = predict_land_use_class(record["prewar_rgb"], model)
    else:
        record["land_use_class"] = ""

    record["land_use_label"] = labels.get(str(record["land_use_class"]), "")

    spatial_context = tile_context_features(record)
    for key, value in spatial_context.items():
        record[key] = value

    new_row = {
        "h5_source": h5_source,
        "h5_index": record["h5_index"],
        "municipality": record["municipality"],
        "source_tif": record["source_tif"],
        "date": record["date"],
        "time_of_day": record["time_of_day"],
        "center_x": record["center_x"],
        "center_y": record["center_y"],
        "tent_count": record["tent_count"],
        **{field: record.get(field, "") for field in SUN_SATELLITE_FIELDS},
        "is_agriculture": record.get("is_agriculture", ""),
        "distance_to_nearest_base": record.get("distance_to_nearest_base", ""),
        "is_destruction": record.get("is_destruction", ""),
        "has_greenhouse": record.get("has_greenhouse", ""),
        "building_density": record.get("building_density", ""),
        "is_humanitarian_zone": record.get("is_humanitarian_zone", ""),
        "distance_to_nearest_outpost": record.get("distance_to_nearest_outpost", ""),
        "distance_to_yellow_line": record.get("distance_to_yellow_line", ""),
        "land_use_class": record.get("land_use_class", ""),
        "land_use_label": record.get("land_use_label", ""),
    }

    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(new_row)
        return

    existing = load_results_csv(CSV_PATH)
    if "h5_index" not in existing.columns:
        existing = existing.copy()
        existing["h5_index"] = None
    if "h5_source" not in existing.columns:
        existing = existing.copy()
        existing["h5_source"] = "default"

    for col in fieldnames:
        if col not in existing.columns:
            existing[col] = ""

    existing_pairs = set(
        zip(
            existing["h5_source"].astype(str).tolist(),
            existing["h5_index"].astype("Int64").astype(str).tolist(),
        )
    )
    candidate_pair = (str(h5_source), str(int(new_row["h5_index"])))
    if candidate_pair in existing_pairs:
        raise ValueError(
            f"HDF5 sample ({h5_source}, {new_row['h5_index']}) already exists in {CSV_PATH}."
        )

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(new_row)


def load_results_csv(csv_path: Path) -> pd.DataFrame:
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return pd.DataFrame()

    header = rows[0]
    body = rows[1:]

    if "h5_source" in header:
        width = len(header)
        cleaned = [r[:width] + [""] * max(0, width - len(r)) for r in body]
        return pd.DataFrame(cleaned, columns=header)

    expected = [
        "h5_source",
        *header,
    ]
    migrated: List[List[str]] = []
    for row in body:
        if len(row) == len(header) + 1:
            migrated.append(row[: len(expected)])
            continue
        if len(row) >= len(header):
            migrated.append(["default", *row[: len(header)]])
            continue
        padded = row + [""] * (len(header) - len(row))
        migrated.append(["default", *padded])

    return pd.DataFrame(migrated, columns=expected)


def normalize_results_csv(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    df = load_results_csv(csv_path)
    ordered = [
        "h5_source",
        "h5_index",
        "municipality",
        "source_tif",
        "date",
        "time_of_day",
        "center_x",
        "center_y",
        "tent_count",
        *SUN_SATELLITE_FIELDS,
        "is_agriculture",
        "distance_to_nearest_base",
        "is_destruction",
        "has_greenhouse",
        "building_density",
        "is_humanitarian_zone",
        "distance_to_nearest_outpost",
        "distance_to_yellow_line",
        "land_use_class",
        "land_use_label",
    ]
    for col in ordered:
        if col not in df.columns:
            df[col] = ""
    df = df[ordered]
    df.to_csv(csv_path, index=False)


def show_pair_and_get_count(sample: Dict[str, Any]) -> int:
    def build_figure():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Sample {sample['h5_index']} | {sample['municipality']} | {sample['source_tif']}",
            fontsize=11,
        )

        axes[0].imshow(sample["prewar_rgb"])
        axes[0].set_title("Prewar")
        axes[0].axis("off")

        axes[1].imshow(sample["historic_rgb"])
        axes[1].set_title("Current")
        axes[1].axis("off")

        return fig, axes

    def read_count_from_prompt(prompt: str) -> int:
        while True:
            raw = input(prompt).strip()
            if raw == "":
                print("No count entered. Please enter a non-negative integer.")
                continue
            try:
                parsed = int(raw)
                if parsed < 0:
                    raise ValueError
                return parsed
            except ValueError:
                print("Invalid input. Please enter a non-negative integer.")

    try:
        fig, axes = build_figure()

        ax_box = fig.add_axes([0.38, 0.02, 0.18, 0.06])
        tb = TextBox(ax_box, "Tents: ", initial="")
        tb.set_active(True)

        submit_ax = fig.add_axes([0.58, 0.02, 0.12, 0.06])
        button = Button(submit_ax, "Next")

        value = {"count": None}

        def submit(_event):
            raw = tb.text.strip()
            try:
                parsed = int(raw)
                if parsed < 0:
                    raise ValueError
                value["count"] = parsed
            except ValueError:
                tb.set_val("")
                tb.label.set_text("Tents (non-negative integer): ")
                return
            plt.close(fig)

        def on_key(event):
            if event.key in {"enter", "return"}:
                submit(None)
                return

        button.on_clicked(submit)
        fig.canvas.mpl_connect("key_press_event", on_key)

        fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.14, wspace=0.08)
        plt.show(block=True)

        if value["count"] is None:
            raise ValueError(f"No tent count entered for sample {sample['h5_index']}.")

        return value["count"]
    except Exception as exc:
        fig, _ = build_figure()
        tmp_dir = Path("/tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        preview_path = tmp_dir / f"sample_{sample['h5_index']}_preview.png"
        fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.04, wspace=0.08)
        fig.savefig(preview_path, dpi=160)
        plt.close(fig)
        print(f"GUI display unavailable ({type(exc).__name__}: {exc})")
        print(f"Saved preview image to {preview_path}")
        print(f"Sample {sample['h5_index']} | {sample['municipality']} | {sample['source_tif']}")
        return read_count_from_prompt(f"Enter tent count for sample {sample['h5_index']}: ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual tent-count validation UI")
    parser.add_argument(
        "--h5-source",
        choices=["default", "high_density"],
        default="default",
        help="Which validation HDF5 dataset to review.",
    )
    return parser


def resolve_h5_source(source_key: str) -> tuple[Path, str]:
    if source_key == "high_density":
        return H5_HIGH_DENSITY_PATH, "high_density"
    return H5_PATH, "default"


def main() -> None:
    args = build_parser().parse_args()
    h5_path, h5_source = resolve_h5_source(args.h5_source)
    normalize_results_csv(CSV_PATH)

    while True:
        next_index = next_unreviewed_index(h5_path=h5_path, h5_source=h5_source)
        if next_index is None:
            print(f"All samples in {h5_path} have already been reviewed for source '{h5_source}'.")
            break

        print(f"Reviewing HDF5 index {next_index} from {h5_path} (source={h5_source})")
        sample = load_h5_sample(next_index, h5_path=h5_path)
        tent_count = show_pair_and_get_count(sample)
        sample["tent_count"] = tent_count
        save_result_row(sample, h5_source=h5_source)

        print(
            f"Saved tent count {tent_count} for sample index {next_index} to {CSV_PATH}"
        )


if __name__ == "__main__":
    main()
