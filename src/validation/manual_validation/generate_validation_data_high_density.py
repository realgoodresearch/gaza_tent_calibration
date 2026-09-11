from __future__ import annotations

import random
import tempfile
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.transform import rowcol
from shapely.geometry import Point
from tqdm import tqdm

from generate_validation_data import (
    BOX_SIZE_METERS,
    MIN_DATA_FRACTION,
    copy_with_retry,
    extract_sun_satellite_metadata,
    get_box_size_pixels,
    image_has_sufficient_data,
    list_raw_tifs,
    load_municipalities,
    parse_date_and_time_from_filename,
    read_rgb_window,
    transform_point_between_crs,
)


if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parents[2]
else:
    REPO_ROOT = Path.cwd()

RAW_TIF_DIR = REPO_ROOT / "data" / "tif_files" / "historic" / "raw"
PREWAR_TIF_PATH = REPO_ROOT / "data" / "tif_files" / "prewar_gaza.tif"
PREDICTION_FOLDER = REPO_ROOT / "data" / "tent_predictions" / "merged"
OUTPUT_H5 = REPO_ROOT / "outputs" / "tile_validation_sample_high_density.h5"

TARGET_SAMPLE_SIZE = 200
TARGET_CRS = "EPSG:32636"


def _list_prediction_gpks(folder: Path) -> List[Path]:
    files = sorted(folder.glob("*.gpkg"))
    if not files:
        raise FileNotFoundError(f"No weekly prediction GeoPackages in: {folder}")
    return files


def _read_prediction_points(gpkg_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg_path)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(TARGET_CRS)


def _date_in_week_range(day: pd.Timestamp, week_name: str) -> bool:
    parts = week_name.split("-")
    if len(parts) != 2:
        return False
    start = pd.to_datetime(parts[0], format="%Y%m%d", errors="coerce")
    end = pd.to_datetime(parts[1], format="%Y%m%d", errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return False
    return bool(start <= day <= end)


def _index_tifs_by_week(tif_paths: List[Path], week_paths: List[Path]) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {p.stem: [] for p in week_paths}
    for tif_path in tif_paths:
        date_text, _ = parse_date_and_time_from_filename(tif_path.name)
        if not date_text:
            continue
        day = pd.to_datetime(date_text, errors="coerce")
        if pd.isna(day):
            continue
        for week_path in week_paths:
            if _date_in_week_range(day, week_path.stem):
                out[week_path.stem].append(tif_path)
                break
    return out


def _municipality_for_xy(x: float, y: float, municipalities: gpd.GeoDataFrame) -> str:
    pt = gpd.GeoDataFrame(geometry=[Point(x, y)], crs=TARGET_CRS)
    hit = gpd.sjoin(pt, municipalities[["municipality", "geometry"]], how="left", predicate="within")
    if not hit.empty and pd.notna(hit["municipality"].iloc[0]):
        return str(hit["municipality"].iloc[0])
    near = gpd.sjoin_nearest(pt, municipalities[["municipality", "geometry"]], how="left")
    if not near.empty and pd.notna(near["municipality"].iloc[0]):
        return str(near["municipality"].iloc[0])
    return "UNKNOWN"


def _sample_from_point(
    point_x_utm: float,
    point_y_utm: float,
    tif_path: Path,
    prewar_raster: rasterio.DatasetReader,
    municipalities: gpd.GeoDataFrame,
    week_file_name: str,
    cache_dir_path: Path,
) -> Optional[Dict[str, Any]]:
    local_tif_path = cache_dir_path / tif_path.name
    try:
        copy_with_retry(tif_path, local_tif_path)
        with rasterio.open(local_tif_path) as src:
            if src.crs is None:
                return None

            x_src, y_src = transform_point_between_crs(point_x_utm, point_y_utm, TARGET_CRS, src.crs)
            row, col = rowcol(src.transform, x_src, y_src)
            if not (0 <= row < src.height and 0 <= col < src.width):
                return None

            box_size_px = get_box_size_pixels(src)
            historic_rgb = read_rgb_window(src, int(row), int(col), box_size_px)
            if not image_has_sufficient_data(historic_rgb, min_fraction=MIN_DATA_FRACTION):
                return None

            x_pre, y_pre = transform_point_between_crs(point_x_utm, point_y_utm, TARGET_CRS, prewar_raster.crs)
            pre_row, pre_col = rowcol(prewar_raster.transform, x_pre, y_pre)
            if not (0 <= pre_row < prewar_raster.height and 0 <= pre_col < prewar_raster.width):
                return None

            prewar_rgb = read_rgb_window(prewar_raster, int(pre_row), int(pre_col), box_size_px)
            if not image_has_sufficient_data(prewar_rgb, min_fraction=MIN_DATA_FRACTION):
                return None

            date_text, time_text = parse_date_and_time_from_filename(tif_path.name)
            metadata = extract_sun_satellite_metadata(str(local_tif_path))
            municipality = _municipality_for_xy(point_x_utm, point_y_utm, municipalities)

            return {
                "municipality": municipality,
                "source_tif": tif_path.name,
                "date": date_text,
                "time_of_day": time_text,
                "center_x": float(point_x_utm),
                "center_y": float(point_y_utm),
                "prediction_week_file": week_file_name,
                "historic_rgb": historic_rgb,
                "prewar_rgb": prewar_rgb,
                **metadata,
            }
    except (RasterioIOError, ValueError, RuntimeError):
        return None
    finally:
        local_tif_path.unlink(missing_ok=True)


def _week_path_for_tif(tif_path: Path, week_files: List[Path]) -> Optional[Path]:
    date_text, _ = parse_date_and_time_from_filename(tif_path.name)
    if not date_text:
        return None
    day = pd.to_datetime(date_text, errors="coerce")
    if pd.isna(day):
        return None
    for week_path in week_files:
        if _date_in_week_range(day, week_path.stem):
            return week_path
    return None


def generate_high_density_samples(target_samples: int = TARGET_SAMPLE_SIZE) -> List[Dict[str, Any]]:
    tif_files = list_raw_tifs(RAW_TIF_DIR)
    if not tif_files:
        raise RuntimeError("No TIFF files found for sampling")

    week_files = _list_prediction_gpks(PREDICTION_FOLDER)
    municipalities = load_municipalities()

    samples_per_tif = max(1, target_samples // len(tif_files))
    remainder = max(0, target_samples - (samples_per_tif * len(tif_files)))

    samples: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="raster_cache_high_density_") as cache_dir:
        cache_dir_path = Path(cache_dir)
        local_prewar_path = cache_dir_path / PREWAR_TIF_PATH.name
        copy_with_retry(PREWAR_TIF_PATH, local_prewar_path)

        with rasterio.open(local_prewar_path) as prewar_src, tqdm(total=target_samples, desc="Generating high-density samples") as pbar:
            week_point_cache: Dict[Path, gpd.GeoDataFrame] = {}

            for tif_index, tif_path in enumerate(tif_files):
                week_path = _week_path_for_tif(tif_path, week_files)
                if week_path is None:
                    continue

                if week_path not in week_point_cache:
                    week_point_cache[week_path] = _read_prediction_points(week_path)
                week_points = week_point_cache[week_path]
                if week_points.empty:
                    continue

                local_tif_path = cache_dir_path / tif_path.name
                try:
                    copy_with_retry(tif_path, local_tif_path)
                    with rasterio.open(local_tif_path) as src:
                        if src.crs is None:
                            continue

                        footprint = gpd.GeoSeries([box(*src.bounds)], crs=src.crs).to_crs(TARGET_CRS).iloc[0]
                        candidate_idx = list(week_points.sindex.intersection(footprint.bounds))
                        if not candidate_idx:
                            continue
                        candidate_points = week_points.iloc[candidate_idx]
                        candidate_points = candidate_points[candidate_points.within(footprint)].copy()
                        if candidate_points.empty:
                            continue

                        n_for_this_tif = samples_per_tif + (1 if tif_index < remainder else 0)
                        n_for_this_tif = max(1, n_for_this_tif)

                        replace = len(candidate_points) < n_for_this_tif
                        chosen = candidate_points.sample(n=n_for_this_tif, replace=replace, random_state=None)

                        for point in chosen.geometry.tolist():
                            sample = _sample_from_point(
                                point_x_utm=float(point.x),
                                point_y_utm=float(point.y),
                                tif_path=tif_path,
                                prewar_raster=prewar_src,
                                municipalities=municipalities,
                                week_file_name=week_path.name,
                                cache_dir_path=cache_dir_path,
                            )
                            if sample is None:
                                continue
                            samples.append(sample)
                            pbar.update(1)
                            if len(samples) >= target_samples:
                                break
                finally:
                    local_tif_path.unlink(missing_ok=True)

                if len(samples) >= target_samples:
                    break

    if len(samples) < target_samples:
        raise RuntimeError(f"Generated {len(samples)} samples out of requested {target_samples}.")

    return samples


def save_hdf5(samples: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["num_samples"] = len(samples)
        h5.attrs["box_size_meters"] = BOX_SIZE_METERS
        h5.attrs["sampling_mode"] = "high_density_weekly_predictions"

        for i, sample in enumerate(tqdm(samples, desc="Writing HDF5")):
            group = h5.create_group(f"sample_{i}")
            group.attrs["municipality"] = sample["municipality"]
            group.attrs["source_tif"] = sample["source_tif"]
            group.attrs["date"] = sample.get("date", "")
            group.attrs["time_of_day"] = sample.get("time_of_day", "")
            group.attrs["center_x"] = float(sample["center_x"])
            group.attrs["center_y"] = float(sample["center_y"])
            group.attrs["prediction_week_file"] = sample.get("prediction_week_file", "")

            for key in ["satellite_azimuth", "satellite_elevation", "sun_azimuth", "sun_elevation"]:
                group.attrs[key] = float(sample.get(key, float("nan")))

            group.create_dataset("historic_rgb", data=sample["historic_rgb"], compression="gzip")
            group.create_dataset("prewar_rgb", data=sample["prewar_rgb"], compression="gzip")


def main() -> None:
    samples = generate_high_density_samples(target_samples=TARGET_SAMPLE_SIZE)
    save_hdf5(samples, OUTPUT_H5)
    print(f"Saved {len(samples)} high-density samples to {OUTPUT_H5}")


if __name__ == "__main__":
    main()
