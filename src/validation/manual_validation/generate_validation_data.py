from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio
import tifffile
from rasterio.transform import rowcol, xy
from rasterio.windows import Window
from rasterio.warp import transform
from rasterio.errors import RasterioIOError
from shapely.geometry import Point, box
from tqdm import tqdm

if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parents[2]
else:
    REPO_ROOT = Path.cwd()

# utils.py lives in validation/, one level up from this manual_validation/ package.
sys.path.insert(0, str(REPO_ROOT / "validation"))
from utils import SUN_SATELLITE_FIELDS

MUNICIPALITY_PATH = (
    REPO_ROOT
    / "data"
    / "gaza_boundaries"
    / "municipal"
    / "GazaStrip_MunicipalBoundaries.shp"
)
RAW_TIF_DIR = REPO_ROOT / "data" / "tif_files" / "historic" / "raw"
PREWAR_TIF_PATH = REPO_ROOT / "data" / "tif_files" / "prewar_gaza.tif"
OUTPUT_H5 = REPO_ROOT / "outputs" / "tile_validation_samples.h5"

TARGET_SAMPLE_SIZE = 1000
BOX_SIZE_METERS = 100
MIN_DATA_FRACTION = 0.90

COPY_RETRY_ATTEMPTS = 3
COPY_RETRY_DELAY_SECONDS = 3.0

# How many candidate (municipality, tif) combinations to try before giving
# up on planning/replacing a single sample slot.
PLANNING_ATTEMPTS_PER_SAMPLE = 20


def load_municipalities() -> gpd.GeoDataFrame:
    """Load, clean, dissolve, and project municipality boundaries."""
    if not MUNICIPALITY_PATH.exists():
        raise FileNotFoundError(
            f"Municipality shapefile not found: {MUNICIPALITY_PATH}"
        )

    gdf = gpd.read_file(MUNICIPALITY_PATH)

    required_columns = {"NAME", "geometry"}
    missing = required_columns - set(gdf.columns)
    if missing:
        raise ValueError(
            f"Municipality shapefile is missing columns: {sorted(missing)}"
        )

    gdf = gdf[["NAME", "geometry"]].rename(
        columns={"NAME": "municipality"}
    )

    gdf = gdf[gdf["municipality"].notna()].copy()
    gdf["municipality"] = gdf["municipality"].astype(str).str.strip()
    gdf = gdf[gdf["municipality"] != ""].copy()

    if gdf.crs is None:
        raise ValueError("Municipality shapefile has no CRS.")

    gdf = gdf.to_crs("EPSG:32636")

    gdf = (
        gdf.dissolve(by="municipality", as_index=False)[
            ["municipality", "geometry"]
        ]
    )

    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

    if gdf.empty:
        raise ValueError("No valid municipality geometries were found.")

    return gdf


def list_raw_tifs(raw_dir: Path) -> List[Path]:
    """Recursively find supported raster files."""
    valid_exts = {".tif", ".tiff", ".img", ".jp2"}

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw TIFF directory not found: {raw_dir}")

    files = [
        p
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    ]

    return sorted(files)


def copy_with_retry(
    src: Path,
    dst: Path,
    attempts: int = COPY_RETRY_ATTEMPTS,
    delay_seconds: float = COPY_RETRY_DELAY_SECONDS,
) -> None:
    """Copy a file, retrying a few times on transient Drive/FUSE hiccups."""
    last_exc: Exception | None = None

    for attempt in range(attempts):
        try:
            shutil.copy2(src, dst)
            return
        except OSError as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(delay_seconds)

    raise RuntimeError(f"F ailed to copy {src} after {attempts} attempts") from last_exc


def get_box_size_pixels(raster: rasterio.DatasetReader) -> int:
    """Convert the desired physical box size into pixels."""
    if raster.res is None or len(raster.res) < 2:
        raise ValueError("Raster does not have valid pixel resolution.")

    pixel_size_x = abs(float(raster.res[0]))
    pixel_size_y = abs(float(raster.res[1]))

    pixel_size = (pixel_size_x + pixel_size_y) / 2.0

    if pixel_size <= 0:
        pixel_size = 1.0

    return max(1, int(round(BOX_SIZE_METERS / pixel_size)))


def random_valid_pixel_from_polygon(
    raster: rasterio.DatasetReader,
    polygon: gpd.GeoSeries,
    max_attempts: int = 200,
) -> Tuple[int, int] | None:
    """
    Select a random raster pixel inside a polygon via rejection sampling.

    The polygon must be in the same CRS as the raster. Random points are
    drawn from the polygon's bounding box and accepted once one both falls
    inside the polygon and maps to a pixel within the raster. This avoids
    masking/cropping the raster against the whole (potentially large)
    polygon for every sample.
    """
    if polygon.empty:
        return None

    geom = polygon.iloc[0]

    if geom is None or geom.is_empty:
        return None

    min_x, min_y, max_x, max_y = geom.bounds

    for _ in range(max_attempts):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)

        if not geom.contains(Point(x, y)):
            continue

        row, col = rowcol(raster.transform, x, y)

        if 0 <= row < raster.height and 0 <= col < raster.width:
            return int(row), int(col)

    return None


def image_has_sufficient_data(
    image: np.ndarray,
    min_fraction: float = MIN_DATA_FRACTION,
) -> bool:
    """Return True if the tile contains enough non-zero information."""
    if image.size == 0:
        return False

    arr = np.asarray(image)
    if arr.ndim == 3:
        valid = np.any(arr != 0, axis=-1)
    else:
        valid = arr != 0

    fraction = float(np.mean(valid)) if valid.size else 0.0
    return fraction >= min_fraction


def read_rgb_window(
    raster: rasterio.DatasetReader,
    center_row: int,
    center_col: int,
    box_size_px: int,
) -> np.ndarray:
    """
    Read a square window centered on a pixel.

    Output shape is (height, width, bands).
    Grayscale rasters are replicated to three channels.
    """
    if box_size_px <= 0:
        raise ValueError(f"Invalid box_size_px: {box_size_px}")

    if not (0 <= center_row < raster.height):
        raise ValueError(
            f"center_row={center_row} outside raster "
            f"height={raster.height}"
        )

    if not (0 <= center_col < raster.width):
        raise ValueError(
            f"center_col={center_col} outside raster "
            f"width={raster.width}"
        )

    half = box_size_px // 2

    row_start = max(0, center_row - half)
    row_stop = min(raster.height, center_row + half + 1)

    col_start = max(0, center_col - half)
    col_stop = min(raster.width, center_col + half + 1)

    height = row_stop - row_start
    width = col_stop - col_start

    if height <= 0 or width <= 0:
        raise ValueError(
            "Invalid raster window: "
            f"rows={row_start}:{row_stop}, "
            f"cols={col_start}:{col_stop}"
        )

    window = Window(
        col_start,
        row_start,
        width,
        height,
    )

    data = raster.read(window=window)

    if data.size == 0:
        raise ValueError("Raster read returned an empty array.")

    # Rasterio returns (bands, height, width).
    if data.ndim == 3:
        data = np.moveaxis(data, 0, -1)

        # Keep only RGB if there are more than 3 bands.
        if data.shape[-1] >= 3:
            data = data[..., :3]

        # If there are only 1 or 2 bands, expand to three channels.
        elif data.shape[-1] == 1:
            data = np.repeat(data, 3, axis=-1)

        elif data.shape[-1] == 2:
            third = data[..., :1]
            data = np.concatenate([data, third], axis=-1)

    elif data.ndim == 2:
        data = np.stack([data, data, data], axis=-1)

    else:
        raise ValueError(
            f"Unexpected raster data dimensions: {data.shape}"
        )

    # Replace NaN/inf before converting to uint8.
    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    return np.clip(data, 0, 255).astype(np.uint8)


def transform_point_between_crs(
    x: float,
    y: float,
    source_crs,
    destination_crs,
) -> Tuple[float, float]:
    """Transform one point between raster CRSs."""
    if source_crs == destination_crs:
        return float(x), float(y)

    xs, ys = transform(
        source_crs,
        destination_crs,
        [x],
        [y],
    )

    return float(xs[0]), float(ys[0])


@lru_cache(maxsize=None)
def extract_sun_satellite_metadata(tif_path: str) -> Dict[str, float]:
    """Read satellite/sun azimuth+elevation from a tif's ImageDescription tag.

    Not all source tifs carry this metadata, so missing/unreadable values
    fall back to NaN rather than raising.
    """
    defaults = {field: float("nan") for field in SUN_SATELLITE_FIELDS}

    try:
        with tifffile.TiffFile(tif_path) as tif:
            description = tif.pages[0].tags["ImageDescription"].value
            props = json.loads(description)["properties"]
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return defaults

    return {
        field: float(props[field]) if field in props else float("nan")
        for field in SUN_SATELLITE_FIELDS
    }


def build_tif_footprint_index(
    tif_paths: List[Path],
    target_crs: str = "EPSG:32636",
) -> gpd.GeoDataFrame:
    """
    Build a lightweight spatial index of each tif's footprint.

    Only raster header metadata (bounds + CRS) is read here, not pixel data,
    so indexing hundreds of large rasters is cheap even though many of them
    don't cover the whole Gaza Strip.
    """
    records: List[Dict[str, Any]] = []

    for tif_path in tqdm(tif_paths, desc="Indexing footprints"):
        try:
            with rasterio.open(tif_path) as src:
                if src.crs is None:
                    continue

                footprint = (
                    gpd.GeoSeries([box(*src.bounds)], crs=src.crs)
                    .to_crs(target_crs)
                    .iloc[0]
                )
        except (RasterioIOError, ValueError, RuntimeError):
            # Corrupt/unreadable rasters just don't contribute a footprint.
            continue

        date_value, time_value = parse_date_and_time_from_filename(tif_path.name)

        records.append(
            {
                "tif_path": tif_path,
                "date": date_value,
                "time_of_day": time_value,
                "geometry": footprint,
            }
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs)


def group_tifs_by_municipality(
    footprints: gpd.GeoDataFrame,
    municipalities: gpd.GeoDataFrame,
) -> Dict[str, pd.DataFrame]:
    """Restrict each municipality's candidate tifs to those whose footprint overlaps it."""
    empty = pd.DataFrame(columns=["tif_path", "date", "time_of_day"])

    if footprints.empty:
        return {name: empty for name in municipalities["municipality"]}

    joined = gpd.sjoin(
        footprints,
        municipalities[["municipality", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    index: Dict[str, pd.DataFrame] = {}
    for name in municipalities["municipality"]:
        subset = joined.loc[
            joined["municipality"] == name,
            ["tif_path", "date", "time_of_day"],
        ]
        index[name] = subset.reset_index(drop=True) if not subset.empty else empty

    return index


def choose_intersection_for_municipality(
    municipality_name: str,
    municipality_geom: Any,
    candidate_tifs: pd.DataFrame,
    footprint_geom_by_tif: Dict[Path, Any],
    intersection_cache: Dict[Tuple[str, Path], Any],
    max_attempts: int = PLANNING_ATTEMPTS_PER_SAMPLE,
) -> Dict[str, Any] | None:
    """
    Pick a (date, time_of_day, tif) for a municipality and resolve its
    municipality-tif intersection geometry (in EPSG:32636), caching the
    intersection per (municipality, tif) pair so it's only computed once.
    """
    if candidate_tifs.empty:
        return None

    acquisitions = (
        candidate_tifs[["date", "time_of_day"]]
        .drop_duplicates()
        .to_records(index=False)
        .tolist()
    )

    for _ in range(max_attempts):
        date_value, time_value = random.choice(acquisitions)
        acquisition_tifs = candidate_tifs[
            (candidate_tifs["date"] == date_value)
            & (candidate_tifs["time_of_day"] == time_value)
        ]["tif_path"].tolist()

        tif_path = random.choice(acquisition_tifs)

        cache_key = (municipality_name, tif_path)
        if cache_key not in intersection_cache:
            footprint_geom = footprint_geom_by_tif.get(tif_path)
            intersection_geom = None
            if footprint_geom is not None:
                candidate_intersection = municipality_geom.intersection(footprint_geom)
                if (
                    candidate_intersection is not None
                    and not candidate_intersection.is_empty
                    and candidate_intersection.area > 0
                ):
                    intersection_geom = candidate_intersection
            intersection_cache[cache_key] = intersection_geom

        intersection_geom = intersection_cache[cache_key]
        if intersection_geom is None:
            continue

        return {
            "municipality": municipality_name,
            "date": date_value,
            "time_of_day": time_value,
            "tif_path": tif_path,
            "intersection_geom": intersection_geom,
        }

    return None


def build_sampling_plan(
    municipality_names: List[str],
    municipality_geoms: Dict[str, Any],
    tifs_by_municipality: Dict[str, pd.DataFrame],
    footprint_geom_by_tif: Dict[Path, Any],
    intersection_cache: Dict[Tuple[str, Path], Any],
    target_samples: int,
) -> List[Dict[str, Any]]:
    """
    Build the complete sampling plan before opening any historic tif.

    Each planned entry records the municipality/acquisition/tif chosen via
    the existing stratified logic, plus the precomputed municipality-tif
    intersection geometry samples will later be drawn from.
    """
    plan: List[Dict[str, Any]] = []

    n_municipalities = len(municipality_names)
    per_municipality = max(1, target_samples // n_municipalities)

    for municipality in municipality_names:
        remaining = target_samples - len(plan)

        if remaining <= 0:
            break

        requested = min(per_municipality, remaining)

        for _ in range(requested):
            entry = choose_intersection_for_municipality(
                municipality,
                municipality_geoms[municipality],
                tifs_by_municipality[municipality],
                footprint_geom_by_tif,
                intersection_cache,
            )
            if entry is not None:
                plan.append(entry)

    # Fill any shortfall with additional random municipalities.
    max_attempts = max(target_samples * 20, 1000)
    attempts = 0

    eligible_municipalities = [
        name for name in municipality_names if not tifs_by_municipality[name].empty
    ]

    while (
        len(plan) < target_samples
        and attempts < max_attempts
        and eligible_municipalities
    ):
        attempts += 1

        municipality = random.choice(eligible_municipalities)

        entry = choose_intersection_for_municipality(
            municipality,
            municipality_geoms[municipality],
            tifs_by_municipality[municipality],
            footprint_geom_by_tif,
            intersection_cache,
        )
        if entry is not None:
            plan.append(entry)

    return plan


def generate_sample_for_entry(
    entry: Dict[str, Any],
    src: rasterio.DatasetReader,
    polygon: gpd.GeoSeries,
    prewar_raster: rasterio.DatasetReader,
    local_tif_path: Path,
) -> Dict[str, Any] | None:
    """Try to produce one sample for a planned entry; None if it fails any check."""
    pixel = random_valid_pixel_from_polygon(src, polygon)

    if pixel is None:
        return None

    center_row, center_col = pixel

    historic_box_size_px = get_box_size_pixels(src)
    prewar_box_size_px = get_box_size_pixels(prewar_raster)

    historic_box = read_rgb_window(
        src,
        center_row,
        center_col,
        historic_box_size_px,
    )

    # Convert historical pixel center to map coordinates.
    center_x, center_y = xy(
        src.transform,
        center_row,
        center_col,
        offset="center",
    )

    # Transform point into pre-war raster CRS if necessary.
    prewar_x, prewar_y = transform_point_between_crs(
        center_x,
        center_y,
        src.crs,
        prewar_raster.crs,
    )

    prewar_row, prewar_col = rowcol(
        prewar_raster.transform,
        prewar_x,
        prewar_y,
    )

    # Skip samples that fall outside the pre-war raster.
    if not (
        0 <= prewar_row < prewar_raster.height
        and 0 <= prewar_col < prewar_raster.width
    ):
        return None

    prewar_box = read_rgb_window(
        prewar_raster,
        prewar_row,
        prewar_col,
        prewar_box_size_px,
    )

    if not image_has_sufficient_data(
        historic_box
    ) or not image_has_sufficient_data(prewar_box):
        return None

    sun_satellite_metadata = extract_sun_satellite_metadata(str(local_tif_path))

    return {
        "municipality": entry["municipality"],
        "source_tif": entry["tif_path"].name,
        "historic_rgb": historic_box,
        "prewar_rgb": prewar_box,
        "center_x": float(center_x),
        "center_y": float(center_y),
        **sun_satellite_metadata,
    }


def process_sampling_plan(
    plan: List[Dict[str, Any]],
    municipality_geoms: Dict[str, Any],
    tifs_by_municipality: Dict[str, pd.DataFrame],
    footprint_geom_by_tif: Dict[Path, Any],
    intersection_cache: Dict[Tuple[str, Path], Any],
    prewar_raster: rasterio.DatasetReader,
    cache_dir_path: Path,
    pbar: tqdm,
) -> List[Dict[str, Any]]:
    """
    Execute a sampling plan, processing one historic tif at a time.

    Every tif referenced by the plan is copied locally, opened once, and all
    samples assigned to it are generated before the local copy is deleted.
    Samples that fail the pixel/data-quality checks are replaced with a
    freshly planned sample for the same municipality so representation isn't
    skewed by failures.
    """
    plan_by_tif: Dict[Path, List[Dict[str, Any]]] = defaultdict(list)
    for entry in plan:
        plan_by_tif[entry["tif_path"]].append(entry)

    tif_queue: "deque[Path]" = deque(plan_by_tif.keys())
    queued_tifs = set(tif_queue)

    samples: List[Dict[str, Any]] = []

    max_replacement_attempts = max(len(plan) * 20, 1000)
    replacement_attempts = 0

    def draw_replacement(municipality_name: str) -> Dict[str, Any] | None:
        nonlocal replacement_attempts
        if replacement_attempts >= max_replacement_attempts:
            return None
        replacement_attempts += 1
        return choose_intersection_for_municipality(
            municipality_name,
            municipality_geoms[municipality_name],
            tifs_by_municipality[municipality_name],
            footprint_geom_by_tif,
            intersection_cache,
        )

    def enqueue(entry: Dict[str, Any]) -> None:
        target_tif = entry["tif_path"]
        plan_by_tif[target_tif].append(entry)
        if target_tif not in queued_tifs:
            tif_queue.append(target_tif)
            queued_tifs.add(target_tif)

    while tif_queue:
        tif_path = tif_queue.popleft()
        queued_tifs.discard(tif_path)
        entries = plan_by_tif.pop(tif_path, [])

        if not entries:
            continue

        local_tif_path = cache_dir_path / tif_path.name

        try:
            copy_with_retry(tif_path, local_tif_path)

            with rasterio.open(local_tif_path) as src:
                if src.crs is None:
                    for entry in entries:
                        replacement = draw_replacement(entry["municipality"])
                        if replacement is not None:
                            enqueue(replacement)
                    continue

                entries_by_municipality: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for entry in entries:
                    entries_by_municipality[entry["municipality"]].append(entry)

                for municipality_name, muni_entries in entries_by_municipality.items():
                    # Reproject the municipality-tif intersection once per
                    # municipality, not once per sample.
                    reprojected_polygon = gpd.GeoSeries(
                        [muni_entries[0]["intersection_geom"]],
                        crs="EPSG:32636",
                    ).to_crs(src.crs)

                    idx = 0
                    while idx < len(muni_entries):
                        entry = muni_entries[idx]
                        idx += 1

                        try:
                            sample = generate_sample_for_entry(
                                entry,
                                src,
                                reprojected_polygon,
                                prewar_raster,
                                local_tif_path,
                            )
                        except (RasterioIOError, ValueError, RuntimeError):
                            sample = None

                        if sample is not None:
                            samples.append(sample)
                            pbar.update(1)
                            continue

                        replacement = draw_replacement(municipality_name)
                        if replacement is None:
                            continue

                        if replacement["tif_path"] == tif_path:
                            # Same tif is still open/local: process it now.
                            muni_entries.append(replacement)
                        else:
                            enqueue(replacement)

        except (RasterioIOError, ValueError, RuntimeError) as exc:
            print(f"Skipping {tif_path.name}: {exc}")
            for entry in entries:
                replacement = draw_replacement(entry["municipality"])
                if replacement is not None:
                    enqueue(replacement)
        finally:
            local_tif_path.unlink(missing_ok=True)

    return samples


def build_validation_dataset(
    raw_tif_dir: Path,
    prewar_tif_path: Path,
    municipalities: gpd.GeoDataFrame,
    target_samples: int = TARGET_SAMPLE_SIZE,
) -> List[Dict[str, Any]]:
    """Build a balanced-ish validation sample across municipalities."""
    if target_samples <= 0:
        raise ValueError(
            f"target_samples must be positive, got {target_samples}"
        )

    tif_paths = list_raw_tifs(raw_tif_dir)
    print(f"Found {len(tif_paths)} raster files")

    if not tif_paths:
        raise FileNotFoundError(
            f"No supported raster files found under: {raw_tif_dir}"
        )

    if not prewar_tif_path.exists():
        raise FileNotFoundError(
            f"Pre-war TIFF not found: {prewar_tif_path}"
        )

    municipality_names = municipalities[
        "municipality"
    ].dropna().tolist()

    if not municipality_names:
        raise ValueError("No municipalities available for sampling.")

    footprints = build_tif_footprint_index(tif_paths)
    tifs_by_municipality = group_tifs_by_municipality(footprints, municipalities)

    footprint_geom_by_tif: Dict[Path, Any] = (
        dict(zip(footprints["tif_path"], footprints["geometry"]))
        if not footprints.empty
        else {}
    )
    municipality_geoms: Dict[str, Any] = dict(
        zip(municipalities["municipality"], municipalities["geometry"])
    )
    intersection_cache: Dict[Tuple[str, Path], Any] = {}

    plan = build_sampling_plan(
        municipality_names=municipality_names,
        municipality_geoms=municipality_geoms,
        tifs_by_municipality=tifs_by_municipality,
        footprint_geom_by_tif=footprint_geom_by_tif,
        intersection_cache=intersection_cache,
        target_samples=target_samples,
    )

    if len(plan) < target_samples:
        raise RuntimeError(
            f"Only planned {len(plan)} samples out of {target_samples} "
            "requested; not enough overlapping imagery was found."
        )

    print(
        f"Planned {len(plan)} samples across "
        f"{len({entry['tif_path'] for entry in plan})} distinct rasters"
    )

    with tempfile.TemporaryDirectory(prefix="raster_cache_") as cache_dir:
        cache_dir_path = Path(cache_dir)

        # Copy the pre-war mosaic locally once: every sample reads from it.
        local_prewar_path = cache_dir_path / prewar_tif_path.name
        copy_with_retry(prewar_tif_path, local_prewar_path)

        with rasterio.open(local_prewar_path) as prewar_raster, tqdm(
            total=target_samples, desc="Generating samples"
        ) as pbar:
            samples = process_sampling_plan(
                plan=plan,
                municipality_geoms=municipality_geoms,
                tifs_by_municipality=tifs_by_municipality,
                footprint_geom_by_tif=footprint_geom_by_tif,
                intersection_cache=intersection_cache,
                prewar_raster=prewar_raster,
                cache_dir_path=cache_dir_path,
                pbar=pbar,
            )

    if len(samples) < target_samples:
        raise RuntimeError(
            f"Only generated {len(samples)} valid samples out of "
            f"{target_samples} requested."
        )

    return samples[:target_samples]


def parse_date_and_time_from_filename(source_tif: str) -> Tuple[str, str]:
    """Extract date and time from a filename like ..._20241014_120709_....tif."""
    if not source_tif:
        return "", ""

    stem = Path(source_tif).stem
    parts = stem.split("_")

    if len(parts) < 2:
        return "", ""

    for idx in range(len(parts) - 1):
        candidate = parts[idx]
        next_candidate = parts[idx + 1]
        if len(candidate) == 8 and candidate.isdigit():
            date_value = f"{candidate[:4]}-{candidate[4:6]}-{candidate[6:8]}"
            if len(next_candidate) == 6 and next_candidate.isdigit():
                time_value = (
                    f"{next_candidate[:2]}:{next_candidate[2:4]}:{next_candidate[4:6]}"
                )
                return date_value, time_value
            return date_value, ""

    return "", ""


def save_hdf5(
    samples: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Save samples to HDF5."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(output_path, "w") as h5:
        h5.attrs["num_samples"] = len(samples)
        h5.attrs["box_size_meters"] = BOX_SIZE_METERS

        for i, sample in enumerate(tqdm(samples, desc="Writing HDF5")):
            group = h5.create_group(f"sample_{i}")

            date_value, time_value = parse_date_and_time_from_filename(
                str(sample.get("source_tif", ""))
            )

            group.attrs["municipality"] = sample["municipality"]
            group.attrs["source_tif"] = sample["source_tif"]
            group.attrs["date"] = date_value
            group.attrs["time_of_day"] = time_value
            group.attrs["center_x"] = float(sample["center_x"])
            group.attrs["center_y"] = float(sample["center_y"])

            for field in SUN_SATELLITE_FIELDS:
                group.attrs[field] = float(sample.get(field, float("nan")))

            group.create_dataset(
                "historic_rgb",
                data=sample["historic_rgb"],
                compression="gzip",
            )

            group.create_dataset(
                "prewar_rgb",
                data=sample["prewar_rgb"],
                compression="gzip",
            )


def main() -> None:
    municipalities = load_municipalities()

    samples = build_validation_dataset(
        raw_tif_dir=RAW_TIF_DIR,
        prewar_tif_path=PREWAR_TIF_PATH,
        municipalities=municipalities,
        target_samples=TARGET_SAMPLE_SIZE,
    )

    save_hdf5(
        samples=samples,
        output_path=OUTPUT_H5,
    )

    print(
        f"Saved {len(samples)} samples to {OUTPUT_H5}"
    )


if __name__ == "__main__":
    main()