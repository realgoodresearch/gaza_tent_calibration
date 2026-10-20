"""Prepare merged tent detections and coverage for calibration application.

This command intentionally has no site-specific input defaults.  It takes an
existing merged point layer and the Planet TIFFs that produced it, then writes
an enriched point GeoPackage and a valid-imagery coverage GeoJSON.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import tifffile
from astral import LocationInfo
from astral.sun import sun
from rasterio.features import shapes
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import shape
from shapely.ops import unary_union
from tqdm.auto import tqdm
from zoneinfo import ZoneInfo


TARGET_CRS = "EPSG:32636"
GAZA_TIMEZONE = "Asia/Gaza"
GAZA_LATITUDE = 31.5
GAZA_LONGITUDE = 34.47
METADATA_COLUMNS = (
    "source_tif",
    "time",
    "hours_from_sunrise",
    "satellite_azimuth",
    "satellite_elevation",
    "sun_azimuth",
    "sun_elevation",
)


@dataclass(frozen=True)
class SourceImage:
    path: Path
    geometry: object
    acquisition: pd.Timestamp
    satellite_azimuth: float
    satellite_elevation: float
    sun_azimuth: float
    sun_elevation: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach TIFF metadata to merged tent points and build valid-imagery coverage."
    )
    parser.add_argument("--points", type=Path, required=True, help="Merged input point GeoPackage.")
    parser.add_argument("--imagery-dir", type=Path, required=True, help="Directory containing source TIFFs.")
    parser.add_argument("--week-key", required=True, help="Output week key, e.g. 20260803-20260809.")
    parser.add_argument("--output-points", type=Path, required=True, help="Output enriched GeoPackage.")
    parser.add_argument("--output-coverage", type=Path, required=True, help="Output coverage GeoJSON.")
    parser.add_argument(
        "--source-timezone",
        default="UTC",
        help="Timezone of TIFF acquisition timestamps; Planet filenames use UTC (default: UTC).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs.")
    return parser.parse_args()


def _read_metadata(path: Path, source_timezone: str) -> dict[str, float | pd.Timestamp]:
    fields = ("satellite_azimuth", "satellite_elevation", "sun_azimuth", "sun_elevation")
    properties: dict = {}
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        description = page.tags.get("ImageDescription")
        if description is not None:
            try:
                properties = json.loads(description.value).get("properties", {})
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        timestamp_tag = page.tags.get("DateTime")
        timestamp_text = timestamp_tag.value if timestamp_tag is not None else None

    timestamp = pd.to_datetime(timestamp_text, format="%Y:%m:%d %H:%M:%S", errors="coerce")
    if pd.isna(timestamp):
        match = re.search(r"(\d{8})_(\d{6})", path.name)
        timestamp = pd.to_datetime("".join(match.groups()), format="%Y%m%d%H%M%S", errors="coerce") if match else pd.NaT
    missing = [field for field in fields if pd.isna(pd.to_numeric(properties.get(field), errors="coerce"))]
    if pd.isna(timestamp) or missing:
        detail = ["acquisition timestamp"] if pd.isna(timestamp) else []
        detail.extend(missing)
        raise ValueError(f"Missing {', '.join(detail)} in {path}")
    try:
        local_acquisition = timestamp.tz_localize(ZoneInfo(source_timezone)).tz_convert(GAZA_TIMEZONE)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid --source-timezone {source_timezone!r}") from exc
    return {"acquisition": local_acquisition, **{field: float(properties[field]) for field in fields}}


def _valid_geometry(path: Path) -> object:
    """Polygonize valid pixels in row chunks to avoid loading a full TIFF mask."""
    geometries = []
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Source TIFF has no CRS: {path}")
        for row_start in tqdm(range(0, src.height, 1024), desc=f"Coverage {path.name}", unit="chunk", leave=False):
            height = min(1024, src.height - row_start)
            window = Window(0, row_start, src.width, height)
            valid = src.dataset_mask(window=window) > 0
            if not valid.any():
                continue
            transform = window_transform(window, src.transform)
            for geom, value in shapes(valid.astype(np.uint8), mask=valid, transform=transform):
                if value:
                    geometries.append(shape(geom))
        if not geometries:
            raise ValueError(f"No valid pixels found in {path}")
        geom = unary_union(geometries)
        return gpd.GeoSeries([geom], crs=src.crs).to_crs(TARGET_CRS).iloc[0]


def _hours_from_sunrise(acquisition: pd.Timestamp) -> float:
    tz = ZoneInfo(GAZA_TIMEZONE)
    local = acquisition.tz_localize(tz) if acquisition.tzinfo is None else acquisition.tz_convert(tz)
    location = LocationInfo("Gaza Strip", "PS", GAZA_TIMEZONE, GAZA_LATITUDE, GAZA_LONGITUDE)
    sunrise = sun(location.observer, date=local.date(), tzinfo=tz)["sunrise"]
    return float((local - sunrise).total_seconds() / 3600.0)


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"\d{8}-\d{8}", args.week_key):
        raise ValueError("--week-key must be YYYYMMDD-YYYYMMDD")
    for output in (args.output_points, args.output_coverage):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
        output.parent.mkdir(parents=True, exist_ok=True)
    tifs = sorted(args.imagery_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No TIFFs found in {args.imagery_dir}")

    sources = []
    for path in tifs:
        metadata = _read_metadata(path, args.source_timezone)
        sources.append(SourceImage(path=path, geometry=_valid_geometry(path), **metadata))
    source_frame = gpd.GeoDataFrame(
        {"source_index": range(len(sources)), "source_tif": [s.path.name for s in sources]},
        geometry=[s.geometry for s in sources],
        crs=TARGET_CRS,
    )
    points = gpd.read_file(args.points)
    if points.empty:
        raise ValueError(f"Input contains no points: {args.points}")
    if points.crs is None:
        raise ValueError(f"Input point layer has no CRS: {args.points}")
    points = points.to_crs(TARGET_CRS).reset_index(names="_point_index")
    matches = gpd.sjoin(points[["_point_index", "geometry"]], source_frame[["source_index", "geometry"]], how="left", predicate="within")
    unmatched = matches["source_index"].isna()
    if unmatched.any():
        raise ValueError(f"{int(unmatched.sum())} points are outside all source TIFF valid-data masks")
    multi_match_count = int(matches["_point_index"].duplicated(keep=False).sum())
    matches = matches.sort_values(["_point_index", "source_index"]).drop_duplicates("_point_index", keep="first")
    if len(matches) != len(points):
        raise RuntimeError("Source-image matching did not return exactly one row per point")
    points = points.merge(matches[["_point_index", "source_index"]], on="_point_index", how="left", validate="one_to_one")
    source_table = pd.DataFrame([{
        "source_index": i,
        "source_tif": source.path.name,
        "time": source.acquisition.strftime("%H:%M:%S"),
        "hours_from_sunrise": _hours_from_sunrise(source.acquisition),
        "satellite_azimuth": source.satellite_azimuth,
        "satellite_elevation": source.satellite_elevation,
        "sun_azimuth": source.sun_azimuth,
        "sun_elevation": source.sun_elevation,
    } for i, source in enumerate(sources)])
    for column in METADATA_COLUMNS:
        if column in points.columns:
            points = points.drop(columns=column)
    points = points.merge(source_table, on="source_index", how="left", validate="many_to_one")
    points = points.drop(columns=["_point_index", "source_index"])
    points.to_file(args.output_points, driver="GPKG")
    coverage = gpd.GeoDataFrame({"week_key": [args.week_key]}, geometry=[unary_union(source_frame.geometry)], crs=TARGET_CRS)
    coverage.to_file(args.output_coverage, driver="GeoJSON")
    print(json.dumps({
        "week_key": args.week_key,
        "input_points": len(points),
        "source_images": len(sources),
        "coverage_path": str(args.output_coverage),
        "points_path": str(args.output_points),
        "overlapping_source_matches_resolved": multi_match_count,
        "sources": source_table.to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
