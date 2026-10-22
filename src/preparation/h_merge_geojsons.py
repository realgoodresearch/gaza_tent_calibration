"""
merge_geojsons.py

Merge multiple point-cloud GeoJSON files from a folder into a single deduplicated
GeoPackage (.gpkg), using the same merge_close_points_global function used during
prediction.

Contributed by GitHub collaborator @rapsoj.

Usage:
    poetry run merge-geojsons config.yaml

All settings come from the ``merge`` section of the config; the input folder
defaults to ``prediction.output_folder`` so the stage chains onto the
prediction output without extra configuration.
"""

import json
import calendar
import re
import shutil
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import click
import geopandas as gpd
from rasterio.features import shapes
import rasterio
import tifffile
from rasterio.enums import Resampling
from shapely.geometry import Point
from shapely.geometry import shape
from shapely.ops import unary_union
from rasterio.warp import transform_geom
import yaml

from displacement_tracker.util.config import flow_option, load_flow_config
from displacement_tracker.util.deduplication import merge_close_points_global
from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.thresholding import filter_points_by_adjusted_peak

LOGGER = setup_logging("merge_geojsons")

DATETIME_PATTERN = re.compile(r"(?:^|_)(\d{8})_(\d{6})(?:_|$)")
DATE_PATTERN = re.compile(r"(?:^|_)(\d{8})(?:_|$)")

SUN_SATELLITE_FIELDS = (
    "sun_azimuth",
    "sun_elevation",
    "satellite_azimuth",
    "satellite_elevation",
)

# Coverage footprints only need to be approximately right, not pixel-perfect.
# Polygonizing at full raster resolution can turn speckled nodata edges into
# hundreds of thousands of tiny polygons, making unary_union take hours.
COVERAGE_MAX_DIMENSION = 2000


def load_thresholds(thresholds_config: str | None) -> dict[str, float]:
    """
    Load per-file adj_peak thresholds from a YAML file.

    Expected format::

        default: 0.05          # optional global fallback (overrides --min-adj-peak)
        per_file:
          some_file.json: 0.10
          other_file.geojson: 0.20
    """
    if not thresholds_config:
        return {}
    path = Path(thresholds_config)
    if not path.exists():
        raise click.ClickException(f"Thresholds config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_threshold(
    filename: str,
    thresholds_data: dict,
    global_threshold: float,
) -> float:
    """Return the effective adj_peak threshold for a given filename."""
    per_file = thresholds_data.get("per_file") or {}
    if filename in per_file:
        return float(per_file[filename])
    if "default" in thresholds_data:
        return float(thresholds_data["default"])
    return global_threshold


def extract_sun_satellite_metadata(tif_path: str | Path | None) -> dict[str, float]:
    """Read sun/satellite azimuth and elevation from a TIFF ImageDescription tag."""
    defaults = {field: float("nan") for field in SUN_SATELLITE_FIELDS}
    if not tif_path:
        return defaults

    try:
        with tifffile.TiffFile(str(tif_path)) as tif:
            description = tif.pages[0].tags["ImageDescription"].value
            props = json.loads(description)["properties"]
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return defaults

    return {
        field: (
            float(props[field])
            if field in props and props[field] is not None
            else float("nan")
        )
        for field in SUN_SATELLITE_FIELDS
    }


def load_points_from_geojson(path: Path) -> list[tuple]:
    """
    Load all Point features from a GeoJSON file.
    Returns a list of (lat, lon, peak_value, adjusted_peak) tuples.
    adjusted_peak defaults to 0.0 when the field is absent.
    """
    with path.open("r", encoding="utf-8") as f:
        gj = json.load(f)

    points = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        props = feat.get("properties") or {}
        peak = float(props.get("peak_value", 0.0))
        adj_raw = props.get("adjusted_peak", 0.0)
        adj_peak = float(adj_raw) if adj_raw is not None else 0.0
        points.append([lat, lon, peak, adj_peak])

    return points


def load_zone_geometry(zones_path: str | None, label: str):
    """Load and unify polygon geometries from a shapefile or GeoPackage file."""
    if not zones_path:
        return None

    path = Path(zones_path)
    if not path.exists():
        raise click.ClickException(f"{label} zones file not found: {path}")

    try:
        zones_gdf = gpd.read_file(path)
    except Exception as exc:
        raise click.ClickException(
            f"Failed to read {label} zones file {path}: {exc}"
        ) from exc

    if zones_gdf.empty:
        LOGGER.warning("%s zones file is empty: %s", label, path)
        return None

    if zones_gdf.crs is None:
        LOGGER.warning(
            "%s zones CRS missing, assuming EPSG:4326: %s", label, path
        )
        zones_gdf = zones_gdf.set_crs("EPSG:4326")
    else:
        zones_gdf = zones_gdf.to_crs("EPSG:4326")

    geom = zones_gdf.geometry.union_all()
    LOGGER.info("Loaded %s zones from %s", label, path)
    return geom


def filter_points_by_exclusion(points: list[tuple], exclusion_geom) -> list[tuple]:
    """Drop points that lie inside exclusion geometry."""
    if exclusion_geom is None:
        return points

    kept = []
    for lat, lon, peak, adj_peak in points:
        pt = Point(lon, lat)
        if exclusion_geom.contains(pt):
            continue
        kept.append((lat, lon, peak, adj_peak))
    return kept


def filter_points_by_inclusion(points: list[tuple], inclusion_geom) -> list[tuple]:
    """Drop points that lie outside inclusion geometry."""
    if inclusion_geom is None:
        return points

    kept = []
    for lat, lon, peak, adj_peak in points:
        pt = Point(lon, lat)
        if not inclusion_geom.contains(pt):
            continue
        kept.append((lat, lon, peak, adj_peak))
    return kept


def save_merged_gpkg(points: list[tuple], out_path: Path) -> None:
    """Save merged tuples to a GeoPackage file.

    Writes to a local temp file first, then moves it into place. GPKG writes
    are many small SQLite transactions; doing that directly against a
    network-mounted path (e.g. a Google Drive mount in Colab) can make the
    write take an extremely long time with no progress output.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for point in points:
        lat, lon, peak, adj_peak = point[:4]
        row = {
            "geometry": Point(lon, lat),
            "name": "tents",
            "peak_value": peak,
            "adjusted_peak": adj_peak,
        }
        if len(point) >= 6:
            row["date"] = point[4]
            row["time"] = point[5]
        if len(point) >= 10:
            for field, value in zip(SUN_SATELLITE_FIELDS, point[6:10]):
                row[field] = float(value)
        rows.append(row)

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    LOGGER.info("Writing GeoPackage with %d points to local temp file...", len(gdf))
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / out_path.name
        gdf.to_file(str(tmp_path), driver="GPKG")
        shutil.move(str(tmp_path), str(out_path))

    LOGGER.info("Merged GeoPackage saved to %s (%d points)", out_path, len(gdf))


def merge_points_with_members(
    flat: list[tuple],
    min_distance_m: float,
    agreement: int,
) -> tuple[list[tuple], list[list[int]]]:
    """Merge points and return member indices; supports old/new dedup signatures."""
    try:
        merged, merged_members = merge_close_points_global(
            flat,
            min_distance_m=min_distance_m,
            agreement=agreement,
            return_members=True,
        )
        return merged, merged_members
    except TypeError:
        LOGGER.warning(
            "deduplication.merge_close_points_global does not accept return_members; "
            "falling back to compatibility mode."
        )

    merged = merge_close_points_global(
        flat,
        min_distance_m=min_distance_m,
        agreement=agreement,
    )

    # Group by (peak, adj_peak) up front so lookup is O(1) average instead of
    # rescanning all of `flat` for every merged point (O(n * len(merged))).
    by_key: dict[tuple[float, float], list[int]] = {}
    for idx, pt in enumerate(flat):
        by_key.setdefault((pt[2], pt[3]), []).append(idx)

    merged_members: list[list[int]] = []
    for lat, lon, peak, adj_peak in merged:
        candidates = by_key.get((peak, adj_peak)) or list(range(len(flat)))

        rep_idx = min(
            candidates,
            key=lambda idx: abs(flat[idx][0] - lat) + abs(flat[idx][1] - lon),
        )
        merged_members.append([rep_idx])

    return merged, merged_members


def process_geojson_files(
    geojson_files: list[Path],
    output_gpkg: Path,
    min_distance_m: float,
    agreement: int,
    min_adj_peak: float,
    adjustment_factor: float,
    thresholds_data: dict,
    exclusion_geom,
    inclusion_geom,
    geotiff_dir: Path | None = None,
) -> None:
    """Process a list of GeoJSON/JSON files and write one merged GPKG."""
    if not geojson_files:
        LOGGER.warning("No GeoJSON files supplied for %s", output_gpkg)
        return

    LOGGER.info(
        "Found %d GeoJSON files for output %s",
        len(geojson_files),
        output_gpkg,
    )

    flat: list[tuple] = []
    source_meta: list[tuple] = []
    for path in geojson_files:
        pts = load_points_from_geojson(path)
        _, source_date, source_time = extract_prediction_datetime(path)
        tif_path = (
            geotiff_dir / f"{path.stem}.tif"
            if geotiff_dir is not None
            else None
        )
        sun_satellite_metadata = extract_sun_satellite_metadata(tif_path)
        threshold = resolve_threshold(path.name, thresholds_data, min_adj_peak)
        loaded = len(pts)
        pts = filter_points_by_adjusted_peak(pts, threshold, adjustment_factor)
        LOGGER.info(
            "  %s: %d points loaded, %d kept (adj_peak >= %.4f)",
            path.name,
            loaded,
            len(pts),
            threshold,
        )

        if exclusion_geom is not None:
            before_exclusion = len(pts)
            pts = filter_points_by_exclusion(pts, exclusion_geom)
            LOGGER.info(
                "  %s: %d kept after exclusion filtering",
                path.name,
                len(pts),
            )
            if before_exclusion != len(pts):
                LOGGER.info(
                    "  %s: %d points removed by exclusion zones",
                    path.name,
                    before_exclusion - len(pts),
                )

        if inclusion_geom is not None:
            before_inclusion = len(pts)
            pts = filter_points_by_inclusion(pts, inclusion_geom)
            LOGGER.info(
                "  %s: %d kept after inclusion filtering",
                path.name,
                len(pts),
            )
            if before_inclusion != len(pts):
                LOGGER.info(
                    "  %s: %d points removed outside inclusion zone",
                    path.name,
                    before_inclusion - len(pts),
                )

        flat.extend(pts)
        source_meta.extend(
            [
                (
                    source_date,
                    source_time,
                    *(sun_satellite_metadata[field] for field in SUN_SATELLITE_FIELDS),
                )
            ]
            * len(pts)
        )

    LOGGER.info("Total points before merge: %d", len(flat))

    merged, merged_members = merge_points_with_members(
        flat=flat,
        min_distance_m=min_distance_m,
        agreement=agreement,
    )

    merged_with_meta = []
    for merged_point, member_indices in zip(merged, merged_members):
        rep_idx = max(member_indices, key=lambda idx: flat[idx][2])
        (
            source_date,
            source_time,
            *sun_satellite_values,
        ) = source_meta[rep_idx]
        lat, lon, peak, adj_peak = merged_point
        merged_with_meta.append(
            (
                lat,
                lon,
                peak,
                adj_peak,
                source_date,
                source_time,
                *sun_satellite_values,
            )
        )

    LOGGER.info("Total points after merge: %d", len(merged_with_meta))
    save_merged_gpkg(merged_with_meta, output_gpkg)


def extract_prediction_date(path: Path) -> date | None:
    """Extract YYYYMMDD date token from filename."""
    parsed_dt, _, _ = extract_prediction_datetime(path)
    if parsed_dt is None:
        return None
    return parsed_dt.date()


def extract_prediction_datetime(path: Path) -> tuple[datetime | None, str | None, str | None]:
    """Extract source datetime and ISO date/time strings from filename."""
    stem = path.stem

    dt_match = DATETIME_PATTERN.search(stem)
    if dt_match:
        raw = f"{dt_match.group(1)}_{dt_match.group(2)}"
        try:
            parsed = datetime.strptime(raw, "%Y%m%d_%H%M%S")
            return parsed, parsed.strftime("%Y-%m-%d"), parsed.strftime("%H:%M:%S")
        except ValueError:
            LOGGER.warning(
                "Skipping invalid YYYYMMDD_HHMMSS token in filename: %s",
                path.name,
            )

    date_match = DATE_PATTERN.search(stem)
    if date_match:
        try:
            parsed_date = datetime.strptime(date_match.group(1), "%Y%m%d")
            return parsed_date, parsed_date.strftime("%Y-%m-%d"), None
        except ValueError:
            LOGGER.warning("Skipping invalid YYYYMMDD token in filename: %s", path.name)

    return None, None, None


def grouping_window(d: date, mode: Literal["day", "week", "month"]) -> tuple[date, date]:
    """Return [start, end] calendar window for one date."""
    if mode == "day":
        return d, d
    if mode == "week":
        start = d - timedelta(days=d.weekday())
        end = start + timedelta(days=6)
        return start, end

    month_days = calendar.monthrange(d.year, d.month)[1]
    start = date(d.year, d.month, 1)
    end = date(d.year, d.month, month_days)
    return start, end


def range_label(start: date, end: date) -> str:
    """Create stable output filename token for a date range."""
    return f"{start:%Y%m%d}-{end:%Y%m%d}"


def collect_prediction_groups(
    input_dir: Path,
    mode: Literal["day", "week", "month"],
) -> list[tuple[date, date, list[Path]]]:
    """Collect prediction JSON files grouped by day/week/month."""
    geojson_files = sorted(input_dir.rglob("*.geojson")) + sorted(
        input_dir.rglob("*.json")
    )
    grouped: dict[tuple[date, date], list[Path]] = {}

    for path in geojson_files:
        d = extract_prediction_date(path)
        if d is None:
            continue
        key = grouping_window(d, mode)
        grouped.setdefault(key, []).append(path)

    return [
        (start, end, sorted(paths))
        for (start, end), paths in sorted(grouped.items(), key=lambda kv: kv[0][0])
    ]


def save_group_coverage_polygon(
    prediction_files: list[Path],
    geotiff_dir: Path,
    output_path: Path,
) -> None:
    """Save a polygon of combined valid-data TIFF coverage for one group."""
    coverage_geoms = []

    for pred_path in prediction_files:
        tif_path = geotiff_dir / f"{pred_path.stem}.tif"
        if not tif_path.exists():
            LOGGER.warning("Missing TIFF for prediction %s: %s", pred_path.name, tif_path)
            continue

        with rasterio.open(tif_path) as src:
            # Downsample before polygonizing: a footprint doesn't need
            # per-pixel precision, and full-resolution masks with speckled
            # nodata edges can explode into huge polygon counts that make
            # unary_union take hours.
            scale = max(
                1, -(-max(src.width, src.height) // COVERAGE_MAX_DIMENSION)
            )
            out_shape = (max(1, src.height // scale), max(1, src.width // scale))
            valid_mask = src.dataset_mask(
                out_shape=out_shape, resampling=Resampling.nearest
            ) > 0
            transform = src.transform * src.transform.scale(
                src.width / out_shape[1], src.height / out_shape[0]
            )
            tif_geoms = [
                shape(geom)
                for geom, value in shapes(
                    valid_mask.astype("uint8"),
                    mask=valid_mask,
                    transform=transform,
                )
                if value
            ]

            if not tif_geoms:
                LOGGER.warning("No valid-data footprint found for TIFF %s", tif_path)
                continue

            geom = unary_union(tif_geoms)
            if src.crs and str(src.crs) != "EPSG:4326":
                geom = shape(
                    transform_geom(src.crs, "EPSG:4326", geom.__geo_interface__)
                )
            coverage_geoms.append(geom)

    if not coverage_geoms:
        LOGGER.warning("No TIFF coverage polygons could be built for %s", output_path)
        return

    union_geom = unary_union(coverage_geoms)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coverage_gdf = gpd.GeoDataFrame(
        [{"geometry": union_geom, "source_tifs": len(coverage_geoms)}],
        crs="EPSG:4326",
    )
    coverage_gdf.to_file(str(output_path), driver="GeoJSON")
    LOGGER.info("Coverage polygon saved to %s", output_path)


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@flow_option(default="predict")
def cli(
    config: str,
    flow: str,
) -> None:
    """Merge prediction GeoJSONs into one deduplicated GeoPackage.

    Reads the ``merge`` section of the YAML config; the input folder
    defaults to ``prediction.output_folder``.
    """
    params = load_flow_config(config, flow)
    merge_cfg = params.get("merge") or {}

    input_folder = merge_cfg.get("input_folder") or (
        params.get("prediction") or {}
    ).get("output_folder")
    if not input_folder:
        raise click.ClickException(
            "Missing required config key: merge.input_folder "
            "(or prediction.output_folder as fallback)"
        )
    output_gpkg = merge_cfg.get("output")
    if not output_gpkg:
        raise click.ClickException("Missing required config key: merge.output")

    # only pass keys the config actually sets — merge_geojsons() signature
    # defaults are the single source of truth for the rest
    kwargs = {
        key: merge_cfg[key]
        for key in (
            "min_distance_m",
            "agreement",
            "min_adj_peak",
            "adjustment_factor",
            "thresholds_config",
            "exclusion_zones_gpkg",
            "inclusion_zone",
            "geotiff_dir",
        )
        if merge_cfg.get(key) is not None
    }

    process_mode = merge_cfg.get("aggregation_period")
    if process_mode is None:
        # Backward-compatible config alias.
        process_mode = merge_cfg.get("process_by")

    if process_mode is not None and process_mode not in {"day", "week", "month"}:
        raise click.ClickException(
            "Invalid merge.aggregation_period value. Use one of: day, week, month"
        )

    resolved_geotiff_dir = (
        params.get("geotiff_dir")
        or merge_cfg.get("geotiff_dir")
        or (params.get("train") or {}).get("geotiff_dir")
    )
    if resolved_geotiff_dir is not None:
        kwargs.setdefault("geotiff_dir", resolved_geotiff_dir)

    if process_mode is not None:
        input_dir = Path(input_folder)
        if not input_dir.is_dir():
            raise click.ClickException(f"Input folder not found: {input_dir}")

        geotiff_dir_raw = params.get("geotiff_dir")
        if not geotiff_dir_raw:
            # Backward-compatible fallbacks for older config layouts.
            geotiff_dir_raw = merge_cfg.get("geotiff_dir") or (
                (params.get("train") or {}).get("geotiff_dir")
            )
        if not geotiff_dir_raw:
            raise click.ClickException(
                "Missing required config key for grouped processing: "
                "geotiff_dir (resolved flow key), merge.geotiff_dir, "
                "or train.geotiff_dir"
            )
        geotiff_dir = Path(geotiff_dir_raw)
        if not geotiff_dir.is_dir():
            raise click.ClickException(f"GeoTIFF folder not found: {geotiff_dir}")

        thresholds_data = load_thresholds(kwargs.get("thresholds_config"))
        exclusion_geom = load_zone_geometry(
            kwargs.get("exclusion_zones_gpkg"), "exclusion"
        )
        inclusion_geom = load_zone_geometry(
            kwargs.get("inclusion_zone"), "inclusion"
        )

        grouped_files = collect_prediction_groups(input_dir, process_mode)
        if not grouped_files:
            raise click.ClickException(
                f"No dated prediction JSON/GeoJSON files found in {input_dir}."
            )

        merge_defaults = {
            "min_distance_m": 3.0,
            "agreement": 1,
            "min_adj_peak": 0.0,
            "adjustment_factor": 1.0,
        }

        for start, end, group_paths in grouped_files:
            group_name = range_label(start, end)
            out_path = input_dir / "merged" / f"{group_name}.gpkg"
            coverage_path = input_dir / "coverage" / f"{group_name}_coverage.geojson"

            if out_path.exists():
                LOGGER.info(
                    "Skipping %s group %s because merged output already exists: %s",
                    process_mode,
                    group_name,
                    out_path,
                )
                continue

            LOGGER.info(
                "Processing %s group %s with %d files -> %s",
                process_mode,
                group_name,
                len(group_paths),
                out_path,
            )

            process_geojson_files(
                geojson_files=group_paths,
                output_gpkg=out_path,
                min_distance_m=kwargs.get(
                    "min_distance_m", merge_defaults["min_distance_m"]
                ),
                agreement=kwargs.get(
                    "agreement", merge_defaults["agreement"]
                ),
                min_adj_peak=kwargs.get(
                    "min_adj_peak", merge_defaults["min_adj_peak"]
                ),
                adjustment_factor=kwargs.get(
                    "adjustment_factor",
                    merge_defaults["adjustment_factor"],
                ),
                thresholds_data=thresholds_data,
                exclusion_geom=exclusion_geom,
                inclusion_geom=inclusion_geom,
                geotiff_dir=geotiff_dir,
            )

            save_group_coverage_polygon(
                prediction_files=group_paths,
                geotiff_dir=geotiff_dir,
                output_path=coverage_path,
            )
        return

    merge_geojsons(input_folder, output_gpkg, **kwargs)


def merge_geojsons(
    input_folder: str,
    output_gpkg: str,
    *,
    min_distance_m: float = 3.0,
    agreement: int = 1,
    min_adj_peak: float = 0.0,
    adjustment_factor: float = 1.0,
    thresholds_config: str | None = None,
    exclusion_zones_gpkg: str | None = None,
    inclusion_zone: str | None = None,
    geotiff_dir: str | None = None,
) -> None:
    input_dir = Path(input_folder)
    if not input_dir.is_dir():
        raise click.ClickException(f"Input folder not found: {input_dir}")
    geojson_files = sorted(input_dir.glob("*.geojson")) + sorted(
        input_dir.glob("*.json")
    )

    if not geojson_files:
        raise click.ClickException(f"No GeoJSON files found in {input_dir}")

    LOGGER.info("Found %d GeoJSON files in %s", len(geojson_files), input_dir)

    thresholds_data = load_thresholds(thresholds_config)
    exclusion_geom = load_zone_geometry(exclusion_zones_gpkg, "exclusion")
    inclusion_geom = load_zone_geometry(inclusion_zone, "inclusion")
    geotiff_path = Path(geotiff_dir) if geotiff_dir else None
    if geotiff_path is not None and not geotiff_path.is_dir():
        raise click.ClickException(f"GeoTIFF folder not found: {geotiff_path}")

    flat: list[tuple] = []
    source_meta: list[tuple] = []
    for path in geojson_files:
        pts = load_points_from_geojson(path)
        _, source_date, source_time = extract_prediction_datetime(path)
        tif_path = (
            geotiff_path / f"{path.stem}.tif"
            if geotiff_path is not None
            else None
        )
        sun_satellite_metadata = extract_sun_satellite_metadata(tif_path)
        threshold = resolve_threshold(path.name, thresholds_data, min_adj_peak)
        loaded = len(pts)
        pts = filter_points_by_adjusted_peak(pts, threshold, adjustment_factor)
        LOGGER.info(
            "  %s: %d points loaded, %d kept (adj_peak >= %.4f)",
            path.name,
            loaded,
            len(pts),
            threshold,
        )

        if exclusion_geom is not None:
            before_exclusion = len(pts)
            pts = filter_points_by_exclusion(pts, exclusion_geom)
            LOGGER.info(
                "  %s: %d kept after exclusion filtering",
                path.name,
                len(pts),
            )
            if before_exclusion != len(pts):
                LOGGER.info(
                    "  %s: %d points removed by exclusion zones",
                    path.name,
                    before_exclusion - len(pts),
                )

        if inclusion_geom is not None:
            before_inclusion = len(pts)
            pts = filter_points_by_inclusion(pts, inclusion_geom)
            LOGGER.info(
                "  %s: %d kept after inclusion filtering",
                path.name,
                len(pts),
            )
            if before_inclusion != len(pts):
                LOGGER.info(
                    "  %s: %d points removed outside inclusion zone",
                    path.name,
                    before_inclusion - len(pts),
                )

        flat.extend(pts)
        source_meta.extend(
            [
                (
                    source_date,
                    source_time,
                    *(sun_satellite_metadata[field] for field in SUN_SATELLITE_FIELDS),
                )
            ]
            * len(pts)
        )

    LOGGER.info("Total points before merge: %d", len(flat))

    merged, merged_members = merge_points_with_members(
        flat=flat,
        min_distance_m=min_distance_m,
        agreement=agreement,
    )

    merged_with_meta = []
    for merged_point, member_indices in zip(merged, merged_members):
        rep_idx = max(member_indices, key=lambda idx: flat[idx][2])
        (
            source_date,
            source_time,
            *sun_satellite_values,
        ) = source_meta[rep_idx]
        lat, lon, peak, adj_peak = merged_point
        merged_with_meta.append(
            (
                lat,
                lon,
                peak,
                adj_peak,
                source_date,
                source_time,
                *sun_satellite_values,
            )
        )

    LOGGER.info("Total points after merge: %d", len(merged_with_meta))
    save_merged_gpkg(merged_with_meta, Path(output_gpkg))


if __name__ == "__main__":
    cli()