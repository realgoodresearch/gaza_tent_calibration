"""Apply a fitted calibration posterior to a supplied raster mastergrid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import xy
from affine import Affine
from shapely.geometry import box

from workflow.application import (
    TARGET_CRS,
    _attach_admin_units,
    _classify_grid_land_use,
    _compute_week_context_features,
    _design_matrix_for_spec,
    _read_to_target,
)
from workflow.postprocessing import apply_weekly_postprocessing, load_yellow_line_context
from workflow.publication import expected_counts
from workflow.specs import load_best_spec


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "src" / "calibration" / "artifacts"
NODATA = np.float32(-9999.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write calibrated tent counts on an aligned raster mastergrid.")
    parser.add_argument("--week-key", required=True, help="Week key matching the point and coverage files.")
    parser.add_argument("--mastergrid", type=Path, default=DATA / "mastergrid" / "mastergrid.tif")
    parser.add_argument("--points", type=Path)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--posterior", type=Path, default=ARTIFACTS / "final_model_posterior.nc")
    parser.add_argument("--spec", type=Path, default=ARTIFACTS / "selected_model.json")
    parser.add_argument("--scaler", type=Path, default=ARTIFACTS / "feature_scaler.json")
    parser.add_argument("--spline-knots", type=Path, default=ARTIFACTS / "spline_knots.json")
    parser.add_argument("--municipalities", type=Path, default=DATA / "gaza_boundaries" / "municipal" / "GazaStrip_MunicipalBoundaries.shp")
    parser.add_argument("--blocks", type=Path, default=DATA / "gaza_boundaries" / "dash_population_blocks.geojson")
    parser.add_argument("--spatial-context", type=Path, default=DATA / "spatial_context")
    parser.add_argument("--yellow-line", type=Path, default=DATA / "spatial_context" / "yellow-line-zone.json")
    parser.add_argument("--prewar-tif", type=Path, default=DATA / "tif_files" / "prewar_gaza.tif")
    parser.add_argument("--land-use-model", type=Path, default=ROOT / "src" / "validation" / "land_use_classification" / "land_use_kmeans.pkl")
    parser.add_argument("--land-use-labels", type=Path, default=ROOT / "src" / "validation" / "land_use_classification" / "manual_land_use_labels.json")
    parser.add_argument("--output", type=Path, default=None, help="Eight-band output GeoTIFF path.")
    parser.add_argument("--postprocessed-points-output", type=Path, default=None, help="GeoPackage containing the points used as model input.")
    parser.add_argument("--land-use-cache", type=Path, default=ROOT / "outputs" / "tent_counts" / "mastergrid_land_use_labels.csv")
    parser.add_argument("--postprocess-eps-m", type=float, default=40.0)
    parser.add_argument("--postprocess-min-samples", type=int, default=3)
    parser.add_argument("--hours-from-sunrise-cap", type=float, default=None, help="Optional upper cap for the model time-since-sunrise feature.")
    parser.add_argument("--hours-from-sunrise-offset", type=float, default=0.0, help="Offset applied only to the model time-since-sunrise feature.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _week_date(key: str) -> str:
    if len(key) < 8 or not key[:8].isdigit():
        raise ValueError("--week-key must start with YYYYMMDD")
    return f"{key[:4]}-{key[4:6]}-{key[6:8]}"


def _load_spec_and_fit(args: argparse.Namespace):
    if not args.posterior.exists() or not args.spec.exists():
        raise FileNotFoundError("Both --posterior and --spec must exist")
    spec = load_best_spec(args.spec)
    payload = json.loads(args.spec.read_text())
    fit = az.from_netcdf(args.posterior)
    if spec.model_kind == "hurdle_nb":
        count_dim = int(fit.posterior["beta_count"].shape[-1])
        zero_dim = int(fit.posterior["beta_zero"].shape[-1])
        if count_dim != len(spec.count_columns or []) or zero_dim != len(spec.zero_columns or []):
            final = payload.get("final_run", {})
            count_cols, zero_cols = final.get("count_columns"), final.get("zero_columns")
            if not isinstance(count_cols, list) or not isinstance(zero_cols, list):
                raise ValueError("Posterior/spec coefficient dimensions differ and final_run columns are unavailable")
            spec.count_columns, spec.zero_columns = count_cols, zero_cols
            spec.final_columns = list(dict.fromkeys(count_cols + zero_cols))
        if count_dim != len(spec.count_columns or []) or zero_dim != len(spec.zero_columns or []):
            raise ValueError("Posterior/spec coefficient dimensions differ")
    elif "beta" in fit.posterior and int(fit.posterior["beta"].shape[-1]) != len(spec.final_columns):
        raise ValueError("Posterior/spec coefficient dimensions differ")
    scaler = json.loads(args.scaler.read_text()) if args.scaler.exists() else {}
    knots = json.loads(args.spline_knots.read_text()) if args.spline_knots.exists() else {}
    return spec, fit, scaler, knots


def _grid_from_raster(path: Path) -> tuple[gpd.GeoDataFrame, dict]:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Mastergrid has no CRS: {path}")
        if src.crs.to_string() != TARGET_CRS:
            raise ValueError(f"Mastergrid CRS must be {TARGET_CRS}; received {src.crs}")
        values = src.read(1)
        valid = src.read_masks(1) > 0
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= values != src.nodata
        valid &= np.isfinite(values)
        rows, cols = np.where(valid)
        geometries, centers_x, centers_y = [], [], []
        for row, col in zip(rows, cols):
            ulx, uly = xy(src.transform, int(row), int(col), offset="ul")
            lrx, lry = xy(src.transform, int(row), int(col), offset="lr")
            geometries.append(box(ulx, lry, lrx, uly))
            cx, cy = xy(src.transform, int(row), int(col), offset="center")
            centers_x.append(cx)
            centers_y.append(cy)
        metadata = {"height": src.height, "width": src.width, "transform": src.transform, "crs": src.crs, "valid": valid}
    grid = gpd.GeoDataFrame({"cell_id": rows * metadata["width"] + cols, "row": rows, "col": cols, "center_x": centers_x, "center_y": centers_y}, geometry=geometries, crs=TARGET_CRS)
    grid["cell_area_m2"] = grid.geometry.area
    return grid, metadata


def _cell_metadata(points: gpd.GeoDataFrame, grid: gpd.GeoDataFrame, transform, height: int, width: int) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(columns=["cell_id", "model_input_points"])
    points = points.to_crs(TARGET_CRS).reset_index(drop=True)
    rows, cols = rasterio.transform.rowcol(transform, points.geometry.x.to_numpy(), points.geometry.y.to_numpy())
    frame = pd.DataFrame({"row": rows, "col": cols})
    frame = frame[(frame.row >= 0) & (frame.row < height) & (frame.col >= 0) & (frame.col < width)].copy()
    frame["cell_id"] = frame.row * width + frame.col
    grid_ids = set(grid.cell_id.tolist())
    frame = frame[frame.cell_id.isin(grid_ids)]
    for column in ("satellite_elevation", "sun_elevation", "satellite_azimuth", "sun_azimuth", "hours_from_sunrise"):
        if column in points.columns:
            frame[column] = pd.to_numeric(points.loc[frame.index, column], errors="coerce")
    out = frame.groupby("cell_id", as_index=False).size().rename(columns={"size": "model_input_points"})
    for column in ("satellite_elevation", "sun_elevation", "satellite_azimuth", "sun_azimuth", "hours_from_sunrise"):
        if column in frame:
            out = out.merge(frame.groupby("cell_id", as_index=False)[column].median(), on="cell_id", how="left")
    return out


def _coverage_arrays(coverage: gpd.GeoDataFrame, metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize coverage once; 10 m subpixels estimate fractional coverage."""
    geometry_values = [(geometry, 1) for geometry in coverage.geometry if geometry is not None and not geometry.is_empty]
    if not geometry_values:
        raise ValueError("Coverage geometry is empty")
    height, width, transform = metadata["height"], metadata["width"], metadata["transform"]
    available = rasterize(geometry_values, out_shape=(height, width), transform=transform, fill=0, dtype="uint8").astype(bool)
    factor = 10
    fine = rasterize(
        geometry_values,
        out_shape=(height * factor, width * factor),
        transform=transform * Affine.scale(1 / factor, 1 / factor),
        fill=0,
        dtype="uint8",
    )
    fraction = fine.reshape(height, factor, width, factor).mean(axis=(1, 3))
    return available, fraction


def _write_raster(path: Path, metadata: dict, bands: list[np.ndarray]) -> None:
    profile = {"driver": "GTiff", "height": metadata["height"], "width": metadata["width"], "count": len(bands), "dtype": "float32", "crs": metadata["crs"], "transform": metadata["transform"], "nodata": NODATA, "compress": "deflate", "predictor": 3, "tiled": True}
    names = ("valid_mastergrid", "coverage_fraction", "prediction_available", "source_points", "model_input_points", "adjusted_count_mean", "adjusted_count_p2_5", "adjusted_count_p97_5")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for index, (name, band) in enumerate(zip(names, bands), start=1):
            dst.write(band.astype(np.float32), index)
            dst.set_band_description(index, name)
        dst.update_tags(coverage_rule="cell center is within coverage", uncertainty="posterior credible interval for expected count")


def main() -> None:
    args = parse_args()
    args.points = args.points or DATA / "tent_predictions" / "merged" / f"{args.week_key}.gpkg"
    args.coverage = args.coverage or DATA / "tent_predictions" / "coverage" / f"{args.week_key}_coverage.geojson"
    output_dir = ROOT / "outputs" / "tent_counts" / "mastergrid" / args.week_key
    output = args.output or output_dir / f"{args.week_key}_mastergrid_predictions.tif"
    postprocessed_output = args.postprocessed_points_output or output_dir / f"{args.week_key}_postprocessed_points.gpkg"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite to replace it")
    if postprocessed_output.exists() and not args.overwrite:
        raise FileExistsError(f"Postprocessed point output exists: {postprocessed_output}; pass --overwrite to replace it")
    for path in (args.mastergrid, args.points, args.coverage, args.municipalities, args.blocks, args.prewar_tif, args.land_use_model, args.land_use_labels):
        if not path.exists():
            raise FileNotFoundError(path)
    spec, fit, scaler, knots = _load_spec_and_fit(args)
    grid, metadata = _grid_from_raster(args.mastergrid)
    grid = _attach_admin_units(grid, args.municipalities, args.blocks)
    land_use = _classify_grid_land_use(grid, args.prewar_tif, args.land_use_model, args.land_use_labels, args.land_use_cache)
    grid = grid.merge(land_use, on="cell_id", how="left")
    coverage = _read_to_target(args.coverage)
    coverage_available, coverage_fraction = _coverage_arrays(coverage, metadata)
    grid["coverage_fraction"] = coverage_fraction[grid.row.to_numpy(dtype=int), grid.col.to_numpy(dtype=int)]
    grid["prediction_available"] = coverage_available[grid.row.to_numpy(dtype=int), grid.col.to_numpy(dtype=int)]
    yellow, windows = load_yellow_line_context(args.yellow_line)
    points = _read_to_target(args.points)
    source_cell_data = _cell_metadata(points, grid, metadata["transform"], metadata["height"], metadata["width"]).rename(columns={"model_input_points": "source_points"})
    points, postprocess = apply_weekly_postprocessing(points, args.week_key, yellow, windows, args.postprocess_eps_m, args.postprocess_min_samples)
    postprocessed_output.parent.mkdir(parents=True, exist_ok=True)
    points.to_file(postprocessed_output, driver="GPKG")
    cell_data = _cell_metadata(points, grid, metadata["transform"], metadata["height"], metadata["width"])
    grid = grid.merge(source_cell_data[["cell_id", "source_points"]], on="cell_id", how="left")
    grid = grid.merge(cell_data, on="cell_id", how="left")
    grid["source_points"] = grid.source_points.fillna(0.0)
    grid["model_input_points"] = grid.model_input_points.fillna(0.0)
    # The shared design-matrix builder retains the legacy raw_points name.
    grid["raw_points"] = grid.model_input_points
    for column in ("satellite_elevation", "sun_elevation", "satellite_azimuth", "sun_azimuth", "hours_from_sunrise"):
        if column in grid.columns and column in cell_data.columns:
            default = pd.to_numeric(cell_data[column], errors="coerce").median()
            grid[column] = pd.to_numeric(grid[column], errors="coerce").fillna(default)
    if args.hours_from_sunrise_offset:
        grid["hours_from_sunrise"] = grid["hours_from_sunrise"] + args.hours_from_sunrise_offset
    if args.hours_from_sunrise_cap is not None:
        if args.hours_from_sunrise_cap < 0:
            raise ValueError("--hours-from-sunrise-cap must be non-negative")
        grid["hours_from_sunrise"] = grid["hours_from_sunrise"].clip(upper=args.hours_from_sunrise_cap)
    grid = _compute_week_context_features(grid, _week_date(args.week_key), args.spatial_context)
    municipality_names = sorted(_read_to_target(args.municipalities)["NAME"].dropna().astype(str).unique())
    active = grid[grid.prediction_available].copy()
    count_cols = spec.count_columns if spec.model_kind == "hurdle_nb" and spec.count_columns else spec.final_columns
    zero_cols = spec.zero_columns if spec.model_kind == "hurdle_nb" and spec.zero_columns else count_cols
    X, muni = _design_matrix_for_spec(active, spec, municipality_names, _week_date(args.week_key), count_cols, scaler_params=scaler, spline_knots=knots)
    X_zero, _ = _design_matrix_for_spec(active, spec, municipality_names, _week_date(args.week_key), zero_cols, scaler_params=scaler, spline_knots=knots)
    adjusted_mean = np.empty(len(active), dtype=float)
    adjusted_p2_5 = np.empty(len(active), dtype=float)
    adjusted_p97_5 = np.empty(len(active), dtype=float)
    for start in range(0, len(active), 1000):
        stop = min(start + 1000, len(active))
        draws = expected_counts(spec, fit, X[start:stop], muni[start:stop], X_zero=X_zero[start:stop] if spec.model_kind == "hurdle_nb" else None)
        adjusted_mean[start:stop] = draws.mean(axis=1)
        adjusted_p2_5[start:stop] = np.quantile(draws, 0.025, axis=1)
        adjusted_p97_5[start:stop] = np.quantile(draws, 0.975, axis=1)
    active["adjusted_count_mean"] = adjusted_mean
    active["adjusted_count_p2_5"] = adjusted_p2_5
    active["adjusted_count_p97_5"] = adjusted_p97_5
    grid = grid.merge(active[["cell_id", "adjusted_count_mean", "adjusted_count_p2_5", "adjusted_count_p97_5"]], on="cell_id", how="left")
    bands = [np.full((metadata["height"], metadata["width"]), NODATA, dtype=np.float32) for _ in range(8)]
    for _, cell in grid.iterrows():
        row, col = int(cell.row), int(cell.col)
        bands[0][row, col] = 1
        bands[1][row, col] = cell.coverage_fraction
        bands[2][row, col] = float(cell.prediction_available)
        # Preserve detector and model-input counts at valid cells for audit, even
        # where the center-based coverage rule prevents a calibrated prediction.
        bands[3][row, col] = cell.source_points
        bands[4][row, col] = cell.model_input_points
        if cell.prediction_available:
            bands[5][row, col] = cell.adjusted_count_mean
            bands[6][row, col] = cell.adjusted_count_p2_5
            bands[7][row, col] = cell.adjusted_count_p97_5
    _write_raster(output, metadata, bands)
    summary = {"week_key": args.week_key, "mastergrid_cells": int(len(grid)), "prediction_cells": int(active.shape[0]), "coverage_area_km2": float((grid.coverage_fraction * grid.cell_area_m2).sum() / 1e6), "source_points": int(grid.source_points.sum()), "model_input_points": int(grid.model_input_points.sum()), "postprocessing": postprocess.__dict__, "output": str(output), "postprocessed_points": str(postprocessed_output), "hours_from_sunrise_offset": args.hours_from_sunrise_offset, "hours_from_sunrise_cap": args.hours_from_sunrise_cap}
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
