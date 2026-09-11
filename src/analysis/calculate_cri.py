"""Calculate 95% credible intervals directly from calibration posterior draws.

Only the municipality/week combinations referenced by summary_statistics.py are
evaluated.  In particular, changes are calculated draw-by-draw, so their
intervals are not produced by subtracting marginal interval endpoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import arviz as az
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "calibration"))

from workflow.application import (  # noqa: E402
    TARGET_CRS,
    _compute_week_context_features,
    _derive_point_time_strings,
    _design_matrix_for_spec,
    _read_to_target,
    _week_date_from_key,
)
from workflow.postprocessing import apply_weekly_postprocessing, load_yellow_line_context  # noqa: E402
from workflow.publication import expected_counts  # noqa: E402
from workflow.specs import load_best_spec  # noqa: E402

ARTIFACTS = REPO_ROOT / "src" / "calibration" / "artifacts"
MERGED_DIR = REPO_ROOT / "data" / "tent_predictions" / "merged"
COVERAGE_DIR = REPO_ROOT / "data" / "tent_predictions" / "coverage"
MUNICIPALITY_PATH = REPO_ROOT / "data" / "gaza_boundaries" / "municipal" / "GazaStrip_MunicipalBoundaries.shp"
BLOCK_PATH = REPO_ROOT / "data" / "gaza_boundaries" / "dash_population_blocks.geojson"
YELLOW_LINE_PATH = REPO_ROOT / "data" / "spatial_context" / "yellow-line-zone.json"
LAND_USE_CACHE = REPO_ROOT / "outputs" / "tent_counts" / "grid_land_use_labels.csv"
SUMMARY_PATH = REPO_ROOT / "outputs" / "analysis" / "summary_statistics.json"
OUTPUT_PATH = REPO_ROOT / "outputs" / "analysis" / "posterior_draws_cri.json"
NORTH = {"Gaza", "Jabalya", "Beit Hanoun", "Beit Lahiya"}
EXCLUDED_WEEKS = {"20260518-20260524", "20260525-20260531", "20260601-20260607"}


def _quantiles(draws: np.ndarray) -> dict[str, float]:
    return {"mean": float(draws.mean()), "p2_5": float(np.quantile(draws, .025)), "p97_5": float(np.quantile(draws, .975))}


def _requested_grid(municipalities: gpd.GeoDataFrame, target_names: set[str]) -> gpd.GeoDataFrame:
    """Build just target cells, retaining the original global grid cell ids."""
    area = municipalities.geometry.union_all()
    minx, miny, maxx, maxy = area.bounds
    xs = np.arange(minx, maxx + 100., 100.)
    ys = np.arange(miny, maxy + 100., 100.)
    x0 = np.repeat(xs[:-1], len(ys) - 1)
    y0 = np.tile(ys[:-1], len(xs) - 1)
    centers = shapely.points(x0 + 50., y0 + 50.)
    inside = shapely.covers(area, centers)
    target_area = municipalities[municipalities["NAME"].isin(target_names)].geometry.union_all()
    keep = inside & shapely.covers(target_area, centers)
    ids = np.cumsum(inside, dtype=int) - 1
    grid = gpd.GeoDataFrame({"cell_id": ids[keep]}, geometry=shapely.box(x0[keep], y0[keep], x0[keep] + 100., y0[keep] + 100.), crs=TARGET_CRS)
    grid["cell_area_m2"] = 10000.
    grid["center_x"] = x0[keep] + 50.
    grid["center_y"] = y0[keep] + 50.
    grid["col"] = np.floor((grid.center_x - minx) / 100.).astype(int)
    grid["row"] = np.floor((grid.center_y - miny) / 100.).astype(int)
    grid.attrs.update(grid_minx=float(minx), grid_miny=float(miny), cell_size=100.)
    return grid


def _load_targets(summary: dict) -> tuple[dict[str, set[str]], dict[str, tuple[str, str]]]:
    """Return named area/week totals and named paired comparisons."""
    counts = pd.read_csv(REPO_ROOT / "outputs" / "tent_counts" / "municipality_counts_by_week.csv")
    all_munis = set(counts["Municipality"].dropna())
    # Event windows in the summary are ISO dates, whereas prediction assets use
    # YYYYMMDD-YYYYMMDD keys.
    date_to_key = {
        pd.to_datetime(key.split("-")[0], format="%Y%m%d").strftime("%Y-%m-%d"): key
        for key in counts["week_key"].dropna().unique()
    }
    regions: dict[str, set[str]] = {}
    comparisons: dict[str, tuple[str, str]] = {}
    largest = summary["largest_change"]
    name = "largest_sustained_change"
    regions[f"{name}:previous"] = {largest["municipality"]}
    regions[f"{name}:current"] = {largest["municipality"]}
    comparisons[name] = (f"{name}:previous", f"{name}:current")
    for event in summary["directional_trends_key_events"]:
        slug = event["event_date"]
        for side, municipalities in (("north", NORTH), ("south", all_munis - NORTH)):
            pre_name, post_name = f"{slug}:{side}:pre", f"{slug}:{side}:post"
            regions[pre_name] = municipalities
            regions[post_name] = municipalities
            comparisons[f"{slug}:{side}"] = (pre_name, post_name)
            # Store week lists on the entries themselves for unambiguous lookup.
            regions[pre_name] = (municipalities, tuple(date_to_key[d] for d in event["pre_weeks"]))  # type: ignore[assignment]
            regions[post_name] = (municipalities, tuple(date_to_key[d] for d in event["post_weeks"]))  # type: ignore[assignment]
    ceasefire = summary["north_gaza_ceasefire_counts"]
    start = ceasefire["start_snapshot"]["municipality_weeks_used"][0]["week_key"]
    end = ceasefire["end_snapshot"]["municipality_weeks_used"][0]["week_key"]
    regions["north_ceasefire:start"] = (NORTH, (start,))  # type: ignore[assignment]
    regions["north_ceasefire:end"] = (NORTH, (end,))  # type: ignore[assignment]
    comparisons["north_ceasefire"] = ("north_ceasefire:start", "north_ceasefire:end")
    regions[f"{name}:previous"] = ({largest["municipality"]}, (largest["from_week"],))  # type: ignore[assignment]
    regions[f"{name}:current"] = ({largest["municipality"]}, (largest["to_week"],))  # type: ignore[assignment]
    return regions, comparisons


def _target_draws(targets: dict[str, tuple[set[str], tuple[str, ...]]]) -> dict[tuple[str, str], np.ndarray]:
    spec = load_best_spec(ARTIFACTS / "selected_model.json")
    fit = az.from_netcdf(ARTIFACTS / "final_model_posterior.nc")
    # A deterministic, evenly spaced subset keeps this focused re-calculation
    # fast while still using actual posterior draws (not fitted marginals).
    posterior_draw_count = fit.posterior.sizes["draw"]
    draw_step = max(1, posterior_draw_count // 100)
    fit = fit.isel(draw=slice(0, None, draw_step))
    spec_payload = json.loads((ARTIFACTS / "selected_model.json").read_text())
    # The final posterior can use the final-run component columns rather than
    # the selection-stage columns stored at the top level of this artifact.
    final_run = spec_payload.get("final_run", {})
    if spec.model_kind == "hurdle_nb" and "beta_count" in fit.posterior and len(spec.count_columns or []) != fit.posterior["beta_count"].shape[-1]:
        spec.count_columns = final_run["count_columns"]
        spec.zero_columns = final_run["zero_columns"]
        spec.final_columns = list(dict.fromkeys(spec.count_columns + spec.zero_columns))
    scaler = json.loads((ARTIFACTS / "feature_scaler.json").read_text())
    knots = json.loads((ARTIFACTS / "spline_knots.json").read_text())
    municipal = _read_to_target(MUNICIPALITY_PATH)
    municipality_names = sorted(municipal["NAME"].dropna().astype(str).unique())
    requested = {(m, week) for municipalities, weeks in targets.values() for m in municipalities for week in weeks}
    requested_weeks = sorted({week for _, week in requested} - EXCLUDED_WEEKS)

    all_target_munis = {m for municipalities, _ in targets.values() for m in municipalities}
    grid = _requested_grid(municipal, all_target_munis)
    grid_attrs = dict(grid.attrs)
    centroids = gpd.GeoDataFrame(grid[["cell_id"]], geometry=gpd.points_from_xy(grid.center_x, grid.center_y), crs=TARGET_CRS)
    municipality_lookup = gpd.sjoin(centroids, municipal[["NAME", "geometry"]], how="left", predicate="within").set_index("cell_id")["NAME"]
    grid["municipality"] = grid.cell_id.map(municipality_lookup)
    grid["block"] = ""
    grid.attrs.update(grid_attrs)
    land_use = pd.read_csv(LAND_USE_CACHE)
    grid = grid.merge(land_use[["cell_id", "land_use_label"]], on="cell_id", how="left")
    grid.attrs.update(grid_attrs)
    grid["land_use_label"] = grid["land_use_label"].fillna("Unknown")
    grid_minx, grid_miny = grid.attrs["grid_minx"], grid.attrs["grid_miny"]
    cell_lookup = {(int(c), int(r)): int(i) for c, r, i in grid[["col", "row", "cell_id"]].itertuples(index=False, name=None)}
    total_area = grid.groupby("municipality")["cell_area_m2"].sum()
    yellow, yellow_windows = load_yellow_line_context(YELLOW_LINE_PATH)
    result: dict[tuple[str, str], np.ndarray] = {}

    for week in requested_weeks:
        print(f"Processing {week}", flush=True)
        needed_munis = {m for m, w in requested if w == week}
        coverage = _read_to_target(COVERAGE_DIR / f"{week}_coverage.geojson")
        centers = gpd.GeoDataFrame(grid[["cell_id"]], geometry=gpd.points_from_xy(grid.center_x, grid.center_y), crs=TARGET_CRS)
        covered_ids = gpd.sjoin(centers, coverage[["geometry"]], how="inner", predicate="within")["cell_id"].unique()
        cells = grid[grid.cell_id.isin(covered_ids) & grid.municipality.isin(needed_munis)].copy()
        if cells.empty:
            continue
        coverage_ratio = cells.groupby("municipality")["cell_area_m2"].sum() / total_area
        cells = cells[cells.municipality.map(coverage_ratio).ge(.8)].reset_index(drop=True)
        if cells.empty:
            continue
        points = _read_to_target(MERGED_DIR / f"{week}.gpkg")
        points, _ = apply_weekly_postprocessing(points, week, yellow, yellow_windows, eps_m=40.0, min_samples=3)
        point_counts = pd.Series(dtype=float)
        feature_maps: dict[str, dict[int, float]] = {}
        defaults: dict[str, float] = {}
        if not points.empty:
            xy = points.geometry
            mapped = pd.Series([cell_lookup.get((int(c), int(r))) for c, r in zip(np.floor((xy.x-grid_minx)/100).astype(int), np.floor((xy.y-grid_miny)/100).astype(int))])
            mapped = mapped[mapped.isin(set(cells.cell_id))].dropna().astype(int)
            point_counts = mapped.value_counts().astype(float)
            for col in ["satellite_elevation", "sun_elevation", "satellite_azimuth", "sun_azimuth", "hours_from_sunrise"]:
                if col in points:
                    vals = pd.to_numeric(points[col], errors="coerce")
                    # Mapping is recomputed without dropping invalid point positions.
                    raw_map = pd.Series([cell_lookup.get((int(c), int(r))) for c, r in zip(np.floor((xy.x-grid_minx)/100).astype(int), np.floor((xy.y-grid_miny)/100).astype(int))])
                    frame = pd.DataFrame({"cell_id": raw_map, col: vals}).dropna()
                    frame = frame[frame.cell_id.isin(set(cells.cell_id))]
                    if not frame.empty:
                        defaults[col] = float(frame[col].median())
                        feature_maps[col] = frame.groupby("cell_id")[col].median().to_dict()
        cells["raw_points"] = cells.cell_id.map(point_counts).fillna(0.0)
        for col, values in feature_maps.items():
            cells[col] = cells.cell_id.map(values).fillna(defaults[col])
        times = _derive_point_time_strings(points)
        if not times.empty:
            # The application code uses median time per cell; retain its exact convention.
            raw_map = pd.Series([cell_lookup.get((int(c), int(r))) for c, r in zip(np.floor((points.geometry.x-grid_minx)/100).astype(int), np.floor((points.geometry.y-grid_miny)/100).astype(int))])
            tf = pd.DataFrame({"cell_id": raw_map, "time": times}).dropna()
            tf = tf[tf.cell_id.isin(set(cells.cell_id))]
            parsed = pd.to_datetime(tf.time, format="%H:%M:%S", errors="coerce")
            if parsed.notna().any():
                seconds = parsed.dt.hour * 3600 + parsed.dt.minute * 60 + parsed.dt.second
                med = seconds.groupby(tf.cell_id).median()
                cells["point_time"] = cells.cell_id.map({int(k): f"{int(v//3600):02d}:{int((v%3600)//60):02d}:{int(v%60):02d}" for k, v in med.items()})
        cells["prediction_count"] = cells.raw_points
        cells["date"] = _week_date_from_key(week)
        cells["week"] = pd.Timestamp(cells.date.iloc[0]).strftime("%G-W%V")
        cells = _compute_week_context_features(cells, _week_date_from_key(week), REPO_ROOT / "data" / "spatial_context")
        count_cols = spec.count_columns if spec.model_kind == "hurdle_nb" and spec.count_columns else spec.final_columns
        zero_cols = spec.zero_columns if spec.model_kind == "hurdle_nb" and spec.zero_columns else count_cols
        x, muni_codes = _design_matrix_for_spec(cells, spec, municipality_names, _week_date_from_key(week), count_cols, None, scaler, knots)
        x0, _ = _design_matrix_for_spec(cells, spec, municipality_names, _week_date_from_key(week), zero_cols, None, scaler, knots)
        # Avoid materialising an all-cells × all-draws array; only running
        # municipality totals are needed for the requested regions.
        n_draws = fit.posterior.sizes["chain"] * fit.posterior.sizes["draw"]
        weekly_totals = {m: np.zeros(n_draws) for m in needed_munis}
        for start in range(0, len(cells), 500):
            stop = min(start + 500, len(cells))
            chunk = expected_counts(spec, fit, x[start:stop], muni_codes[start:stop], X_zero=x0[start:stop] if spec.model_kind == "hurdle_nb" else None)
            for municipality in needed_munis:
                mask = cells.municipality.iloc[start:stop].eq(municipality).to_numpy()
                if mask.any():
                    weekly_totals[municipality] += np.nan_to_num(chunk[mask], nan=0., posinf=1e6, neginf=0.).sum(axis=0)
        result.update({(municipality, week): total for municipality, total in weekly_totals.items()})
    return result


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    targets, comparisons = _load_targets(summary)
    draws = _target_draws(targets)  # Posterior draw index is shared across every requested total.
    totals = {}
    for name, (municipalities, weeks) in targets.items():
        per_week = [sum(draws[(m, week)] for m in municipalities if (m, week) in draws) for week in weeks]
        if len(per_week) != len(weeks):
            raise ValueError(f"Missing posterior totals for {name}")
        totals[name] = np.mean(per_week, axis=0)
    output = {"method": "95% CrIs are quantiles of draw-by-draw posterior totals and changes; no marginal bounds are subtracted.", "totals": {name: _quantiles(value) for name, value in totals.items()}, "changes": {}}
    for name, (before, after) in comparisons.items():
        change = totals[after] - totals[before]
        output["changes"][name] = {"from": before, "to": after, "absolute_change": _quantiles(change), "percent_change": _quantiles(change / np.maximum(totals[before], 1e-9) * 100)}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
