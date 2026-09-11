from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
import pickle
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, box
from tqdm.auto import tqdm

from .publication import expected_counts
from .postprocessing import apply_weekly_postprocessing, load_yellow_line_context
from .types import ModelSpec

TARGET_CRS = "EPSG:32636"
GAZA_LATITUDE = 31.5
GAZA_LONGITUDE = 34.47
GAZA_TIMEZONE = "Asia/Gaza"


def parse_date_value(value) -> pd.Timestamp:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def _resolve_week_anchor_date(week_date_value) -> pd.Timestamp:
    """Resolve a usable anchor date from week keys like YYYYMMDD-YYYYMMDD or ordinary date strings."""
    dt = parse_date_value(week_date_value)
    if pd.notna(dt):
        return dt
    text = str(week_date_value)
    if len(text) >= 8 and text[:8].isdigit():
        start = pd.to_datetime(text[:8], format="%Y%m%d", errors="coerce")
        if pd.notna(start):
            return start
    return pd.NaT


def _parse_time_of_day(value) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    parsed = pd.to_datetime(str(value), format="%H:%M:%S", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def _extract_hhmmss_from_text(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    # Prefer YYYYMMDD_HHMMSS patterns commonly used in tif names.
    match = re.search(r"\d{8}_(\d{6})", text)
    if match:
        hhmmss = match.group(1)
        return f"{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
    # Fallback to any standalone 6-digit time token.
    match = re.search(r"(?:^|[^0-9])(\d{6})(?:[^0-9]|$)", text)
    if match:
        hhmmss = match.group(1)
        hh = int(hhmmss[0:2])
        mm = int(hhmmss[2:4])
        ss = int(hhmmss[4:6])
        if 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59:
            return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return None


def _derive_point_time_strings(points: gpd.GeoDataFrame) -> pd.Series:
    """Return per-point HH:MM:SS strings using explicit time or filename-derived fallback."""
    n = len(points)
    if n == 0:
        return pd.Series(dtype="object")

    out = pd.Series([None] * n, index=points.index, dtype="object")
    if "time" in points.columns:
        parsed = pd.to_datetime(points["time"].astype(str), format="%H:%M:%S", errors="coerce")
        mask = parsed.notna()
        if mask.any():
            out.loc[mask] = parsed.loc[mask].dt.strftime("%H:%M:%S")

    fallback_cols = [
        "source_tif",
        "source_file",
        "filename",
        "image_name",
        "scene_name",
        "name",
    ]
    missing_mask = out.isna()
    for col in fallback_cols:
        if not missing_mask.any():
            break
        if col not in points.columns:
            continue
        extracted = points.loc[missing_mask, col].apply(_extract_hhmmss_from_text)
        fill_mask = extracted.notna()
        if fill_mask.any():
            out.loc[extracted.index[fill_mask]] = extracted.loc[fill_mask]
            missing_mask = out.isna()

    return out


def _hours_from_sunrise(date_value, time_of_day) -> float:
    parsed_date = pd.to_datetime(date_value, errors="coerce")
    parsed_time = _parse_time_of_day(time_of_day)
    if pd.isna(parsed_date) or parsed_time is None:
        return np.nan

    try:
        from astral import LocationInfo
        from astral.sun import sun
    except ImportError as exc:
        raise ImportError("astral is required to compute hours_from_sunrise during application") from exc

    location = LocationInfo(
        name="Gaza Strip",
        region="PS",
        timezone=GAZA_TIMEZONE,
        latitude=GAZA_LATITUDE,
        longitude=GAZA_LONGITUDE,
    )
    tz = ZoneInfo(GAZA_TIMEZONE)
    sun_times = sun(location.observer, date=parsed_date.date(), tzinfo=tz)
    sunrise = sun_times["sunrise"]
    observation_dt = pd.Timestamp(
        year=parsed_date.year,
        month=parsed_date.month,
        day=parsed_date.day,
        hour=parsed_time.hour,
        minute=parsed_time.minute,
        second=parsed_time.second,
        tz=tz,
    )
    return float((observation_dt - sunrise).total_seconds() / 3600.0)


@dataclass
class ApplicationConfig:
    merged_dir: Path
    coverage_dir: Path
    municipality_path: Path
    block_path: Path
    spatial_context_dir: Path
    output_dir: Path
    apply_municipality: bool = True
    apply_block: bool = True
    resume: bool = True
    municipality_coverage_threshold: float = 0.8
    predictor_bounds: Dict[str, Tuple[float, float]] | None = None
    scaler_params: Dict[str, Dict[str, float]] | None = None
    spline_knots: Dict[str, List[float]] | None = None
    prewar_tif_path: Path | None = None
    land_use_model_path: Path | None = None
    land_use_labels_path: Path | None = None
    yellow_line_path: Path | None = None
    postprocess_eps_m: float = 40.0
    postprocess_min_samples: int = 3


def _week_key_from_merged_name(path: Path) -> str:
    return path.stem


def _week_key_from_coverage_name(path: Path) -> str:
    return path.name.replace("_coverage.geojson", "")


def _week_label_from_key(key: str) -> str:
    start = key.split("-")[0]
    dt = pd.to_datetime(start, format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        return ""
    iso = dt.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"


def _week_date_from_key(key: str) -> str:
    start = key.split("-")[0]
    dt = pd.to_datetime(start, format="%Y%m%d", errors="coerce")
    return "" if pd.isna(dt) else dt.strftime("%Y-%m-%d")


def _read_to_target(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(TARGET_CRS)


def _build_fixed_grid(study_area: gpd.GeoDataFrame, cell_size: float = 100.0) -> gpd.GeoDataFrame:
    if study_area.empty:
        raise ValueError("Study-area geometry is empty")
    union_geom = study_area.geometry.union_all()
    if union_geom is None or union_geom.is_empty:
        raise ValueError("No valid study-area geometry found to build fixed grid")

    minx, miny, maxx, maxy = union_geom.bounds
    xs = np.arange(minx, maxx + cell_size, cell_size)
    ys = np.arange(miny, maxy + cell_size, cell_size)

    cells = []
    ids = []
    idx = 0
    for ix, x0 in enumerate(xs[:-1]):
        for iy, y0 in enumerate(ys[:-1]):
            geom = box(x0, y0, x0 + cell_size, y0 + cell_size)
            center = Point(x0 + (cell_size / 2.0), y0 + (cell_size / 2.0))
            if union_geom.covers(center):
                cells.append(geom)
                ids.append(idx)
                idx += 1

    grid = gpd.GeoDataFrame({"cell_id": ids}, geometry=cells, crs=TARGET_CRS)
    grid["cell_area_m2"] = grid.geometry.area
    cent = grid.geometry.centroid
    grid["center_x"] = cent.x
    grid["center_y"] = cent.y
    grid["col"] = np.floor((grid["center_x"] - minx) / cell_size).astype(int)
    grid["row"] = np.floor((grid["center_y"] - miny) / cell_size).astype(int)
    grid.attrs["grid_minx"] = float(minx)
    grid.attrs["grid_miny"] = float(miny)
    grid.attrs["cell_size"] = float(cell_size)
    return grid


def _attach_admin_units(grid: gpd.GeoDataFrame, municipality_path: Path, block_path: Path) -> gpd.GeoDataFrame:
    out = grid.copy()
    muni = _read_to_target(municipality_path)
    block = _read_to_target(block_path)

    pts = gpd.GeoDataFrame(
        out[["cell_id"]].copy(),
        geometry=gpd.points_from_xy(out["center_x"], out["center_y"]),
        crs=TARGET_CRS,
    )

    if not muni.empty:
        muni_col = "NAME" if "NAME" in muni.columns else muni.columns[0]
        muni_source = muni[[muni_col, "geometry"]].copy()
        muni_source = muni_source[muni_source[muni_col].notna() & (muni_source[muni_col].astype(str).str.strip() != "")]
        if muni_source.empty:
            muni_source = muni[[muni_col, "geometry"]]
        nearest = gpd.sjoin_nearest(pts, muni_source, how="left")
        muni_map = nearest.groupby("cell_id", dropna=False)[muni_col].first()
        out["municipality"] = out["cell_id"].map(muni_map).fillna("UNKNOWN").astype(str).values
    else:
        out["municipality"] = "UNKNOWN"

    if not block.empty:
        block_col = "Name" if "Name" in block.columns else block.columns[0]
        joined = gpd.sjoin(pts, block[[block_col, "geometry"]], how="left", predicate="within")
        block_map = joined.groupby("cell_id", dropna=False)[block_col].first()
        out["block"] = out["cell_id"].map(block_map).fillna("UNKNOWN").astype(str).values
        missing = out["block"].eq("UNKNOWN")
        if missing.any():
            nearest = gpd.sjoin_nearest(pts[missing], block[[block_col, "geometry"]], how="left")
            near_map = nearest.groupby("cell_id", dropna=False)[block_col].first()
            out.loc[missing, "block"] = out.loc[missing, "cell_id"].map(near_map).fillna("UNKNOWN").astype(str).values
    else:
        out["block"] = "UNKNOWN"

    return out


def _load_spatial_context_layer(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    gdf = gpd.read_file(path)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(TARGET_CRS)


def _filter_by_date_start(gdf: gpd.GeoDataFrame, date_str: str) -> gpd.GeoDataFrame:
    if gdf.empty or "date_start" not in gdf.columns:
        return gdf
    dt = parse_date_value(date_str)
    if dt is None:
        return gdf
    out = gdf.copy()
    out["date_start_parsed"] = out["date_start"].apply(parse_date_value)
    return out[out["date_start_parsed"].notna() & (out["date_start_parsed"] >= dt)]


def _compute_week_context_features(cells_week: gpd.GeoDataFrame, week_date: str, spatial_context_dir: Path) -> gpd.GeoDataFrame:
    out = cells_week.copy()
    out["distance_to_nearest_base"] = np.nan
    out["distance_to_nearest_outpost"] = np.nan
    out["distance_to_yellow_line"] = np.nan
    out["is_humanitarian_zone"] = False
    out["is_agriculture"] = False

    centers = gpd.GeoSeries(
        gpd.points_from_xy(out["center_x"], out["center_y"]),
        index=out.index,
        crs=TARGET_CRS,
    )

    week_anchor = _resolve_week_anchor_date(week_date)

    bases = _load_spatial_context_layer(spatial_context_dir / "bases.json")
    bases = _filter_by_date_start(bases, week_anchor)
    if not bases.empty:
        base_union = bases.geometry.union_all()
        if base_union is not None and not base_union.is_empty:
            out["distance_to_nearest_base"] = centers.distance(base_union).astype(float)

    outposts = _load_spatial_context_layer(spatial_context_dir / "outposts.json")
    if not outposts.empty:
        dt = week_anchor
        filtered_outposts = outposts.copy()
        if "date_start" in filtered_outposts.columns:
            filtered_outposts["date_start_parsed"] = filtered_outposts["date_start"].apply(parse_date_value)
            filtered_outposts = filtered_outposts[
                filtered_outposts["date_start_parsed"].notna()
                & (filtered_outposts["date_start_parsed"] <= dt)
            ]
        if "date_end" in filtered_outposts.columns:
            filtered_outposts["date_end_parsed"] = filtered_outposts["date_end"].apply(parse_date_value)
            filtered_outposts = filtered_outposts[
                filtered_outposts["date_end_parsed"].isna()
                | (filtered_outposts["date_end_parsed"] >= dt)
            ]
        outpost_source = filtered_outposts if not filtered_outposts.empty else outposts
        outpost_union = outpost_source.geometry.union_all() if not outpost_source.empty else None
        if outpost_union is not None and not outpost_union.is_empty:
            out["distance_to_nearest_outpost"] = centers.distance(outpost_union).astype(float)

    yellow = _load_spatial_context_layer(spatial_context_dir / "yellow-line-zone.json")
    if not yellow.empty:
        dt = week_anchor
        filtered_yellow = yellow.copy()
        if "date_start" in filtered_yellow.columns:
            filtered_yellow["date_start_parsed"] = filtered_yellow["date_start"].apply(parse_date_value)
            filtered_yellow = filtered_yellow[
                filtered_yellow["date_start_parsed"].notna()
                & (filtered_yellow["date_start_parsed"] <= dt)
            ]
        if "date_end" in filtered_yellow.columns:
            filtered_yellow["date_end_parsed"] = filtered_yellow["date_end"].apply(parse_date_value)
            filtered_yellow = filtered_yellow[
                filtered_yellow["date_end_parsed"].isna()
                | (filtered_yellow["date_end_parsed"] >= dt)
            ]
        yellow_source = filtered_yellow if not filtered_yellow.empty else yellow
        yellow_union = yellow_source.geometry.union_all() if not yellow_source.empty else None
        if yellow_union is not None and not yellow_union.is_empty:
            out["distance_to_yellow_line"] = centers.distance(yellow_union).astype(float)

    agriculture = _load_spatial_context_layer(spatial_context_dir / "agriculture.json")
    if not agriculture.empty:
        centers_df = gpd.GeoDataFrame(
            out[["cell_id"]].copy(),
            geometry=gpd.points_from_xy(out["center_x"], out["center_y"]),
            crs=TARGET_CRS,
        )
        ag_join = gpd.sjoin(centers_df, agriculture[["geometry"]], how="left", predicate="within")
        out["is_agriculture"] = out["cell_id"].isin(ag_join[ag_join["index_right"].notna()]["cell_id"]).astype(bool)

    humanitarian = _load_spatial_context_layer(spatial_context_dir / "humanitarian_zone.json")
    if not humanitarian.empty:
        filtered = humanitarian.copy()
        if {"safe_date", "end_date"}.issubset(filtered.columns):
            dt = week_anchor
            filtered["safe_date_parsed"] = filtered["safe_date"].apply(parse_date_value)
            filtered["end_date_parsed"] = filtered["end_date"].apply(parse_date_value)
            filtered = filtered[
                filtered["safe_date_parsed"].notna()
                & filtered["end_date_parsed"].notna()
                & (filtered["safe_date_parsed"] <= dt)
                & (filtered["end_date_parsed"] >= dt)
            ]

        if not filtered.empty:
            centers_df = gpd.GeoDataFrame(
                out[["cell_id"]].copy(),
                geometry=gpd.points_from_xy(out["center_x"], out["center_y"]),
                crs=TARGET_CRS,
            )
            joined = gpd.sjoin(centers_df, filtered[["geometry"]], how="left", predicate="within")
            out["is_humanitarian_zone"] = out["cell_id"].isin(joined[joined["index_right"].notna()]["cell_id"]).astype(bool)

    return out


def _load_land_use_label_map(labels_path: Path | None) -> Dict[int, str]:
    if labels_path is None or not labels_path.exists():
        return {}
    try:
        payload = json.loads(labels_path.read_text())
    except Exception:
        return {}
    out: Dict[int, str] = {}
    if isinstance(payload, dict):
        for k, v in payload.items():
            try:
                idx = int(k)
            except Exception:
                continue
            label = str(v).strip() if v is not None else ""
            if label:
                out[idx] = label
    return out


def _classify_grid_land_use(
    grid: gpd.GeoDataFrame,
    prewar_tif_path: Path,
    land_use_model_path: Path,
    land_use_labels_path: Path | None,
    cache_path: Path,
    sample_stride: int = 4,
) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if {"cell_id", "land_use_label"}.issubset(cached.columns):
            return cached[["cell_id", "land_use_label"]]

    if not prewar_tif_path.exists():
        raise FileNotFoundError(f"Prewar tif not found: {prewar_tif_path}")
    if not land_use_model_path.exists():
        raise FileNotFoundError(f"Land-use model not found: {land_use_model_path}")

    with open(land_use_model_path, "rb") as f:
        model = pickle.load(f)

    label_map = _load_land_use_label_map(land_use_labels_path)
    labels: List[str] = []

    with rasterio.open(prewar_tif_path) as src:
        if src.crs is None:
            raise ValueError("prewar tif has no CRS")
        grid_for_read = grid.to_crs(src.crs) if str(grid.crs) != str(src.crs) else grid

        bar = tqdm(total=len(grid_for_read), desc="Classifying prewar land-use", unit="cell", leave=False)
        for geom in grid_for_read.geometry:
            minx, miny, maxx, maxy = geom.bounds
            row_min, col_min = src.index(minx, maxy)
            row_max, col_max = src.index(maxx, miny)
            r0, r1 = sorted((int(row_min), int(row_max)))
            c0, c1 = sorted((int(col_min), int(col_max)))
            r0 = max(0, min(r0, src.height - 1))
            r1 = max(0, min(r1, src.height - 1))
            c0 = max(0, min(c0, src.width - 1))
            c1 = max(0, min(c1, src.width - 1))
            h = max(1, r1 - r0 + 1)
            w = max(1, c1 - c0 + 1)

            window = rasterio.windows.Window(col_off=c0, row_off=r0, width=w, height=h)
            arr = src.read([1, 2, 3], window=window)
            if arr.size == 0:
                labels.append("Unknown")
                bar.update(1)
                continue
            arr = arr[:, ::sample_stride, ::sample_stride]
            pix = np.transpose(arr, (1, 2, 0)).reshape(-1, 3).astype(np.float32)
            if pix.size == 0:
                labels.append("Unknown")
                bar.update(1)
                continue
            if np.nanmax(pix) <= 1.0:
                pix = pix * 255.0
            pred = model.predict(pix)
            if pred.size == 0:
                labels.append("Unknown")
            else:
                major = int(np.bincount(pred.astype(int)).argmax())
                labels.append(label_map.get(major, f"cluster_{major}"))
            bar.update(1)
        bar.close()

    out = pd.DataFrame({"cell_id": grid["cell_id"].to_numpy(dtype=int), "land_use_label": labels})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


def _summarize_aggregated_draws(
    grouped_df: pd.DataFrame,
    grouped_draws: np.ndarray,
) -> pd.DataFrame:
    out = grouped_df.copy()
    out["corrected_count_median"] = np.median(grouped_draws, axis=1)
    out["corrected_count_mean"] = np.mean(grouped_draws, axis=1)
    out["corrected_count_p2_5"] = np.quantile(grouped_draws, 0.025, axis=1)
    out["corrected_count_p97_5"] = np.quantile(grouped_draws, 0.975, axis=1)
    area_km2 = out["covered_area_m2"].to_numpy(dtype=float) / 1_000_000.0
    density_draws = grouped_draws / area_km2[:, None]
    out["density_per_km2_median"] = np.median(density_draws, axis=1)
    out["density_per_km2_p2_5"] = np.quantile(density_draws, 0.025, axis=1)
    out["density_per_km2_p97_5"] = np.quantile(density_draws, 0.975, axis=1)
    out["corrected_count_95_cri"] = out["corrected_count_p2_5"].round(0).astype(int).astype(str) + "-" + out["corrected_count_p97_5"].round(0).astype(int).astype(str)
    return out


def _completed_week_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=lambda c: c in {"week_key", "Week"})
    if "week_key" in df.columns:
        return set(df["week_key"].dropna().astype(str).unique().tolist())
    if "Week" in df.columns:
        return set(df["Week"].dropna().astype(str).unique().tolist())
    return set()


def _append_rows(csv_path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        existing = pd.read_csv(csv_path, usecols=lambda c: c in {"week_key", "Municipality", "Week", "block"})
        if "week_key" in existing.columns:
            done = set(existing["week_key"].dropna().astype(str).unique().tolist())
            df = df[~df["week_key"].astype(str).isin(done)].copy()
    if df.empty:
        return
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def _design_matrix_for_spec(
    week_cells: pd.DataFrame,
    spec: ModelSpec,
    municipality_names: List[str],
    week_date: str,
    columns: List[str] | None = None,
    predictor_bounds: Dict[str, Tuple[float, float]] | None = None,
    scaler_params: Dict[str, Dict[str, float]] | None = None,
    spline_knots: Dict[str, List[float]] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    selected_columns = list(columns if columns is not None else spec.final_columns)
    features = pd.DataFrame(index=week_cells.index)
    dt = _resolve_week_anchor_date(week_date)
    missing_requirements: list[str] = []

    if "log1p_model_tent_count" in selected_columns:
        features["log1p_model_tent_count"] = np.log1p(np.clip(pd.to_numeric(week_cells["raw_points"], errors="coerce").fillna(0.0), 0.0, None))
    if "year" in selected_columns:
        year_value = float(dt.year) if pd.notna(dt) else np.nan
        if predictor_bounds and "year" in predictor_bounds and pd.notna(year_value):
            lo, hi = predictor_bounds["year"]
            year_value = float(np.clip(year_value, lo, hi))
        features["year"] = year_value
    if "month_sin" in selected_columns or "month_cos" in selected_columns:
        month_num = float(dt.month) if pd.notna(dt) else np.nan
        angle = 2.0 * np.pi * (month_num / 12.0) if pd.notna(month_num) else np.nan
        if "month_sin" in selected_columns:
            features["month_sin"] = np.sin(angle) if pd.notna(angle) else np.nan
        if "month_cos" in selected_columns:
            features["month_cos"] = np.cos(angle) if pd.notna(angle) else np.nan
    if "log1p_distance_to_nearest_base" in selected_columns:
        features["log1p_distance_to_nearest_base"] = np.log1p(np.clip(pd.to_numeric(week_cells["distance_to_nearest_base"], errors="coerce"), 0.0, None))
    if "log1p_distance_to_nearest_outpost" in selected_columns:
        if "distance_to_nearest_outpost" not in week_cells.columns:
            missing_requirements.append("distance_to_nearest_outpost")
        else:
            features["log1p_distance_to_nearest_outpost"] = np.log1p(np.clip(pd.to_numeric(week_cells["distance_to_nearest_outpost"], errors="coerce"), 0.0, None))
    if "log1p_distance_to_yellow_line" in selected_columns:
        if "distance_to_yellow_line" not in week_cells.columns:
            missing_requirements.append("distance_to_yellow_line")
        else:
            features["log1p_distance_to_yellow_line"] = np.log1p(np.clip(pd.to_numeric(week_cells["distance_to_yellow_line"], errors="coerce"), 0.0, None))

    if "is_humanitarian_zone" in selected_columns:
        features["is_humanitarian_zone"] = week_cells["is_humanitarian_zone"].astype(bool).astype(int)

    for bool_col in ["is_agriculture", "is_destruction", "has_greenhouse"]:
        if bool_col in selected_columns:
            if bool_col not in week_cells.columns:
                missing_requirements.append(bool_col)
            else:
                features[bool_col] = week_cells[bool_col].astype(bool).astype(int)

    for pass_col in ["satellite_elevation", "sun_elevation"]:
        if pass_col in selected_columns:
            if pass_col not in week_cells.columns:
                missing_requirements.append(pass_col)
            else:
                features[pass_col] = pd.to_numeric(week_cells[pass_col], errors="coerce")

    if any(c in selected_columns for c in ["sat_az_sin", "sat_az_cos"]):
        if "satellite_azimuth" not in week_cells.columns:
            missing_requirements.append("satellite_azimuth")
        else:
            sat = pd.to_numeric(week_cells["satellite_azimuth"], errors="coerce")
            ang = 2.0 * np.pi * (sat % 360.0) / 360.0
            if "sat_az_sin" in selected_columns:
                features["sat_az_sin"] = np.sin(ang)
            if "sat_az_cos" in selected_columns:
                features["sat_az_cos"] = np.cos(ang)

    if any(c in selected_columns for c in ["sun_az_sin", "sun_az_cos"]):
        if "sun_azimuth" not in week_cells.columns:
            missing_requirements.append("sun_azimuth")
        else:
            sun = pd.to_numeric(week_cells["sun_azimuth"], errors="coerce")
            ang = 2.0 * np.pi * (sun % 360.0) / 360.0
            if "sun_az_sin" in selected_columns:
                features["sun_az_sin"] = np.sin(ang)
            if "sun_az_cos" in selected_columns:
                features["sun_az_cos"] = np.cos(ang)

    if "log1p_local_pred_density" in selected_columns:
        if "local_pred_density" in week_cells.columns:
            features["log1p_local_pred_density"] = np.log1p(np.clip(pd.to_numeric(week_cells["local_pred_density"], errors="coerce"), 0.0, None))
        elif "log1p_local_pred_density" in week_cells.columns:
            features["log1p_local_pred_density"] = pd.to_numeric(week_cells["log1p_local_pred_density"], errors="coerce")
        else:
            missing_requirements.append("local_pred_density")

    land_cols = [c for c in selected_columns if c.startswith("land_use_")]
    if land_cols:
        if "land_use_label" not in week_cells.columns:
            missing_requirements.append("land_use_label")
        else:
            labels = week_cells["land_use_label"].fillna("nan").astype(str)
            for c in land_cols:
                cat = c.replace("land_use_", "")
                features[c] = (labels == cat).astype(int)

    if any(c.startswith("hours_from_sunrise_cand_s") for c in selected_columns):
        need_hours_fallback = (
            "hours_from_sunrise" not in week_cells.columns
            or pd.to_numeric(week_cells["hours_from_sunrise"], errors="coerce").isna().all()
        )
        if need_hours_fallback and "point_time" in week_cells.columns:
            week_cells = week_cells.copy()
            week_cells["hours_from_sunrise"] = [
                _hours_from_sunrise(dt, t) for t in week_cells["point_time"].tolist()
            ]

        if "hours_from_sunrise" not in week_cells.columns:
            missing_requirements.append("hours_from_sunrise")
        elif not spline_knots or "hours_from_sunrise" not in spline_knots:
            missing_requirements.append("spline_knots.hours_from_sunrise")
        else:
            x = pd.to_numeric(week_cells["hours_from_sunrise"], errors="coerce")
            if x.notna().any():
                x = x.fillna(float(x.median()))
            for j, knot in enumerate(spline_knots["hours_from_sunrise"]):
                col = f"hours_from_sunrise_cand_s{j}"
                if col in selected_columns:
                    features[col] = np.maximum(x - float(knot), 0.0)

    if any(c.startswith("building_density_cand_s") for c in selected_columns):
        if "building_density" not in week_cells.columns:
            missing_requirements.append("building_density")
        elif not spline_knots or "building_density" not in spline_knots:
            missing_requirements.append("spline_knots.building_density")
        else:
            x = pd.to_numeric(week_cells["building_density"], errors="coerce")
            for j, knot in enumerate(spline_knots["building_density"]):
                col = f"building_density_cand_s{j}"
                if col in selected_columns:
                    features[col] = np.maximum(x - float(knot), 0.0)

    for col in selected_columns:
        if col in features.columns:
            continue
        if col in week_cells.columns:
            features[col] = pd.to_numeric(week_cells[col], errors="coerce")

    unresolved = [c for c in selected_columns if c not in features.columns]
    if unresolved:
        missing_requirements.extend(unresolved)
    if missing_requirements:
        unique_missing = sorted(set(missing_requirements))
        raise ValueError(
            "Cannot build required selected predictors for full-application design matrix. "
            f"Missing/unsupported inputs: {unique_missing}. "
            "Re-run model selection with deployable features or add these covariates to merged predictions/context layers."
        )

    features = features[selected_columns].apply(pd.to_numeric, errors="coerce")
    if predictor_bounds:
        for col, (lo, hi) in predictor_bounds.items():
            if col in features.columns:
                features[col] = features[col].clip(lower=lo, upper=hi)
    if scaler_params:
        for col, cfg in scaler_params.items():
            if col in features.columns:
                mean = float(cfg.get("mean", 0.0))
                std = float(cfg.get("std", 1.0))
                if np.isfinite(std) and std > 1e-12:
                    features[col] = (features[col] - mean) / std

    nan_cols = [col for col in selected_columns if col in features.columns and features[col].isna().any()]
    if nan_cols:
        missing_detail = {
            col: int(features[col].isna().sum())
            for col in nan_cols
        }
        raise ValueError(
            "Cannot compute required selected predictors without missing values for full-application design matrix. "
            f"NaN counts by feature: {missing_detail}."
        )

    muni_map = {name: i for i, name in enumerate(municipality_names)}
    muni_codes = week_cells["municipality"].astype(str).map(muni_map).fillna(0).to_numpy(dtype=int)
    return features.to_numpy(dtype=float), muni_codes


def apply_calibration_to_full_predictions(
    spec: ModelSpec,
    fit,
    municipality_names: List[str],
    config: ApplicationConfig,
) -> Dict[str, int]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    merged_paths = sorted(config.merged_dir.glob("*.gpkg"))
    coverage_paths = sorted(config.coverage_dir.glob("*_coverage.geojson"))
    coverage_map = {_week_key_from_coverage_name(p): p for p in coverage_paths}
    merged_keys = {_week_key_from_merged_name(p) for p in merged_paths}
    coverage_keys = set(coverage_map.keys())
    common_keys = sorted(merged_keys & coverage_keys)
    yellow, yellow_windows = load_yellow_line_context(config.yellow_line_path) if config.yellow_line_path else (gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS), [])

    study_area = _read_to_target(config.municipality_path)
    grid = _build_fixed_grid(study_area=study_area, cell_size=100.0)
    grid = _attach_admin_units(
        grid=grid,
        municipality_path=config.municipality_path,
        block_path=config.block_path,
    )

    if config.prewar_tif_path and config.land_use_model_path:
        saved_attrs = dict(grid.attrs)
        land_use_cache = config.output_dir / "grid_land_use_labels.csv"
        land_use_df = _classify_grid_land_use(
            grid=grid,
            prewar_tif_path=config.prewar_tif_path,
            land_use_model_path=config.land_use_model_path,
            land_use_labels_path=config.land_use_labels_path,
            cache_path=land_use_cache,
        )
        grid = grid.merge(land_use_df, on="cell_id", how="left")
        grid.attrs.update(saved_attrs)
    else:
        grid["land_use_label"] = "Unknown"
    grid_minx = float(grid.attrs["grid_minx"])
    grid_miny = float(grid.attrs["grid_miny"])
    cell_size = float(grid.attrs["cell_size"])
    cell_lookup = {(int(c), int(r)): int(cid) for c, r, cid in grid[["col", "row", "cell_id"]].itertuples(index=False, name=None)}
    centers_points = gpd.GeoDataFrame(grid[["cell_id"]].copy(), geometry=gpd.points_from_xy(grid["center_x"], grid["center_y"]), crs=TARGET_CRS)
    municipality_total_area = grid.groupby("municipality", dropna=False)["cell_area_m2"].sum()

    municipal_out = config.output_dir / "municipality_counts_by_week.csv"
    block_out = config.output_dir / "block_counts_by_week.csv"
    if not config.resume:
        if config.apply_municipality and municipal_out.exists():
            municipal_out.unlink()
        if config.apply_block and block_out.exists():
            block_out.unlink()

    completed_muni = _completed_week_keys(municipal_out) if config.apply_municipality and config.resume else set()
    todo_keys = [k for k in common_keys if k not in completed_muni] if config.apply_municipality else common_keys

    municipal_bar = tqdm(total=len(todo_keys), desc="Municipal corrections", unit="week") if config.apply_municipality else None
    block_bar = tqdm(total=len(todo_keys), desc="Block corrections", unit="week") if config.apply_block else None

    for key in todo_keys:
        merged_path = config.merged_dir / f"{key}.gpkg"
        coverage_path = coverage_map[key]

        pts = _read_to_target(merged_path)
        pts, _ = apply_weekly_postprocessing(
            points=pts,
            week_key=key,
            yellow=yellow,
            yellow_windows=yellow_windows,
            eps_m=config.postprocess_eps_m,
            min_samples=config.postprocess_min_samples,
        )
        cov = _read_to_target(coverage_path)
        if cov.empty:
            continue

        covered_join = gpd.sjoin(centers_points, cov[["geometry"]], how="inner", predicate="within")
        covered_ids = covered_join["cell_id"].drop_duplicates()
        keep_cols = ["cell_id", "cell_area_m2", "center_x", "center_y", "municipality", "block", "geometry"]
        if "land_use_label" in grid.columns:
            keep_cols.append("land_use_label")
        covered_cells = grid[grid["cell_id"].isin(covered_ids)][keep_cols].reset_index(drop=True)
        covered_area_by_muni = covered_cells.groupby("municipality", dropna=False)["cell_area_m2"].sum()

        point_counts = pd.Series(dtype=float)
        point_time_map: dict[int, str] = {}
        point_feature_maps: dict[str, dict[int, float]] = {}
        point_feature_defaults: dict[str, float] = {}
        if not pts.empty:
            pxy = pts.geometry
            cols = np.floor((pxy.x.to_numpy() - grid_minx) / cell_size).astype(int)
            rows = np.floor((pxy.y.to_numpy() - grid_miny) / cell_size).astype(int)
            mapped = [cell_lookup.get((int(c), int(r))) for c, r in zip(cols, rows)]
            point_cell_ids = pd.Series(mapped).dropna().astype(int)
            covered_cell_set = set(covered_cells["cell_id"].tolist())
            if not point_cell_ids.empty:
                point_cell_ids = point_cell_ids[point_cell_ids.isin(covered_cell_set)]
                point_counts = point_cell_ids.value_counts().astype(float)

            pt_frame_base = pd.DataFrame({"cell_id": pd.Series(mapped)})
            pt_frame_base = pt_frame_base.dropna(subset=["cell_id"])
            if not pt_frame_base.empty:
                pt_frame_base["cell_id"] = pt_frame_base["cell_id"].astype(int)
                pt_frame_base = pt_frame_base[pt_frame_base["cell_id"].isin(covered_cell_set)]

            for col in ["satellite_elevation", "sun_elevation", "satellite_azimuth", "sun_azimuth", "hours_from_sunrise"]:
                if col not in pts.columns or pt_frame_base.empty:
                    continue
                tmp = pt_frame_base.copy()
                tmp[col] = pd.to_numeric(pts[col], errors="coerce")
                tmp = tmp.dropna(subset=[col])
                if tmp.empty:
                    continue
                point_feature_defaults[col] = float(tmp[col].median())
                med = tmp.groupby("cell_id", dropna=False)[col].median()
                point_feature_maps[col] = med.to_dict()

            point_times = _derive_point_time_strings(pts)
            if not point_times.empty:
                pt_frame = pd.DataFrame({
                    "cell_id": pd.Series(mapped),
                    "time": point_times.tolist(),
                }).dropna(subset=["cell_id"])
                pt_frame["cell_id"] = pt_frame["cell_id"].astype(int)
                pt_frame = pt_frame[pt_frame["cell_id"].isin(covered_cell_set)]
                parsed = pd.to_datetime(pt_frame["time"], format="%H:%M:%S", errors="coerce")
                pt_frame = pt_frame.loc[parsed.notna()].copy()
                pt_frame["seconds"] = parsed[parsed.notna()].dt.hour * 3600 + parsed[parsed.notna()].dt.minute * 60 + parsed[parsed.notna()].dt.second
                if not pt_frame.empty:
                    med_sec = pt_frame.groupby("cell_id", dropna=False)["seconds"].median()
                    hh = (med_sec // 3600).astype(int)
                    mm = ((med_sec % 3600) // 60).astype(int)
                    ss = (med_sec % 60).astype(int)
                    point_time_map = {
                        int(cid): f"{h:02d}:{m:02d}:{s:02d}"
                        for cid, h, m, s in zip(med_sec.index.tolist(), hh.tolist(), mm.tolist(), ss.tolist())
                    }

        week_date = _week_date_from_key(key)
        week_label = _week_label_from_key(key)

        week_cells = covered_cells.copy()
        week_cells["raw_points"] = week_cells["cell_id"].map(point_counts).fillna(0.0)
        for col, value_map in point_feature_maps.items():
            week_cells[col] = week_cells["cell_id"].map(value_map)
            if col in point_feature_defaults:
                week_cells[col] = week_cells[col].fillna(point_feature_defaults[col])
        if point_time_map:
            week_cells["point_time"] = week_cells["cell_id"].map(point_time_map)
        week_cells["prediction_count"] = week_cells["raw_points"]
        week_cells["date"] = week_date
        week_cells["week"] = week_label

        week_cells = _compute_week_context_features(week_cells, week_date=week_date, spatial_context_dir=config.spatial_context_dir)

        count_columns = spec.count_columns if spec.model_kind == "hurdle_nb" and spec.count_columns else spec.final_columns
        zero_columns = spec.zero_columns if spec.model_kind == "hurdle_nb" and spec.zero_columns else count_columns

        design_matrix, municipality_codes = _design_matrix_for_spec(
            week_cells=week_cells,
            spec=spec,
            municipality_names=municipality_names,
            week_date=week_date,
            columns=count_columns,
            predictor_bounds=config.predictor_bounds,
            scaler_params=config.scaler_params,
            spline_knots=config.spline_knots,
        )
        design_matrix_zero, _ = _design_matrix_for_spec(
            week_cells=week_cells,
            spec=spec,
            municipality_names=municipality_names,
            week_date=week_date,
            columns=zero_columns,
            predictor_bounds=config.predictor_bounds,
            scaler_params=config.scaler_params,
            spline_knots=config.spline_knots,
        )

        key_df = week_cells[["municipality", "block", "week"]].copy().reset_index(drop=True)
        key_df["raw_points"] = week_cells["raw_points"].to_numpy(dtype=float)
        key_df["cell_area_m2"] = week_cells["cell_area_m2"].to_numpy(dtype=float)

        muni_labels, muni_uniques = pd.factorize(pd.MultiIndex.from_frame(key_df[["municipality", "week"]]), sort=True)
        block_labels, block_uniques = pd.factorize(pd.MultiIndex.from_frame(key_df[["municipality", "block", "week"]]), sort=True)

        n_draws = int(fit.posterior.sizes.get("chain", 1) * fit.posterior.sizes.get("draw", 1))
        muni_group_draws = np.zeros((len(muni_uniques), n_draws), dtype=float)
        block_group_draws = np.zeros((len(block_uniques), n_draws), dtype=float)
        chunk_size = 20000

        X = design_matrix
        M = municipality_codes
        X_zero = design_matrix_zero
        for start in range(0, X.shape[0], chunk_size):
            stop = min(start + chunk_size, X.shape[0])
            chunk_draws = expected_counts(spec, fit, X[start:stop], M[start:stop], X_zero=X_zero[start:stop] if spec.model_kind == "hurdle_nb" else None)
            chunk_draws = np.nan_to_num(chunk_draws, nan=0.0, posinf=1e6, neginf=0.0)
            np.add.at(muni_group_draws, muni_labels[start:stop], chunk_draws)
            np.add.at(block_group_draws, block_labels[start:stop], chunk_draws)

        muni_group_df = muni_uniques.to_frame(index=False)
        muni_group_df.columns = ["municipality", "week"]
        muni_group_df["raw_points"] = key_df.groupby(["municipality", "week"], dropna=False)["raw_points"].sum().values
        muni_group_df["covered_area_m2"] = key_df.groupby(["municipality", "week"], dropna=False)["cell_area_m2"].sum().values
        if config.apply_municipality:
            muni_summary = _summarize_aggregated_draws(
                grouped_df=muni_group_df,
                grouped_draws=muni_group_draws,
            )
            muni_summary["municipality_total_area_m2"] = muni_summary["municipality"].map(municipality_total_area).fillna(0.0)
            muni_summary["municipality_covered_area_m2"] = muni_summary["municipality"].map(covered_area_by_muni).fillna(0.0)
            denom = muni_summary["municipality_total_area_m2"].replace(0.0, np.nan)
            muni_summary["municipality_coverage_ratio"] = (muni_summary["municipality_covered_area_m2"] / denom).fillna(0.0)
            muni_summary = muni_summary[muni_summary["municipality_coverage_ratio"] >= float(config.municipality_coverage_threshold)].copy()
            if muni_summary.empty:
                municipal_bar.update(1)
                continue
            muni_summary["week_key"] = key
            muni_summary.rename(columns={"municipality": "Municipality", "week": "Week", "raw_points": "Raw points", "corrected_count_median": "Corrected count", "corrected_count_95_cri": "95% CrI"}, inplace=True)
            preferred_order = [
                "Municipality", "Week", "week_key", "Raw points", "Corrected count", "95% CrI",
                "corrected_count_mean", "corrected_count_p2_5", "corrected_count_p97_5",
                "density_per_km2_median", "density_per_km2_p2_5", "density_per_km2_p97_5", "covered_area_m2",
                "municipality_coverage_ratio", "municipality_covered_area_m2", "municipality_total_area_m2",
            ]
            muni_summary = muni_summary[[c for c in preferred_order if c in muni_summary.columns]]
            _append_rows(municipal_out, muni_summary)
            municipal_bar.update(1)

        block_group_df = block_uniques.to_frame(index=False)
        block_group_df.columns = ["municipality", "block", "week"]
        block_group_df["raw_points"] = key_df.groupby(["municipality", "block", "week"], dropna=False)["raw_points"].sum().values
        block_group_df["covered_area_m2"] = key_df.groupby(["municipality", "block", "week"], dropna=False)["cell_area_m2"].sum().values
        if config.apply_block:
            block_summary = _summarize_aggregated_draws(
                grouped_df=block_group_df,
                grouped_draws=block_group_draws,
            )
            block_summary["week_key"] = key
            _append_rows(block_out, block_summary)
            block_bar.update(1)

    if municipal_bar is not None:
        municipal_bar.close()
    if block_bar is not None:
        block_bar.close()

    return {
        "weeks_merged": len(merged_keys),
        "weeks_coverage": len(coverage_keys),
        "weeks_processed": len(todo_keys),
        "weeks_missing_coverage": len(merged_keys - coverage_keys),
        "weeks_missing_merged": len(coverage_keys - merged_keys),
        "n_grid_cells": int(len(grid)),
        "resume_enabled": int(config.resume),
        "weeks_already_done": len(completed_muni),
    }
