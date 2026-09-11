from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from .types import FeatureBundle

TARGET_CRS = "EPSG:32636"


def _is_binary_series(s: pd.Series) -> bool:
    vals = pd.to_numeric(s, errors="coerce").dropna().astype(float).to_numpy()
    if len(vals) == 0:
        return False
    rounded = np.round(vals.astype(float), 10)
    return set(rounded.tolist()).issubset({0.0, 1.0})


def fit_scaler_params(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, Dict[str, float]]:
    params: Dict[str, Dict[str, float]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.dropna().empty:
            continue
        if _is_binary_series(s):
            continue
        mean = float(s.mean())
        std = float(s.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            continue
        params[col] = {"mean": mean, "std": std}
    return params


def apply_scaler_params(df: pd.DataFrame, scaler_params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, cfg in scaler_params.items():
        if col not in out.columns:
            continue
        mean = float(cfg.get("mean", 0.0))
        std = float(cfg.get("std", 1.0))
        if not np.isfinite(std) or std <= 1e-12:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        out[col] = (s - mean) / std
    return out


def add_cyclic_terms(df: pd.DataFrame, col: str, period: float, prefix: str) -> None:
    angle = 2.0 * np.pi * (pd.to_numeric(df[col], errors="coerce") % period) / period
    df[f"{prefix}_sin"] = np.sin(angle)
    df[f"{prefix}_cos"] = np.cos(angle)


def add_distance_features(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce")
        df[f"log1p_{col}"] = np.log1p(np.clip(x, a_min=0.0, a_max=None))


def add_spline_features(
    df: pd.DataFrame,
    cols: Iterable[str],
    suffix: str,
    knots: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    transformers: Dict[str, Any] = {}
    for col in cols:
        raw = pd.to_numeric(out[col], errors="coerce")
        med = float(raw.median()) if raw.notna().any() else 0.0
        x = raw.fillna(med).to_numpy(dtype=float)
        use_knots = knots if knots is not None else [
            float(np.nanquantile(x, q)) for q in [0.2, 0.4, 0.6, 0.8]
        ]
        transformers[col] = {"knots": use_knots}
        for j, knot in enumerate(use_knots):
            out[f"{col}_{suffix}_s{j}"] = np.maximum(x - knot, 0.0)
    return out, transformers


def add_local_density(df: pd.DataFrame, radius_m: float = 250.0) -> pd.DataFrame:
    out = df.copy()
    out["week"] = pd.to_datetime(out["date"], errors="coerce").dt.to_period("W-MON").astype(str)
    out["local_pred_density"] = 0.0
    for week, chunk in out.groupby("week", dropna=False):
        if pd.isna(week):
            continue
        valid = chunk[["center_x", "center_y", "prediction_count"]].dropna()
        if valid.empty:
            continue
        coords = valid[["center_x", "center_y"]].to_numpy(dtype=float)
        pred = valid["prediction_count"].to_numpy(dtype=float)
        d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        within = d2 <= (radius_m**2)
        density = within @ pred
        out.loc[valid.index, "local_pred_density"] = density
    out["log1p_local_pred_density"] = np.log1p(np.clip(out["local_pred_density"], 0.0, None))
    return out


def attach_block_from_centers(df: pd.DataFrame, block_path) -> pd.DataFrame:
    out = df.copy()
    if "block" in out.columns and out["block"].notna().any():
        return out
    if not {"center_x", "center_y"}.issubset(out.columns):
        out["block"] = "UNKNOWN"
        return out
    block = gpd.read_file(block_path)
    if block.empty or block.crs is None:
        out["block"] = "UNKNOWN"
        return out
    if block.crs.to_string() != TARGET_CRS:
        block = block.to_crs(TARGET_CRS)
    block_col = "Name" if "Name" in block.columns else block.columns[0]
    pts = gpd.GeoDataFrame(
        out.reset_index().rename(columns={"index": "row_index"}),
        geometry=gpd.points_from_xy(out["center_x"], out["center_y"]),
        crs=TARGET_CRS,
    )
    joined = gpd.sjoin(pts[["row_index", "geometry"]], block[[block_col, "geometry"]], how="left", predicate="within")
    block_map = joined.set_index("row_index")[block_col]
    out["block"] = out.index.to_series().map(block_map).fillna("UNKNOWN").astype(str)
    return out


def featurize_validation(df: pd.DataFrame, block_path, require_target: bool = True) -> pd.DataFrame:
    out = attach_block_from_centers(df.copy(), block_path=block_path)
    out["prediction_count"] = pd.to_numeric(out["prediction_count"], errors="coerce")
    if "tent_count" not in out.columns:
        out["tent_count"] = np.nan
    out["tent_count"] = pd.to_numeric(out["tent_count"], errors="coerce")
    if require_target:
        out = out[out["prediction_count"].notna() & out["tent_count"].notna()].copy()
    else:
        out = out[out["prediction_count"].notna()].copy()

    out["log1p_model_tent_count"] = np.log1p(np.clip(out["prediction_count"], 0.0, None))
    out["month_num"] = pd.to_datetime(out["date"], errors="coerce").dt.month
    out["year"] = pd.to_datetime(out["date"], errors="coerce").dt.year
    add_cyclic_terms(out, "month_num", 12.0, "month")
    add_cyclic_terms(out, "satellite_azimuth", 360.0, "sat_az")
    add_cyclic_terms(out, "sun_azimuth", 360.0, "sun_az")
    out["hours_from_sunrise"] = pd.to_numeric(out["hours_from_sunrise"], errors="coerce")

    bool_cols = ["is_agriculture", "is_destruction", "has_greenhouse", "is_humanitarian_zone"]
    for c in bool_cols:
        out[c] = out[c].astype(str).str.lower().isin(["1", "true", "t", "yes"]).astype(int)

    add_distance_features(out, ["distance_to_nearest_base", "distance_to_nearest_outpost", "distance_to_yellow_line"])
    out = add_local_density(out)
    return out


def build_candidate_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    out = df.copy()
    land = pd.get_dummies(out["land_use_label"], prefix="land_use", dummy_na=True, drop_first=False)
    non_nan_land_cols = [c for c in land.columns if c != "land_use_nan"]
    if non_nan_land_cols:
        ref_col = sorted(non_nan_land_cols)[0]
        land = land.drop(columns=[ref_col])
    out = pd.concat([out, land], axis=1)
    out, _ = add_spline_features(out, ["hours_from_sunrise"], suffix="cand")
    out, _ = add_spline_features(out, ["building_density"], suffix="cand")

    candidate_groups: Dict[str, List[str]] = {
        "land_use": land.columns.tolist(),
        "month_cycle": ["month_sin", "month_cos"],
        "year": ["year"],
        "hours_from_sunrise_spline": [c for c in out.columns if c.startswith("hours_from_sunrise_cand_s")],
        "building_density_spline": [c for c in out.columns if c.startswith("building_density_cand_s")],
        "is_agriculture": ["is_agriculture"],
        "is_destruction": ["is_destruction"],
        "has_greenhouse": ["has_greenhouse"],
        "is_humanitarian_zone": ["is_humanitarian_zone"],
        "dist_base_log1p": ["log1p_distance_to_nearest_base"],
        "dist_outpost_log1p": ["log1p_distance_to_nearest_outpost"],
        "dist_yellow_line_log1p": ["log1p_distance_to_yellow_line"],
        "satellite_elevation": ["satellite_elevation"],
        "sun_elevation": ["sun_elevation"],
        "satellite_azimuth_sin_cos": ["sat_az_sin", "sat_az_cos"],
        "sun_azimuth_sin_cos": ["sun_az_sin", "sun_az_cos"],
        "local_density_log1p": ["log1p_local_pred_density"],
    }
    return out, candidate_groups


def make_bundle_from_columns(
    df: pd.DataFrame,
    columns: List[str],
    municipality_names: Optional[List[str]] = None,
) -> FeatureBundle:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = 0.0
    out[columns] = out[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if municipality_names is None:
        municipality_names = sorted(out["municipality"].astype(str).unique().tolist())
    muni_map = {name: i for i, name in enumerate(municipality_names)}
    muni_codes = out["municipality"].astype(str).map(muni_map).fillna(0).to_numpy(dtype=int)
    week_group = pd.to_datetime(out["date"], errors="coerce").dt.to_period("W-MON").astype(str).to_numpy()

    return FeatureBundle(
        frame=out,
        municipality_codes=muni_codes,
        municipality_names=municipality_names,
        design_matrix=out[columns].to_numpy(dtype=float),
        design_columns=columns,
        target=out["tent_count"].to_numpy(dtype=int),
        week_group=week_group,
    )
