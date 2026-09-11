from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm

TARGET_CRS = "EPSG:32636"
EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class YellowWindow:
    start: pd.Timestamp
    end: pd.Timestamp | None


@dataclass
class WeeklyPostprocessResult:
    week_key: str
    n_input: int
    n_after_dbscan: int
    n_removed_yellow: int
    n_output: int



def parse_week_key_range(week_key: str) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
    parts = str(week_key).split("-")
    if len(parts) < 2:
        return pd.NaT, pd.NaT
    start = pd.to_datetime(parts[0], format="%Y%m%d", errors="coerce")
    end = pd.to_datetime(parts[1], format="%Y%m%d", errors="coerce")
    return start, end



def load_yellow_line_context(yellow_line_path: Path) -> tuple[gpd.GeoDataFrame, list[YellowWindow]]:
    if not yellow_line_path.exists():
        raise FileNotFoundError(f"Yellow line file not found: {yellow_line_path}")

    yellow = gpd.read_file(yellow_line_path)
    if yellow.empty:
        return yellow.set_crs("EPSG:4326").to_crs(TARGET_CRS), []
    if yellow.crs is None:
        yellow = yellow.set_crs("EPSG:4326")
    yellow = yellow.to_crs(TARGET_CRS)

    windows: list[YellowWindow] = []
    for _, row in yellow.iterrows():
        props_start = pd.to_datetime(row.get("date_start"), errors="coerce")
        props_end = pd.to_datetime(row.get("date_end"), errors="coerce")
        if pd.isna(props_start):
            continue
        windows.append(YellowWindow(start=props_start.normalize(), end=None if pd.isna(props_end) else props_end.normalize()))

    return yellow, windows



def _week_overlaps_any_window(week_start: pd.Timestamp, week_end: pd.Timestamp, windows: list[YellowWindow]) -> bool:
    if pd.isna(week_start) or pd.isna(week_end) or not windows:
        return False
    for w in windows:
        end = w.end if w.end is not None else pd.Timestamp.max
        if week_end >= w.start and week_start <= end:
            return True
    return False



def _dbscan_retain_non_noise(points: gpd.GeoDataFrame, eps_m: float = 40.0, min_samples: int = 3) -> gpd.GeoDataFrame:
    if points.empty:
        return points

    geo = points.to_crs("EPSG:4326")
    coords = np.column_stack([geo.geometry.y.to_numpy(dtype=float), geo.geometry.x.to_numpy(dtype=float)])
    if coords.shape[0] == 0:
        return points.iloc[0:0].copy()

    coords_rad = np.radians(coords)
    eps_rad = float(eps_m) / EARTH_RADIUS_M
    model = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric="haversine", algorithm="ball_tree")
    labels = model.fit_predict(coords_rad)
    keep = labels != -1
    if not np.any(keep):
        return points.iloc[0:0].copy()
    return points.loc[keep].copy()



def _remove_points_inside_yellow(
    points: gpd.GeoDataFrame,
    yellow: gpd.GeoDataFrame,
    apply_filter: bool,
) -> tuple[gpd.GeoDataFrame, int]:
    if points.empty or not apply_filter or yellow.empty:
        return points, 0

    joined = gpd.sjoin(points[["geometry"]], yellow[["geometry"]], how="left", predicate="within")
    inside_mask = joined["index_right"].notna().to_numpy()
    removed = int(np.sum(inside_mask))
    if removed == 0:
        return points, 0
    out = points.loc[~inside_mask].copy()
    return out, removed



def apply_weekly_postprocessing(
    points: gpd.GeoDataFrame,
    week_key: str,
    yellow: gpd.GeoDataFrame,
    yellow_windows: list[YellowWindow],
    eps_m: float = 40.0,
    min_samples: int = 3,
) -> tuple[gpd.GeoDataFrame, WeeklyPostprocessResult]:
    if points.crs is None:
        points = points.set_crs("EPSG:4326")
    points = points.to_crs(TARGET_CRS)

    start, end = parse_week_key_range(week_key)
    if pd.isna(start) or pd.isna(end):
        start = pd.Timestamp.min
        end = pd.Timestamp.max

    n_input = int(len(points))
    clustered = _dbscan_retain_non_noise(points, eps_m=eps_m, min_samples=min_samples)
    n_after_dbscan = int(len(clustered))

    apply_yellow = _week_overlaps_any_window(start.normalize(), end.normalize(), yellow_windows)
    filtered, n_removed_yellow = _remove_points_inside_yellow(clustered, yellow=yellow, apply_filter=apply_yellow)

    result = WeeklyPostprocessResult(
        week_key=week_key,
        n_input=n_input,
        n_after_dbscan=n_after_dbscan,
        n_removed_yellow=int(n_removed_yellow),
        n_output=int(len(filtered)),
    )
    return filtered, result



def recompute_validation_prediction_counts(
    validation_df: pd.DataFrame,
    merged_dir: Path,
    yellow_line_path: Path,
    tile_size_m: float = 100.0,
    eps_m: float = 40.0,
    min_samples: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = validation_df.copy()
    out["prediction_count"] = np.nan
    out["prediction_available"] = False
    out["prediction_status"] = "missing_week_file"

    if "date" not in out.columns or "center_x" not in out.columns or "center_y" not in out.columns:
        raise ValueError("Validation data must contain date, center_x, center_y")

    out["_date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out[out["_date"].notna()].copy()

    merged_paths = sorted(merged_dir.glob("*.gpkg"))
    week_meta = []
    for p in merged_paths:
        wk = p.stem
        ws, we = parse_week_key_range(wk)
        if pd.isna(ws) or pd.isna(we):
            continue
        week_meta.append((wk, ws.normalize(), we.normalize(), p))

    yellow, windows = load_yellow_line_context(yellow_line_path)
    logs: list[dict] = []

    date_to_week: dict[pd.Timestamp, str] = {}
    for dt in out["_date"].dt.normalize().unique():
        match = None
        for wk, ws, we, _ in week_meta:
            if ws <= dt <= we:
                match = wk
                break
        if match is not None:
            date_to_week[dt] = match

    out["_week_key"] = out["_date"].dt.normalize().map(date_to_week)

    half = float(tile_size_m) / 2.0
    for wk, sub in tqdm(out.groupby("_week_key", dropna=False), desc="Postprocess validation weeks", unit="week"):
        if wk is None or (isinstance(wk, float) and np.isnan(wk)):
            continue

        week_rows = sub.index.to_numpy()
        week_path = merged_dir / f"{wk}.gpkg"
        if not week_path.exists():
            continue

        pts = gpd.read_file(week_path)
        if pts.empty:
            out.loc[week_rows, "prediction_count"] = 0.0
            out.loc[week_rows, "prediction_available"] = True
            out.loc[week_rows, "prediction_status"] = "ok"
            logs.append({"week_key": wk, "n_tiles": int(len(week_rows)), "n_points_post": 0})
            continue

        if pts.crs is None:
            pts = pts.set_crs("EPSG:4326")
        pts = pts.to_crs(TARGET_CRS)
        pts_post, res = apply_weekly_postprocessing(pts, wk, yellow, windows, eps_m=eps_m, min_samples=min_samples)

        tiles = gpd.GeoDataFrame(
            {
                "row_index": week_rows,
            },
            geometry=[
                box(float(out.loc[i, "center_x"]) - half, float(out.loc[i, "center_y"]) - half, float(out.loc[i, "center_x"]) + half, float(out.loc[i, "center_y"]) + half)
                for i in week_rows
            ],
            crs=TARGET_CRS,
        )

        if pts_post.empty:
            counts = pd.Series(0, index=tiles["row_index"])
        else:
            j = gpd.sjoin(pts_post[["geometry"]], tiles[["row_index", "geometry"]], how="inner", predicate="within")
            counts = j.groupby("row_index").size()

        out.loc[week_rows, "prediction_count"] = out.loc[week_rows].index.to_series().map(counts).fillna(0).astype(float).to_numpy()
        out.loc[week_rows, "prediction_available"] = True
        out.loc[week_rows, "prediction_status"] = "ok"

        logs.append(
            {
                "week_key": wk,
                "n_tiles": int(len(week_rows)),
                "n_input_points": res.n_input,
                "n_after_dbscan": res.n_after_dbscan,
                "n_removed_yellow": res.n_removed_yellow,
                "n_output_points": res.n_output,
                "n_nonzero_tiles": int((out.loc[week_rows, "prediction_count"] > 0).sum()),
            }
        )

    out["error"] = pd.to_numeric(out["prediction_count"], errors="coerce") - pd.to_numeric(out.get("tent_count"), errors="coerce")
    out["abs_error"] = np.abs(out["error"])
    out["correct"] = (out["error"] == 0).astype(float)
    out = out.drop(columns=["_date", "_week_key"], errors="ignore")
    return out, pd.DataFrame(logs)
