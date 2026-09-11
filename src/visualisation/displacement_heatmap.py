from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
MUNICIPALITY_PATH = REPO_ROOT / "data" / "gaza_boundaries" / "municipal" / "GazaStrip_MunicipalBoundaries.shp"
MERGED_DIR = REPO_ROOT / "data" / "tent_predictions" / "merged"
COVERAGE_DIR = REPO_ROOT / "data" / "tent_predictions" / "coverage"
CORRECTED_COUNTS_PATH = REPO_ROOT / "outputs" / "tent_counts" / "municipality_counts_by_week.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "visualisation"
HEATMAP_PATH = OUTPUT_DIR / "gaza_municipality_tent_count_change_heatmap.png"
CSV_PATH = OUTPUT_DIR / "gaza_municipality_tent_count_change.csv"
GAZA_WIDE_COUNTS_PATH = REPO_ROOT / "outputs" / "analysis" / "gaza_wide_counts_by_week.csv"
TARGET_CRS = "EPSG:32636"
EVENTS = [
    ("2023-10-13", "North Gaza evacuation order"),
    ("2024-05-06", "Rafah evacuation order"),
    ("2025-01-19", "First ceasefire"),
    ("2025-03-18", "End of ceasefire"),
    ("2025-09-09", "Gaza City evacuation order"),
    ("2025-10-10", "Second ceasefire"),
]
EXCLUDE_GAZA_WIDE_WEEKS = {
    "20260518-20260524",
    "20260525-20260531",
    "20260601-20260607",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Municipality weekly tent-density change heatmap")
    p.add_argument("--value-source", choices=["corrected", "raw_mean"], default="corrected")
    p.add_argument("--corrected-csv", type=Path, default=CORRECTED_COUNTS_PATH)
    p.add_argument("--output-image", type=Path, default=HEATMAP_PATH)
    p.add_argument("--output-csv", type=Path, default=CSV_PATH)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--robust-scale", action="store_true", default=True)
    p.add_argument("--no-robust-scale", dest="robust_scale", action="store_false")
    p.add_argument("--robust-low-quantile", type=float, default=0.02)
    p.add_argument("--robust-high-quantile", type=float, default=0.98)
    return p


def parse_week_start_from_key(stem: str) -> pd.Timestamp:
    start = stem.replace("_coverage", "").split("-")[0]
    return pd.to_datetime(start, format="%Y%m%d", errors="coerce")


def load_municipalities() -> gpd.GeoDataFrame:
    municipalities = gpd.read_file(MUNICIPALITY_PATH)
    municipalities = municipalities[["NAME", "geometry"]].rename(columns={"NAME": "municipality"})
    municipalities = municipalities[municipalities["municipality"].notna()].copy()
    municipalities = municipalities.to_crs(TARGET_CRS)
    municipalities = municipalities.dissolve(by="municipality", as_index=False)
    municipalities = municipalities[["municipality", "geometry"]]

    cent = municipalities.geometry.centroid
    municipalities["_cx"] = cent.x
    municipalities["_cy"] = cent.y
    municipalities = municipalities.sort_values(["_cy", "_cx"], ascending=[False, True]).reset_index(drop=True)
    return municipalities


def compute_weekly_raw_counts_and_coverage(municipalities: gpd.GeoDataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    muni_area = municipalities.set_index("municipality").geometry.area

    for prediction_file in sorted(MERGED_DIR.glob("*.gpkg")):
        week_key = prediction_file.stem
        week_start = parse_week_start_from_key(week_key)
        coverage_file = COVERAGE_DIR / f"{week_key}_coverage.geojson"
        if not coverage_file.exists() or pd.isna(week_start):
            continue

        coverage = gpd.read_file(coverage_file)
        coverage = coverage.to_crs(TARGET_CRS) if not coverage.empty else coverage
        points = gpd.read_file(prediction_file)
        points = points.to_crs(TARGET_CRS) if not points.empty else points

        coverage_area = pd.Series(0.0, index=municipalities["municipality"])
        if not coverage.empty:
            inter = gpd.overlay(coverage[["geometry"]], municipalities[["municipality", "geometry"]], how="intersection")
            if not inter.empty:
                inter["intersection_area"] = inter.geometry.area
                coverage_area = inter.groupby("municipality")["intersection_area"].sum().reindex(municipalities["municipality"]).fillna(0.0)

        raw_counts = pd.Series(0.0, index=municipalities["municipality"])
        if not points.empty:
            points = points[points.geometry.notna()].copy()
            points = points[points.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
            joined = gpd.sjoin(points, municipalities[["municipality", "geometry"]], how="left", predicate="within")
            if not joined.empty:
                raw_counts = joined.groupby("municipality").size().reindex(municipalities["municipality"]).fillna(0.0)

        for municipality in municipalities["municipality"]:
            covered_area = float(coverage_area.get(municipality, 0.0))
            total_area = float(muni_area.get(municipality, np.nan))
            coverage_ratio = 0.0 if not np.isfinite(total_area) or total_area <= 0 else covered_area / total_area
            rows.append(
                {
                    "municipality": municipality,
                    "week_key": week_key,
                    "week_start": week_start,
                    "raw_points": float(raw_counts.get(municipality, 0.0)),
                    "covered_area_m2": covered_area,
                    "municipality_total_area_m2": total_area,
                    "municipality_coverage_ratio": coverage_ratio,
                }
            )

    return pd.DataFrame(rows)


def merge_corrected_if_available(base_df: pd.DataFrame, corrected_csv: Path) -> pd.DataFrame:
    if not corrected_csv.exists():
        return base_df
    corrected = pd.read_csv(corrected_csv)
    required = {"Municipality", "week_key", "Corrected count"}
    if not required.issubset(corrected.columns):
        return base_df
    out = base_df.merge(
        corrected[["Municipality", "week_key", "Corrected count"]].rename(
            columns={"Municipality": "municipality", "Corrected count": "corrected_count"}
        ),
        on=["municipality", "week_key"],
        how="left",
    )
    return out


def build_analysis_frame(value_source: str, corrected_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    municipalities = load_municipalities()
    order = municipalities["municipality"].tolist()
    df = compute_weekly_raw_counts_and_coverage(municipalities)
    df = merge_corrected_if_available(df, corrected_csv)

    df = df[df["municipality_coverage_ratio"] >= 0.8].copy()
    if df.empty:
        raise ValueError("No municipality-week rows with >=80% coverage")

    if value_source == "corrected":
        if "corrected_count" not in df.columns or df["corrected_count"].notna().sum() == 0:
            raise ValueError("Corrected counts not available. Use --value-source raw_mean or generate corrected CSV first.")
        df["selected_count"] = pd.to_numeric(df["corrected_count"], errors="coerce")
    else:
        df["selected_count"] = pd.to_numeric(df["raw_points"], errors="coerce")

    df = df.sort_values(["municipality", "week_start"]).copy()
    baseline = df.groupby("municipality")["selected_count"].transform("first")
    df["count_change"] = df["selected_count"] - baseline
    return df, order


def save_heatmap(
    df: pd.DataFrame,
    municipality_order: list[str],
    output_path: Path,
    dpi: int,
    robust_scale: bool,
    robust_low_quantile: float,
    robust_high_quantile: float,
) -> None:
    heatmap_df = df.pivot(index="municipality", columns="week_start", values="count_change")
    heatmap_df = heatmap_df.reindex(municipality_order)
    heatmap_df = heatmap_df.loc[heatmap_df.notna().any(axis=1)]
    heatmap_df = heatmap_df.sort_index(axis=1)

    # Ensure equal temporal spacing even when some weeks are missing from data.
    col_dates = pd.to_datetime(heatmap_df.columns)
    if len(col_dates) > 0:
        full_week_index = pd.date_range(start=col_dates.min(), end=col_dates.max(), freq="W-MON")
        heatmap_df = heatmap_df.reindex(columns=full_week_index)

    cols = len(heatmap_df.columns)
    rows = len(heatmap_df.index)
    fig_w = max(5.5, cols * 0.10)
    fig_h = max(6.2, rows * 0.32 + 1.7)

    data = heatmap_df.values.astype(float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    elif robust_scale:
        vmin = float(np.nanquantile(finite, robust_low_quantile))
        vmax = float(np.nanquantile(finite, robust_high_quantile))
        if vmin == vmax:
            delta = max(abs(vmin), 1.0)
            vmin, vmax = -delta, delta
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmin == vmax:
            delta = max(abs(vmin), 1.0)
            vmin, vmax = -delta, delta

    if vmin >= 0:
        vmin = -max(1e-6, 0.05 * vmax)
    if vmax <= 0:
        vmax = max(1e-6, -0.05 * vmin)
    if vmin > -500:
        vmin = -500.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=[1.0, 0.03],
        height_ratios=[5.0, 1.35],
        hspace=0.34,
        wspace=0.05,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[1, 0], sharex=ax)
    cax = fig.add_subplot(gs[0, 1])
    ax_unused = fig.add_subplot(gs[1, 1])
    ax_unused.axis("off")

    mesh = ax.pcolormesh(
        np.arange(cols + 1),
        np.arange(rows + 1),
        np.ma.masked_invalid(data),
        cmap="RdBu_r",
        norm=norm,
        shading="flat",
        linewidth=0.25,
        edgecolors=(1, 1, 1, 0.35),
        zorder=2,
    )

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)

    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels(heatmap_df.index.tolist(), fontsize=8)

    col_dates = pd.to_datetime(heatmap_df.columns)
    quarter_ticks = pd.date_range(start=pd.Timestamp("2023-10-01"), end=col_dates.max(), freq="QS-OCT")
    tick_positions = []
    tick_labels = []
    for tick_date in quarter_ticks:
        idx = int(np.argmin(np.abs((col_dates - tick_date).days)))
        if idx not in tick_positions:
            tick_positions.append(idx)
            tick_labels.append(tick_date.strftime("%b %Y"))
    xtick_pos = np.array(tick_positions, dtype=float) + 0.5
    shared_locator = FixedLocator(xtick_pos)
    ax.xaxis.set_major_locator(shared_locator)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_zorder(30)
        lbl.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 0.25})

    ax.set_xlabel("")
    ax.set_ylabel("Municipality", fontsize=9)
    ax.set_title("Weekly change in detected tent counts across Gaza municipalities, 2023-2026", fontsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(mesh, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7)
    neg_ticks = np.arange(-500, vmin - 1, -500)
    pos_ticks = np.arange(0, max(vmax, 0) + 1, 20000)
    tick_vals = np.unique(np.concatenate([neg_ticks, pos_ticks]))
    tick_vals = tick_vals[(tick_vals >= vmin) & (tick_vals <= vmax)]
    if 0.0 not in tick_vals:
        tick_vals = np.sort(np.append(tick_vals, 0.0))
    cbar.set_ticks(tick_vals)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    cbar.ax.set_title("Change in\ndetected\ntents", fontsize=7, pad=5)

    event_label_offsets = [2.35, 3.35]
    offset_zorder = {2.35: 13, 3.35: 12}
    for event_idx, (event_date_str, event_label) in enumerate(EVENTS):
        event_date = pd.to_datetime(event_date_str)
        idx = int(np.argmin(np.abs((col_dates - event_date).days)))
        x = float(idx)
        offset = event_label_offsets[event_idx % 2]
        y_label = rows + offset
        y_line_end = y_label - 0.16
        ax.plot([x, x], [0.0, y_line_end], color="black", linewidth=0.9, alpha=0.65, zorder=8, clip_on=False)
        ax.text(
            x,
            y_label,
            event_label,
            rotation=0,
            ha="center",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25},
            zorder=offset_zorder.get(offset, 12),
            clip_on=False,
        )

    # Bottom panel: Gaza-wide weekly counts.
    gaza_wide = pd.read_csv(GAZA_WIDE_COUNTS_PATH)
    gaza_wide["week_id"] = gaza_wide["week_id"].astype(str)
    gaza_wide["week_start"] = pd.to_datetime(gaza_wide["week_start"], errors="coerce")
    gaza_wide = gaza_wide.dropna(subset=["week_start"]).copy()
    gaza_wide = gaza_wide.sort_values("week_start")
    gaza_wide = gaza_wide[["week_id", "week_start", "gaza_wide_corrected_count_mean"]]

    full_index = pd.DataFrame({"week_start": full_week_index})
    gaza_wide = full_index.merge(gaza_wide, on="week_start", how="left")
    gaza_wide.loc[gaza_wide["week_id"].isin(EXCLUDE_GAZA_WIDE_WEEKS), "gaza_wide_corrected_count_mean"] = np.nan

    ax_ts.plot(
        np.arange(len(full_week_index)) + 0.5,
        gaza_wide["gaza_wide_corrected_count_mean"].to_numpy(float),
        color="#1f3d7a",
        linewidth=1.6,
    )
    ax_ts.set_ylabel("Gaza-wide\ncount", fontsize=8)
    ax_ts.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    ax_ts.tick_params(axis="y", labelsize=7)

    ax_ts.xaxis.set_major_locator(shared_locator)
    ax_ts.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=8)
    for lbl in ax_ts.get_xticklabels():
        lbl.set_zorder(30)
        lbl.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 0.25})

    for spine in ax_ts.spines.values():
        spine.set_visible(False)
    ax_ts.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax_ts.set_xlim(ax.get_xlim())

    fig.subplots_adjust(left=0.08, right=0.94, top=0.93, bottom=0.10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, order = build_analysis_frame(value_source=args.value_source, corrected_csv=args.corrected_csv)
    df.to_csv(args.output_csv, index=False)
    save_heatmap(
        df,
        municipality_order=order,
        output_path=args.output_image,
        dpi=args.dpi,
        robust_scale=args.robust_scale,
        robust_low_quantile=args.robust_low_quantile,
        robust_high_quantile=args.robust_high_quantile,
    )
    print(f"Saved analysis CSV: {args.output_csv}")
    print(f"Saved heatmap: {args.output_image}")


if __name__ == "__main__":
    main()
