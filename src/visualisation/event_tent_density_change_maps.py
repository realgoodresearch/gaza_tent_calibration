"""Map corrected tent-density changes at the six major events."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 0.3
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

REPO_ROOT = Path(__file__).resolve().parents[2]
MUNICIPALITIES = REPO_ROOT / "data/gaza_boundaries/municipal/GazaStrip_MunicipalBoundaries.shp"
MERGED_DIR = REPO_ROOT / "data/tent_predictions/merged"
COVERAGE_DIR = REPO_ROOT / "data/tent_predictions/coverage"
COUNTS = REPO_ROOT / "outputs/tent_counts/municipality_counts_by_week.csv"
BLOCK_COUNTS = REPO_ROOT / "outputs/tent_counts/block_counts_by_week.csv"
BLOCKS = REPO_ROOT / "data/gaza_boundaries/dash_population_blocks.geojson"
HUMANITARIAN_ZONE = REPO_ROOT / "data/spatial_context/humanitarian_zone.json"
YELLOW_LINE = REPO_ROOT / "data/spatial_context/yellow-line-zone.json"
OUTPUT = REPO_ROOT / "outputs/visualisation/event_tent_density_change_maps.png"
TARGET_CRS = "EPSG:32636"
EVENTS = [
    ("2023-10-13", "North Gaza evacuation order"),
    ("2024-05-06", "Rafah evacuation order"),
    ("2025-01-19", "First ceasefire"),
    ("2025-03-18", "End of ceasefire"),
    ("2025-09-09", "Gaza City evacuation order"),
    ("2025-10-10", "Second ceasefire"),
]
RNG = np.random.default_rng(2026)
DISPLAY_TITLES = {
    "North Gaza evacuation order": "North Gaza evacuation",
    "Rafah evacuation order": "Rafah evacuation",
    "Gaza City evacuation order": "Gaza City evacuation",
}


def _read(path: Path) -> gpd.GeoDataFrame:
    data = gpd.read_file(path)
    return data.set_crs("EPSG:4326") if data.crs is None else data.to_crs(TARGET_CRS)


def _solid_territory(geometry):
    """Remove interior rings after merging all source municipal polygons."""
    if isinstance(geometry, Polygon):
        return Polygon(geometry.exterior)
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon([Polygon(part.exterior) for part in geometry.geoms])
    if isinstance(geometry, GeometryCollection):
        return _solid_territory(gpd.GeoSeries([part for part in geometry.geoms if part.geom_type in {"Polygon", "MultiPolygon"}]).union_all())
    return geometry


def _active(layer: gpd.GeoDataFrame, date: pd.Timestamp) -> gpd.GeoDataFrame:
    """Return features active on date, accepting the two source date schemas."""
    start = pd.to_datetime(layer.get("start_date", layer.get("date_start", layer.get("safe_date"))), errors="coerce")
    end = pd.to_datetime(layer.get("end_date", layer.get("date_end")), errors="coerce")
    return layer[(start.isna() | (start <= date)) & (end.isna() | (end >= date))]


def _event_week(counts: pd.DataFrame, event_date: pd.Timestamp) -> str:
    keys = counts["week_key"].dropna().unique()
    for key in keys:
        start, end = (pd.to_datetime(x, format="%Y%m%d") for x in key.split("-"))
        if start <= event_date <= end:
            return str(key)
    raise ValueError(f"No corrected-count week contains {event_date.date()}")


def _panel_data(counts: pd.DataFrame, date: pd.Timestamp, municipalities: gpd.GeoDataFrame) -> tuple[str, pd.Series]:
    week = _event_week(counts, date)
    current = counts[counts.week_key.eq(week)].set_index("Municipality")
    adjusted_count = pd.to_numeric(current["corrected_count_mean"], errors="coerce").reindex(municipalities.municipality)
    # Where the event week has no municipal estimate, use the first later
    # corrected estimate for that municipality; leave it gray if none exists.
    starts = pd.to_datetime(counts.week_key.str.split("-").str[0], format="%Y%m%d")
    later = counts.loc[starts > date].copy()
    later["week_start"] = pd.to_datetime(later.week_key.str.split("-").str[0], format="%Y%m%d")
    fallback = later.sort_values("week_start").groupby("Municipality")["corrected_count_mean"].first()
    corrected_count = adjusted_count.fillna(fallback.reindex(municipalities.municipality))
    area_km2 = municipalities.geometry.area.to_numpy() / 1e6
    return week, corrected_count / area_km2


def _block_panel_data(counts: pd.DataFrame, date: pd.Timestamp, blocks: gpd.GeoDataFrame) -> tuple[str, pd.Series]:
    week = _event_week(counts, date)
    current = counts[counts.week_key.eq(week)].set_index("block_id")["corrected_count_mean"]
    starts = pd.to_datetime(counts.week_key.str.split("-").str[0], format="%Y%m%d")
    later = counts.loc[starts > date].copy()
    later["week_start"] = pd.to_datetime(later.week_key.str.split("-").str[0], format="%Y%m%d")
    fallback = later.sort_values("week_start").groupby("block_id")["corrected_count_mean"].first()
    corrected_count = current.reindex(blocks.block_id).fillna(fallback.reindex(blocks.block_id))
    return week, corrected_count / (blocks.geometry.area.to_numpy() / 1e6)


def _corrected_points(week: str, counts: pd.Series, yellow: gpd.GeoDataFrame, municipalities: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = _read(MERGED_DIR / f"{week}.gpkg")
    points = points[points.geometry.notna()].copy()
    points = gpd.sjoin(points, municipalities[["municipality", "geometry"]], how="left", predicate="within").drop(columns="index_right")
    if not yellow.empty:
        blocked = gpd.sjoin(points[["geometry"]], yellow[["geometry"]], how="left", predicate="within")["index_right"].notna()
        points = points.loc[~blocked].copy()
    selected = []
    for municipality, group in points.groupby("municipality", sort=False):
        n = max(0, int(round(float(counts.get(municipality, 0)))))
        if n == 0 or group.empty:
            continue
        ranked = group.nlargest(min(n, len(group)), "adjusted_peak")
        if n > len(group):
            # Repeated randomly selected locations encode the corrected total
            # when the correction exceeds the detected-point count.
            extra = group.iloc[RNG.integers(0, len(group), size=n - len(group))]
            ranked = pd.concat([ranked, extra])
        selected.append(ranked)
    # Some areas are outside a named municipality or have no corrected count.
    # Keep their observed points rather than rendering them as background gaps.
    uncorrected = points[points["municipality"].isna() | ~points["municipality"].isin(counts.index)]
    if not uncorrected.empty:
        selected.append(uncorrected)
    return gpd.GeoDataFrame(pd.concat(selected, ignore_index=True), crs=points.crs) if selected else points.iloc[0:0]


def _following_week_with_coverage(date: pd.Timestamp) -> str:
    """First available coverage week starting strictly after an event date."""
    options = []
    for path in COVERAGE_DIR.glob("*_coverage.geojson"):
        key = path.name.replace("_coverage.geojson", "")
        start = pd.to_datetime(key.split("-")[0], format="%Y%m%d", errors="coerce")
        if pd.notna(start) and start > date and (MERGED_DIR / f"{key}.gpkg").exists():
            options.append((start, key))
    if not options:
        raise ValueError(f"No later coverage is available after {date.date()}")
    return min(options)[1]


def _later_coverage_weeks(date: pd.Timestamp) -> list[str]:
    options = []
    for path in COVERAGE_DIR.glob("*_coverage.geojson"):
        key = path.name.replace("_coverage.geojson", "")
        start = pd.to_datetime(key.split("-")[0], format="%Y%m%d", errors="coerce")
        if pd.notna(start) and start > date and (MERGED_DIR / f"{key}.gpkg").exists():
            options.append((start, key))
    return [key for _, key in sorted(options)]


def main() -> None:
    # Merge the complete shapefile first. Named-municipality filtering below is
    # only for count matching and labels; it must not alter the Strip base map.
    all_boundaries = _read(MUNICIPALITIES)
    outer_boundary = _solid_territory(all_boundaries.geometry.union_all()).buffer(75).buffer(-75)
    municipalities = all_boundaries[["NAME", "geometry"]].dropna(subset=["NAME"]).rename(columns={"NAME": "municipality"}).dissolve(by="municipality", as_index=False)
    block_counts = pd.read_csv(BLOCK_COUNTS)
    block_counts["block_id"] = pd.to_numeric(block_counts["block"], errors="coerce").astype("Int64")
    for column in ["corrected_count_mean"]:
        block_counts[column] = pd.to_numeric(block_counts[column], errors="coerce")
    block_counts = block_counts.dropna(subset=["block_id", "week_key", "corrected_count_mean"])
    # A few population-block geometries span municipal reporting units. Sum
    # their corrected components back to the unique block id for mapping.
    block_counts = block_counts.groupby(["week_key", "block_id"], as_index=False)["corrected_count_mean"].sum()
    blocks = _read(BLOCKS)[["Name", "geometry"]].rename(columns={"Name": "block_id"})
    blocks["block_id"] = pd.to_numeric(blocks["block_id"], errors="coerce").astype("Int64")
    blocks = blocks.dropna(subset=["block_id"])
    humanitarian = _read(HUMANITARIAN_ZONE)
    yellow_line = _read(YELLOW_LINE)
    panel = []
    for date_text, label in EVENTS:
        date = pd.Timestamp(date_text)
        week, adjusted = _block_panel_data(block_counts, date, blocks)
        panel.append((date, label, week, adjusted))
    max_count = max(float(adjusted.max()) for *_, adjusted in panel if adjusted.notna().any())
    norm = Normalize(vmin=0, vmax=max_count)
    cmap = plt.get_cmap("Reds")
    minx, miny, maxx, maxy = municipalities.total_bounds
    pad = 3000
    label_municipalities = municipalities.assign(_area=municipalities.geometry.area).nlargest(7, "_area")
    fig, axes = plt.subplots(2, 3, figsize=(20, 16), facecolor="white", gridspec_kw={"wspace": 0, "hspace": .16})
    fig.patch.set_facecolor("white")
    for panel_index, (ax, (date, label, week, adjusted)) in enumerate(zip(axes.flat, panel)):
        ax.set_facecolor("white")
        gpd.GeoSeries([outer_boundary], crs=TARGET_CRS).plot(ax=ax, color="#f7f4ed", edgecolor="none")
        hz = _active(humanitarian, date)
        if not hz.empty:
            hz = hz.copy()
            hz["geometry"] = hz.geometry.intersection(outer_boundary)
            hz = hz[~hz.geometry.is_empty]
            hz = gpd.GeoDataFrame(geometry=[hz.geometry.union_all()], crs=TARGET_CRS)
        yellow = _active(yellow_line, date)
        # The Yellow Line is pertinent to the second ceasefire panel even
        # though the source's end-date metadata stops before it.
        if date == pd.Timestamp("2025-10-10"):
            yellow = yellow_line
        values = blocks[["block_id", "geometry"]].copy()
        values["corrected_density"] = values.block_id.map(adjusted)
        values.plot(ax=ax, column="corrected_density", cmap=cmap, norm=norm, missing_kwds={"color": "#9aa0a5"}, edgecolor="none", zorder=4)
        if not hz.empty:
            hz.plot(ax=ax, facecolor="none", edgecolor="#6b3f87", hatch="////", linewidth=.9, zorder=6)
        if not yellow.empty:
            yellow = yellow.copy()
            yellow["geometry"] = yellow.geometry.intersection(outer_boundary)
            yellow = yellow[~yellow.geometry.is_empty]
            yellow.plot(ax=ax, color="#f2c230", alpha=1, edgecolor="#6f5200", linewidth=.9, zorder=5)
        municipalities.boundary.plot(ax=ax, color="#737b80", linewidth=.35, alpha=.5, zorder=5)
        gpd.GeoSeries([outer_boundary], crs=TARGET_CRS).boundary.plot(ax=ax, color="#38434a", linewidth=1.1, zorder=7)
        for _, row in label_municipalities.iterrows():
            point = row.geometry.representative_point()
            name = str(row.municipality).replace(" ", "\n")
            fontsize = 8 if row["_area"] > 12_000_000 else 6.5
            ax.text(point.x, point.y, name, ha="center", va="center", fontsize=fontsize, color="#171717", weight="medium", zorder=7,
                    path_effects=[path_effects.withStroke(linewidth=2.2, foreground="white")])
        if panel_index == len(panel) - 1:
            cx, cy, length = maxx - 900, miny + 1000, 2400
            ax.annotate("", xy=(cx, cy + length), xytext=(cx, cy), arrowprops={"arrowstyle": "-|>", "color": "#263238", "lw": 1.8}, zorder=8)
            ax.text(cx, cy + length + 300, "N", ha="center", va="bottom", fontsize=12, weight="bold", color="#263238", zorder=8)
            ax.legend(handles=[
                Patch(facecolor="none", edgecolor="#6b3f87", hatch="////", label="Humanitarian zone"),
                Patch(facecolor="#f2c230", edgecolor="#6f5200", label="Yellow Line"),
            ], loc="lower right", bbox_to_anchor=(.78, .02), frameon=True, facecolor="white", edgecolor="#d8d8d8", framealpha=1, fontsize=11)
        title = DISPLAY_TITLES.get(label, label)
        ax.set(xlim=(minx-pad, maxx+pad), ylim=(miny-pad, maxy+pad))
        ax.set_title(f"{title} — {date.day} {date:%B} {date.year}", fontsize=14, y=1.025, pad=7)
        ax.set_axis_off()
    color_ax = fig.add_axes([.955, .15, .014, .70])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax, orientation="vertical")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=10)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    colorbar.set_label("Tent density per km²", fontsize=11, labelpad=10)
    fig.subplots_adjust(left=.005, right=.94, bottom=.005, top=.985, wspace=0, hspace=.16)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
