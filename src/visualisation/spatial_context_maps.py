"""Map the static spatial-context layers used in the displacement analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape


REPO_ROOT = Path(__file__).resolve().parents[2]
MUNICIPALITIES = REPO_ROOT / "data/gaza_boundaries/municipal/GazaStrip_MunicipalBoundaries.shp"
LAND_USE = REPO_ROOT / "src/validation/land_use_classification/prewar_gaza_land_use.tif"
EVAC_ORDERS = REPO_ROOT / "data/military_activity/evac-orders-grid.geojson"
BLOCKS = REPO_ROOT / "data/gaza_boundaries/dash_population_blocks.geojson"
INCIDENTS = REPO_ROOT / "data/incidents/gaza_incidents.csv"
HUMANITARIAN_ZONE = REPO_ROOT / "data/spatial_context/humanitarian_zone.json"
AGRICULTURE = REPO_ROOT / "data/spatial_context/agriculture.json"
GREENHOUSES = REPO_ROOT / "data/spatial_context/greenhouses-combined.json"
DESTRUCTION = REPO_ROOT / "data/spatial_context/destruction.json"
OUTPOSTS = REPO_ROOT / "data/spatial_context/outposts.json"
TANKS = REPO_ROOT / "data/military_activity/tanks.geojson"
BERMS = REPO_ROOT / "data/military_activity/berms.geojson"
YELLOW_LINE = REPO_ROOT / "data/spatial_context/yellow-line-zone.json"
OUTPUT = REPO_ROOT / "outputs/visualisation/spatial_context_maps.png"
TARGET_CRS = "EPSG:32636"

LAND_USE_COLORS = ["#f4e7a1", "#28743c", "#8d9297"]
NAVY, NAVY_EDGE = "#174a7e", "#082d50"
RED, RED_EDGE = "#d73027", "#78110c"
PURPLE, PURPLE_EDGE = "#8e63a9", "#4f2868"
LIME, LIME_EDGE = "#79cf3a", "#367817"
BLACK, BLACK_EDGE = "#252525", "#000000"
ORANGE, ORANGE_EDGE = "#ed8b2c", "#8f4300"
YELLOW, YELLOW_EDGE = "#f2d33c", "#806b00"


def _read(path: Path) -> gpd.GeoDataFrame:
    data = gpd.read_file(path)
    if data.crs is None:
        data = data.set_crs("EPSG:4326")
    return data.to_crs(TARGET_CRS)


def _read_permissive_geojson(path: Path) -> gpd.GeoDataFrame:
    """Read valid geometries while skipping malformed one-coordinate lines."""
    payload = json.loads(path.read_text())
    rows = []
    for feature in payload.get("features", []):
        try:
            geometry = shape(feature["geometry"])
        except Exception:
            # The source contains a few one-coordinate LineStrings, which are
            # invalid GeoJSON line geometries and cannot be rendered.
            continue
        if geometry.is_empty:
            continue
        rows.append({**feature.get("properties", {}), "geometry": geometry})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(TARGET_CRS)


def _solid_territory(geometry):
    if isinstance(geometry, Polygon):
        return Polygon(geometry.exterior)
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon([Polygon(part.exterior) for part in geometry.geoms])
    if isinstance(geometry, GeometryCollection):
        polygons = [part for part in geometry.geoms if part.geom_type in {"Polygon", "MultiPolygon"}]
        return _solid_territory(gpd.GeoSeries(polygons).union_all())
    return geometry


def _merged_polygons(layer: gpd.GeoDataFrame, boundary) -> gpd.GeoDataFrame:
    """Clip and dissolve polygons so overlaps have no internal outlines."""
    polygons = layer[layer.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if polygons.empty:
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    polygons["geometry"] = shapely.make_valid(polygons.geometry.to_numpy())
    polygons["geometry"] = polygons.geometry.intersection(boundary)
    polygons = polygons[~polygons.geometry.is_empty & polygons.geometry.notna()]
    if polygons.empty:
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    return gpd.GeoDataFrame(geometry=[polygons.geometry.union_all()], crs=TARGET_CRS)


def _base(ax: plt.Axes, outer_boundary, municipalities: gpd.GeoDataFrame) -> None:
    gpd.GeoSeries([outer_boundary], crs=TARGET_CRS).plot(
        ax=ax, color="#faf9f5", edgecolor="none", zorder=0
    )


def _foreground(ax: plt.Axes, outer_boundary, municipalities: gpd.GeoDataFrame,
                labels: gpd.GeoDataFrame) -> None:
    municipalities.boundary.plot(ax=ax, color="#555e63", linewidth=0.42, alpha=0.78, zorder=20)
    gpd.GeoSeries([outer_boundary], crs=TARGET_CRS).boundary.plot(
        ax=ax, color="#263238", linewidth=1.65, zorder=22
    )
    for _, row in labels.iterrows():
        point = row.geometry.representative_point()
        name = str(row.municipality).replace(" ", "\n")
        fontsize = 7.2 if row["_area"] > 12_000_000 else 6.2
        ax.text(
            point.x, point.y, name, ha="center", va="center", fontsize=fontsize,
            color="#171717", weight="medium", zorder=25,
            path_effects=[path_effects.withStroke(linewidth=2.4, foreground="white")],
        )


def _plot_polygon_source(ax: plt.Axes, layer: gpd.GeoDataFrame, boundary,
                         color: str, edge: str, alpha: float = 1.0) -> None:
    merged = _merged_polygons(layer, boundary)
    if not merged.empty:
        merged.plot(ax=ax, color=color, edgecolor=edge, linewidth=0.75, alpha=alpha, zorder=5)


def _plot_land_use(ax: plt.Axes, boundary) -> None:
    # The source raster is extremely high resolution. Nearest-neighbour
    # downsampling retains its categorical values without loading 5.4bn cells.
    with rasterio.open(LAND_USE) as source:
        scale = max(source.width, source.height) / 1800
        height, width = int(np.ceil(source.height / scale)), int(np.ceil(source.width / scale))
        values = source.read(1, out_shape=(height, width), resampling=Resampling.nearest)
        extent = [source.bounds.left, source.bounds.right, source.bounds.bottom, source.bounds.top]
    # Source coding: 0=nodata, 1=empty field, 2=trees/vegetation,
    # 3=buildings/roads.
    values = np.ma.masked_where((values < 1) | (values > 3), values)
    image = ax.imshow(
        values, extent=extent, origin="upper", interpolation="nearest",
        cmap=ListedColormap(LAND_USE_COLORS), norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5], 3),
        zorder=3,
    )
    # Clip the rectangular raster to the Gaza Strip.
    patch = gpd.GeoSeries([boundary], crs=TARGET_CRS).plot(ax=ax, facecolor="none").collections[-1]
    image.set_clip_path(patch.get_paths()[0], transform=ax.transData)
    patch.set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=LAND_USE_COLORS[0], label="Empty field"),
        Patch(facecolor=LAND_USE_COLORS[1], label="Trees / vegetation"),
        Patch(facecolor=LAND_USE_COLORS[2], label="Buildings / roads"),
    ], loc="lower right", frameon=False, fontsize=7)


def _incident_points() -> gpd.GeoDataFrame:
    data = pd.read_csv(INCIDENTS)
    searchable = data[["Incident Type", "Incident Tags"]].fillna("").astype(str).agg(" ".join, axis=1)
    keep = searchable.str.contains(r"air\s*strike|airstrike|attack(?:s)? on civilian", case=False, regex=True)
    selected = data.loc[keep].copy()
    lon = pd.to_numeric(selected["Lon"], errors="coerce").fillna(
        pd.to_numeric(selected["Display Longitude"], errors="coerce")
    )
    lat = pd.to_numeric(selected["Lat"], errors="coerce").fillna(
        pd.to_numeric(selected["Display Latitude"], errors="coerce")
    )
    selected = selected[lon.notna() & lat.notna()].copy()
    return gpd.GeoDataFrame(
        selected, geometry=gpd.points_from_xy(lon[lon.notna() & lat.notna()], lat[lon.notna() & lat.notna()]),
        crs="EPSG:4326",
    ).to_crs(TARGET_CRS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    all_boundaries = _read(MUNICIPALITIES)
    outer_boundary = _solid_territory(all_boundaries.geometry.union_all()).buffer(75).buffer(-75)
    municipalities = (
        all_boundaries[["NAME", "geometry"]].dropna(subset=["NAME"])
        .rename(columns={"NAME": "municipality"}).dissolve(by="municipality", as_index=False)
    )
    labels = municipalities.assign(_area=municipalities.geometry.area).nlargest(7, "_area")
    minx, miny, maxx, maxy = municipalities.total_bounds
    pad = 2600

    orders = _read(EVAC_ORDERS)
    ordered_ids = set(pd.to_numeric(orders.loc[orders["order_date"].notna(), "name"], errors="coerce").dropna())
    blocks = _read(BLOCKS)
    block_ids = pd.to_numeric(blocks["Name"], errors="coerce")
    ordered_blocks = blocks.loc[block_ids.isin(ordered_ids)]
    incidents = _incident_points()
    humanitarian = _read(HUMANITARIAN_ZONE)
    agriculture = _read(AGRICULTURE)
    greenhouses = _read(GREENHOUSES)
    destruction = _read(DESTRUCTION)
    outposts = _read(OUTPOSTS)
    tanks = _read(TANKS)
    berms = _read_permissive_geojson(BERMS)
    yellow = _read(YELLOW_LINE)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 12})
    fig, axes = plt.subplots(2, 4, figsize=(17, 12), facecolor="white",
                             gridspec_kw={"wspace": 0.02, "hspace": 0.10})
    titles = [
        "Pre-war land use", "Evacuation blocks", "Selected incidents", "Humanitarian zone",
        "Agriculture and greenhouses", "Destruction", "Military installations", "Yellow Line",
    ]
    for index, (ax, title) in enumerate(zip(axes.flat, titles)):
        _base(ax, outer_boundary, municipalities)
        if index == 0:
            _plot_land_use(ax, outer_boundary)
        elif index == 1:
            _plot_polygon_source(ax, ordered_blocks, outer_boundary, NAVY, NAVY_EDGE)
        elif index == 2:
            inside = incidents[incidents.geometry.within(outer_boundary)]
            inside.plot(ax=ax, color=RED, edgecolor=RED_EDGE, linewidth=0.25, markersize=7, alpha=0.72, zorder=6)
        elif index == 3:
            _plot_polygon_source(ax, humanitarian, outer_boundary, PURPLE, PURPLE_EDGE, alpha=0.82)
        elif index == 4:
            _plot_polygon_source(ax, agriculture, outer_boundary, LIME, LIME_EDGE, alpha=0.82)
            greenhouse_points = greenhouses[greenhouses.geometry.within(outer_boundary)]
            greenhouse_points.plot(ax=ax, color=LIME, edgecolor=LIME_EDGE, linewidth=0.15,
                                   markersize=2.5, alpha=0.7, zorder=6)
        elif index == 5:
            _plot_polygon_source(ax, destruction, outer_boundary, BLACK, BLACK_EDGE, alpha=0.9)
        elif index == 6:
            _plot_polygon_source(ax, outposts, outer_boundary, ORANGE, ORANGE_EDGE, alpha=0.9)
            tank_points = tanks[tanks.geometry.within(outer_boundary)]
            tank_points.plot(ax=ax, color=ORANGE, edgecolor=ORANGE_EDGE, linewidth=0.12,
                             markersize=2.2, alpha=0.6, zorder=6)
            berm_lines = berms[berms.geometry.geom_type.isin(["LineString", "MultiLineString"])]
            berm_lines.plot(ax=ax, color=ORANGE_EDGE, linewidth=0.45, alpha=0.8, zorder=6)
        else:
            _plot_polygon_source(ax, yellow, outer_boundary, YELLOW, YELLOW_EDGE, alpha=0.95)
        _foreground(ax, outer_boundary, municipalities, labels)
        ax.set(xlim=(minx - pad, maxx + pad), ylim=(miny - pad, maxy + pad))
        ax.set_title(f"{chr(65 + index)}   {title}", loc="left", fontweight="normal", pad=5)
        ax.set_axis_off()

    axes.flat[2].legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=RED,
               markeredgecolor=RED_EDGE, markersize=6, label="Airstrike / attack on civilians")
    ], loc="lower right", frameon=False, fontsize=7)
    axes.flat[4].legend(handles=[
        Patch(facecolor=LIME, edgecolor=LIME_EDGE, label="Agricultural land"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=LIME,
               markeredgecolor=LIME_EDGE, markersize=5, label="Greenhouse")
    ], loc="lower right", frameon=False, fontsize=7)
    axes.flat[6].legend(handles=[
        Patch(facecolor=ORANGE, edgecolor=ORANGE_EDGE, label="Outpost"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=ORANGE,
               markeredgecolor=ORANGE_EDGE, markersize=5, label="Tank"),
        Line2D([0], [0], color=ORANGE_EDGE, lw=1.2, label="Berm"),
    ], loc="lower right", frameon=False, fontsize=7)
    fig.suptitle("Spatial context used in the displacement analysis", fontsize=17,
                 fontweight="normal", x=0.02, ha="left", y=0.985)
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.01, top=0.95, wspace=0.02, hspace=0.10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
