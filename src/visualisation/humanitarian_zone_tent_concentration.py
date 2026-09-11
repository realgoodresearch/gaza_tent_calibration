"""Plot humanitarian-zone tent concentration and relative displacement burden."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSON = REPO_ROOT / "outputs/analysis/humanitarian_zone_tent_concentration.json"
ZONE_JSON = REPO_ROOT / "data/spatial_context/humanitarian_zone.json"
OUTPUT_DIR = REPO_ROOT / "outputs/visualisation"
OUTPUT_PNG = OUTPUT_DIR / "humanitarian_zone_tent_concentration.png"

START_DATE = pd.Timestamp("2023-10-01")
END_DATE = pd.Timestamp("2026-06-30")
PEAK_DATE = pd.Timestamp("2024-07-14")
MAJOR_EVENTS = [
    ("2023-10-13", "North Gaza evacuation order"),
    ("2024-05-06", "Rafah evacuation order"),
    ("2025-01-19", "First ceasefire"),
    ("2025-03-18", "End of ceasefire"),
    ("2025-09-09", "Gaza City evacuation order"),
    ("2025-10-10", "Second ceasefire"),
]

PURPLE = "#6f3c8f"
PURPLE_LIGHT = "#c9afd8"
GOLD = "#b77b16"
TEAL = "#087f8c"
GRID = "#d9d9d9"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=INPUT_JSON)
    parser.add_argument("--output-png", type=Path, default=OUTPUT_PNG)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _load(path: Path) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(path.read_text())
    frame = pd.DataFrame(payload["weeks"])
    frame["date"] = pd.to_datetime(frame.week_end)
    frame = frame[frame.date.between(START_DATE, END_DATE)].sort_values("date")
    if frame.empty:
        raise ValueError(f"No concentration observations between {START_DATE.date()} and {END_DATE.date()}")
    return frame, payload


def _event_lines(axes: list[plt.Axes]) -> None:
    for date_text, _ in MAJOR_EVENTS:
        date = pd.Timestamp(date_text)
        for axis in axes:
            axis.axvline(date, color="#777777", lw=0.8, ls=(0, (3, 3)), alpha=0.65, zorder=0)


def _unofficial_periods() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    payload = json.loads(ZONE_JSON.read_text())
    intervals = []
    for feature in payload["features"]:
        properties = feature.get("properties", {})
        start = pd.to_datetime(properties.get("safe_date"), errors="coerce")
        end = pd.to_datetime(properties.get("end_date"), errors="coerce")
        if pd.notna(start) and pd.notna(end):
            intervals.append((start, end))
    intervals.sort()
    merged: list[list[pd.Timestamp]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + pd.Timedelta(days=1):
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gaps = []
    for left, right in zip(merged, merged[1:]):
        start = max(left[1] + pd.Timedelta(days=1), START_DATE)
        end = min(right[0] - pd.Timedelta(days=1), END_DATE)
        if start <= end:
            gaps.append((start, end))
    return gaps


def main() -> None:
    args = build_parser().parse_args()
    data, _ = _load(args.input_json)
    ci = pd.DataFrame(
        data.percent_of_tents_in_humanitarian_zone_95_cri.tolist(),
        columns=["ci_low", "ci_high"], index=data.index,
    )
    data["segment"] = data.date.diff().dt.days.gt(90).cumsum()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
    })
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(12, 10.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.1], "hspace": 0.18},
    )
    fig.patch.set_facecolor("white")
    unofficial_periods = _unofficial_periods()
    for axis in (ax_b,):
        for start, end in unofficial_periods:
            axis.axvspan(start, end, color="#eeeeee", alpha=0.9, zorder=-2)

    # Panel A: primary tent share with uncertainty, plus the zone's land share.
    for _, segment in data.groupby("segment"):
        ax_a.fill_between(
            segment.date, ci.loc[segment.index, "ci_low"], ci.loc[segment.index, "ci_high"],
            color=PURPLE_LIGHT, alpha=0.55, linewidth=0,
        )
        ax_a.plot(
            segment.date, segment.percent_of_tents_in_humanitarian_zone,
            color=PURPLE, lw=2.2, marker="o", ms=4.3,
        )
    ax_a.set_ylim(0, 115)
    ax_a.set_ylabel("Tents inside humanitarian zone (%)", color=PURPLE)
    ax_a.tick_params(axis="y", colors=PURPLE)
    ax_a.grid(axis="y", color=GRID, lw=0.7, alpha=0.75)
    ax_a.set_title("A   Concentration", loc="left")

    ax_area = ax_a.twinx()
    ax_area.set_zorder(1)
    ax_a.set_zorder(2)
    ax_a.patch.set_visible(False)
    ax_area.spines["top"].set_visible(False)
    for start, end in unofficial_periods:
        ax_area.axvspan(start, end, color="#eeeeee", alpha=0.9, zorder=0)
    for _, segment in data.groupby("segment"):
        ax_area.fill_between(
            segment.date,
            0,
            segment.humanitarian_zone_percent_of_gaza_land_area,
            color=GOLD,
            alpha=0.28,
            linewidth=0,
            zorder=1,
        )
    area_max = max(25, float(data.humanitarian_zone_percent_of_gaza_land_area.max()) * 1.25)
    ax_area.set_ylim(0, area_max)
    ax_area.set_ylabel("Gaza land inside humanitarian zone (%)", color=GOLD)
    ax_area.tick_params(axis="y", colors=GOLD)
    ax_area.spines["right"].set_color(GOLD)

    handles = [
        Line2D([0], [0], color=PURPLE, lw=2.2, marker="o", ms=4, label="Tent share"),
        Line2D([0], [0], color=PURPLE_LIGHT, lw=7, alpha=0.55, label="95% CrI"),
        Patch(facecolor=GOLD, edgecolor="none", alpha=0.28, label="Zone land share"),
        Patch(facecolor="#eeeeee", edgecolor="none", label="No official zone; previous borders used"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.985), frameon=False, ncol=4)

    # Panel B: zone-versus-remainder density ratio.
    ratio_col = "tent_density_ratio_zone_vs_remainder_of_covered_gaza"
    for _, segment in data.groupby("segment"):
        ax_b.plot(segment.date, segment[ratio_col], color=TEAL, lw=2.2, marker="o", ms=4.3)
    ax_b.set_ylim(0, max(40, float(data[ratio_col].max()) * 1.13))
    ax_b.set_ylabel(
        "Tent-density ratio\n"
        "(density of tents in humanitarian zone vs. rest of Gaza Strip)",
        fontsize=8.5,
    )
    ax_b.set_title("B   Relative displacement burden", loc="left")
    ax_b.grid(axis="y", color=GRID, lw=0.7, alpha=0.75)

    peak = data.loc[(data.date - PEAK_DATE).abs().idxmin()]
    ax_b.scatter([peak.date], [peak[ratio_col]], s=54, color=TEAL, edgecolor="white", zorder=4)
    ax_b.annotate(
        f"{peak[ratio_col]:.1f}× peak\n{peak.date:%d %b %Y}",
        xy=(peak.date, peak[ratio_col]), xytext=(35, -8), textcoords="offset points",
        ha="left", va="top", color=TEAL,
        arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.0},
    )

    _event_lines([ax_a, ax_b])
    for index, (date_text, label) in enumerate(MAJOR_EVENTS):
        date = pd.Timestamp(date_text)
        ax_b.annotate(
            label, xy=(date, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -42 if index % 2 == 0 else -66), textcoords="offset points",
            rotation=0, ha="center", va="top", fontsize=7.5, color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            arrowprops={
                "arrowstyle": "-", "color": "#777777", "lw": 0.8,
                "linestyle": (0, (3, 3)),
            },
            annotation_clip=False,
            zorder=5,
        )

    ax_b.set_xlim(START_DATE, END_DATE)
    ax_b.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    for tick_label in ax_b.get_xticklabels():
        tick_label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 1.0})
        tick_label.set_zorder(5)
    ax_b.spines["right"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.15, right=0.90, top=0.92, bottom=0.20)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output_png}")

if __name__ == "__main__":
    main()
