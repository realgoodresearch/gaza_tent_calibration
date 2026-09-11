"""Plot corrected weekly tent counts for selected Gaza municipalities."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "outputs/tent_counts/municipality_counts_by_week.csv"
OUTPUT_PATH = REPO_ROOT / "outputs/visualisation/municipality_corrected_tent_counts.png"
OUTPUT_DATA_PATH = REPO_ROOT / "outputs/visualisation/municipality_corrected_tent_counts.csv"
START_DATE = pd.Timestamp("2023-10-01")
END_DATE = pd.Timestamp("2026-06-30")

MUNICIPALITIES = {
    "Rafah": "Rafah",
    "Khan Yunis": "Khan Yunis",
    "Deir al Balah": "Deir el Balah",
    "Gaza": "Gaza",
    "Jabalya": "Jabalia",
    "Beit Lahiya": "Beit Lahiya",
    "Beit Hanoun": "Beit Hanoun",
}
COLORS = {
    "Rafah": "#264653",
    "Khan Yunis": "#8c3b3b",
    "Deir el Balah": "#3f5f8a",
    "Gaza": "#604c78",
    "Jabalia": "#526b52",
    "Beit Lahiya": "#80623a",
    "Beit Hanoun": "#4f6268",
}
EVENTS = [
    ("2023-10-13", "North Gaza evacuation order"),
    ("2024-05-06", "Rafah evacuation order"),
    ("2025-01-19", "First ceasefire"),
    ("2025-03-18", "End of ceasefire"),
    ("2025-09-09", "Gaza City evacuation order"),
    ("2025-10-10", "Second ceasefire"),
]
CLOUD_COVER_WEEKS = {"20260518-20260524", "20260601-20260607"}


def main() -> None:
    data = pd.read_csv(INPUT_PATH)
    required = {
        "Municipality", "week_key", "corrected_count_mean",
        "corrected_count_p2_5", "corrected_count_p97_5",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data[data.Municipality.isin(MUNICIPALITIES)].copy()
    data["date"] = pd.to_datetime(
        data.week_key.str.split("-").str[1], format="%Y%m%d", errors="coerce"
    )
    for column in ("corrected_count_mean", "corrected_count_p2_5", "corrected_count_p97_5"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=[
        "date", "corrected_count_mean", "corrected_count_p2_5", "corrected_count_p97_5"
    ])
    data["display_municipality"] = data.Municipality.map(MUNICIPALITIES)
    data = data[data.date.between(START_DATE, END_DATE)].sort_values(["display_municipality", "date"])
    value_columns = ["corrected_count_mean", "corrected_count_p2_5", "corrected_count_p97_5"]
    data.loc[data.week_key.isin(CLOUD_COVER_WEEKS), value_columns] = pd.NA
    smoothed = []
    for municipality, group in data.groupby("display_municipality", sort=False):
        group = group.sort_values("date").copy()
        for column in value_columns:
            smooth_column = f"{column}_smoothed"
            group[smooth_column] = group[column].rolling(3, center=True, min_periods=1).mean()
            # Explicitly preserve cloud-covered observations as gaps.
            group.loc[group.week_key.isin(CLOUD_COVER_WEEKS), smooth_column] = pd.NA
        smoothed.append(group)
    data = pd.concat(smoothed, ignore_index=True)
    data.to_csv(OUTPUT_DATA_PATH, index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.spines.top": False,
    })
    fig, axes = plt.subplots(
        len(COLORS), 1, figsize=(7, 10), sharex=True, facecolor="white",
        gridspec_kw={"hspace": 0.5},
    )
    for ax, (display_name, color) in zip(axes, COLORS.items()):
        series = data[data.display_municipality == display_name]
        if series.empty:
            continue
        ax.fill_between(
            series.date,
            series.corrected_count_p2_5_smoothed,
            series.corrected_count_p97_5_smoothed,
            color=color, alpha=0.12, linewidth=0, zorder=1,
        )
        ax.plot(
            series.date, series.corrected_count_mean_smoothed,
            color=color, lw=2.0, zorder=2,
        )
        ax.set_title(display_name, loc="left", color=color)
        ax.set_xlim(START_DATE, END_DATE)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#d9d9d9", lw=0.7, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        for date_text, _ in EVENTS:
            ax.axvline(
                pd.Timestamp(date_text), color="#777777", lw=0.8,
                ls=(0, (3, 3)), alpha=0.65, zorder=0,
            )

    bottom_ax = axes[-1]
    bottom_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    for tick_label in bottom_ax.get_xticklabels():
        tick_label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 1.0})
        tick_label.set_zorder(5)

    for index, (date_text, label) in enumerate(EVENTS):
        date = pd.Timestamp(date_text)
        bottom_ax.annotate(
            label,
            xy=(date, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -42 if index % 2 == 0 else -66),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.5,
            color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            arrowprops={
                "arrowstyle": "-", "color": "#777777", "lw": 0.8,
                "linestyle": (0, (3, 3)),
            },
            annotation_clip=False,
            zorder=5,
        )

    fig.supylabel("Corrected tents", x=-0.02, fontsize=11)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.98, bottom=0.10)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")
    print(f"Saved {OUTPUT_DATA_PATH}")


if __name__ == "__main__":
    main()
