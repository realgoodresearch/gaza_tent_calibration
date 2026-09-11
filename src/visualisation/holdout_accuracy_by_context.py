"""Plot calibrated holdout accuracy across five spatial and imaging contexts."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = REPO_ROOT / "src/calibration/artifacts/holdout_predictions.csv"
OUTPUT_PNG = REPO_ROOT / "outputs/visualisation/holdout_accuracy_by_context.png"

GRID = "#dddddd"
BIAS_CMAP = "RdBu_r"  # blue = underprediction; red = overprediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-png", type=Path, default=OUTPUT_PNG)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "tent_count", "pred_mean", "land_use_label", "hours_from_sunrise",
        "is_agriculture", "distance_to_yellow_line", "is_humanitarian_zone",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    for column in ["tent_count", "pred_mean", "hours_from_sunrise",
                   "distance_to_yellow_line", "is_agriculture", "is_humanitarian_zone"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["tent_count", "pred_mean"]).copy()
    frame["error"] = frame["pred_mean"] - frame["tent_count"]
    frame["absolute_error"] = frame["error"].abs()

    # Substantively interpretable fixed bins are preferable to sample-specific
    # tertiles and remain comparable if the holdout artifact is regenerated.
    frame["sunrise_group"] = pd.cut(
        frame["hours_from_sunrise"],
        bins=[-np.inf, 1, 4, np.inf],
        labels=["≤1 hour", ">1–4 hours", ">4 hours"],
        right=True,
    )
    distance = frame["distance_to_yellow_line"]
    frame["yellow_line_group"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    frame.loc[distance.eq(0), "yellow_line_group"] = "On line (0 km)"
    frame.loc[distance.gt(0) & distance.le(2_000), "yellow_line_group"] = "Near (>0–2 km)"
    frame.loc[distance.gt(2_000), "yellow_line_group"] = "Far (>2 km)"
    frame["yellow_line_group"] = pd.Categorical(
        frame["yellow_line_group"],
        categories=["On line (0 km)", "Near (>0–2 km)", "Far (>2 km)"],
        ordered=True,
    )
    frame["agriculture_group"] = frame["is_agriculture"].map(
        {0: "Outside agriculture", 1: "Agriculture"}
    )
    frame["humanitarian_group"] = frame["is_humanitarian_zone"].map(
        {0: "Outside zone", 1: "Inside zone"}
    )
    return frame


def _summarise(
    frame: pd.DataFrame, group_column: str, order: list[str], draws: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows = []
    for label in order:
        group = frame.loc[frame[group_column].astype("object").eq(label)]
        if group.empty:
            continue
        absolute_error = group["absolute_error"].to_numpy(float)
        boot_indices = rng.integers(0, len(group), size=(draws, len(group)))
        boot_mae = absolute_error[boot_indices].mean(axis=1)
        rows.append({
            "label": label,
            "n": len(group),
            "mae": absolute_error.mean(),
            "mae_low": np.quantile(boot_mae, 0.025),
            "mae_high": np.quantile(boot_mae, 0.975),
            "bias": group["error"].mean(),
        })
    return pd.DataFrame(rows)


def _panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    title: str,
    letter: str,
    overall_mae: float,
    bias_norm: TwoSlopeNorm,
) -> None:
    # Keep category estimates away from panel boundaries. Binary panels use a
    # deliberately tighter central spacing so their two rows do not sit at the
    # top and bottom extremes of an otherwise full-height subplot.
    if len(summary) == 2:
        y = np.array([0.36, 0.64])
    else:
        y = np.linspace(0.16, 0.84, len(summary))
    errors = np.vstack([
        summary["mae"] - summary["mae_low"],
        summary["mae_high"] - summary["mae"],
    ])
    ax.errorbar(
        summary["mae"], y, xerr=errors, fmt="none", ecolor="#777777",
        elinewidth=1.4, capsize=3, zorder=2,
    )
    ax.scatter(
        summary["mae"], y,
        c=plt.get_cmap(BIAS_CMAP)(bias_norm(summary["bias"].to_numpy())),
        s=82, edgecolor="black", linewidth=0.55,
        zorder=3,
    )
    ax.axvline(overall_mae, color="#888888", ls=(0, (3, 3)), lw=1.2, alpha=0.9)
    ax.set_yticks(y, [f"{label}  (n={n})" for label, n in zip(summary["label"], summary["n"])])
    ax.set_ylim(1, 0)
    ax.set_title(f"{letter}   {title}", loc="left", fontweight="normal")
    ax.set_xlabel("Mean absolute error (tents per tile)")
    ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    args = build_parser().parse_args()
    frame = _load(args.input_csv)
    rng = np.random.default_rng(args.seed)
    specifications = [
        ("land_use_label", ["Buildings/Roads", "Empty Field", "Trees/Vegetation"],
         "Pre-war land use"),
        ("sunrise_group", ["≤1 hour", ">1–4 hours", ">4 hours"],
         "Time since sunrise"),
        ("agriculture_group", ["Outside agriculture", "Agriculture"],
         "Pre-war agricultural land"),
        ("humanitarian_group", ["Outside zone", "Inside zone"],
         "Humanitarian-zone status"),
        ("yellow_line_group", ["On line (0 km)", "Near (>0–2 km)", "Far (>2 km)"],
         "Distance to Yellow Line"),
    ]
    summaries = [
        _summarise(frame, column, order, args.bootstrap_draws, rng)
        for column, order, _ in specifications
    ]
    all_biases = np.concatenate([summary["bias"].to_numpy() for summary in summaries])
    bias_min = float(all_biases.min())
    bias_max = float(all_biases.max())
    if not bias_min < 0 < bias_max:
        raise ValueError("Signed-bias colour scale requires biases on both sides of zero")
    bias_norm = TwoSlopeNorm(vmin=bias_min, vcenter=0, vmax=bias_max)
    overall_mae = float(frame["absolute_error"].mean())

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 11.5, "axes.labelsize": 9.5,
    })
    fig = plt.figure(figsize=(13, 10), facecolor="white")
    grid = fig.add_gridspec(3, 2, hspace=0.65, wspace=0.52)
    # Land use and acquisition time occupy row one; the two binary contexts
    # share row two; the three-level Yellow Line context spans row three.
    axes = [
        fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, :]),
    ]
    for index, (ax, summary, specification) in enumerate(zip(axes, summaries, specifications)):
        _panel(ax, summary, specification[2], chr(65 + index), overall_mae, bias_norm)
    for ax in axes:
        ax.set_xlim(left=0)

    colorbar = fig.colorbar(
        ScalarMappable(norm=bias_norm, cmap=BIAS_CMAP), ax=axes,
        orientation="horizontal", fraction=0.025, pad=0.075, aspect=55,
    )
    colorbar.set_label("Mean signed prediction bias (predicted − observed tents per tile)")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0)
    fig.suptitle(
        "Calibrated prediction accuracy in independently held-out tiles",
        x=0.08, y=0.98, ha="left", fontsize=16, fontweight="normal",
    )
    missing_distance = int(frame["distance_to_yellow_line"].isna().sum())
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output_png}")


if __name__ == "__main__":
    main()
