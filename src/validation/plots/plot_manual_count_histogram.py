from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot histogram of manual tent counts.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("validation/prediction_accuracy_results.csv"),
        help="Input CSV containing manual tent counts.",
    )
    parser.add_argument(
        "--manual-col",
        type=str,
        default="tent_count",
        help="Column name for manual tent counts.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/manual_count_histogram.png"),
        help="Output image path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.input)
    if args.manual_col not in df.columns:
        raise ValueError(f"Column not found: {args.manual_col}")

    counts = pd.to_numeric(df[args.manual_col], errors="coerce").dropna()
    if counts.empty:
        raise ValueError("No valid manual count values found")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(counts, bins=args.bins, alpha=0.85)
    ax.set_title("Histogram of Manual Tent Counts")
    ax.set_xlabel("Manual tent count")
    ax.set_ylabel("Number of tiles")
    ax.grid(True, axis="y", alpha=0.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"Saved plot: {args.output}")
    print(f"Rows plotted: {len(counts)}")


if __name__ == "__main__":
    main()

