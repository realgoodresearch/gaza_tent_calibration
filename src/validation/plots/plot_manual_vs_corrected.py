from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot manual tent counts vs corrected predicted counts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/validation/manual_validation_results_with_raw_and_postprocessed_predictions.csv"),
        help="CSV with manual and raw/postprocessed prediction columns.",
    )
    parser.add_argument(
        "--manual-col",
        type=str,
        default="tent_count",
        help="Column name for manual count.",
    )
    parser.add_argument(
        "--corrected-col",
        type=str,
        default="prediction_count_corrected_hurdle",
        help="Column name for corrected predicted count.",
    )
    parser.add_argument(
        "--uncorrected-col",
        type=str,
        default="prediction_count_original",
        help="Column name for uncorrected model-predicted count.",
    )
    parser.add_argument(
        "--postprocessed-col",
        type=str,
        default="prediction_count_postprocessed",
        help="Column name for postprocessed raw prediction count (DBSCAN + yellow-line).",
    )
    parser.add_argument(
        "--holdout-csv",
        type=Path,
        default=Path("src/calibration/artifacts/holdout_predictions.csv"),
        help="Holdout predictions CSV used for automatic corrected-column merge when needed.",
    )
    parser.add_argument(
        "--disable-auto-corrected-merge",
        action="store_true",
        help="Disable auto-merge of corrected values from holdout CSV when corrected column is missing.",
    )
    parser.add_argument(
        "--coord-tol-m",
        type=float,
        default=1e-3,
        help="Tolerance for center_x/center_y safety check in automatic corrected merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/validation/plots/manual_vs_corrected_predicted_count.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--output-linear",
        type=Path,
        default=Path("src/validation/plots/manual_vs_corrected_predicted_count_linear.png"),
        help="Output image path for linear-scale version.",
    )
    return parser


def _draw_two_panel(
    plot_df: pd.DataFrame,
    manual_col: str,
    corrected_col: str,
    uncorrected_col: str,
    postprocessed_col: str | None,
    output_path: Path,
    log_scale: bool,
) -> tuple[float, float, float, float, float | None, float | None]:
    manual = plot_df[manual_col]
    corrected = plot_df[corrected_col]
    uncorrected = plot_df[uncorrected_col]
    has_post = postprocessed_col is not None and postprocessed_col in plot_df.columns
    postprocessed = plot_df[postprocessed_col] if has_post else None
    max_axis = float(max(manual.max(), corrected.max(), uncorrected.max(), (postprocessed.max() if has_post and postprocessed is not None else 0)))

    manual_vals = manual.to_numpy(dtype=float)
    corrected_vals = corrected.to_numpy(dtype=float)
    uncorrected_vals = uncorrected.to_numpy(dtype=float)

    # Fit manual ~ predicted (x=predicted, y=manual)
    corr_slope, corr_intercept = np.polyfit(corrected_vals, manual_vals, deg=1)
    raw_slope, raw_intercept = np.polyfit(uncorrected_vals, manual_vals, deg=1)
    post_slope = None
    post_intercept = None
    if has_post and postprocessed is not None:
        post_vals = postprocessed.to_numpy(dtype=float)
        post_slope, post_intercept = np.polyfit(post_vals, manual_vals, deg=1)
    x_line = np.linspace(0.0, max_axis, 200)
    y_fit_corrected = np.clip(corr_intercept + corr_slope * x_line, 0.0, None)
    y_fit_uncorrected = np.clip(raw_intercept + raw_slope * x_line, 0.0, None)
    y_fit_post = np.clip((post_intercept or 0.0) + (post_slope or 0.0) * x_line, 0.0, None) if has_post else None

    eps = 1.0
    if log_scale:
        x_corr_plot = corrected + eps
        x_raw_plot = uncorrected + eps
        x_post_plot = (postprocessed + eps) if has_post and postprocessed is not None else None
        y_plot = manual + eps
        x_line_plot = x_line + eps
        y_fit_corrected_plot = y_fit_corrected + eps
        y_fit_uncorrected_plot = y_fit_uncorrected + eps
        y_fit_post_plot = (y_fit_post + eps) if has_post and y_fit_post is not None else None
    else:
        x_corr_plot = corrected
        x_raw_plot = uncorrected
        x_post_plot = postprocessed if has_post and postprocessed is not None else None
        y_plot = manual
        x_line_plot = x_line
        y_fit_corrected_plot = y_fit_corrected
        y_fit_uncorrected_plot = y_fit_uncorrected
        y_fit_post_plot = y_fit_post if has_post else None

    if has_post:
        fig, (ax_raw, ax_post, ax_corr) = plt.subplots(1, 3, figsize=(18, 6.5), sharex=True, sharey=True)
    else:
        fig, (ax_raw, ax_corr) = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
        ax_post = None

    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if sst <= 1e-12:
            return float("nan")
        sse = float(np.sum((y_true - y_pred) ** 2))
        return 1.0 - (sse / sst)

    raw_r2 = _r2(manual_vals, raw_intercept + raw_slope * uncorrected_vals)
    corr_r2 = _r2(manual_vals, corr_intercept + corr_slope * corrected_vals)
    post_r2 = None
    if has_post and postprocessed is not None and post_slope is not None and post_intercept is not None:
        post_r2 = _r2(manual_vals, post_intercept + post_slope * postprocessed.to_numpy(dtype=float))

    ax_raw.scatter(x_raw_plot, y_plot, alpha=0.5, s=18, label="Uncorrected points")
    ax_raw.plot(x_line_plot, x_line_plot, linestyle="--", linewidth=1.4, label="1:1 line")
    ax_raw.plot(x_line_plot, y_fit_uncorrected_plot, linewidth=1.9, label=f"Fitted (manual={raw_slope:.2f}·pred+{raw_intercept:.2f})")
    ax_raw.set_title("Uncorrected vs Manual")
    ax_raw.grid(True, alpha=0.2)
    ax_raw.legend(loc="upper left")

    if has_post and ax_post is not None and x_post_plot is not None:
        ax_post.scatter(x_post_plot, y_plot, alpha=0.5, s=18, label="Postprocessed raw points")
        ax_post.plot(x_line_plot, x_line_plot, linestyle="--", linewidth=1.4, label="1:1 line")
        ax_post.plot(
            x_line_plot,
            y_fit_post_plot,
            linewidth=1.9,
            label=f"Fitted (manual={post_slope:.2f}·pred+{post_intercept:.2f})",
        )
        ax_post.set_title("Postprocessed Raw vs Manual")
        ax_post.grid(True, alpha=0.2)
        ax_post.legend(loc="upper left")

    ax_corr.scatter(x_corr_plot, y_plot, alpha=0.5, s=18, label="Corrected points")
    ax_corr.plot(x_line_plot, x_line_plot, linestyle="--", linewidth=1.4, label="1:1 line")
    ax_corr.plot(x_line_plot, y_fit_corrected_plot, linewidth=1.9, label=f"Fitted (manual={corr_slope:.2f}·pred+{corr_intercept:.2f})")
    ax_corr.set_title("Corrected vs Manual")
    ax_corr.grid(True, alpha=0.2)
    ax_corr.legend(loc="upper left")

    if log_scale:
        axes = [ax_raw, ax_corr] if ax_post is None else [ax_raw, ax_post, ax_corr]
        for axis in axes:
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(eps, (max_axis * 1.03) + eps)
            axis.set_ylim(eps, (max_axis * 1.03) + eps)
        xlab = "Predicted tent count (+1), log scale"
        ylab = "Manual tent count (+1), log scale"
    else:
        axes = [ax_raw, ax_corr] if ax_post is None else [ax_raw, ax_post, ax_corr]
        for axis in axes:
            axis.set_xlim(0, max_axis * 1.03)
            axis.set_ylim(0, max_axis * 1.03)
        xlab = "Predicted tent count"
        ylab = "Manual tent count"

    axes = [ax_raw, ax_corr] if ax_post is None else [ax_raw, ax_post, ax_corr]
    for axis in axes:
        axis.set_xlabel(xlab)
        axis.set_ylabel(ylab)
    fig.suptitle(
        "Holdout Tiles: Manual vs Predicted Tent Counts"
        + (" (log scale)" if log_scale else " (linear scale)"),
        y=1.02,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return corr_slope, corr_intercept, raw_slope, raw_intercept, post_slope, post_intercept


def _auto_attach_corrected(
    df: pd.DataFrame,
    corrected_col: str,
    holdout_csv: Path,
    coord_tol_m: float,
) -> pd.DataFrame:
    needed = {"h5_index", "date", "center_x", "center_y"}
    if not needed.issubset(df.columns):
        print(f"Auto-merge skipped: input missing required columns {sorted(needed - set(df.columns))}")
        return df
    if not holdout_csv.exists():
        print(f"Auto-merge skipped: holdout CSV not found at {holdout_csv}")
        return df

    hold = pd.read_csv(holdout_csv)
    if "pred_mean" not in hold.columns:
        print("Auto-merge skipped: holdout CSV missing pred_mean")
        return df

    out = df.copy()
    if "h5_source" not in out.columns:
        out["h5_source"] = "default"
    if "h5_source" not in hold.columns:
        hold["h5_source"] = "default"

    out["h5_source"] = out["h5_source"].astype(str).str.strip().str.lower()
    hold["h5_source"] = hold["h5_source"].astype(str).str.strip().str.lower()

    out["h5_index"] = pd.to_numeric(out["h5_index"], errors="coerce").astype("Int64")
    hold["h5_index"] = pd.to_numeric(hold["h5_index"], errors="coerce").astype("Int64")

    out["_date_norm"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    hold["_date_norm_hold"] = pd.to_datetime(hold["date"], errors="coerce").dt.normalize()
    out["center_x"] = pd.to_numeric(out["center_x"], errors="coerce")
    out["center_y"] = pd.to_numeric(out["center_y"], errors="coerce")
    hold["center_x_hold"] = pd.to_numeric(hold["center_x"], errors="coerce")
    hold["center_y_hold"] = pd.to_numeric(hold["center_y"], errors="coerce")

    key = ["h5_source", "h5_index"]
    hold_slim = hold[key + ["pred_mean", "_date_norm_hold", "center_x_hold", "center_y_hold"]].copy()
    dup = hold_slim.duplicated(key, keep=False)
    if dup.any():
        raise ValueError(
            "Auto-merge aborted: holdout CSV has duplicate (h5_source,h5_index) keys; "
            f"duplicates={int(dup.sum())}"
        )

    merged = out.merge(hold_slim, on=key, how="left")
    has_pred = merged["pred_mean"].notna()
    date_match = merged["_date_norm"].eq(merged["_date_norm_hold"])
    x_match = np.isclose(merged["center_x"], merged["center_x_hold"], equal_nan=False, atol=float(coord_tol_m), rtol=0.0)
    y_match = np.isclose(merged["center_y"], merged["center_y_hold"], equal_nan=False, atol=float(coord_tol_m), rtol=0.0)
    valid = has_pred & date_match & x_match & y_match

    merged[corrected_col] = np.where(valid, pd.to_numeric(merged["pred_mean"], errors="coerce"), np.nan)
    ok = int(np.sum(valid))
    print(f"Auto-merged corrected values into '{corrected_col}': {ok} rows")

    drop_cols = ["_date_norm", "_date_norm_hold", "center_x_hold", "center_y_hold", "pred_mean"]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
    return merged


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.input)

    if args.corrected_col not in df.columns and not args.disable_auto_corrected_merge:
        df = _auto_attach_corrected(
            df=df,
            corrected_col=args.corrected_col,
            holdout_csv=args.holdout_csv,
            coord_tol_m=args.coord_tol_m,
        )

    if args.manual_col not in df.columns:
        raise ValueError(f"Manual column not found: {args.manual_col}")
    if args.corrected_col not in df.columns:
        raise ValueError(f"Corrected column not found: {args.corrected_col}")
    if args.uncorrected_col not in df.columns:
        raise ValueError(f"Uncorrected column not found: {args.uncorrected_col}")
    has_post = args.postprocessed_col in df.columns

    keep_cols = [args.manual_col, args.corrected_col, args.uncorrected_col]
    if has_post:
        keep_cols.append(args.postprocessed_col)
    keep_cols = list(dict.fromkeys(keep_cols))
    plot_df = df[keep_cols].copy()
    plot_df[args.manual_col] = pd.to_numeric(plot_df[args.manual_col], errors="coerce")
    plot_df[args.corrected_col] = pd.to_numeric(plot_df[args.corrected_col], errors="coerce")
    plot_df[args.uncorrected_col] = pd.to_numeric(plot_df[args.uncorrected_col], errors="coerce")
    if has_post:
        plot_df[args.postprocessed_col] = pd.to_numeric(plot_df[args.postprocessed_col], errors="coerce")
    plot_df = plot_df.dropna().reset_index(drop=True)

    if plot_df.empty:
        raise ValueError("No valid rows to plot after dropping NaNs")

    corr_slope, corr_intercept, raw_slope, raw_intercept, post_slope, post_intercept = _draw_two_panel(
        plot_df=plot_df,
        manual_col=args.manual_col,
        corrected_col=args.corrected_col,
        uncorrected_col=args.uncorrected_col,
        postprocessed_col=(args.postprocessed_col if has_post else None),
        output_path=args.output,
        log_scale=True,
    )
    _draw_two_panel(
        plot_df=plot_df,
        manual_col=args.manual_col,
        corrected_col=args.corrected_col,
        uncorrected_col=args.uncorrected_col,
        postprocessed_col=(args.postprocessed_col if has_post else None),
        output_path=args.output_linear,
        log_scale=False,
    )

    print(f"Saved plot: {args.output}")
    print(f"Saved plot: {args.output_linear}")
    print(f"Rows plotted: {len(plot_df)}")
    print(f"Fitted corrected (manual~pred): manual = {corr_slope:.6f} * pred + {corr_intercept:.6f}")
    print(f"Fitted uncorrected (manual~pred): manual = {raw_slope:.6f} * pred + {raw_intercept:.6f}")
    if has_post and post_slope is not None and post_intercept is not None:
        print(f"Fitted postprocessed raw (manual~pred): manual = {post_slope:.6f} * pred + {post_intercept:.6f}")
    else:
        print("Postprocessed raw column not found; generated two-panel plot.")


if __name__ == "__main__":
    main()
