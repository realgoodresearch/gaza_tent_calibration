from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge calibrated corrected predictions (pred_mean) into the manual-validation "
            "comparison CSV using stable keys with safety checks."
        )
    )
    parser.add_argument(
        "--manual-comparison-csv",
        type=Path,
        default=Path("outputs/validation/manual_validation_results_with_raw_and_postprocessed_predictions.csv"),
        help="Manual comparison CSV containing h5_source/h5_index/date/center_x/center_y.",
    )
    parser.add_argument(
        "--holdout-csv",
        type=Path,
        default=Path("src/calibration/artifacts/holdout_predictions.csv"),
        help="Calibration holdout predictions CSV containing pred_mean.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/validation/manual_validation_results_with_raw_postprocessed_and_corrected.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--corrected-col",
        type=str,
        default="prediction_count_corrected_hurdle",
        help="Output column name for corrected calibrated prediction.",
    )
    parser.add_argument(
        "--coord-tol-m",
        type=float,
        default=1e-3,
        help="Absolute tolerance for center_x/center_y consistency checks (in projected units).",
    )
    parser.add_argument(
        "--date-strict",
        action="store_true",
        help="Require exact same normalized date between sources; mismatches are dropped from corrected merge.",
    )
    return parser


def _prep_manual(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "h5_source" not in out.columns:
        out["h5_source"] = "default"
    out["h5_source"] = out["h5_source"].astype(str).str.strip().str.lower()
    out["h5_index"] = pd.to_numeric(out["h5_index"], errors="coerce").astype("Int64")
    out["_date_norm"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["center_x"] = pd.to_numeric(out["center_x"], errors="coerce")
    out["center_y"] = pd.to_numeric(out["center_y"], errors="coerce")
    return out


def _prep_holdout(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "h5_source" not in out.columns:
        out["h5_source"] = "default"
    out["h5_source"] = out["h5_source"].astype(str).str.strip().str.lower()
    out["h5_index"] = pd.to_numeric(out["h5_index"], errors="coerce").astype("Int64")
    out["_date_norm_holdout"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["center_x_holdout"] = pd.to_numeric(out["center_x"], errors="coerce")
    out["center_y_holdout"] = pd.to_numeric(out["center_y"], errors="coerce")
    return out


def main() -> None:
    args = build_parser().parse_args()

    manual = pd.read_csv(args.manual_comparison_csv)
    holdout = pd.read_csv(args.holdout_csv)

    if "pred_mean" not in holdout.columns:
        raise ValueError("Holdout CSV must contain 'pred_mean'.")

    manual = _prep_manual(manual)
    holdout = _prep_holdout(holdout)

    key_cols = ["h5_source", "h5_index"]

    hold_keep = key_cols + ["pred_mean", "_date_norm_holdout", "center_x_holdout", "center_y_holdout"]
    hold_slim = holdout[hold_keep].copy()

    dup = hold_slim.duplicated(key_cols, keep=False)
    if dup.any():
        raise ValueError(
            "Holdout CSV has duplicate (h5_source, h5_index) keys; cannot safely merge. "
            f"Duplicate rows: {int(dup.sum())}"
        )

    merged = manual.merge(hold_slim, on=key_cols, how="left")

    date_match = merged["_date_norm"].eq(merged["_date_norm_holdout"])
    x_match = np.isclose(merged["center_x"], merged["center_x_holdout"], equal_nan=False, atol=float(args.coord_tol_m), rtol=0.0)
    y_match = np.isclose(merged["center_y"], merged["center_y_holdout"], equal_nan=False, atol=float(args.coord_tol_m), rtol=0.0)
    coord_match = x_match & y_match

    has_pred = merged["pred_mean"].notna()
    valid = has_pred & coord_match
    if args.date_strict:
        valid = valid & date_match

    merged[args.corrected_col] = np.where(valid, pd.to_numeric(merged["pred_mean"], errors="coerce"), np.nan)
    merged["corrected_merge_status"] = np.select(
        [
            ~has_pred,
            has_pred & ~coord_match,
            has_pred & coord_match & ~date_match,
            valid,
        ],
        [
            "no_holdout_match",
            "coord_mismatch",
            "date_mismatch",
            "ok",
        ],
        default="no_holdout_match",
    )

    drop_cols = ["_date_norm", "_date_norm_holdout", "center_x_holdout", "center_y_holdout", "pred_mean"]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)

    total = len(merged)
    ok = int((merged["corrected_merge_status"] == "ok").sum())
    print(f"Saved: {args.output_csv}")
    print(f"Rows: {total}")
    print(f"Corrected merged (status=ok): {ok}")
    print("Status counts:")
    print(merged["corrected_merge_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

