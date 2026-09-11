from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/tent_counts/municipality_counts_by_week.csv")
DEFAULT_OUTPUT = Path("outputs/analysis/gaza_wide_counts_by_week.csv")
DEFAULT_EXTREMES_JSON = Path("outputs/analysis/gaza_wide_counts_extremes.json")


def _extract_week_start(week_value: str) -> pd.Timestamp:
    value = str(week_value)
    if "-" in value and len(value.split("-", 1)[0]) == 8:
        return pd.to_datetime(value.split("-", 1)[0], format="%Y%m%d", errors="raise")
    if value.startswith("20") and "-W" in value:
        return pd.to_datetime(value + "-1", format="%G-W%V-%u", errors="raise")
    return pd.to_datetime(value, errors="raise")


def _normal_sd_from_95(p2_5: np.ndarray, p97_5: np.ndarray) -> np.ndarray:
    width = np.maximum(p97_5 - p2_5, 0.0)
    sd = width / (2.0 * 1.959963984540054)
    return np.where(sd <= 0, 1e-8, sd)


def _build_municipality_week_frame(municipality_df: pd.DataFrame) -> pd.DataFrame:
    week_column = "week_key" if "week_key" in municipality_df.columns else "Week"
    municipality_df = municipality_df.copy()
    municipality_df["week_id"] = municipality_df[week_column].astype(str)
    municipality_df["week_start"] = municipality_df["week_id"].map(_extract_week_start)

    for col in ["corrected_count_mean", "corrected_count_p2_5", "corrected_count_p97_5"]:
        if col not in municipality_df.columns:
            raise ValueError(f"Expected column '{col}' in municipality counts CSV.")
        municipality_df[col] = pd.to_numeric(municipality_df[col], errors="coerce")

    municipalities = sorted(municipality_df["Municipality"].dropna().astype(str).unique().tolist())
    all_weeks = (
        municipality_df[["week_id", "week_start"]]
        .drop_duplicates()
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    grid = all_weeks.assign(_key=1).merge(
        pd.DataFrame({"Municipality": municipalities, "_key": 1}), on="_key", how="outer"
    )
    grid = grid.drop(columns="_key")

    merged = grid.merge(
        municipality_df[
            [
                "Municipality",
                "week_id",
                "corrected_count_mean",
                "corrected_count_p2_5",
                "corrected_count_p97_5",
            ]
        ],
        on=["Municipality", "week_id"],
        how="left",
    )
    merged = merged.sort_values(["Municipality", "week_start"]).reset_index(drop=True)

    merged["source_week_id"] = merged["week_id"].where(merged["corrected_count_mean"].notna(), pd.NA)

    merged["imputed_from_previous_week"] = merged["corrected_count_mean"].isna()
    for col in ["corrected_count_mean", "corrected_count_p2_5", "corrected_count_p97_5", "source_week_id"]:
        merged[f"{col}_filled"] = merged.groupby("Municipality")[col].ffill()

    merged["imputed_from_next_week"] = False
    still_missing = merged["corrected_count_mean_filled"].isna()
    if still_missing.any():
        for col in ["corrected_count_mean", "corrected_count_p2_5", "corrected_count_p97_5", "source_week_id"]:
            b = merged.groupby("Municipality")[col].bfill()
            merged.loc[still_missing, f"{col}_filled"] = b[still_missing]
        merged.loc[still_missing, "imputed_from_next_week"] = True

    if merged["corrected_count_mean_filled"].isna().any():
        missing_muni = sorted(
            merged.loc[merged["corrected_count_mean_filled"].isna(), "Municipality"].unique().tolist()
        )
        raise ValueError(f"Municipality has no usable corrected counts at all: {missing_muni}")

    return merged


def build_gaza_wide_counts(input_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    municipality_df = pd.read_csv(input_csv)
    if "Municipality" not in municipality_df.columns:
        raise ValueError("Expected column 'Municipality' in municipality counts CSV.")

    merged = _build_municipality_week_frame(municipality_df)

    out = (
        merged.groupby(["week_id", "week_start"], as_index=False)
        .agg(
            gaza_wide_corrected_count_mean=("corrected_count_mean_filled", "sum"),
            municipalities_with_imputation=("imputed_from_previous_week", "sum"),
            municipalities_in_sum=("Municipality", "nunique"),
            municipalities_imputed_from_next_week=("imputed_from_next_week", "sum"),
        )
        .sort_values("week_start")
    )
    return out, merged


def build_extremes_json(
    weekly_totals: pd.DataFrame,
    municipality_week_frame: pd.DataFrame,
    output_json: Path,
    draws: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    min_idx = weekly_totals["gaza_wide_corrected_count_mean"].idxmin()
    max_idx = weekly_totals["gaza_wide_corrected_count_mean"].idxmax()
    extremes = {
        "min_week": weekly_totals.loc[min_idx, "week_id"],
        "max_week": weekly_totals.loc[max_idx, "week_id"],
    }

    def _week_ci(week_id: str) -> dict:
        wk = municipality_week_frame.loc[municipality_week_frame["week_id"] == week_id].copy()
        means = wk["corrected_count_mean_filled"].to_numpy(float)
        p2 = wk["corrected_count_p2_5_filled"].to_numpy(float)
        p97 = wk["corrected_count_p97_5_filled"].to_numpy(float)
        sds = _normal_sd_from_95(p2, p97)

        sim = rng.normal(loc=means, scale=sds, size=(draws, len(means)))
        sim = np.clip(sim, 0.0, None)
        totals = sim.sum(axis=1)

        return {
            "week_id": week_id,
            "territory_corrected_count_mean": float(np.mean(means)),
            "territory_corrected_count_sum_of_means": float(np.sum(means)),
            "territory_corrected_count_p2_5": float(np.quantile(totals, 0.025)),
            "territory_corrected_count_p50": float(np.quantile(totals, 0.5)),
            "territory_corrected_count_p97_5": float(np.quantile(totals, 0.975)),
            "municipality_week_sources": [
                {
                    "municipality": str(r["Municipality"]),
                    "target_week": str(r["week_id"]),
                    "source_week": str(r["source_week_id_filled"]),
                    "imputed_from_previous_week": bool(r["imputed_from_previous_week"]),
                    "imputed_from_next_week": bool(r["imputed_from_next_week"]),
                }
                for _, r in wk.sort_values("Municipality").iterrows()
            ],
        }

    payload = {
        "method_note": (
            "Week-level territory CI for min/max weeks estimated from municipality-level means and 95% CrI "
            "using normal approximation per municipality and Monte Carlo summation."
        ),
        "draws": draws,
        "seed": seed,
        "extremes": {
            "min": _week_ci(extremes["min_week"]),
            "max": _week_ci(extremes["max_week"]),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Gaza-wide weekly corrected counts from municipality-level counts."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extremes-json", type=Path, default=DEFAULT_EXTREMES_JSON)
    parser.add_argument("--ci-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    weekly, muni_week = build_gaza_wide_counts(args.input_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.output_csv, index=False)
    build_extremes_json(
        weekly_totals=weekly,
        municipality_week_frame=muni_week,
        output_json=args.extremes_json,
        draws=args.ci_draws,
        seed=args.seed,
    )
    print(f"Saved Gaza-wide weekly counts: {args.output_csv}")
    print(f"Saved min/max week CI summary: {args.extremes_json}")


if __name__ == "__main__":
    main()
