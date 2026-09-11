"""Summarise calibration accuracy and uncertainty for manuscript reporting.

The tile-level accuracy summaries use the independent holdout predictions from
the development fit.  Interval-width summaries use the week-level calibrated
count outputs produced from the final fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOLDOUT = REPO_ROOT / "src/calibration/artifacts/holdout_predictions.csv"
DEFAULT_MUNICIPALITY_COUNTS = REPO_ROOT / "outputs/tent_counts/municipality_counts_by_week.csv"
DEFAULT_BLOCK_COUNTS = REPO_ROOT / "outputs/tent_counts/block_counts_by_week.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/analysis/count_accuracy_and_confidence.json"

# The validation artifact contains municipal boundaries.  These groupings are
# the five governorate-scale spatial contexts named in the manuscript draft.
GOVERNORATE_BY_MUNICIPALITY = {
    "Beit Hanoun": "North Gaza", "Beit Lahiya": "North Gaza",
    "Jabalya": "North Gaza", "Al Qaraya al Badawiya": "North Gaza",
    "Gaza": "Gaza", "Al Zahra": "Gaza", "Al Mughraqa": "Gaza",
    "Deir al Balah": "Deir al-Balah", "An Nuseirat Camp": "Deir al-Balah",
    "Al Bureij Camp": "Deir al-Balah", "Al Maghazi Camp": "Deir al-Balah",
    "Al Musaddar": "Deir al-Balah", "Az Zawayda": "Deir al-Balah",
    "Wadi as Salqa": "Deir al-Balah",
    "Khan Yunis": "Khan Younis", "Khuza'a": "Khan Younis",
    "Al Qarara": "Khan Younis", "Al Fukhkhari": "Khan Younis",
    "Rafah": "Rafah", "Shokat as Sufi": "Rafah", "An Naser": "Rafah",
}


def _number(value: Any) -> float | int | None:
    """Return a JSON-safe Python scalar."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


def _bootstrap_bias_ci(errors: np.ndarray, seed: int, draws: int = 20_000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(errors, size=(draws, len(errors)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _accuracy(group: pd.DataFrame, seed: int = 2026) -> dict[str, Any]:
    error = (group["pred_mean"] - group["tent_count"]).to_numpy(float)
    slope, intercept = np.polyfit(group["pred_mean"], group["tent_count"], 1)
    ci = _bootstrap_bias_ci(error, seed)
    return {
        "n_tiles": int(len(group)),
        "observed_tents": float(group["tent_count"].sum()),
        "predicted_tents": float(group["pred_mean"].sum()),
        "mean_bias_predicted_minus_observed": float(error.mean()),
        "mean_bias_bootstrap_95_ci": ci,
        "bias_ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "observed_on_predicted_slope": float(slope),
        "observed_on_predicted_intercept": float(intercept),
    }


def _group_accuracy(df: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    return {
        str(name): _accuracy(group, seed=2026 + i)
        for i, (name, group) in enumerate(df.groupby(column, dropna=False, sort=True))
    }


def _aggregate_errors(df: pd.DataFrame, keys: list[str]) -> dict[str, Any]:
    grouped = df.groupby(keys, dropna=False).agg(
        observed=("tent_count", "sum"), predicted=("pred_mean", "sum")
    )
    absolute_error = (grouped["predicted"] - grouped["observed"]).abs()
    return {
        "unit": "tents",
        "n_groups": int(len(grouped)),
        "mean_absolute_error": float(absolute_error.mean()),
        "95th_percentile_absolute_error": float(absolute_error.quantile(0.95)),
    }


def _interval_widths(path: Path, level: str) -> dict[str, Any]:
    df = pd.read_csv(path)
    if level == "municipality":
        median_col = "Corrected count"
    else:
        median_col = "corrected_count_median"
    needed = [median_col, "corrected_count_p2_5", "corrected_count_p97_5"]
    values = df[needed].apply(pd.to_numeric, errors="coerce").dropna()
    width = values["corrected_count_p97_5"] - values["corrected_count_p2_5"]
    relative = 100 * width / values[median_col].where(values[median_col] > 0)

    def distribution(series: pd.Series) -> dict[str, float]:
        q = series.dropna().quantile([0.25, 0.5, 0.75, 0.9, 0.95])
        return {"p25": float(q.loc[0.25]), "median": float(q.loc[0.5]),
                "p75": float(q.loc[0.75]), "p90": float(q.loc[0.9]),
                "p95": float(q.loc[0.95])}

    return {
        "n_area_weeks": int(len(values)),
        "absolute_width_tents": distribution(width),
        "width_as_percent_of_posterior_median": distribution(relative),
        "definition": "upper 97.5% bound minus lower 2.5% bound for each area-week",
    }


def build_summary(holdout_path: Path, municipality_path: Path, block_path: Path) -> dict[str, Any]:
    holdout = pd.read_csv(holdout_path)
    required = {"tent_count", "pred_mean", "municipality", "block",
                "land_use_label", "is_humanitarian_zone"}
    missing = required - set(holdout.columns)
    if missing:
        raise ValueError(f"Missing holdout columns: {sorted(missing)}")
    holdout[["tent_count", "pred_mean"]] = holdout[["tent_count", "pred_mean"]].apply(
        pd.to_numeric, errors="coerce"
    )
    holdout = holdout.dropna(subset=["tent_count", "pred_mean"]).copy()
    holdout["governorate"] = holdout["municipality"].map(GOVERNORATE_BY_MUNICIPALITY)
    unmapped = sorted(holdout.loc[holdout["governorate"].isna(), "municipality"].unique())
    if unmapped:
        raise ValueError(f"Municipalities missing from governorate mapping: {unmapped}")

    # Fixed, interpretable bins avoid quantile cut points collapsing at zero.
    holdout["observed_density_band"] = pd.cut(
        holdout["tent_count"], bins=[-np.inf, 0, 9, 49, np.inf],
        labels=["zero (0 tents)", "low (1-9 tents)", "high (10-49 tents)",
                "highest (50+ tents)"],
    )
    by_density = _group_accuracy(holdout, "observed_density_band")
    overall = _accuracy(holdout)
    by_governorate = _group_accuracy(holdout, "governorate")
    materially_biased = [k for k, v in by_governorate.items() if v["bias_ci_excludes_zero"]]

    manuscript_values = {
        "holdout_n_tiles": int(len(holdout)),
        "overall": {k: round(overall[k], 2) for k in (
            "mean_bias_predicted_minus_observed", "mae", "rmse",
            "observed_on_predicted_slope", "observed_on_predicted_intercept")},
        "mae_by_governorate": {k: round(v["mae"], 2) for k, v in by_governorate.items()},
        "geography_bias_sentence": (
            "North Gaza showed a small positive mean bias whose bootstrap 95% CI excluded zero; "
            "no other governorate did."
            if materially_biased == ["North Gaza"] else
            f"Governorates whose bootstrap mean-bias CI excluded zero: {', '.join(materially_biased) or 'none'}."
        ),
        "density": {k: {"mae": round(v["mae"], 2),
                         "mean_bias": round(v["mean_bias_predicted_minus_observed"], 2)}
                    for k, v in by_density.items()},
        "humanitarian_zone": {
            "inside_mae": round(_accuracy(holdout[holdout["is_humanitarian_zone"].astype(bool)], 2027)["mae"], 2),
            "outside_mae": round(_accuracy(holdout[~holdout["is_humanitarian_zone"].astype(bool)], 2028)["mae"], 2),
            "inside_mean_bias": round(_accuracy(holdout[holdout["is_humanitarian_zone"].astype(bool)], 2027)["mean_bias_predicted_minus_observed"], 2),
            "outside_mean_bias": round(_accuracy(holdout[~holdout["is_humanitarian_zone"].astype(bool)], 2028)["mean_bias_predicted_minus_observed"], 2),
        },
    }

    return {
        "sources": {
            "holdout_predictions": str(holdout_path.relative_to(REPO_ROOT)),
            "municipality_week_counts": str(municipality_path.relative_to(REPO_ROOT)),
            "evacuation_block_week_counts": str(block_path.relative_to(REPO_ROOT)),
        },
        "definitions": {
            "error": "calibrated posterior mean prediction minus manual tent count",
            "calibration_line": "ordinary least-squares regression of observed counts on calibrated predictions",
            "spatial_context_note": "The five contexts requested in the draft are governorates, not municipalities; municipal validation labels are mapped to them.",
            "density_bands": "Observed manual tent count per 100 x 100 m tile: zero=0, low=1-9, high=10-49, highest=50+.",
            "systematic_bias_evidence": "A 20,000-replicate tile bootstrap 95% CI for mean bias excluding zero.",
        },
        "manuscript_values_rounded_2dp": manuscript_values,
        "holdout_tile_accuracy": {
            "overall": overall,
            "by_governorate": by_governorate,
            "by_municipality": _group_accuracy(holdout, "municipality"),
            "geographical_heterogeneity": {
                "mae_min": min(v["mae"] for v in by_governorate.values()),
                "mae_max": max(v["mae"] for v in by_governorate.values()),
                "governorates_with_bias_ci_excluding_zero": materially_biased,
            },
            "by_observed_density": by_density,
            "humanitarian_zone": {
                "inside": _accuracy(holdout[holdout["is_humanitarian_zone"].astype(bool)], 2027),
                "outside": _accuracy(holdout[~holdout["is_humanitarian_zone"].astype(bool)], 2028),
            },
            "by_land_use": _group_accuracy(holdout, "land_use_label"),
        },
        "holdout_aggregate_accuracy": {
            "municipality": _aggregate_errors(holdout, ["municipality"]),
            "evacuation_block": _aggregate_errors(holdout, ["municipality", "block"]),
            "note": "Each held-out tile is summed once by spatial unit across the complete holdout; these reproduce the calibration pipeline metric definition.",
        },
        "week_specific_95pct_credible_interval_widths": {
            "municipality": _interval_widths(municipality_path, "municipality"),
            "evacuation_block": _interval_widths(block_path, "block"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--municipality-counts", type=Path, default=DEFAULT_MUNICIPALITY_COUNTS)
    parser.add_argument("--block-counts", type=Path, default=DEFAULT_BLOCK_COUNTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args.holdout, args.municipality_counts, args.block_counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
