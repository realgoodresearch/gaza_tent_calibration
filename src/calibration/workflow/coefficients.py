from __future__ import annotations

import sys
import os

# Avoid shadowing stdlib modules (e.g., local `types.py`) when this file is
# executed directly via `python src/calibration/workflow/coefficients.py`.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [
    p
    for p in sys.path
    if (os.path.abspath(p or ".") != _THIS_DIR)
]

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import arviz as az
import pandas as pd


def _summary_table(idata, var_names: Iterable[str], hdi_prob: float) -> pd.DataFrame:
    use_vars = [v for v in var_names if v in idata.posterior.data_vars]
    if not use_vars:
        return pd.DataFrame()
    try:
        summary = az.summary(idata, var_names=use_vars, hdi_prob=hdi_prob)
    except TypeError:
        summary = az.summary(idata, var_names=use_vars, ci_prob=hdi_prob)
    return summary.reset_index().rename(columns={"index": "term"})


def _attach_feature_names(summary_df: pd.DataFrame, var_name: str, feature_names: List[str]) -> pd.DataFrame:
    out = summary_df.copy()
    out = out[out["term"].str.startswith(f"{var_name}[")].copy()
    if out.empty:
        return out
    idx = out["term"].str.extract(r"\[(\d+)\]").astype(float)[0]
    out["coef_index"] = idx.astype("Int64")
    out["feature"] = out["coef_index"].map(dict(enumerate(feature_names)))
    cols = [
        "feature",
        "coef_index",
        "term",
        "mean",
        "sd",
        "hdi_2.5%",
        "hdi_97.5%",
        "r_hat",
        "ess_bulk",
        "ess_tail",
    ]
    return out[[c for c in cols if c in out.columns]].sort_values("coef_index")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export calibration model coefficient summaries from posterior NetCDF")
    parser.add_argument(
        "--posterior",
        type=Path,
        default=Path("src/calibration/artifacts/final_model_posterior.nc"),
        help="Path to final posterior NetCDF.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("src/calibration/artifacts/effective_model_spec.json"),
        help="Path to effective model spec JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/calibration/artifacts"),
        help="Directory to write coefficient CSV summaries.",
    )
    parser.add_argument("--hdi-prob", type=float, default=0.95, help="HDI probability for interval summaries.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.posterior.exists():
        raise FileNotFoundError(f"Posterior file not found: {args.posterior}")
    if not args.spec.exists():
        raise FileNotFoundError(f"Spec file not found: {args.spec}")

    idata = az.from_netcdf(args.posterior)
    spec_payload = json.loads(args.spec.read_text())
    feature_names = list(spec_payload.get("final_columns", []))
    count_feature_names = list(spec_payload.get("count_columns", feature_names))
    zero_feature_names = list(spec_payload.get("zero_columns", feature_names))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_vars = ["alpha_muni", "sigma_muni", "alpha"]
    base_summary = _summary_table(idata, base_vars, args.hdi_prob)
    if not base_summary.empty:
        base_path = args.output_dir / "final_model_global_parameters.csv"
        base_summary.to_csv(base_path, index=False)
        print(f"Saved: {base_path}")

    if "beta" in idata.posterior.data_vars:
        beta_summary = _summary_table(idata, ["beta"], args.hdi_prob)
        beta_table = _attach_feature_names(beta_summary, "beta", feature_names)
        beta_path = args.output_dir / "final_model_beta.csv"
        beta_table.to_csv(beta_path, index=False)
        print(f"Saved: {beta_path}")

    for var_name, out_name in [("beta_count", "final_model_beta_count.csv"), ("beta_zero", "final_model_beta_zero.csv")]:
        if var_name not in idata.posterior.data_vars:
            continue
        var_summary = _summary_table(idata, [var_name], args.hdi_prob)
        names = count_feature_names if var_name == "beta_count" else zero_feature_names
        var_table = _attach_feature_names(var_summary, var_name, names)
        out_path = args.output_dir / out_name
        var_table.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
