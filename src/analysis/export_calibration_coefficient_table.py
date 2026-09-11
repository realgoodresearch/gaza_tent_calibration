"""Export publication-table summaries from the final calibration posterior."""
from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "src/calibration/artifacts"
OUTPUT_DIR = ROOT / "outputs/calibration"


def draws(array: object) -> np.ndarray:
    return np.asarray(array).reshape(-1)


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "sd": float(np.std(values, ddof=1)),
        "credible_interval_2_5": float(np.quantile(values, 0.025)),
        "credible_interval_97_5": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    posterior = az.from_netcdf(ARTIFACTS / "final_model_posterior.nc").posterior
    spec = json.loads((ARTIFACTS / "effective_model_spec.json").read_text())
    count_columns = spec["count_columns"]
    zero_columns = spec["zero_columns"]
    beta_count = np.asarray(posterior["beta_count"])
    beta_zero = np.asarray(posterior["beta_zero"])
    shared_intercept = draws(posterior["alpha_muni"])
    if beta_count.shape[-1] != len(count_columns) or beta_zero.shape[-1] != len(zero_columns):
        raise ValueError("Posterior coefficient dimensions do not match effective_model_spec.json")

    feature_labels = {
        "log1p_model_tent_count": "log(1+predicted count)",
        "land_use_Empty Field": "Land use: Empty Field",
        "land_use_Trees/Vegetation": "Land use: Trees/Vegetation",
        "is_humanitarian_zone": "Humanitarian zone",
        "sun_elevation": "Solar elevation",
        "log1p_distance_to_nearest_outpost": "log(1+distance to nearest outpost)",
        "is_agriculture": "Agricultural land",
        "log1p_distance_to_yellow_line": "log(1+distance to Yellow Line)",
        "month_sin": "Month: sine term",
        "month_cos": "Month: cosine term",
        "hours_from_sunrise_cand_s0": "Hours from sunrise: spline 1",
        "hours_from_sunrise_cand_s1": "Hours from sunrise: spline 2",
        "hours_from_sunrise_cand_s2": "Hours from sunrise: spline 3",
        "hours_from_sunrise_cand_s3": "Hours from sunrise: spline 4",
    }

    rows: list[dict[str, object]] = [{
        "parameter": "Intercept", "feature_name": "alpha_muni",
        "component": "Shared: hurdle and positive count", "coefficient_index": np.nan,
        **summary(shared_intercept), "status": "Estimated shared intercept",
    }]
    for component, columns, beta in (
        ("Hurdle", zero_columns, beta_zero),
        ("Positive count", count_columns, beta_count),
    ):
        rows.append({
            "parameter": "Land use: Buildings/Roads",
            "feature_name": "reference_land_use_Buildings/Roads",
            "component": component, "coefficient_index": np.nan,
            "median": 0.0, "sd": 0.0,
            "credible_interval_2_5": 0.0, "credible_interval_97_5": 0.0,
            "status": "Reference category; coefficient fixed to zero",
        })
        for index, feature in enumerate(columns):
            rows.append({"parameter": feature_labels.get(feature, feature),
                         "feature_name": feature, "component": component,
                         "coefficient_index": index,
                         **summary(beta[:, :, index].reshape(-1)),
                         "status": "Estimated"})

    for parameter, component, variable in (
        ("sigma_muni", "Municipality SD", "sigma_muni"),
        ("phi", "Dispersion", "alpha"),
    ):
        rows.append({"parameter": parameter, "feature_name": variable,
                     "component": component, "coefficient_index": np.nan,
                     **summary(draws(posterior[variable])), "status": "Estimated"})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "calibration_coefficient_table.csv", index=False)

    # Municipality-specific shared intercepts aid interpretation of alpha_muni.
    offsets = np.asarray(posterior["muni_offset"])
    sigma = np.asarray(posterior["sigma_muni"])[..., None]
    intercept = np.asarray(posterior["alpha_muni"])[..., None] + sigma * offsets
    municipality_names = sorted(pd.read_csv(ROOT / "src/validation/manual_validation_results.csv")["municipality"].astype(str).unique())
    muni_rows = []
    for index in range(intercept.shape[-1]):
        name = municipality_names[index] if index < len(municipality_names) else f"municipality_{index}"
        muni_rows.append({"municipality": name, **summary(intercept[..., index].reshape(-1))})
    pd.DataFrame(muni_rows).to_csv(OUTPUT_DIR / "municipality_intercept_summaries.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'calibration_coefficient_table.csv'}")
    print(f"Saved {OUTPUT_DIR / 'municipality_intercept_summaries.csv'}")


if __name__ == "__main__":
    main()
