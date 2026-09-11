from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import arviz as az
import numpy as np
import pandas as pd

from .evaluation import holdout_metrics
from .features import make_bundle_from_columns
from .models import (
    expected_count_hurdle,
    expected_count_nb,
    expected_count_nb_offset,
    expected_count_nb_offset_features,
    fit_hurdle_nb,
    fit_nb,
    fit_nb_offset,
    fit_nb_offset_features,
)
from .types import ModelSpec


def fit_model(
    spec: ModelSpec,
    X: np.ndarray,
    X_zero: np.ndarray | None,
    y: np.ndarray,
    municipality_codes: np.ndarray,
    n_muni: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    mean_coef_sigma: float = 0.15,
    random_seed: int | None = None,
):
    if spec.model_kind == "nb_offset":
        offset_log = X[:, 0] if X.shape[1] else np.zeros(X.shape[0], dtype=float)
        return fit_nb_offset(offset_log, y, municipality_codes, n_muni, draws, tune, chains, target_accept, random_seed=random_seed)
    if spec.model_kind == "nb_offset_features":
        offset_log = X[:, 0] if X.shape[1] else np.zeros(X.shape[0], dtype=float)
        X_features = X[:, 1:] if X.shape[1] > 1 else np.zeros((X.shape[0], 0), dtype=float)
        return fit_nb_offset_features(
            offset_log,
            X_features,
            y,
            municipality_codes,
            n_muni,
            draws,
            tune,
            chains,
            target_accept,
            mean_coef_sigma=mean_coef_sigma,
            random_seed=random_seed,
        )
    if spec.model_kind == "nb":
        return fit_nb(X, y, municipality_codes, n_muni, draws, tune, chains, target_accept, random_seed=random_seed)
    return fit_hurdle_nb(X, X if X_zero is None else X_zero, y, municipality_codes, n_muni, draws, tune, chains, target_accept, random_seed=random_seed)


def fit_hurdle_model(
    X_count: np.ndarray,
    X_zero: np.ndarray,
    y: np.ndarray,
    municipality_codes: np.ndarray,
    n_muni: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    random_seed: int | None = None,
):
    return fit_hurdle_nb(X_count, X_zero, y, municipality_codes, n_muni, draws, tune, chains, target_accept, random_seed=random_seed)


def expected_counts(spec: ModelSpec, idata, X: np.ndarray, municipality_codes: np.ndarray, X_zero: np.ndarray | None = None) -> np.ndarray:
    if spec.model_kind == "nb_offset":
        offset_log = X[:, 0] if X.shape[1] else np.zeros(X.shape[0], dtype=float)
        return expected_count_nb_offset(idata, offset_log, municipality_codes)
    if spec.model_kind == "nb_offset_features":
        offset_log = X[:, 0] if X.shape[1] else np.zeros(X.shape[0], dtype=float)
        X_features = X[:, 1:] if X.shape[1] > 1 else np.zeros((X.shape[0], 0), dtype=float)
        return expected_count_nb_offset_features(idata, offset_log, X_features, municipality_codes)
    if spec.model_kind == "nb":
        return expected_count_nb(idata, X, municipality_codes)
    return expected_count_hurdle(idata, X, X if X_zero is None else X_zero, municipality_codes)


def summarize_mcmc_diagnostics(idata: az.InferenceData) -> Dict[str, float]:
    summary = az.summary(idata, kind="diagnostics", round_to=None)
    rhat_series = summary["r_hat"].replace([np.inf, -np.inf], np.nan).dropna() if "r_hat" in summary.columns else pd.Series(dtype=float)
    ess_bulk_series = summary["ess_bulk"].replace([np.inf, -np.inf], np.nan).dropna() if "ess_bulk" in summary.columns else pd.Series(dtype=float)
    ess_tail_series = summary["ess_tail"].replace([np.inf, -np.inf], np.nan).dropna() if "ess_tail" in summary.columns else pd.Series(dtype=float)

    diagnostics = {
        "n_chains": int(idata.posterior.sizes.get("chain", 0)),
        "n_draws": int(idata.posterior.sizes.get("draw", 0)),
        "max_rhat": float(rhat_series.max()) if not rhat_series.empty else np.nan,
        "min_ess_bulk": float(ess_bulk_series.min()) if not ess_bulk_series.empty else np.nan,
        "min_ess_tail": float(ess_tail_series.min()) if not ess_tail_series.empty else np.nan,
    }

    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        diagnostics["n_divergences"] = int(idata.sample_stats["diverging"].sum().values)
    if hasattr(idata, "sample_stats") and "tree_depth" in idata.sample_stats:
        diagnostics["max_tree_depth_observed"] = int(idata.sample_stats["tree_depth"].max().values)

    return diagnostics


def posterior_predictive_checks(y_true: np.ndarray, pred_draws: np.ndarray) -> Dict[str, float]:
    pred_mean = pred_draws.mean(axis=1)
    q05, q25, q75, q95 = np.quantile(pred_draws, [0.05, 0.25, 0.75, 0.95], axis=1)
    return {
        "obs_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(pred_mean)),
        "obs_var": float(np.var(y_true)),
        "pred_var": float(np.var(pred_mean)),
        "coverage_50": float(np.mean((y_true >= q25) & (y_true <= q75))),
        "coverage_90": float(np.mean((y_true >= q05) & (y_true <= q95))),
    }


def evaluate_holdout_from_dev_fit(
    spec: ModelSpec,
    dev: pd.DataFrame,
    holdout: pd.DataFrame,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    mean_coef_sigma: float = 0.15,
    seed: int = 2026,
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, float], Dict[str, float], az.InferenceData]:
    count_cols = spec.count_columns if spec.model_kind == "hurdle_nb" and spec.count_columns else spec.final_columns
    zero_cols = spec.zero_columns if spec.model_kind == "hurdle_nb" and spec.zero_columns else count_cols

    dev_bundle = make_bundle_from_columns(dev, count_cols)
    dev_bundle_zero = make_bundle_from_columns(dev, zero_cols, municipality_names=dev_bundle.municipality_names)

    if spec.model_kind == "nb":
        fit_dev = fit_nb(
            dev_bundle.design_matrix,
            dev_bundle.target,
            dev_bundle.municipality_codes,
            len(dev_bundle.municipality_names),
            draws,
            tune,
            chains,
            target_accept,
            random_seed=seed + 100_000,
        )
    elif spec.model_kind == "nb_offset":
        fit_dev = fit_nb_offset(
            dev_bundle.design_matrix[:, 0] if dev_bundle.design_matrix.shape[1] else np.zeros(dev_bundle.design_matrix.shape[0], dtype=float),
            dev_bundle.target,
            dev_bundle.municipality_codes,
            len(dev_bundle.municipality_names),
            draws,
            tune,
            chains,
            target_accept,
            random_seed=seed + 100_000,
        )
    elif spec.model_kind == "nb_offset_features":
        fit_dev = fit_nb_offset_features(
            dev_bundle.design_matrix[:, 0] if dev_bundle.design_matrix.shape[1] else np.zeros(dev_bundle.design_matrix.shape[0], dtype=float),
            dev_bundle.design_matrix[:, 1:] if dev_bundle.design_matrix.shape[1] > 1 else np.zeros((dev_bundle.design_matrix.shape[0], 0), dtype=float),
            dev_bundle.target,
            dev_bundle.municipality_codes,
            len(dev_bundle.municipality_names),
            draws,
            tune,
            chains,
            target_accept,
            mean_coef_sigma=mean_coef_sigma,
            random_seed=seed + 100_000,
        )
    else:
        fit_dev = fit_hurdle_model(
            X_count=dev_bundle.design_matrix,
            X_zero=dev_bundle_zero.design_matrix,
            y=dev_bundle.target,
            municipality_codes=dev_bundle.municipality_codes,
            n_muni=len(dev_bundle.municipality_names),
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed + 100_000,
        )

    hold_bundle = make_bundle_from_columns(holdout, count_cols, municipality_names=dev_bundle.municipality_names)
    hold_bundle_zero = make_bundle_from_columns(holdout, zero_cols, municipality_names=dev_bundle.municipality_names)
    hold_draws = expected_counts(spec, fit_dev, hold_bundle.design_matrix, hold_bundle.municipality_codes, X_zero=hold_bundle_zero.design_matrix)
    hold_pred_mean = np.nan_to_num(hold_draws.mean(axis=1), nan=0.0, posinf=1e6, neginf=0.0)

    hold_df = hold_bundle.frame.copy()
    hold_df["pred_mean"] = hold_pred_mean
    hold_metrics = holdout_metrics(hold_bundle.target, hold_pred_mean, hold_df)

    dev_draws = expected_counts(spec, fit_dev, dev_bundle.design_matrix, dev_bundle.municipality_codes, X_zero=dev_bundle_zero.design_matrix)
    dev_ppc = posterior_predictive_checks(dev_bundle.target, dev_draws)
    dev_diag = summarize_mcmc_diagnostics(fit_dev)
    return hold_metrics, hold_df, dev_ppc, dev_diag, fit_dev


def refit_all_validated(
    spec: ModelSpec,
    full_df: pd.DataFrame,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    mean_coef_sigma: float = 0.15,
    seed: int = 2026,
) -> Tuple[az.InferenceData, Dict[str, float], List[str]]:
    count_cols = spec.count_columns if spec.model_kind == "hurdle_nb" and spec.count_columns else spec.final_columns
    zero_cols = spec.zero_columns if spec.model_kind == "hurdle_nb" and spec.zero_columns else count_cols

    full_bundle = make_bundle_from_columns(full_df, count_cols)
    full_bundle_zero = make_bundle_from_columns(full_df, zero_cols, municipality_names=full_bundle.municipality_names)
    if spec.model_kind == "nb":
        idata = fit_nb(
            full_bundle.design_matrix,
            full_bundle.target,
            full_bundle.municipality_codes,
            len(full_bundle.municipality_names),
            draws,
            tune,
            chains,
            target_accept,
            random_seed=seed + 200_000,
        )
    elif spec.model_kind == "nb_offset":
        idata = fit_nb_offset(
            full_bundle.design_matrix[:, 0] if full_bundle.design_matrix.shape[1] else np.zeros(full_bundle.design_matrix.shape[0], dtype=float),
            full_bundle.target,
            full_bundle.municipality_codes,
            len(full_bundle.municipality_names),
            draws,
            tune,
            chains,
            target_accept,
            random_seed=seed + 200_000,
        )
    elif spec.model_kind == "nb_offset_features":
        idata = fit_nb_offset_features(
            full_bundle.design_matrix[:, 0] if full_bundle.design_matrix.shape[1] else np.zeros(full_bundle.design_matrix.shape[0], dtype=float),
            full_bundle.design_matrix[:, 1:] if full_bundle.design_matrix.shape[1] > 1 else np.zeros((full_bundle.design_matrix.shape[0], 0), dtype=float),
            full_bundle.target,
            full_bundle.municipality_codes,
            len(full_bundle.municipality_names),
            draws,
            tune,
            chains,
            target_accept,
            mean_coef_sigma=mean_coef_sigma,
            random_seed=seed + 200_000,
        )
    else:
        idata = fit_hurdle_model(
            X_count=full_bundle.design_matrix,
            X_zero=full_bundle_zero.design_matrix,
            y=full_bundle.target,
            municipality_codes=full_bundle.municipality_codes,
            n_muni=len(full_bundle.municipality_names),
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed + 200_000,
        )
    return idata, summarize_mcmc_diagnostics(idata), full_bundle.municipality_names


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2))
