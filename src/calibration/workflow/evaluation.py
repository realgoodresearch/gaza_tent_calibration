from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from tqdm.auto import tqdm

from .models import expected_count_hurdle, expected_count_nb, fit_hurdle_nb, fit_nb
from .types import FeatureBundle


def fold_aggregate_metrics(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eval_df = frame[["municipality", "date", "block"]].copy() if "block" in frame.columns else frame[["municipality", "date"]].copy()
    if "block" not in eval_df.columns:
        eval_df["block"] = "UNKNOWN"
    eval_df["week"] = pd.to_datetime(eval_df["date"], errors="coerce").dt.to_period("W-MON").astype(str)
    eval_df["actual"] = y_true
    eval_df["pred"] = y_pred
    grouped = eval_df.groupby(["municipality", "block", "week"], dropna=False).agg(actual_sum=("actual", "sum"), pred_sum=("pred", "sum"))
    diff = grouped["pred_sum"] - grouped["actual_sum"]
    return {"agg_abs_error_mean": float(np.mean(np.abs(diff))), "agg_bias_mean": float(np.mean(diff))}


def blocked_cv_with_metrics(
    bundle: FeatureBundle,
    model_kind: str,
    repeats: int,
    folds: int,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    seed: int = 2026,
    split_mode: str = "random",
    verbose: bool = False,
    log_prefix: str = "",
    bundle_zero: FeatureBundle | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    groups = bundle.week_group
    total_fits = max(1, repeats * folds)
    progress = tqdm(total=total_fits, desc=f"{log_prefix}CV ({model_kind})", leave=False, disable=not verbose)

    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        unique_groups = np.array(sorted(pd.Series(groups).dropna().unique()))
        rng.shuffle(unique_groups)
        shuffled_group_order = {g: i for i, g in enumerate(unique_groups)}
        order = pd.Series(groups).map(shuffled_group_order).fillna(10**9).to_numpy()
        idx = np.argsort(order)

        X = bundle.design_matrix[idx]
        X_zero = bundle_zero.design_matrix[idx] if bundle_zero is not None else X
        y = bundle.target[idx]
        muni = bundle.municipality_codes[idx]
        grp = groups[idx]
        frame = bundle.frame.iloc[idx].reset_index(drop=True)

        if split_mode == "blocked_week":
            splitter = GroupKFold(n_splits=folds)
            split_iter = splitter.split(X, y, groups=grp)
        else:
            splitter = KFold(n_splits=folds, shuffle=True, random_state=seed + rep)
            split_iter = splitter.split(X, y)

        for fold_id, (tr, te) in enumerate(split_iter, start=1):
            fold_seed = int(seed + (rep * 10_000) + fold_id)
            if model_kind == "nb":
                idata = fit_nb(
                    X[tr], y[tr], muni[tr], len(bundle.municipality_names), draws, tune, chains, target_accept, random_seed=fold_seed
                )
                pred_draws = expected_count_nb(idata, X[te], muni[te])
            else:
                idata = fit_hurdle_nb(
                    X[tr], X_zero[tr], y[tr], muni[tr], len(bundle.municipality_names), draws, tune, chains, target_accept, random_seed=fold_seed
                )
                pred_draws = expected_count_hurdle(idata, X[te], X_zero[te], muni[te])

            pred_mean = pred_draws.mean(axis=1)
            pred_mean = np.nan_to_num(pred_mean, nan=0.0, posinf=1e6, neginf=0.0)
            agg = fold_aggregate_metrics(frame.iloc[te].copy(), y[te], pred_mean)
            rows.append(
                {
                    "repeat": rep,
                    "fold": fold_id,
                    "model": model_kind,
                    "mae": float(mean_absolute_error(y[te], pred_mean)),
                    "rmse": float(np.sqrt(mean_squared_error(y[te], pred_mean))),
                    "bias": float(np.mean(pred_mean - y[te])),
                    "agg_abs_error_mean": agg["agg_abs_error_mean"],
                    "agg_bias_mean": agg["agg_bias_mean"],
                    "n_test": int(len(te)),
                }
            )
            progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def evaluate_candidate_against_current(
    current_cv: pd.DataFrame,
    candidate_cv: pd.DataFrame,
    min_improvement: float,
    min_win_rate: float,
) -> Dict[str, Any]:
    merged = current_cv.merge(candidate_cv, on=["repeat", "fold"], suffixes=("_cur", "_cand"))
    improvement = merged["agg_abs_error_mean_cur"] - merged["agg_abs_error_mean_cand"]
    mae_improvement = merged["mae_cur"] - merged["mae_cand"]
    return {
        "mean_primary_improvement": float(improvement.mean()),
        "win_rate": float((improvement > 0).mean()),
        "mean_mae_improvement": float(mae_improvement.mean()),
        "consistent": bool((improvement > 0).mean() >= min_win_rate),
    }


def holdout_metrics(y_true: np.ndarray, y_pred: np.ndarray, hold_df: pd.DataFrame) -> Dict[str, float]:
    metrics = {
        "count_bias": float(np.mean(y_pred - y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    slope, intercept = np.polyfit(y_pred, y_true, deg=1)
    metrics["calibration_slope"] = float(slope)
    metrics["calibration_intercept"] = float(intercept)

    eval_df = hold_df.copy()

    muni_grouped = eval_df.groupby(["municipality"], dropna=False).agg(actual_sum=("tent_count", "sum"), pred_sum=("pred_mean", "sum"))
    muni_abs_err = (muni_grouped["pred_sum"] - muni_grouped["actual_sum"]).abs()
    metrics["municipality_abs_error_mean"] = float(muni_abs_err.mean())
    metrics["municipality_abs_error_p95"] = float(np.quantile(muni_abs_err, 0.95))

    if "block" not in eval_df.columns:
        eval_df["block"] = "UNKNOWN"
    block_grouped = eval_df.groupby(["municipality", "block"], dropna=False).agg(actual_sum=("tent_count", "sum"), pred_sum=("pred_mean", "sum"))
    block_abs_err = (block_grouped["pred_sum"] - block_grouped["actual_sum"]).abs()
    metrics["block_abs_error_mean"] = float(block_abs_err.mean())
    metrics["block_abs_error_p95"] = float(np.quantile(block_abs_err, 0.95))
    return metrics
