from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

from .evaluation import blocked_cv_with_metrics, evaluate_candidate_against_current
from .features import make_bundle_from_columns


def run_forward_selection(
    dev: pd.DataFrame,
    candidate_groups: Dict[str, List[str]],
    baseline_columns: List[str],
    cv_args: Dict[str, Any],
    min_primary_improvement: float,
    min_fold_win_rate: float,
    mae_epsilon: float,
) -> Tuple[List[str], List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    selected_candidates: List[str] = []
    remaining_candidates = sorted(candidate_groups.keys())

    baseline_bundle = make_bundle_from_columns(dev, baseline_columns)
    print("[selection] Baseline CV")
    current_cv = blocked_cv_with_metrics(bundle=baseline_bundle, log_prefix="baseline", verbose=True, **cv_args)

    forward_rows: List[Dict[str, Any]] = []
    selected_steps: List[Dict[str, Any]] = []
    step = 0

    while remaining_candidates:
        step += 1
        print(f"[selection] Step {step}: {len(remaining_candidates)} candidates")
        best_candidate: Optional[str] = None
        best_candidate_cv: Optional[pd.DataFrame] = None
        best_eval: Optional[Dict[str, Any]] = None

        for candidate in tqdm(remaining_candidates, desc=f"Selection step {step}", leave=False):
            columns = baseline_columns + [c for name in selected_candidates for c in candidate_groups[name]] + candidate_groups[candidate]
            bundle = make_bundle_from_columns(dev, columns, municipality_names=baseline_bundle.municipality_names)
            cand_cv = blocked_cv_with_metrics(bundle=bundle, log_prefix=f"{candidate}", verbose=True, **cv_args)
            eval_result = evaluate_candidate_against_current(
                current_cv=current_cv,
                candidate_cv=cand_cv,
                min_improvement=min_primary_improvement,
                min_win_rate=min_fold_win_rate,
            )
            forward_rows.append(
                {
                    "step": step,
                    "candidate": candidate,
                    "selected_so_far": ",".join(selected_candidates),
                    **eval_result,
                    "candidate_primary": float(cand_cv["agg_abs_error_mean"].mean()),
                    "candidate_mae": float(cand_cv["mae"].mean()),
                    "candidate_rmse": float(cand_cv["rmse"].mean()),
                    "candidate_agg_bias_abs": float(cand_cv["agg_bias_mean"].abs().mean()),
                }
            )

            if not eval_result["consistent"]:
                continue

            if best_eval is None:
                best_candidate = candidate
                best_candidate_cv = cand_cv
                best_eval = eval_result
                continue

            better = (
                eval_result["mean_mae_improvement"],
                eval_result["win_rate"],
                eval_result["mean_primary_improvement"],
            ) > (
                best_eval["mean_mae_improvement"],
                best_eval["win_rate"],
                best_eval["mean_primary_improvement"],
            )
            if better:
                best_candidate = candidate
                best_candidate_cv = cand_cv
                best_eval = eval_result

        if best_candidate is None:
            print("[selection] No candidate passed consistency + improvement thresholds")
            break

        if best_eval is not None and float(best_eval["mean_mae_improvement"]) <= -float(mae_epsilon):
            print(
                "[selection] Stopping: best candidate fails MAE epsilon rule "
                f"(best mean_mae_impr={best_eval['mean_mae_improvement']:.4f}, epsilon={mae_epsilon:.4f})"
            )
            break

        selected_candidates.append(best_candidate)
        remaining_candidates.remove(best_candidate)
        current_cv = best_candidate_cv if best_candidate_cv is not None else current_cv
        selected_steps.append(
            {
                "step": step,
                "candidate": best_candidate,
                "marginal_primary_improvement": float(best_eval["mean_primary_improvement"] if best_eval else 0.0),
                "win_rate": float(best_eval["win_rate"] if best_eval else 0.0),
                "marginal_mae_improvement": float(best_eval["mean_mae_improvement"] if best_eval else 0.0),
                "post_add_primary": float(current_cv["agg_abs_error_mean"].mean()),
                "post_add_mae": float(current_cv["mae"].mean()),
                "post_add_rmse": float(current_cv["rmse"].mean()),
            }
        )
        print(
            f"[selection] Added {best_candidate} "
            f"(win_rate={best_eval['win_rate']:.2f}, mean_mae_impr={best_eval['mean_mae_improvement']:.4f})"
        )

    final_columns = baseline_columns + [c for name in selected_candidates for c in candidate_groups[name]]
    return (
        selected_candidates,
        final_columns,
        pd.DataFrame(forward_rows),
        pd.DataFrame(selected_steps),
        current_cv,
        remaining_candidates,
    )


def run_forward_selection_hurdle_components(
    dev: pd.DataFrame,
    candidate_groups: Dict[str, List[str]],
    baseline_columns: List[str],
    cv_args: Dict[str, Any],
    min_primary_improvement: float,
    min_fold_win_rate: float,
    mae_epsilon: float,
) -> Tuple[List[str], List[str], List[str], List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv_args_no_kind = {k: v for k, v in cv_args.items() if k != "model_kind"}
    selected_count: List[str] = []
    selected_zero: List[str] = []
    remaining_count = sorted(candidate_groups.keys())
    remaining_zero = sorted(candidate_groups.keys())

    baseline_bundle = make_bundle_from_columns(dev, baseline_columns)
    print("[selection] Baseline CV (hurdle, count/zero baselines)")
    current_cv = blocked_cv_with_metrics(
        bundle=baseline_bundle,
        bundle_zero=baseline_bundle,
        model_kind="hurdle_nb",
        log_prefix="baseline",
        verbose=True,
        **cv_args_no_kind,
    )

    forward_rows: List[Dict[str, Any]] = []
    selected_steps: List[Dict[str, Any]] = []
    step = 0

    while remaining_count or remaining_zero:
        step += 1
        print(f"[selection] Step {step}: count_remaining={len(remaining_count)} zero_remaining={len(remaining_zero)}")

        best_candidate: Optional[str] = None
        best_component: Optional[str] = None
        best_candidate_cv: Optional[pd.DataFrame] = None
        best_eval: Optional[Dict[str, Any]] = None

        eval_list: List[Tuple[str, str]] = [("count", c) for c in remaining_count] + [("zero", c) for c in remaining_zero]
        for component, candidate in tqdm(eval_list, desc=f"Selection step {step}", leave=False):
            count_cols = baseline_columns + [c for name in selected_count for c in candidate_groups[name]]
            zero_cols = baseline_columns + [c for name in selected_zero for c in candidate_groups[name]]
            if component == "count":
                count_cols = count_cols + candidate_groups[candidate]
            else:
                zero_cols = zero_cols + candidate_groups[candidate]

            bundle_count = make_bundle_from_columns(dev, count_cols, municipality_names=baseline_bundle.municipality_names)
            bundle_zero = make_bundle_from_columns(dev, zero_cols, municipality_names=baseline_bundle.municipality_names)
            cand_cv = blocked_cv_with_metrics(
                bundle=bundle_count,
                bundle_zero=bundle_zero,
                model_kind="hurdle_nb",
                log_prefix=f"{candidate}:{component}",
                verbose=True,
                **cv_args_no_kind,
            )
            eval_result = evaluate_candidate_against_current(
                current_cv=current_cv,
                candidate_cv=cand_cv,
                min_improvement=min_primary_improvement,
                min_win_rate=min_fold_win_rate,
            )
            forward_rows.append(
                {
                    "step": step,
                    "component": component,
                    "candidate": candidate,
                    "selected_count_so_far": ",".join(selected_count),
                    "selected_zero_so_far": ",".join(selected_zero),
                    **eval_result,
                    "candidate_primary": float(cand_cv["agg_abs_error_mean"].mean()),
                    "candidate_mae": float(cand_cv["mae"].mean()),
                    "candidate_rmse": float(cand_cv["rmse"].mean()),
                    "candidate_agg_bias_abs": float(cand_cv["agg_bias_mean"].abs().mean()),
                }
            )

            if not eval_result["consistent"]:
                continue

            if best_eval is None:
                best_candidate, best_component, best_candidate_cv, best_eval = candidate, component, cand_cv, eval_result
                continue

            better = (
                eval_result["mean_mae_improvement"],
                eval_result["win_rate"],
                eval_result["mean_primary_improvement"],
            ) > (
                best_eval["mean_mae_improvement"],
                best_eval["win_rate"],
                best_eval["mean_primary_improvement"],
            )
            if better:
                best_candidate, best_component, best_candidate_cv, best_eval = candidate, component, cand_cv, eval_result

        if best_candidate is None or best_component is None:
            print("[selection] No candidate-component passed consistency + improvement thresholds")
            break

        if best_eval is not None and float(best_eval["mean_mae_improvement"]) <= -float(mae_epsilon):
            print(
                "[selection] Stopping: best candidate-component fails MAE epsilon rule "
                f"(best mean_mae_impr={best_eval['mean_mae_improvement']:.4f}, epsilon={mae_epsilon:.4f})"
            )
            break

        if best_component == "count":
            selected_count.append(best_candidate)
            if best_candidate in remaining_count:
                remaining_count.remove(best_candidate)
        else:
            selected_zero.append(best_candidate)
            if best_candidate in remaining_zero:
                remaining_zero.remove(best_candidate)

        current_cv = best_candidate_cv if best_candidate_cv is not None else current_cv
        selected_steps.append(
            {
                "step": step,
                "component": best_component,
                "candidate": best_candidate,
                "marginal_primary_improvement": float(best_eval["mean_primary_improvement"] if best_eval else 0.0),
                "win_rate": float(best_eval["win_rate"] if best_eval else 0.0),
                "marginal_mae_improvement": float(best_eval["mean_mae_improvement"] if best_eval else 0.0),
                "post_add_primary": float(current_cv["agg_abs_error_mean"].mean()),
                "post_add_mae": float(current_cv["mae"].mean()),
                "post_add_rmse": float(current_cv["rmse"].mean()),
            }
        )
        print(
            f"[selection] Added {best_candidate} to {best_component} "
            f"(win_rate={best_eval['win_rate']:.2f}, mean_mae_impr={best_eval['mean_mae_improvement']:.4f})"
        )

    count_columns = baseline_columns + [c for name in selected_count for c in candidate_groups[name]]
    zero_columns = baseline_columns + [c for name in selected_zero for c in candidate_groups[name]]
    return (
        selected_count,
        selected_zero,
        count_columns,
        zero_columns,
        pd.DataFrame(forward_rows),
        pd.DataFrame(selected_steps),
        current_cv,
    )
