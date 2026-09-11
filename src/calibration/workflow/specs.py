from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone

import pandas as pd

from .evaluation import blocked_cv_with_metrics
from .features import make_bundle_from_columns
from .selection import run_forward_selection, run_forward_selection_hurdle_components
from .types import ModelSpec


def run_and_select_spec(
    dev: pd.DataFrame,
    candidate_groups: Dict[str, List[str]],
    baseline_columns: List[str],
    cv_args: Dict[str, Any],
    min_primary_improvement: float,
    min_fold_win_rate: float,
    mae_epsilon: float,
    zero_rate_threshold: float,
    artifacts_dir: Path,
    compare_hurdle_against_nb: bool = True,
) -> Tuple[ModelSpec, pd.DataFrame, pd.DataFrame, List[str], float]:
    cv_model_kind = str(cv_args.get("model_kind", "nb"))
    count_columns: List[str] | None = None
    zero_columns: List[str] | None = None
    selected_count: List[str] | None = None
    selected_zero: List[str] | None = None
    if cv_model_kind == "hurdle_nb":
        selected_count, selected_zero, count_columns, zero_columns, forward_df, selected_steps_df, _ = run_forward_selection_hurdle_components(
            dev=dev,
            candidate_groups=candidate_groups,
            baseline_columns=baseline_columns,
            cv_args=cv_args,
            min_primary_improvement=min_primary_improvement,
            min_fold_win_rate=min_fold_win_rate,
            mae_epsilon=mae_epsilon,
        )
        selected_candidates = sorted(set(selected_count + selected_zero))
        final_columns = list(dict.fromkeys((count_columns or []) + (zero_columns or [])))
        remaining = sorted(set(candidate_groups.keys()) - set(selected_candidates))
    else:
        selected_candidates, final_columns, forward_df, selected_steps_df, _, remaining = run_forward_selection(
            dev=dev,
            candidate_groups=candidate_groups,
            baseline_columns=baseline_columns,
            cv_args=cv_args,
            min_primary_improvement=min_primary_improvement,
            min_fold_win_rate=min_fold_win_rate,
            mae_epsilon=mae_epsilon,
        )
    forward_df.to_csv(artifacts_dir / "forward_selection_results.csv", index=False)
    selected_steps_df.to_csv(artifacts_dir / "selected_variable_contributions.csv", index=False)

    zero_rate = float((dev["tent_count"] == 0).mean())
    selected_model_kind = cv_model_kind
    final_bundle_dev = make_bundle_from_columns(dev, final_columns)

    if compare_hurdle_against_nb and cv_model_kind == "nb" and zero_rate >= zero_rate_threshold:
        non_kind_args = {k: v for k, v in cv_args.items() if k != "model_kind"}
        cv_nb = blocked_cv_with_metrics(bundle=final_bundle_dev, model_kind="nb", **non_kind_args)
        cv_hurdle = blocked_cv_with_metrics(bundle=final_bundle_dev, model_kind="hurdle_nb", **non_kind_args)
        cv_nb.to_csv(artifacts_dir / "cv_results_selected_nb.csv", index=False)
        cv_hurdle.to_csv(artifacts_dir / "cv_results_selected_hurdle_nb.csv", index=False)
        if cv_hurdle["agg_abs_error_mean"].mean() < cv_nb["agg_abs_error_mean"].mean():
            selected_model_kind = "hurdle_nb"

    spec_name = "baseline" if not selected_candidates else "baseline_plus_" + "__".join(selected_candidates)
    spec = ModelSpec(
        name=spec_name,
        selected_candidates=selected_candidates,
        final_columns=final_columns,
        model_kind=selected_model_kind,
        selected_candidates_count=selected_count,
        selected_candidates_zero=selected_zero,
        count_columns=count_columns,
        zero_columns=zero_columns,
    )
    return spec, forward_df, selected_steps_df, remaining, zero_rate


def load_best_spec(path: Path) -> ModelSpec:
    payload = json.loads(path.read_text())
    model_kind = payload.get("selected_model_kind", "nb")
    count_columns = list(payload.get("count_columns", [])) or None
    zero_columns = list(payload.get("zero_columns", [])) or None
    final_columns = list(payload["final_columns"])
    if model_kind == "hurdle_nb":
        if count_columns is None:
            count_columns = final_columns
        if zero_columns is None:
            zero_columns = final_columns
    return ModelSpec(
        name=payload["selected_spec"],
        selected_candidates=list(payload.get("selected_candidates", [])),
        final_columns=final_columns,
        model_kind=model_kind,
        selected_candidates_count=list(payload.get("selected_candidates_count", [])) or None,
        selected_candidates_zero=list(payload.get("selected_candidates_zero", [])) or None,
        count_columns=count_columns,
        zero_columns=zero_columns,
    )


def manual_spec(
    candidate_groups: Dict[str, List[str]],
    baseline_columns: List[str],
    candidate_names: List[str],
    model_kind: str,
    spec_name: str,
) -> ModelSpec:
    unknown = sorted(set(candidate_names) - set(candidate_groups.keys()))
    if unknown:
        raise ValueError(f"Unknown manual candidates: {unknown}")
    final_columns = baseline_columns + [col for name in candidate_names for col in candidate_groups[name]]
    return ModelSpec(
        name=spec_name,
        selected_candidates=candidate_names,
        final_columns=final_columns,
        model_kind=model_kind,
        selected_candidates_count=candidate_names if model_kind == "hurdle_nb" else None,
        selected_candidates_zero=candidate_names if model_kind == "hurdle_nb" else None,
        count_columns=final_columns if model_kind == "hurdle_nb" else None,
        zero_columns=final_columns if model_kind == "hurdle_nb" else None,
    )


def write_selected_payload(
    path: Path,
    spec: ModelSpec,
    zero_rate_development: float,
    remaining: List[str],
    selection_mode: str,
    run_config: Dict[str, Any] | None = None,
) -> None:
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}

    payload = {
        "selected_spec": spec.name,
        "selected_model_kind": spec.model_kind,
        "zero_rate_development": zero_rate_development,
        "selection_mode": selection_mode,
        "selected_candidates": spec.selected_candidates,
        "selected_candidates_count": spec.selected_candidates_count,
        "selected_candidates_zero": spec.selected_candidates_zero,
        "remaining_untested_or_not_selected": remaining,
        "final_columns": spec.final_columns,
        "count_columns": spec.count_columns,
        "zero_columns": spec.zero_columns,
        "selection_run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_config": run_config or {},
        },
    }
    merged = {**existing, **payload}
    path.write_text(json.dumps(merged, indent=2))


def write_final_run_payload(
    path: Path,
    final_info: Dict[str, Any],
) -> None:
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}

    existing["final_run"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **final_info,
    }
    path.write_text(json.dumps(existing, indent=2))
