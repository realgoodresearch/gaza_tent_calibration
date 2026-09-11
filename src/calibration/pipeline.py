from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from workflow.config import ARTIFACTS_DIR, PROJECT_ROOT, VALIDATION_PATH
from workflow.application import ApplicationConfig, _read_to_target, apply_calibration_to_full_predictions
from workflow.features import apply_scaler_params, build_candidate_feature_table, featurize_validation, fit_scaler_params
from workflow.postprocessing import recompute_validation_prediction_counts
from workflow.publication import evaluate_holdout_from_dev_fit, refit_all_validated, write_json
from workflow.specs import load_best_spec, manual_spec, run_and_select_spec, write_final_run_payload, write_selected_payload
from workflow.types import ModelSpec


BLOCK_PATH = PROJECT_ROOT / "data" / "gaza_boundaries" / "dash_population_blocks.geojson"
MUNICIPALITY_PATH = PROJECT_ROOT / "data" / "gaza_boundaries" / "municipal" / "GazaStrip_MunicipalBoundaries.shp"
MERGED_PREDICTIONS_DIR = PROJECT_ROOT / "data" / "tent_predictions" / "merged"
YELLOW_LINE_PATH = PROJECT_ROOT / "data" / "spatial_context" / "yellow-line-zone.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bayesian calibration pipeline for tent-count predictions")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cv-repeats", type=int, default=2)
    p.add_argument("--cv-folds", type=int, default=4)
    p.add_argument("--cv-draws", type=int, default=300)
    p.add_argument("--cv-tune", type=int, default=300)
    p.add_argument("--cv-chains", type=int, default=2)
    p.add_argument("--cv-model-kind", choices=["nb", "hurdle_nb"], default="nb")
    p.add_argument("--cv-split", choices=["random", "blocked_week"], default="random")
    p.add_argument("--target-accept", type=float, default=0.9)
    p.add_argument("--zero-rate-threshold", type=float, default=0.35)
    p.add_argument("--min-primary-improvement", type=float, default=0.05)
    p.add_argument("--min-fold-win-rate", type=float, default=0.67)
    p.add_argument("--mae-epsilon", type=float, default=0.005)
    p.add_argument("--selection-mode", choices=["run", "use-best", "manual"], default="run")
    p.add_argument("--best-spec-path", type=Path, default=ARTIFACTS_DIR / "selected_model.json")
    p.add_argument("--manual-candidates", type=str, default="")
    p.add_argument("--manual-model-kind", choices=["nb", "hurdle_nb"], default="nb")
    p.add_argument("--manual-spec-name", type=str, default="manual_spec")
    p.add_argument("--pub-draws", type=int, default=1200)
    p.add_argument("--pub-tune", type=int, default=1200)
    p.add_argument("--pub-chains", type=int, default=4)
    p.add_argument("--pub-target-accept", type=float, default=0.95)
    p.add_argument("--skip-full-application", action="store_true")
    p.add_argument("--apply-level", choices=["both", "municipality", "block"], default="both")
    p.add_argument("--no-resume-application", action="store_true")
    p.add_argument("--municipality-coverage-threshold", type=float, default=0.8)
    p.add_argument("--allow-poor-mcmc", action="store_true")
    p.add_argument("--no-postprocessing", action="store_true")
    p.add_argument("--postprocess-eps-m", type=float, default=40.0)
    p.add_argument("--postprocess-min-samples", type=int, default=3)
    p.add_argument("--uncertainty-only", action="store_true", help="Fit NB with log1p(raw) offset for uncertainty intervals, skipping feature selection")
    p.add_argument(
        "--uncertainty-features",
        action="store_true",
        help="Fit NB offset model with constrained mean coefficients and covariate-driven dispersion",
    )
    p.add_argument(
        "--uncertainty-mean-sigma",
        type=float,
        default=0.15,
        help="Shrinkage sigma for uncertainty-feature mean coefficients",
    )
    p.add_argument("--application-only", action="store_true", help="Skip model fitting and run only municipality/block application using an existing posterior")
    p.add_argument("--posterior-path", type=Path, default=ARTIFACTS_DIR / "final_model_posterior.nc", help="Posterior NetCDF path used when --application-only is set")
    p.add_argument("--spec-path", type=Path, default=ARTIFACTS_DIR / "selected_model.json", help="Model spec JSON path used when --application-only is set")
    p.add_argument("--scaler-path", type=Path, default=ARTIFACTS_DIR / "feature_scaler.json", help="Scaler params JSON path used when --application-only is set")
    p.add_argument("--spline-knots-path", type=Path, default=ARTIFACTS_DIR / "spline_knots.json", help="Spline knots JSON path used when --application-only is set")
    return p


def _mcmc_issues(diag: dict, min_chains: int = 2, min_ess: float = 100.0, max_rhat: float = 1.01) -> list[str]:
    issues = []
    if int(diag.get("n_chains", 0)) < min_chains:
        issues.append(f"n_chains<{min_chains}")
    rhat = diag.get("max_rhat")
    if rhat is None or (isinstance(rhat, float) and np.isnan(rhat)):
        issues.append("max_rhat is NaN")
    elif float(rhat) > max_rhat:
        issues.append(f"max_rhat>{max_rhat}")
    if float(diag.get("min_ess_bulk", 0.0)) < min_ess:
        issues.append(f"min_ess_bulk<{min_ess}")
    if float(diag.get("min_ess_tail", 0.0)) < min_ess:
        issues.append(f"min_ess_tail<{min_ess}")
    if int(diag.get("n_divergences", 0)) > 0:
        issues.append("divergences>0")
    return issues


def run(args: argparse.Namespace) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cli_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    np.random.seed(args.seed)

    if args.application_only:
        print(f"[pipeline] Reproducibility seed={args.seed}")
        print("[pipeline] Application-only mode")
        spec = load_best_spec(args.spec_path)
        spec_payload = json.loads(args.spec_path.read_text()) if args.spec_path.exists() else {}
        if not args.posterior_path.exists():
            raise FileNotFoundError(f"Posterior file not found: {args.posterior_path}")
        fit = az.from_netcdf(args.posterior_path)

        def _posterior_beta_dims() -> tuple[int | None, int | None]:
            count_dim = None
            zero_dim = None
            if "beta_count" in fit.posterior:
                count_dim = int(fit.posterior["beta_count"].shape[-1])
            if "beta_zero" in fit.posterior:
                zero_dim = int(fit.posterior["beta_zero"].shape[-1])
            if "beta" in fit.posterior:
                count_dim = int(fit.posterior["beta"].shape[-1])
            if "beta_mean" in fit.posterior:
                count_dim = int(fit.posterior["beta_mean"].shape[-1]) + 1
            return count_dim, zero_dim

        def _spec_dims() -> tuple[int, int | None]:
            count_cols = spec.count_columns if (spec.model_kind == "hurdle_nb" and spec.count_columns) else spec.final_columns
            zero_cols = spec.zero_columns if (spec.model_kind == "hurdle_nb" and spec.zero_columns) else None
            return len(count_cols), (len(zero_cols) if zero_cols is not None else None)

        post_count_dim, post_zero_dim = _posterior_beta_dims()
        spec_count_dim, spec_zero_dim = _spec_dims()

        if spec.model_kind == "hurdle_nb" and (
            (post_count_dim is not None and post_count_dim != spec_count_dim)
            or (post_zero_dim is not None and spec_zero_dim is not None and post_zero_dim != spec_zero_dim)
        ):
            fr = spec_payload.get("final_run", {}) if isinstance(spec_payload, dict) else {}
            fr_count = fr.get("count_columns")
            fr_zero = fr.get("zero_columns")
            if isinstance(fr_count, list) and isinstance(fr_zero, list):
                if (post_count_dim is None or len(fr_count) == post_count_dim) and (post_zero_dim is None or len(fr_zero) == post_zero_dim):
                    print("[pipeline] Spec/posterior mismatch detected; using final_run component columns from spec JSON to match posterior")
                    spec.count_columns = fr_count
                    spec.zero_columns = fr_zero
                    spec.final_columns = list(dict.fromkeys(fr_count + fr_zero))
                    spec.selected_candidates_count = fr.get("selected_candidates_count", spec.selected_candidates_count)
                    spec.selected_candidates_zero = fr.get("selected_candidates_zero", spec.selected_candidates_zero)
                    spec.selected_candidates = sorted(set((spec.selected_candidates_count or []) + (spec.selected_candidates_zero or [])))

        post_count_dim, post_zero_dim = _posterior_beta_dims()
        spec_count_dim, spec_zero_dim = _spec_dims()
        if spec.model_kind == "hurdle_nb":
            if post_count_dim is not None and post_count_dim != spec_count_dim:
                raise ValueError(
                    "Posterior/spec mismatch for hurdle count component: "
                    f"posterior beta_count has {post_count_dim} coefficients, spec count columns has {spec_count_dim}. "
                    "Use matching spec+posterior artifacts from the same final run."
                )
            if post_zero_dim is not None and spec_zero_dim is not None and post_zero_dim != spec_zero_dim:
                raise ValueError(
                    "Posterior/spec mismatch for hurdle zero component: "
                    f"posterior beta_zero has {post_zero_dim} coefficients, spec zero columns has {spec_zero_dim}. "
                    "Use matching spec+posterior artifacts from the same final run."
                )
        elif post_count_dim is not None and post_count_dim != len(spec.final_columns):
            raise ValueError(
                "Posterior/spec mismatch for NB model: "
                f"posterior beta has {post_count_dim} coefficients, spec final columns has {len(spec.final_columns)}. "
                "Use matching spec+posterior artifacts from the same final run."
            )

        scaler_params = {}
        if args.scaler_path.exists():
            scaler_params = json.loads(args.scaler_path.read_text())
        spline_knots = {}
        if args.spline_knots_path.exists():
            spline_knots = json.loads(args.spline_knots_path.read_text())

        municipality_frame = _read_to_target(MUNICIPALITY_PATH)
        muni_series = municipality_frame["NAME"] if "NAME" in municipality_frame.columns else municipality_frame.iloc[:, 0]
        full_municipality_names = sorted(muni_series.dropna().astype(str).unique().tolist())

        full_apply_summary = apply_calibration_to_full_predictions(
            spec=spec,
            fit=fit,
            municipality_names=full_municipality_names,
            config=ApplicationConfig(
                merged_dir=PROJECT_ROOT / "data" / "tent_predictions" / "merged",
                coverage_dir=PROJECT_ROOT / "data" / "tent_predictions" / "coverage",
                municipality_path=MUNICIPALITY_PATH,
                block_path=BLOCK_PATH,
                spatial_context_dir=PROJECT_ROOT / "data" / "spatial_context",
                output_dir=PROJECT_ROOT / "outputs" / "tent_counts",
                apply_municipality=args.apply_level in {"both", "municipality"},
                apply_block=args.apply_level in {"both", "block"},
                resume=not args.no_resume_application,
                municipality_coverage_threshold=args.municipality_coverage_threshold,
                predictor_bounds=None,
                scaler_params=scaler_params,
                spline_knots=spline_knots,
                prewar_tif_path=PROJECT_ROOT / "data" / "tif_files" / "prewar_gaza.tif",
                land_use_model_path=PROJECT_ROOT / "src" / "validation" / "land_use_classification" / "land_use_kmeans.pkl",
                land_use_labels_path=PROJECT_ROOT / "src" / "validation" / "land_use_classification" / "manual_land_use_labels.json",
                yellow_line_path=YELLOW_LINE_PATH,
                postprocess_eps_m=args.postprocess_eps_m,
                postprocess_min_samples=args.postprocess_min_samples,
            ),
        )
        write_json(ARTIFACTS_DIR / "full_prediction_application_summary.json", full_apply_summary)
        print("Saved artifacts to", ARTIFACTS_DIR)
        print("Selected model kind:", spec.model_kind)
        print("Application summary:", json.dumps(full_apply_summary, indent=2))
        return

    print(f"[pipeline] Reproducibility seed={args.seed}")
    print("[pipeline] Loading validation data")

    df = pd.read_csv(VALIDATION_PATH)
    if not args.no_postprocessing:
        print("[pipeline] Recomputing validation prediction_count with point postprocessing")
        df, post_logs = recompute_validation_prediction_counts(
            validation_df=df,
            merged_dir=MERGED_PREDICTIONS_DIR,
            yellow_line_path=YELLOW_LINE_PATH,
            eps_m=args.postprocess_eps_m,
            min_samples=args.postprocess_min_samples,
        )
        post_logs.to_csv(ARTIFACTS_DIR / "validation_postprocessing_summary.csv", index=False)
        print(f"[pipeline] Postprocessing done for {len(post_logs)} validation weeks")
    df = featurize_validation(df, block_path=BLOCK_PATH)
    df, candidate_groups = build_candidate_feature_table(df)
    print(f"[pipeline] Rows={len(df)}; candidates={len(candidate_groups)}")

    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=args.seed)
    groups = pd.to_datetime(df["date"], errors="coerce").dt.to_period("W-MON").astype(str)
    train_idx, hold_idx = next(splitter.split(df, groups=groups))
    dev = df.iloc[train_idx].copy().reset_index(drop=True)
    holdout = df.iloc[hold_idx].copy().reset_index(drop=True)
    print(f"[pipeline] Split: development={len(dev)} holdout={len(holdout)}")

    def _quantile_knots(frame: pd.DataFrame, col: str) -> list[float]:
        if col not in frame.columns:
            return []
        vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return []
        return [float(np.nanquantile(vals, q)) for q in [0.2, 0.4, 0.6, 0.8]]

    spline_knots = {
        "hours_from_sunrise": _quantile_knots(dev, "hours_from_sunrise"),
        "building_density": _quantile_knots(dev, "building_density"),
    }
    write_json(ARTIFACTS_DIR / "spline_knots.json", spline_knots)

    dev[["h5_index", "date", "municipality"]].assign(split="dev").to_csv(ARTIFACTS_DIR / "development_split.csv", index=False)
    holdout[["h5_index", "date", "municipality"]].assign(split="holdout").to_csv(ARTIFACTS_DIR / "holdout_split.csv", index=False)

    baseline_columns = ["log1p_model_tent_count"]
    all_predictor_cols = sorted({c for c in baseline_columns} | {c for cols in candidate_groups.values() for c in cols})
    scaler_params = fit_scaler_params(dev, all_predictor_cols)
    write_json(ARTIFACTS_DIR / "feature_scaler.json", scaler_params)
    dev = apply_scaler_params(dev, scaler_params)
    holdout = apply_scaler_params(holdout, scaler_params)
    df = apply_scaler_params(df, scaler_params)
    print(f"[pipeline] Applied centering/scaling to {len(scaler_params)} predictor columns")
    cv_args = {
        "model_kind": args.cv_model_kind,
        "split_mode": args.cv_split,
        "repeats": args.cv_repeats,
        "folds": args.cv_folds,
        "draws": args.cv_draws,
        "tune": args.cv_tune,
        "chains": args.cv_chains,
        "target_accept": args.target_accept,
        "seed": args.seed,
    }
    if args.uncertainty_features:
        print("[pipeline] Uncertainty-features mode: NB offset with constrained mean + covariate dispersion")
        uncertainty_feature_cols = sorted({c for cols in candidate_groups.values() for c in cols})
        uncertainty_columns = ["log1p_model_tent_count", *uncertainty_feature_cols]
        spec = ModelSpec(
            name="uncertainty_features_nb_offset",
            selected_candidates=sorted(candidate_groups.keys()),
            final_columns=uncertainty_columns,
            model_kind="nb_offset_features",
            count_columns=uncertainty_columns,
            zero_columns=None,
            selected_candidates_count=sorted(candidate_groups.keys()),
            selected_candidates_zero=None,
        )
        cli_args["uncertainty_feature_columns"] = uncertainty_feature_cols
        write_selected_payload(
            path=ARTIFACTS_DIR / "selected_model.json",
            spec=spec,
            zero_rate_development=float((dev["tent_count"] == 0).mean()),
            remaining=[],
            selection_mode="uncertainty_features",
            run_config=cli_args,
        )
    elif args.uncertainty_only:
        print("[pipeline] Uncertainty-only mode: NB with raw-count offset")
        spec = ModelSpec(
            name="uncertainty_only_nb_offset",
            selected_candidates=[],
            final_columns=["log1p_model_tent_count"],
            model_kind="nb_offset",
            count_columns=["log1p_model_tent_count"],
            zero_columns=None,
            selected_candidates_count=[],
            selected_candidates_zero=None,
        )
        write_selected_payload(
            path=ARTIFACTS_DIR / "selected_model.json",
            spec=spec,
            zero_rate_development=float((dev["tent_count"] == 0).mean()),
            remaining=[],
            selection_mode="uncertainty_only",
            run_config=cli_args,
        )
    elif args.selection_mode == "run":
        print(f"[pipeline] Running forward selection (model_kind={args.cv_model_kind}, cv_split={args.cv_split})")
        spec, _, _, remaining, zero_rate = run_and_select_spec(
            dev=dev,
            candidate_groups=candidate_groups,
            baseline_columns=baseline_columns,
            cv_args=cv_args,
            min_primary_improvement=args.min_primary_improvement,
            min_fold_win_rate=args.min_fold_win_rate,
            mae_epsilon=args.mae_epsilon,
            zero_rate_threshold=args.zero_rate_threshold,
            compare_hurdle_against_nb=(args.cv_model_kind == "nb"),
            artifacts_dir=ARTIFACTS_DIR,
        )
        write_selected_payload(
            path=ARTIFACTS_DIR / "selected_model.json",
            spec=spec,
            zero_rate_development=zero_rate,
            remaining=remaining,
            selection_mode="forward_one_at_a_time",
            run_config=cli_args,
        )
        print(f"[pipeline] Selection complete; selected={spec.selected_candidates}")
    elif args.selection_mode == "use-best":
        print(f"[pipeline] Loading preselected spec from {args.best_spec_path}")
        spec = load_best_spec(args.best_spec_path)
        write_selected_payload(
            path=ARTIFACTS_DIR / "selected_model.json",
            spec=spec,
            zero_rate_development=float((dev["tent_count"] == 0).mean()),
            remaining=[],
            selection_mode="preselected_from_file",
            run_config=cli_args,
        )
    else:
        print("[pipeline] Using manual spec")
        manual_candidates = [x.strip() for x in args.manual_candidates.split(",") if x.strip()]
        spec = manual_spec(
            candidate_groups=candidate_groups,
            baseline_columns=baseline_columns,
            candidate_names=manual_candidates,
            model_kind=args.manual_model_kind,
            spec_name=args.manual_spec_name,
        )
        write_selected_payload(
            path=ARTIFACTS_DIR / "selected_model.json",
            spec=spec,
            zero_rate_development=float((dev["tent_count"] == 0).mean()),
            remaining=[],
            selection_mode="manual",
            run_config=cli_args,
        )

    dropped_columns = []
    for col in list(spec.final_columns):
        series = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
        series = series[np.isfinite(series)] if len(series) else series
        if len(series) and series.nunique(dropna=True) <= 1:
            dropped_columns.append(col)
    if dropped_columns:
        spec.final_columns = [c for c in spec.final_columns if c not in set(dropped_columns)]
        if spec.count_columns is not None:
            spec.count_columns = [c for c in spec.count_columns if c not in set(dropped_columns)]
        if spec.zero_columns is not None:
            spec.zero_columns = [c for c in spec.zero_columns if c not in set(dropped_columns)]
        if "year" in dropped_columns:
            spec.selected_candidates = [c for c in spec.selected_candidates if c != "year"]
        if not spec.final_columns:
            raise ValueError("All selected predictors were constant in validated data; cannot fit calibration model")
        write_json(
            ARTIFACTS_DIR / "dropped_constant_predictors.json",
            {"dropped_columns": dropped_columns, "effective_final_columns": spec.final_columns},
        )
    write_json(
        ARTIFACTS_DIR / "effective_model_spec.json",
        {
            "selected_model_kind": spec.model_kind,
            "selected_candidates": spec.selected_candidates,
            "selected_candidates_count": spec.selected_candidates_count,
            "selected_candidates_zero": spec.selected_candidates_zero,
            "final_columns": spec.final_columns,
            "count_columns": spec.count_columns,
            "zero_columns": spec.zero_columns,
            "spec_name": spec.name,
        },
    )

    hold_metrics, hold_df, dev_ppc, dev_diag, dev_fit = evaluate_holdout_from_dev_fit(
        spec=spec,
        dev=dev,
        holdout=holdout,
        draws=args.pub_draws,
        tune=args.pub_tune,
        chains=args.pub_chains,
        target_accept=args.pub_target_accept,
        mean_coef_sigma=args.uncertainty_mean_sigma,
        seed=args.seed,
    )
    print("[pipeline] Development fit + holdout evaluation complete")
    write_json(ARTIFACTS_DIR / "holdout_metrics.json", hold_metrics)
    write_json(ARTIFACTS_DIR / "development_ppc_metrics.json", dev_ppc)
    write_json(ARTIFACTS_DIR / "development_fit_diagnostics.json", dev_diag)
    hold_df.to_csv(ARTIFACTS_DIR / "holdout_predictions.csv", index=False)
    dev_fit.to_netcdf(ARTIFACTS_DIR / "development_model_posterior.nc")

    full_fit, full_diag, full_municipality_names = refit_all_validated(
        spec=spec,
        full_df=df,
        draws=args.pub_draws,
        tune=args.pub_tune,
        chains=args.pub_chains,
        target_accept=args.pub_target_accept,
        mean_coef_sigma=args.uncertainty_mean_sigma,
        seed=args.seed,
    )
    print("[pipeline] Full-data refit complete")
    write_json(ARTIFACTS_DIR / "final_fit_diagnostics.json", full_diag)
    full_fit.to_netcdf(ARTIFACTS_DIR / "final_model_posterior.nc")

    dev_issues = _mcmc_issues(dev_diag)
    full_issues = _mcmc_issues(full_diag)
    if (dev_issues or full_issues) and not args.allow_poor_mcmc:
        raise RuntimeError(
            "MCMC quality gate failed. "
            f"Development issues: {dev_issues or ['none']}. "
            f"Full-data issues: {full_issues or ['none']}. "
            "Increase --pub-draws/--pub-tune/--pub-chains and adjust --pub-target-accept, "
            "or rerun with --allow-poor-mcmc for debugging only."
        )

    full_apply_summary = {}
    if not args.skip_full_application:
        predictor_bounds = {}
        for col in spec.final_columns:
            series = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
            series = series[np.isfinite(series)] if len(series) else series
            if len(series):
                predictor_bounds[col] = (float(series.min()), float(series.max()))

        full_apply_summary = apply_calibration_to_full_predictions(
            spec=spec,
            fit=full_fit,
            municipality_names=full_municipality_names,
            config=ApplicationConfig(
                merged_dir=PROJECT_ROOT / "data" / "tent_predictions" / "merged",
                coverage_dir=PROJECT_ROOT / "data" / "tent_predictions" / "coverage",
                municipality_path=MUNICIPALITY_PATH,
                block_path=BLOCK_PATH,
                spatial_context_dir=PROJECT_ROOT / "data" / "spatial_context",
                output_dir=PROJECT_ROOT / "outputs" / "tent_counts",
                apply_municipality=args.apply_level in {"both", "municipality"},
                apply_block=args.apply_level in {"both", "block"},
                resume=not args.no_resume_application,
                municipality_coverage_threshold=args.municipality_coverage_threshold,
                predictor_bounds=predictor_bounds,
                scaler_params=scaler_params,
                spline_knots=spline_knots,
                prewar_tif_path=PROJECT_ROOT / "data" / "tif_files" / "prewar_gaza.tif",
                land_use_model_path=PROJECT_ROOT / "src" / "validation" / "land_use_classification" / "land_use_kmeans.pkl",
                land_use_labels_path=PROJECT_ROOT / "src" / "validation" / "land_use_classification" / "manual_land_use_labels.json",
                yellow_line_path=YELLOW_LINE_PATH,
                postprocess_eps_m=args.postprocess_eps_m,
                postprocess_min_samples=args.postprocess_min_samples,
            ),
        )
        write_json(ARTIFACTS_DIR / "full_prediction_application_summary.json", full_apply_summary)

    write_final_run_payload(
        path=ARTIFACTS_DIR / "selected_model.json",
        final_info={
            "cli_args": cli_args,
            "selected_spec": spec.name,
            "selected_model_kind": spec.model_kind,
            "selected_candidates": spec.selected_candidates,
            "selected_candidates_count": spec.selected_candidates_count,
            "selected_candidates_zero": spec.selected_candidates_zero,
            "final_columns": spec.final_columns,
            "count_columns": spec.count_columns,
            "zero_columns": spec.zero_columns,
            "holdout_metrics": hold_metrics,
            "development_fit_diagnostics": dev_diag,
            "full_fit_diagnostics": full_diag,
            "mcmc_quality_gate": {
                "allow_poor_mcmc": bool(args.allow_poor_mcmc),
                "development_issues": dev_issues,
                "full_issues": full_issues,
            },
            "full_prediction_application_summary": full_apply_summary,
            "artifacts_dir": str(ARTIFACTS_DIR),
        },
    )

    print("Saved artifacts to", ARTIFACTS_DIR)
    print("Selected candidates:", spec.selected_candidates)
    print("Selected model kind:", spec.model_kind)
    print("Holdout metrics:", json.dumps(hold_metrics, indent=2))
    print("Development fit diagnostics:", json.dumps(dev_diag, indent=2))
    print("Full-data fit diagnostics:", json.dumps(full_diag, indent=2))
    if full_apply_summary:
        print("Full prediction application summary:", json.dumps(full_apply_summary, indent=2))


if __name__ == "__main__":
    run(build_parser().parse_args())
