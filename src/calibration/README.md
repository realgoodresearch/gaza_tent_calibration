# Calibration pipeline

Run:

`python calibration/pipeline.py`

Use a previously selected best spec without rerunning selection:

`python calibration/pipeline.py --selection-mode use-best`

Run only municipal-level weekly correction output (resumable by default):

`python calibration/pipeline.py --selection-mode use-best --apply-level municipality`

Manually specify feature groups:

`python calibration/pipeline.py --selection-mode manual --manual-candidates year,dist_base_log1p,is_humanitarian_zone --manual-model-kind nb`

Code structure:

- `calibration/pipeline.py` - thin orchestration entrypoint
- `calibration/workflow/features.py` - feature engineering and candidate groups
- `calibration/workflow/models.py` - Bayesian NB / hurdle-NB model definitions
- `calibration/workflow/evaluation.py` - blocked CV + holdout metrics
- `calibration/workflow/selection.py` - forward one-at-a-time variable selection
- `calibration/workflow/specs.py` - model-spec selection/loading/writing
- `calibration/workflow/publication.py` - publication-level fit, diagnostics, PPC, holdout evaluation, full-data refit
- `calibration/workflow/application.py` - apply fixed calibrated model to weekly point predictions on 100m grid and aggregate posterior counts

This pipeline does the following:

1. Loads `validation/prediction_accuracy_results.csv`.
2. Splits validated tiles into 80/20 development/holdout using week-blocked grouping.
3. Fits repeated blocked cross-validation on a mandatory baseline model (`log1p(prediction_count)` + partially pooled municipality effect).
4. Tests one candidate predictor group at a time against the current model (splines as grouped terms, azimuth as sin/cos pairs), ranked by aggregate municipality/block/week count error.
5. Greedily adds predictors only when improvement is both meaningful and consistent across CV folds/repeats.
6. Optionally compares the selected NB feature set against hurdle-NB when zero-count prevalence is high.
7. Locks the selected model/specification.
8. Fits the locked specification on development tiles only using publication MCMC settings (separate `--pub-*` CLI flags).
9. Generates held-out predictions from the development posterior draws with no refit/tuning on holdout and writes holdout metrics.
10. Writes MCMC diagnostics and lightweight posterior predictive check summaries.
11. Refits the identical locked specification on all validated tiles using the same publication MCMC settings.
12. Applies the fixed calibrated model to the full `data/tent_predictions/merged` point predictions by week on a fixed 100m grid, and writes municipality/block weekly posterior count summaries with credible intervals and change metrics.

Artifacts are written to `calibration/artifacts/`:

- `development_split.csv`
- `holdout_split.csv`
- `forward_selection_results.csv`
- `selected_variable_contributions.csv`
- `cv_results_selected_nb.csv` (when zero-rate gate triggers)
- `cv_results_selected_hurdle_nb.csv` (when zero-rate gate triggers)
- `selected_model.json`
- `holdout_metrics.json`
- `holdout_predictions.csv`
- `development_ppc_metrics.json`
- `development_fit_diagnostics.json`
- `final_fit_diagnostics.json`
- `development_model_posterior.nc`
- `final_model_posterior.nc`
- `full_prediction_application_summary.json`

Main output tables:

- `outputs/tent_counts/municipality_counts_by_week.csv`
- `outputs/tent_counts/block_counts_by_week.csv`

Notes:

- Full-prediction correction writes CSV rows incrementally by week.
- If interrupted, rerun with the same command; completed `week_key` values in `municipality_counts_by_week.csv` are skipped automatically.
- Use `--no-resume-application` to force restart from scratch.
- Municipality rows are written only when weekly coverage is at least the threshold (default `--municipality-coverage-threshold 0.8`).

Variable contribution outputs:

- `forward_selection_results.csv` contains every candidate tested at every selection step with its marginal improvement and consistency flags.
- `selected_variable_contributions.csv` contains only accepted predictors and their stepwise marginal gains.

Recommended first run for quick smoke test:

`python calibration/pipeline.py --cv-draws 100 --cv-tune 100 --pub-draws 200 --pub-tune 200 --pub-chains 1`

Recommended publication-style run after spec selection exists:

`python calibration/pipeline.py --selection-mode use-best --pub-draws 1200 --pub-tune 1200 --pub-chains 4 --pub-target-accept 0.95`
