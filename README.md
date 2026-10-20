# Gaza displacement analysis

This repository estimates the spatial and temporal distribution of tents in the Gaza Strip from satellite-derived detections. It validates automated detections against manually counted 100 × 100 m tiles, fits a Bayesian calibration model, propagates posterior uncertainty into weekly municipality and evacuation-block estimates, and relates those estimates to evacuation orders, humanitarian zones, incidents, destruction, and military activity.

## Method overview

The workflow has four stages:

1. **Validation:** compare automated detections with manual tent counts in sampled 100 × 100 m tiles.
2. **Post-processing:** remove isolated detections with DBSCAN and detections excluded by the contemporaneous Yellow Line.
3. **Calibration:** fit negative-binomial or hurdle negative-binomial models in PyMC, with a partially pooled municipality effect and selected spatial and acquisition-context predictors. Model selection is performed on the development set; the locked specification is evaluated on an independent holdout set before being refitted to all validated tiles.
4. **Aggregation and analysis:** apply the final calibration model to weekly detections, aggregate posterior draws to municipalities and evacuation blocks, and calculate longitudinal and spatial-context measures.

## Repository structure

```text
data/
  gaza_boundaries/       Municipal and evacuation-block boundaries
  incidents/             Geolocated incident records
  military_activity/     Evacuation orders, arrows, tanks, berms, etc.
  spatial_context/       Humanitarian zone, Yellow Line, land and damage layers
  tent_predictions/      Weekly detections and coverage layers
outputs/
  analysis/              JSON and CSV analysis results
  calibration/           Publication-ready calibration tables
  tent_counts/           Weekly calibrated municipality and block counts
  validation/            Validation comparison tables
  visualisation/         Figures and figure data
src/
  analysis/              Statistical and substantive analysis scripts
  calibration/           Bayesian calibration pipeline and model code
  validation/            Validation sampling, review, and diagnostic scripts
  visualisation/         Publication figure scripts
```

## Installation

Python 3.11 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The geospatial dependencies may also require GDAL/GEOS system libraries, depending on the Python distribution and operating system.

## Calibration workflow

The calibration entry point is `src/calibration/pipeline.py`. A quick smoke test can be run with reduced MCMC settings:

```bash
PYTENSOR_FLAGS='base_compiledir=/tmp/pytensor' \
MPLCONFIGDIR=.mplconfig \
python src/calibration/pipeline.py \
  --cv-draws 100 \
  --cv-tune 100 \
  --pub-draws 200 \
  --pub-tune 200 \
  --pub-chains 1
```

To run the locked selected specification with publication settings:

```bash
PYTENSOR_FLAGS='base_compiledir=/tmp/pytensor' \
MPLCONFIGDIR=.mplconfig \
python src/calibration/pipeline.py \
  --seed 2026 \
  --selection-mode use-best \
  --best-spec-path src/calibration/artifacts/selected_model.json \
  --pub-draws 2000 \
  --pub-tune 2000 \
  --pub-chains 4 \
  --pub-target-accept 0.995 \
  --apply-level both \
  --municipality-coverage-threshold 0.8 \
  --no-resume-application
```

The principal calibration outputs are:

- `src/calibration/artifacts/holdout_metrics.json` — final out-of-sample performance.
- `src/calibration/artifacts/holdout_predictions.csv` — tile-level holdout predictions.
- `src/calibration/artifacts/selected_model.json` — locked model specification and run metadata.
- `src/calibration/artifacts/development_fit_diagnostics.json` and `final_fit_diagnostics.json` — MCMC diagnostics.
- `outputs/tent_counts/municipality_counts_by_week.csv` — weekly municipality posterior summaries.
- `outputs/tent_counts/block_counts_by_week.csv` — weekly evacuation-block posterior summaries.

See `src/calibration/README.md` for model-selection and artifact details.

## Mastergrid application

Apply the fitted posterior to a supplied EPSG:32636 mastergrid without refitting:

```bash
python src/calibration/apply_to_mastergrid.py --week-key 20260803-20260809
```

The command reads the matching point GeoPackage and coverage GeoJSON from
`data/tent_predictions/`, preserves the mastergrid transform and valid-data mask,
and writes an eight-band GeoTIFF plus a GeoPackage of postprocessed model-input points
to `outputs/tent_counts/mastergrid/`. Its bands are valid-mastergrid mask, approximately
10-m coverage fraction, center-based prediction availability, source TentNetFA points,
postprocessed model-input points, posterior mean adjusted count, and 2.5%/97.5% credible
bounds for the expected count. The first run builds a
mastergrid-specific land-use cache. Use `--help` for explicit path overrides.

## Analysis scripts

Scripts are designed to run from the repository root. Important outputs include:

```bash
# Holdout accuracy, contextual performance, aggregate error, and CrI widths
python src/analysis/count_accuracy_and_confidence.py

# Gaza-wide weekly mean count estimates
python src/analysis/gaza_wide_counts.py

# Municipality trends and manuscript summary statistics
python src/analysis/summary_statistics.py

# Humanitarian-zone concentration
python src/analysis/humanitarian_zone_tent_concentration.py

# Evacuation direction and settlement turnover
python src/analysis/evacuation_direction_tent_concentration.py
python src/analysis/evacuation_order_settlement_turnover.py

# Incident exposure and subsequent military activity
python src/analysis/incident_tent_exposure.py
python src/analysis/military_action_tent_change.py
```

`outputs/analysis/gaza_wide_counts_by_week.csv` contains the Gaza-wide weekly corrected means. `outputs/analysis/count_accuracy_and_confidence.json` contains the values used to report holdout and uncertainty performance.

## Figures

Publication figures are written to `outputs/visualisation/`. Examples include:

```bash
MPLCONFIGDIR=.mplconfig python src/visualisation/displacement_heatmap.py
MPLCONFIGDIR=.mplconfig python src/visualisation/event_tent_density_change_maps.py
MPLCONFIGDIR=.mplconfig python src/visualisation/holdout_accuracy_by_context.py
MPLCONFIGDIR=.mplconfig python src/visualisation/example_predictions.py
MPLCONFIGDIR=.mplconfig python src/visualisation/spatial_context_maps.py
MPLCONFIGDIR=.mplconfig python src/visualisation/humanitarian_zone_tent_concentration.py
MPLCONFIGDIR=.mplconfig python src/visualisation/tent_exposure_to_military_activity.py
```

Most scripts provide optional path and rendering arguments; run a script with `--help` for details.

## Validation data semantics

`src/validation/validate_predictions.py` writes tile-level matching results to `src/validation/prediction_accuracy_results.csv`:

- `prediction_count` is numeric only when a tile has matched prediction coverage.
- `prediction_count` is missing when prediction matching is unavailable; missing coverage is not treated as zero tents.
- `prediction_available` identifies rows included in accuracy calculations.
- `prediction_status` records conditions such as `ok`, `missing_prediction_week`, `missing_date_coverage`, or `missing_region_coverage`.

The validation comparison tables in `outputs/validation/` distinguish original and post-processed detections. Calibration holdout predictions are generated from the locked development-set model without refitting or tuning on the holdout data.

## Large local artifacts

Several required inputs and generated posterior files are intentionally excluded from Git because of their size:

- `data/tent_predictions/**/*.gpkg`
- `src/validation/*.h5`
- `src/calibration/artifacts/*.nc`
- raster `.tif` files

These files must be transferred or regenerated separately before rerunning the complete workflow. Do not remove them from the local working directory merely because they are ignored by Git.

## Reproducibility notes

- The primary analysis CRS is EPSG:32636.
- Tile-level validation and calibration operate on 100 × 100 m cells.
- Municipality estimates are emitted only when weekly imagery covers at least 80% of the municipality by default.
- Weekly correction can resume from existing output rows unless `--no-resume-application` is supplied.
- MCMC seeds are configurable; the publication workflow uses seed 2026.
- Posterior NetCDF files are needed for analyses that operate directly on draws rather than stored marginal summaries.
