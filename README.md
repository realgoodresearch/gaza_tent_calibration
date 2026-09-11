# Displacement Analysis: Gaza Strip

This repo is organized for exploratory analysis of displacement patterns in the Gaza Strip using tent prediction coverage data and related spatial context.

## Structure

- `data/` — raw and processed datasets
  - `incidents/` — incident data and conversion scripts
  - `evac_orders/` — evacuation order extraction and related CSVs
  - `spatial_context/` — spatial GIS layers, maps, and zone data
  - `tent_predictions/` — tent coverage and prediction geospatial outputs
- `src/` — analysis code
  - `analysis/` — displacement and movement analysis scripts
  - `visualization/` — plotting and map generation scripts
  - `utils/` — helper functions and data-processing utilities
- `notebooks/` — analysis notebooks
- `outputs/` — generated charts, maps, and summary tables

## Typical workflow

1. Review source data in `data/`
2. Build preprocessing/aggregation scripts in `src/analysis/`
3. Explore results in `notebooks/`
4. Save outputs to `outputs/`

## Notes

This is intentionally a simple repo structure for focused analysis without over-engineering.

## Validation output semantics

`validation/validate_predictions.py` writes per-tile prediction-matching fields to
`validation/prediction_accuracy_results.csv`:

- `prediction_count` is numeric only when predictions are available for that tile/week.
- `prediction_count` is `NaN` when prediction matching is unavailable (for example,
  no week-level prediction file or no temporal/regional coverage).
- `prediction_available` indicates whether a row is scored (`True`) or excluded from
  accuracy/error metrics (`False`).
- `prediction_status` explains why a row is scored or not, with values such as:
  - `ok`
  - `missing_prediction_week`
  - `missing_date_coverage`
  - `missing_region_coverage`

This ensures missing prediction coverage is not silently treated as zero tents.
