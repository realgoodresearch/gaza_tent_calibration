"""Relate repeated evacuation orders to block-level tent settlement turnover."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_ind

REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTS_PATH = REPO_ROOT / "outputs/tent_counts/block_counts_by_week.csv"
ORDERS_PATH = REPO_ROOT / "data/military_activity/evac-orders-grid.geojson"
BLOCKS_PATH = REPO_ROOT / "data/gaza_boundaries/dash_population_blocks.geojson"
OUTPUT_PATH = REPO_ROOT / "outputs/analysis/evacuation_order_settlement_turnover.json"

DECLINE_FRACTION = 0.50
RECOVERY_FRACTION_OF_LOSS = 0.50
RECOVERY_MIN_WEEKS = 4
RECOVERY_MAX_WEEKS = 12


def _week_start(week_key: str) -> pd.Timestamp:
    return pd.to_datetime(str(week_key).split("-")[0], format="%Y%m%d")


def _find_turnover_episodes(block_weeks: pd.DataFrame) -> list[dict]:
    """Return non-overlapping decline/recovery episodes for one block."""
    series = block_weeks.sort_values("week_start").reset_index(drop=True)
    episodes: list[dict] = []
    i = 1
    while i < len(series):
        prior = series.iloc[i - 1]
        trough = series.iloc[i]
        consecutive = (trough.week_start - prior.week_start).days == 7
        qualifies = (
            consecutive
            and prior.posterior_median_tents > 0
            and trough.posterior_median_tents <= (1 - DECLINE_FRACTION) * prior.posterior_median_tents
        )
        if not qualifies:
            i += 1
            continue

        lost = prior.posterior_median_tents - trough.posterior_median_tents
        recovery_threshold = trough.posterior_median_tents + RECOVERY_FRACTION_OF_LOSS * lost
        days_after = (series.week_start - trough.week_start).dt.days
        candidates = series[
            days_after.between(7 * RECOVERY_MIN_WEEKS, 7 * RECOVERY_MAX_WEEKS)
            & (series.posterior_median_tents >= recovery_threshold)
        ]
        if candidates.empty:
            i += 1
            continue

        recovery = candidates.iloc[0]
        episodes.append({
            "decline_from_week": prior.week_key,
            "decline_to_week": trough.week_key,
            "recovery_week": recovery.week_key,
            "pre_decline_posterior_median_tents": float(prior.posterior_median_tents),
            "post_decline_posterior_median_tents": float(trough.posterior_median_tents),
            "decline_percent": float(
                (trough.posterior_median_tents - prior.posterior_median_tents)
                / prior.posterior_median_tents * 100
            ),
            "recovery_posterior_median_tents": float(recovery.posterior_median_tents),
            "weeks_to_recovery": int((recovery.week_start - trough.week_start).days // 7),
        })
        # A recovery closes the episode; scanning resumes afterward so one
        # settlement cycle cannot be counted more than once.
        i = int(recovery.name) + 1
    return episodes


def _welch_mean_difference(high: np.ndarray, low: np.ndarray) -> dict:
    test = ttest_ind(high, low, equal_var=False)
    difference = float(np.mean(high) - np.mean(low))
    standard_error = float(np.sqrt(np.var(high, ddof=1) / len(high) + np.var(low, ddof=1) / len(low)))
    degrees_of_freedom = float(test.df)
    margin = float(student_t.ppf(0.975, degrees_of_freedom) * standard_error)
    return {
        "mean_difference_episodes_per_block": difference,
        "mean_difference_95_percent_confidence_interval": [difference - margin, difference + margin],
        "t_statistic": float(test.statistic),
        "degrees_of_freedom": degrees_of_freedom,
        "p_value_two_sided": float(test.pvalue),
    }


def main() -> None:
    counts = pd.read_csv(COUNTS_PATH)
    counts["block_id"] = pd.to_numeric(counts.block, errors="coerce").astype("Int64")
    counts["corrected_count_median"] = pd.to_numeric(counts.corrected_count_median, errors="coerce")
    counts = counts.dropna(subset=["block_id", "week_key", "corrected_count_median"])
    # A block can be split across municipality/tile rows; counts are additive.
    counts = counts.groupby(["block_id", "week_key"], as_index=False).agg(
        posterior_median_tents=("corrected_count_median", "sum")
    )
    counts["week_start"] = counts.week_key.map(_week_start)

    blocks = gpd.read_file(BLOCKS_PATH).rename(columns={"Name": "block_id"})
    blocks["block_id"] = pd.to_numeric(blocks.block_id, errors="coerce").astype("Int64")
    block_ids = sorted(set(blocks.block_id.dropna().astype(int)))

    orders = gpd.read_file(ORDERS_PATH)
    orders["block_id"] = pd.to_numeric(orders.name, errors="coerce").astype("Int64")
    orders["order_date"] = pd.to_datetime(orders.order_date, errors="coerce")
    distinct_orders = (
        orders.dropna(subset=["block_id", "order_date"])
        .drop_duplicates(["block_id", "order_date"])
        .groupby("block_id").order_date.nunique()
    )

    block_results = []
    for block_id in block_ids:
        episodes = _find_turnover_episodes(counts[counts.block_id == block_id])
        block_results.append({
            "block_id": block_id,
            "distinct_evacuation_events": int(distinct_orders.get(block_id, 0)),
            "turnover_episodes": len(episodes),
            "episodes": episodes,
        })

    block_frame = pd.DataFrame([
        {key: row[key] for key in ("block_id", "distinct_evacuation_events", "turnover_episodes")}
        for row in block_results
    ])
    high = block_frame[block_frame.distinct_evacuation_events.between(2, 3)].turnover_episodes.to_numpy(float)
    low = block_frame[block_frame.distinct_evacuation_events.between(0, 1)].turnover_episodes.to_numpy(float)
    high_mean, low_mean = float(np.mean(high)), float(np.mean(low))
    high_any, low_any = float(np.mean(high > 0)), float(np.mean(low > 0))

    output = {
        "analysis": "Association between repeated evacuation orders and repeated block-level tent settlement turnover.",
        "turnover_definition": (
            "A non-overlapping episode is a >=50% decline in aggregated posterior-median tent count from the immediately preceding consecutive week, followed 4-12 weeks after the decline by recovery of >=50% of the number of tents lost."
        ),
        "evacuation_event_definition": "A distinct dated evacuation order containing the block.",
        "comparison": "Blocks with 2-3 distinct evacuation events versus blocks with 0-1 distinct events; blocks with more than 3 events are excluded from the comparison.",
        "groups": {
            "two_to_three_events": {
                "n_blocks": int(len(high)),
                "mean_turnover_episodes_per_block": high_mean,
                "blocks_with_at_least_one_episode": int(np.sum(high > 0)),
                "proportion_with_at_least_one_episode": high_any,
            },
            "zero_to_one_events": {
                "n_blocks": int(len(low)),
                "mean_turnover_episodes_per_block": low_mean,
                "blocks_with_at_least_one_episode": int(np.sum(low > 0)),
                "proportion_with_at_least_one_episode": low_any,
            },
        },
        "effect_estimates": {
            "times_as_many_mean_turnover_episodes": high_mean / low_mean if low_mean > 0 else None,
            "difference_in_mean_turnover_episodes_per_block": high_mean - low_mean,
            "difference_in_proportion_with_any_episode_percentage_points": (high_any - low_any) * 100,
            "welch_t_test_of_mean_episode_counts": _welch_mean_difference(high, low),
        },
        "blocks": block_results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
