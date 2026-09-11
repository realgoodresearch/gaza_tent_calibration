from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = REPO_ROOT / "outputs" / "tent_counts" / "municipality_counts_by_week.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis"
OUTPUT_JSON = OUTPUT_DIR / "summary_statistics.json"
OUTPUT_CHANGES_CSV = OUTPUT_DIR / "municipality_week_to_week_changes.csv"

MUNICIPALITY_PATH = REPO_ROOT / "data" / "gaza_boundaries" / "municipal" / "GazaStrip_MunicipalBoundaries.shp"
MERGED_DIR = REPO_ROOT / "data" / "tent_predictions" / "merged"
DESTRUCTION_PATH = REPO_ROOT / "data" / "spatial_context" / "destruction.json"
TARGET_CRS = "EPSG:32636"

EXCLUDE_WEEKS = {
    "20260518-20260524",
    "20260525-20260531",
    "20260601-20260607",
}
BOUNCE_BACK_TOLERANCE = 0.30
CHANGE_CI_DRAWS = 50000
CHANGE_CI_SEED = 2026
EVENTS = [
    ("2023-10-13", "North Gaza evacuation order"),
    ("2024-05-06", "Rafah evacuation order"),
    ("2025-03-18", "End of ceasefire"),
    ("2025-09-09", "Gaza City evacuation order"),
]
NORTH_MUNICIPALITIES = {"Gaza", "Jabalya", "Beit Hanoun", "Beit Lahiya"}
CEASEFIRE_START = pd.Timestamp("2025-01-19")
CEASEFIRE_END = pd.Timestamp("2025-03-18")
DESTRUCTION_CUTOFF = pd.Timestamp("2025-01-19")


def _parse_week_start(week_key: str) -> pd.Timestamp:
    return pd.to_datetime(str(week_key).split("-")[0], format="%Y%m%d", errors="coerce")


def _parse_week_end(week_key: str) -> pd.Timestamp:
    return pd.to_datetime(str(week_key).split("-")[1], format="%Y%m%d", errors="coerce")


def _normal_sd_from_95(p2_5: np.ndarray, p97_5: np.ndarray) -> np.ndarray:
    z = 1.959963984540054
    width = np.maximum(p97_5 - p2_5, 0.0)
    sd = width / (2.0 * z)
    return np.where(sd <= 0, 1e-9, sd)


def _load_base_df() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    required = {
        "Municipality",
        "week_key",
        "corrected_count_mean",
        "corrected_count_p2_5",
        "corrected_count_p97_5",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {sorted(missing)}")

    out = df[
        [
            "Municipality",
            "week_key",
            "corrected_count_mean",
            "corrected_count_p2_5",
            "corrected_count_p97_5",
        ]
    ].copy()
    out = out.rename(columns={"Municipality": "municipality", "corrected_count_mean": "count"})
    out["count"] = pd.to_numeric(out["count"], errors="coerce")
    out["count_p2_5"] = pd.to_numeric(out["corrected_count_p2_5"], errors="coerce")
    out["count_p97_5"] = pd.to_numeric(out["corrected_count_p97_5"], errors="coerce")
    out = out[
        out["week_key"].notna()
        & out["municipality"].notna()
        & out["count"].notna()
        & out["count_p2_5"].notna()
        & out["count_p97_5"].notna()
    ].copy()
    out = out[~out["week_key"].astype(str).isin(EXCLUDE_WEEKS)].copy()
    out["week_start"] = out["week_key"].map(_parse_week_start)
    out["week_end"] = out["week_key"].map(_parse_week_end)
    out = out[out["week_start"].notna()].sort_values(["municipality", "week_start"]).copy()
    return out


def _sum_with_mc_ci(rows: pd.DataFrame, seed: int = 2026, draws: int = 30000) -> dict:
    means = rows["count"].to_numpy(float)
    p2 = rows["count_p2_5"].to_numpy(float)
    p97 = rows["count_p97_5"].to_numpy(float)
    sds = _normal_sd_from_95(p2, p97)
    rng = np.random.default_rng(seed)
    sim = rng.normal(loc=means, scale=sds, size=(draws, len(means)))
    sim = np.clip(sim, 0.0, None)
    totals = sim.sum(axis=1)
    return {
        "mean": float(np.sum(means)),
        "p2_5": float(np.quantile(totals, 0.025)),
        "p97_5": float(np.quantile(totals, 0.975)),
    }


def _nearest_week_per_muni(df: pd.DataFrame, target_date: pd.Timestamp, munis: set[str]) -> pd.DataFrame:
    pick = []
    for muni in sorted(munis):
        g = df[df["municipality"] == muni].copy()
        if g.empty:
            continue
        idx = (g["week_start"] - target_date).abs().idxmin()
        pick.append(df.loc[idx])
    if not pick:
        return pd.DataFrame(columns=df.columns)
    return pd.DataFrame(pick)


def _global_week_pre_post(df: pd.DataFrame, target_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    weeks = sorted(df["week_start"].dropna().unique().tolist())
    pre = [w for w in weeks if w <= target_date]
    post = [w for w in weeks if w >= target_date]
    pre_w = pre[-1] if pre else weeks[0]
    post_w = post[0] if post else weeks[-1]
    return pd.Timestamp(pre_w), pd.Timestamp(post_w)


def _event_windows(
    df: pd.DataFrame, target_date: pd.Timestamp, event_label: str
) -> tuple[list[pd.Timestamp], list[pd.Timestamp], bool, bool]:
    weeks = sorted(pd.to_datetime(df["week_start"].dropna().unique()).tolist())
    pre_weeks = [w for w in weeks if w < target_date]
    post_weeks = [w for w in weeks if w >= target_date]
    post_n = 8 if event_label == "North Gaza evacuation order" else 3
    pre_n = 3
    pre_short = len(pre_weeks) < pre_n
    post_short = len(post_weeks) < post_n
    if len(pre_weeks) == 0 or len(post_weeks) == 0:
        raise ValueError(f"No usable pre or post weeks around event {target_date.date()}.")
    return pre_weeks[-pre_n:], post_weeks[:post_n], pre_short, post_short


def _directional_event_stats(df: pd.DataFrame) -> list[dict]:
    all_munis = set(df["municipality"].unique().tolist())
    south_munis = all_munis - NORTH_MUNICIPALITIES
    out = []
    for i, (d, label) in enumerate(EVENTS, start=1):
        event_date = pd.Timestamp(d)
        pre_window, post_window, pre_short, post_short = _event_windows(df, event_date, label)
        pre = df[df["week_start"].isin(pre_window)].copy()
        post = df[df["week_start"].isin(post_window)].copy()

        # Average municipality-level counts/CrIs over each 3-week window first,
        # then aggregate north/south to reduce week-to-week noise.
        pre = (
            pre.groupby("municipality", as_index=False)
            .agg(count=("count", "mean"), count_p2_5=("count_p2_5", "mean"), count_p97_5=("count_p97_5", "mean"))
        )
        post = (
            post.groupby("municipality", as_index=False)
            .agg(count=("count", "mean"), count_p2_5=("count_p2_5", "mean"), count_p97_5=("count_p97_5", "mean"))
        )

        pre_north = _sum_with_mc_ci(pre[pre["municipality"].isin(NORTH_MUNICIPALITIES)], seed=1000 + i)
        pre_south = _sum_with_mc_ci(pre[pre["municipality"].isin(south_munis)], seed=2000 + i)
        post_north = _sum_with_mc_ci(post[post["municipality"].isin(NORTH_MUNICIPALITIES)], seed=3000 + i)
        post_south = _sum_with_mc_ci(post[post["municipality"].isin(south_munis)], seed=4000 + i)

        # MC on shares/direction
        rng = np.random.default_rng(5000 + i)
        def _sample_tot(rows: pd.DataFrame):
            means = rows["count"].to_numpy(float)
            sds = _normal_sd_from_95(rows["count_p2_5"].to_numpy(float), rows["count_p97_5"].to_numpy(float))
            sim = rng.normal(means, sds, size=(20000, len(means)))
            return np.clip(sim, 0.0, None).sum(axis=1)

        pre_n = _sample_tot(pre[pre["municipality"].isin(NORTH_MUNICIPALITIES)])
        pre_s = _sample_tot(pre[pre["municipality"].isin(south_munis)])
        post_n = _sample_tot(post[post["municipality"].isin(NORTH_MUNICIPALITIES)])
        post_s = _sample_tot(post[post["municipality"].isin(south_munis)])
        pre_share = pre_n / np.maximum(pre_n + pre_s, 1e-9)
        post_share = post_n / np.maximum(post_n + post_s, 1e-9)
        delta_share = post_share - pre_share
        delta_s_minus_n = (post_s - post_n) - (pre_s - pre_n)

        # Municipality-level largest change across pre/post windows.
        muni = pre[["municipality", "count", "count_p2_5", "count_p97_5"]].rename(
            columns={
                "count": "pre_count",
                "count_p2_5": "pre_count_p2_5",
                "count_p97_5": "pre_count_p97_5",
            }
        ).merge(
            post[["municipality", "count", "count_p2_5", "count_p97_5"]].rename(
                columns={
                    "count": "post_count",
                    "count_p2_5": "post_count_p2_5",
                    "count_p97_5": "post_count_p97_5",
                }
            ),
            on="municipality",
            how="inner",
        )
        muni["change"] = muni["post_count"] - muni["pre_count"]
        muni["pct_change"] = np.where(
            np.abs(muni["pre_count"].to_numpy(float)) > 1e-9,
            muni["change"] / muni["pre_count"] * 100.0,
            np.nan,
        )
        muni_top = muni.iloc[int(np.nanargmax(np.abs(muni["change"].to_numpy(float))))]

        north_mag = (post_n - pre_n)
        south_mag = (post_s - pre_s)
        north_pct = north_mag / np.maximum(pre_n, 1e-9) * 100.0
        south_pct = south_mag / np.maximum(pre_s, 1e-9) * 100.0
        region_selected = "north" if np.mean(np.abs(north_mag)) >= np.mean(np.abs(south_mag)) else "south"
        region_mag = north_mag if region_selected == "north" else south_mag
        region_pct = north_pct if region_selected == "north" else south_pct

        # CI for municipality-level top change (conservative bound from marginal CrIs).
        muni_change_low = float(muni_top["post_count_p2_5"] - muni_top["pre_count_p97_5"])
        muni_change_high = float(muni_top["post_count_p97_5"] - muni_top["pre_count_p2_5"])

        out.append(
            {
                "event": label,
                "event_date": d,
                "pre_weeks": [pd.Timestamp(x).strftime("%Y-%m-%d") for x in pre_window],
                "post_weeks": [pd.Timestamp(x).strftime("%Y-%m-%d") for x in post_window],
                "window_truncation": {
                    "pre_shorter_than_3_weeks": pre_short,
                    "post_shorter_than_3_weeks": post_short,
                },
                "north_total_pre": pre_north,
                "north_total_post": post_north,
                "south_total_pre": pre_south,
                "south_total_post": post_south,
                "regional_magnitude_shift": {
                    "region": region_selected,
                    "direction": "increase" if float(np.mean(region_mag)) >= 0 else "decrease",
                    "percent_change_mean": float(np.mean(region_pct)),
                    "mag_change_mean": float(np.mean(region_mag)),
                    "mag_change_p2_5": float(np.quantile(region_mag, 0.025)),
                    "mag_change_p97_5": float(np.quantile(region_mag, 0.975)),
                    "abs_mag_change_mean": float(np.mean(np.abs(region_mag))),
                    "abs_mag_change_p2_5": float(np.quantile(np.abs(region_mag), 0.025)),
                    "abs_mag_change_p97_5": float(np.quantile(np.abs(region_mag), 0.975)),
                    "percent_change_p2_5": float(np.quantile(region_pct, 0.025)),
                    "percent_change_p97_5": float(np.quantile(region_pct, 0.975)),
                },
                "municipality_greatest_change": {
                    "municipality": str(muni_top["municipality"]),
                    "pre_count": float(muni_top["pre_count"]),
                    "post_count": float(muni_top["post_count"]),
                    "change": float(muni_top["change"]),
                    "change_95_cri_conservative": [muni_change_low, muni_change_high],
                    "percent_change": None if pd.isna(muni_top["pct_change"]) else float(muni_top["pct_change"]),
                },
            }
        )
    return out


def _north_ceasefire_stats(df: pd.DataFrame) -> dict:
    start_rows = _nearest_week_per_muni(df, CEASEFIRE_START, NORTH_MUNICIPALITIES)
    end_rows = _nearest_week_per_muni(df, CEASEFIRE_END, NORTH_MUNICIPALITIES)
    start_sum = _sum_with_mc_ci(start_rows, seed=7101)
    end_sum = _sum_with_mc_ci(end_rows, seed=7102)

    rng = np.random.default_rng(7103)
    def _sim(rows: pd.DataFrame):
        m = rows["count"].to_numpy(float)
        s = _normal_sd_from_95(rows["count_p2_5"].to_numpy(float), rows["count_p97_5"].to_numpy(float))
        z = np.clip(rng.normal(m, s, size=(30000, len(m))), 0.0, None)
        return z.sum(axis=1)

    s0 = _sim(start_rows)
    s1 = _sim(end_rows)
    pct = (s1 - s0) / np.maximum(s0, 1e-9) * 100.0

    return {
        "after_ceasefire_date": str(CEASEFIRE_START.date()),
        "end_ceasefire_date": str(CEASEFIRE_END.date()),
        "north_municipalities": sorted(NORTH_MUNICIPALITIES),
        "start_snapshot": {
            "sum": start_sum,
            "municipality_weeks_used": start_rows[["municipality", "week_key"]].to_dict("records"),
        },
        "end_snapshot": {
            "sum": end_sum,
            "municipality_weeks_used": end_rows[["municipality", "week_key"]].to_dict("records"),
        },
        "percent_increase": {
            "mean": float(np.mean(pct)),
            "p2_5": float(np.quantile(pct, 0.025)),
            "p97_5": float(np.quantile(pct, 0.975)),
        },
    }


def _load_municipalities() -> gpd.GeoDataFrame:
    m = gpd.read_file(MUNICIPALITY_PATH)[["NAME", "geometry"]].rename(columns={"NAME": "municipality"})
    m = m[m["municipality"].notna()].to_crs(TARGET_CRS)
    return m


def _load_destruction_buffer() -> gpd.GeoDataFrame:
    d = gpd.read_file(DESTRUCTION_PATH)
    d["date_start"] = pd.to_datetime(d["date_start"], errors="coerce")
    d = d[d["date_start"].notna() & (d["date_start"] <= DESTRUCTION_CUTOFF)].copy()
    d = d.to_crs(TARGET_CRS)
    d = d[["geometry"]]
    d["geometry"] = d.geometry.buffer(50.0)
    return d


def _points_within_50m_destroyed_for_weeks(week_keys: list[str], muni_filter: set[str]) -> dict:
    municipalities = _load_municipalities()
    municipalities = municipalities[municipalities["municipality"].isin(muni_filter)].copy()
    destruction = _load_destruction_buffer()

    total_points = 0
    near_points = 0
    used_weeks = []
    for week_key in week_keys:
        gpkg = MERGED_DIR / f"{week_key}.gpkg"
        if not gpkg.exists():
            continue
        pts = gpd.read_file(gpkg)
        if pts.empty:
            used_weeks.append({"week_key": week_key, "points": 0, "near": 0})
            continue
        pts = pts[pts.geometry.notna()].copy().to_crs(TARGET_CRS)
        pts = pts.reset_index(drop=True)
        pts["_pt_id"] = np.arange(len(pts), dtype=int)
        pts = gpd.sjoin(pts, municipalities[["municipality", "geometry"]], how="inner", predicate="within")
        if pts.empty:
            used_weeks.append({"week_key": week_key, "points": 0, "near": 0})
            continue
        total = int(pts["_pt_id"].nunique())
        near_join = gpd.sjoin(pts[["_pt_id", "geometry"]], destruction, how="inner", predicate="within")
        near = int(near_join["_pt_id"].nunique()) if not near_join.empty else 0
        total_points += total
        near_points += near
        used_weeks.append({"week_key": week_key, "points": total, "near": near})

    pct = (near_points / total_points * 100.0) if total_points > 0 else np.nan
    return {
        "weeks": used_weeks,
        "total_points": int(total_points),
        "near_destroyed_50m_points": int(near_points),
        "percent_near_destroyed_50m": None if not np.isfinite(pct) else float(pct),
    }


def _weeks_overlapping(start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[str]:
    week_keys = []
    for gpkg in sorted(MERGED_DIR.glob("*.gpkg")):
        wk = gpkg.stem
        ws = _parse_week_start(wk)
        we = _parse_week_end(wk)
        if pd.isna(ws) or pd.isna(we):
            continue
        if ws <= end_date and we >= start_date:
            week_keys.append(wk)
    return week_keys


def _sustained_largest_change(out: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    out = out.sort_values(["municipality", "week_start"]).copy()
    out["prev_week_key"] = out.groupby("municipality")["week_key"].shift(1)
    out["prev_count"] = out.groupby("municipality")["count"].shift(1)
    out["prev_count_p2_5"] = out.groupby("municipality")["count_p2_5"].shift(1)
    out["prev_count_p97_5"] = out.groupby("municipality")["count_p97_5"].shift(1)
    out["change"] = out["count"] - out["prev_count"]
    out["abs_change"] = out["change"].abs()

    changes = out[out["prev_count"].notna()].copy()
    if changes.empty:
        raise ValueError("No week-to-week municipal changes available after filtering.")

    sustained_rows = []
    by_muni = out.groupby("municipality", sort=False)
    for municipality, group in by_muni:
        group = group.reset_index(drop=True)
        c = group["count"].to_numpy(float)
        wk = group["week_key"].astype(str).to_numpy()

        for j in range(1, len(group)):
            if j - 4 < 0 or j + 3 >= len(group):
                continue
            baseline = c[j - 1]
            current = c[j]
            delta = current - baseline
            if delta == 0:
                continue
            future = c[j + 1 : j + 4]
            history = c[j - 4 : j - 1]
            lower = baseline * (1.0 - BOUNCE_BACK_TOLERANCE)
            upper = baseline * (1.0 + BOUNCE_BACK_TOLERANCE)
            if delta > 0:
                forward_ok = bool((future >= upper).all())
                backward_ok = bool((history <= upper).all())
            else:
                forward_ok = bool((future <= lower).all())
                backward_ok = bool((history >= lower).all())
            if not (forward_ok and backward_ok):
                continue
            sustained_rows.append(
                {
                    "municipality": municipality,
                    "prev_week_key": wk[j - 1],
                    "week_key": wk[j],
                    "prev_count": baseline,
                    "prev_count_p2_5": float(group.loc[j, "prev_count_p2_5"]),
                    "prev_count_p97_5": float(group.loc[j, "prev_count_p97_5"]),
                    "count": current,
                    "count_p2_5": float(group.loc[j, "count_p2_5"]),
                    "count_p97_5": float(group.loc[j, "count_p97_5"]),
                    "change": delta,
                    "abs_change": abs(delta),
                    "week_start": group.loc[j, "week_start"],
                    "sustained_weeks_checked": 3,
                    "history_window_start": wk[j - 4],
                    "history_window_end": wk[j - 2],
                    "future_window_start": wk[j + 1],
                    "future_window_end": wk[j + 3],
                }
            )

    sustained_df = pd.DataFrame(sustained_rows)
    if sustained_df.empty:
        raise ValueError("No sustained week-to-week municipal changes met the >3-week no-bounce rule.")
    sustained_df["abs_change"] = sustained_df["change"].abs()
    largest = sustained_df.iloc[int(sustained_df["abs_change"].to_numpy().argmax())]
    return changes, largest


def main() -> None:
    out = _load_base_df()
    changes, largest = _sustained_largest_change(out)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    changes.to_csv(OUTPUT_CHANGES_CSV, index=False)

    z = 1.959963984540054
    prev_sd = max((largest["prev_count_p97_5"] - largest["prev_count_p2_5"]) / (2.0 * z), 1e-9)
    curr_sd = max((largest["count_p97_5"] - largest["count_p2_5"]) / (2.0 * z), 1e-9)
    rng = np.random.default_rng(CHANGE_CI_SEED)
    prev_draws = rng.normal(float(largest["prev_count"]), float(prev_sd), size=CHANGE_CI_DRAWS)
    curr_draws = rng.normal(float(largest["count"]), float(curr_sd), size=CHANGE_CI_DRAWS)
    change_draws = curr_draws - prev_draws

    event_stats = _directional_event_stats(out)
    north_stats = _north_ceasefire_stats(out)

    north_start_week = north_stats["start_snapshot"]["municipality_weeks_used"][0]["week_key"]
    north_end_week = north_stats["end_snapshot"]["municipality_weeks_used"][0]["week_key"]
    north_destroyed = _points_within_50m_destroyed_for_weeks(
        [north_start_week, north_end_week],
        NORTH_MUNICIPALITIES,
    )

    rafah_weeks = _weeks_overlapping(CEASEFIRE_START, CEASEFIRE_END)
    rafah_destroyed = _points_within_50m_destroyed_for_weeks(rafah_weeks, {"Rafah"})

    summary = {
        "metric": "largest_sustained_week_to_week_change_municipal",
        "source": str(INPUT_CSV),
        "excluded_weeks": sorted(EXCLUDE_WEEKS),
        "count_column": "corrected_count_mean",
        "sustained_rule": {
            "weeks_required_after_change": 3,
            "weeks_required_before_from_week": 3,
            "bounce_back_tolerance_fraction": BOUNCE_BACK_TOLERANCE,
            "definition": (
                "No bounce-back beyond a ±30% tolerance window around the pre-change level in the next 3 observed weeks, "
                "and pre-change level itself must be sustained for prior 3 observed weeks under the same tolerance rule"
            ),
        },
        "largest_change": {
            "municipality": str(largest["municipality"]),
            "from_week": str(largest["prev_week_key"]),
            "to_week": str(largest["week_key"]),
            "previous_count": float(largest["prev_count"]),
            "previous_count_95_cri": [
                float(largest["prev_count_p2_5"]),
                float(largest["prev_count_p97_5"]),
            ],
            "current_count": float(largest["count"]),
            "current_count_95_cri": [
                float(largest["count_p2_5"]),
                float(largest["count_p97_5"]),
            ],
            "change": float(largest["change"]),
            "change_95_cri_conservative": [
                float(largest["count_p2_5"] - largest["prev_count_p97_5"]),
                float(largest["count_p97_5"] - largest["prev_count_p2_5"]),
            ],
            "change_95_cri_independence_normal_mc": [
                float(np.quantile(change_draws, 0.025)),
                float(np.quantile(change_draws, 0.975)),
            ],
            "abs_change": float(largest["abs_change"]),
            "sustained_weeks_checked": int(largest["sustained_weeks_checked"]),
            "history_window": {
                "start_week": str(largest["history_window_start"]),
                "end_week": str(largest["history_window_end"]),
            },
            "future_window": {
                "start_week": str(largest["future_window_start"]),
                "end_week": str(largest["future_window_end"]),
            },
            "change_ci_method_note": (
                "Conservative interval = [current_p2.5 - previous_p97.5, current_p97.5 - previous_p2.5]. "
                "MC interval assumes independent normal marginals reconstructed from each week's 95% CrI."
            ),
        },
        "directional_trends_key_events": event_stats,
        "north_gaza_ceasefire_counts": north_stats,
        "north_gaza_percent_within_50m_destroyed_pre_2025_01_19": north_destroyed,
        "rafah_percent_within_50m_destroyed_pre_2025_01_19_ceasefire_period": {
            "period_start": str(CEASEFIRE_START.date()),
            "period_end": str(CEASEFIRE_END.date()),
            **rafah_destroyed,
        },
    }

    OUTPUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Saved changes table: {OUTPUT_CHANGES_CSV}")
    print(f"Saved summary JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
