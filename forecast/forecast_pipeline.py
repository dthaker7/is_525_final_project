"""
forecast_pipeline.py — Recency-Weighted Profile + HGBR Residual Forecast
-------------------------------------------------------------------------
Reads clean CSVs (output of etl_pipeline.py), builds per-portfolio intraday
profiles, trains HistGradientBoosting residual correction models, and
generates the full 1,488-row August 2025 forecast.

Output:
    /forecast/forecast_v24.csv   — submission-ready CSV
    Supabase: fact_forecast_calls — long-format forecast rows (optional)

Usage:
    python forecast/forecast_pipeline.py
    python forecast/forecast_pipeline.py --no-db   # skip Supabase load
"""

import os
import sys
import logging
import warnings
from datetime import date

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CLEAN_DAILY_FILE, CLEAN_INTERVAL_FILE, OUTPUT_FILE,
    PORTFOLIOS, TARGET_MONTH, MONTH_WEIGHTS, ALL_INTERVALS,
    CV_BIAS, ABD_BIAS, CCT_BIAS, PROFILE_BLEND,
    TABLE_FORECAST, FORECAST_DIR
)
from db import get_engine

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD CLEAN DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load clean_daily.csv and clean_interval.csv (output of ETL pipeline).
    Renames snake_case columns back to the internal names expected by the
    profile builder and forecast function.
    """
    logger.info("Loading clean CSVs...")

    if not os.path.exists(CLEAN_DAILY_FILE):
        raise FileNotFoundError(
            f"{CLEAN_DAILY_FILE} not found. Run etl_pipeline.py first."
        )
    if not os.path.exists(CLEAN_INTERVAL_FILE):
        raise FileNotFoundError(
            f"{CLEAN_INTERVAL_FILE} not found. Run etl_pipeline.py first."
        )

    daily = pd.read_csv(CLEAN_DAILY_FILE, parse_dates=["date"])
    daily = daily.rename(columns={
        "calls_offered":     "Call Volume",
        "cct_seconds":       "CCT",
        "abandon_rate":      "Abandon Rate",
        "day_of_week":       "dow",
        "day_of_week_name":  "dow_name",
        "month_num":         "month",
        "day_num":           "day",
    })

    interval = pd.read_csv(CLEAN_INTERVAL_FILE, parse_dates=["date"])
    interval = interval.rename(columns={
        "calls_offered":    "Call Volume",
        "cct_seconds":      "CCT",
        "abandon_rate":     "Abandoned Rate",
        "abandoned_calls":  "Abandoned Calls",
        "month_name":       "Month",
        "day_of_week":      "dow",
        "day_of_week_name": "dow_name",
        "interval":         "interval_str",
    })
    interval["month_w"] = interval["Month"].map(MONTH_WEIGHTS)

    logger.info(
        "✓ Loaded — daily: %d rows | interval: %d rows",
        len(daily), len(interval)
    )
    return daily, interval


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PROFILE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_profiles(df_int: pd.DataFrame) -> dict:
    """
    Builds three recency-weighted intraday profiles indexed by (dow, interval_str):

    - cv_frac : fraction of daily call volume per 30-min slot
                (recency-weighted median, Gaussian-smoothed σ=1.5, renormalized per DOW)
    - cct     : volume-weighted mean CCT per slot
                (recency-weighted, Gaussian-smoothed σ=1.0)
    - abd     : median abandon rate per slot
                (Gaussian-smoothed σ=1.5)

    Recency weighting is achieved by duplicating rows: April=1×, May=2×, June=3×.
    """
    logger.info("  Building recency-weighted intraday profiles...")

    # Compute cv_frac = slot volume / daily total
    day_sums = df_int.groupby("date")["Call Volume"].sum().rename("day_total")
    df_w = df_int.merge(day_sums, on="date")
    df_w["cv_frac"] = np.where(df_w["day_total"] > 0,
                               df_w["Call Volume"] / df_w["day_total"], 0)

    # Duplicate rows for recency weighting
    rows_weighted = []
    for _, grp in df_w.groupby("date"):
        w = int(grp["month_w"].iloc[0])
        for _ in range(w):
            rows_weighted.append(grp)
    df_w2 = pd.concat(rows_weighted, ignore_index=True)

    # ── cv_frac profile ───────────────────────────────────────────
    cv_profile = (df_w2.groupby(["dow", "interval_str"])["cv_frac"]
                       .median()
                       .unstack(fill_value=0))

    # Gaussian smooth + renormalize per DOW
    for dow in cv_profile.index:
        smoothed = gaussian_filter1d(cv_profile.loc[dow].values.astype(float), sigma=1.5)
        total = smoothed.sum()
        cv_profile.loc[dow] = smoothed / total if total > 0 else smoothed

    # ── CCT profile ───────────────────────────────────────────────
    df_w2["weighted_cct"] = df_w2["CCT"] * df_w2["Call Volume"]
    cct_num = (df_w2.groupby(["dow", "interval_str"])["weighted_cct"].sum()
                    .unstack(fill_value=0))
    cct_den = (df_w2.groupby(["dow", "interval_str"])["Call Volume"].sum()
                    .unstack(fill_value=1))
    cct_profile = (cct_num / cct_den.replace(0, np.nan)).fillna(0)

    for dow in cct_profile.index:
        cct_profile.loc[dow] = gaussian_filter1d(
            cct_profile.loc[dow].values.astype(float), sigma=1.0
        )

    # ── Abandon rate profile ──────────────────────────────────────
    abd_profile = (df_w2.groupby(["dow", "interval_str"])["Abandoned Rate"]
                        .median()
                        .unstack(fill_value=0))

    for dow in abd_profile.index:
        smoothed = gaussian_filter1d(abd_profile.loc[dow].values.astype(float), sigma=1.5)
        abd_profile.loc[dow] = np.clip(smoothed, 0, 1)

    return {"cv_frac": cv_profile, "cct": cct_profile, "abd": abd_profile}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — HGBR RESIDUAL CORRECTION MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_residual_model(df_int: pd.DataFrame, profiles: dict) -> dict:
    """
    Trains two HistGradientBoostingRegressor models on residuals between the
    recency-weighted profile prediction and observed actuals:
        - "cv"  : residual in cv_frac (call volume fraction)
        - "abd" : residual in Abandoned Rate

    Features: hour, dow, week_of_month, month_w, sin/cos(hour), sin/cos(dow),
              profile prediction, daily_cv

    Returns a dict: {target: (fitted_model, feature_column_names)}
    """
    logger.info("  Training HGBR residual correction models...")

    day_sums = df_int.groupby("date")["Call Volume"].sum().rename("day_total")
    df = df_int.merge(day_sums, on="date")
    df["cv_frac"]      = np.where(df["day_total"] > 0,
                                  df["Call Volume"] / df["day_total"], 0)
    df["hour"]         = df["interval_str"].apply(lambda x: int(x.split(":")[0]))
    df["minute"]       = df["interval_str"].apply(lambda x: int(x.split(":")[1]))
    df["week_of_month"] = df["date"].dt.day.apply(lambda d: (d - 1) // 7 + 1)
    df["sin_hour"]     = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"]     = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dow"]      = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"]      = np.cos(2 * np.pi * df["dow"] / 7)
    df["daily_cv"]     = df["day_total"]

    # Look up profile value for each row
    def lookup(profile_df, dow_arr, iv_arr):
        vals = np.zeros(len(dow_arr))
        for i, (d, iv) in enumerate(zip(dow_arr, iv_arr)):
            if d in profile_df.index and iv in profile_df.columns:
                vals[i] = profile_df.loc[d, iv]
        return vals

    df["pf_cv"]  = lookup(profiles["cv_frac"], df["dow"].values, df["interval_str"].values)
    df["pf_abd"] = lookup(profiles["abd"],     df["dow"].values, df["interval_str"].values)

    feat_cols = ["hour", "dow", "week_of_month", "month_w",
                 "sin_hour", "cos_hour", "sin_dow", "cos_dow",
                 "pf_cv", "daily_cv"]

    models = {}
    targets = {
        "cv":  ("cv_frac",       "pf_cv"),
        "abd": ("Abandoned Rate", "pf_abd"),
    }

    for key, (actual_col, profile_col) in targets.items():
        subset = df[df[actual_col].notna() & df[profile_col].notna()].copy()
        X = subset[feat_cols].values
        y = (subset[actual_col] - subset[profile_col]).values

        model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=8, l2_regularization=2.0, random_state=42
        )
        model.fit(X, y)
        models[key] = (model, feat_cols)
        logger.info(
            "    residual '%s': trained on %d rows  resid range [%.4f, %.4f]",
            key, len(X), y.min(), y.max()
        )

    return models


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — AUGUST FORECAST ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def forecast_august(portfolio: str,
                    profiles: dict,
                    resid_models: dict,
                    df_daily_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the full 1,488-row (31 days × 48 intervals) August 2025 forecast
    for one portfolio.

    Strategy:
    - Call volume : 90% profile + 10% HGBR residual correction, peak-hour
                   boosting, ramp smoothing, CV_BIAS multiplier
    - CCT         : pure profile × CCT_BIAS
    - Abandon rate: 90% profile + 10% HGBR residual correction × ABD_BIAS
    - Abandoned calls: derived from rate × volume (always consistent)
    """
    logger.info("  Assembling August forecast for Portfolio %s...", portfolio)

    # Pull August daily actuals (used as the daily total anchor)
    aug_daily = df_daily_clean[
        (df_daily_clean["year"] == 2025) & (df_daily_clean["month"] == 8)
    ].set_index("day")[["Call Volume", "CCT", "Abandon Rate", "dow"]]

    # Build the 31 × 48 grid
    aug_dates = [date(2025, 8, d) for d in range(1, 32)]
    grid_rows = [
        {"date": pd.Timestamp(d), "Day": d.day, "interval_str": iv}
        for d in aug_dates for iv in ALL_INTERVALS
    ]
    df_aug = pd.DataFrame(grid_rows)

    # Time features
    df_aug["dow"]          = df_aug["date"].dt.dayofweek
    df_aug["hour"]         = df_aug["interval_str"].apply(lambda x: int(x.split(":")[0]))
    df_aug["minute"]       = df_aug["interval_str"].apply(lambda x: int(x.split(":")[1]))
    df_aug["week_of_month"] = df_aug["Day"].apply(lambda d: (d - 1) // 7 + 1)
    df_aug["month_num"]    = 8
    df_aug["sin_hour"]     = np.sin(2 * np.pi * df_aug["hour"] / 24)
    df_aug["cos_hour"]     = np.cos(2 * np.pi * df_aug["hour"] / 24)
    df_aug["sin_dow"]      = np.sin(2 * np.pi * df_aug["dow"] / 7)
    df_aug["cos_dow"]      = np.cos(2 * np.pi * df_aug["dow"] / 7)

    # Anchor: actual daily call volume for August
    df_aug["daily_cv"] = df_aug["Day"].map(aug_daily["Call Volume"].to_dict())

    # Profile lookup helper
    def lookup_profile(profile_df, dow_arr, iv_arr):
        vals = np.zeros(len(dow_arr))
        for i, (d, iv) in enumerate(zip(dow_arr, iv_arr)):
            if d in profile_df.index and iv in profile_df.columns:
                vals[i] = profile_df.loc[d, iv]
        return vals

    df_aug["pf_cv"]  = lookup_profile(profiles["cv_frac"], df_aug["dow"].values, df_aug["interval_str"].values)
    df_aug["pf_cct"] = lookup_profile(profiles["cct"],     df_aug["dow"].values, df_aug["interval_str"].values)
    df_aug["pf_abd"] = lookup_profile(profiles["abd"],     df_aug["dow"].values, df_aug["interval_str"].values)

    # HGBR residual corrections
    feat_cols = ["hour", "dow", "week_of_month", "month_num",
                 "sin_hour", "cos_hour", "sin_dow", "cos_dow",
                 "pf_cv", "daily_cv"]
    X = df_aug[feat_cols].values

    cv_corr  = resid_models["cv"][0].predict(X)
    abd_corr = resid_models["abd"][0].predict(X)

    # Blend: 90% profile + 10% (profile + HGBR correction)
    df_aug["cv_frac"] = np.clip(
        PROFILE_BLEND * df_aug["pf_cv"] + (1 - PROFILE_BLEND) * (df_aug["pf_cv"] + cv_corr),
        0, None
    )
    df_aug["cct_raw"] = np.clip(df_aug["pf_cct"], 0, None)
    df_aug["abd_raw"] = np.clip(
        PROFILE_BLEND * df_aug["pf_abd"] + (1 - PROFILE_BLEND) * (df_aug["pf_abd"] + abd_corr),
        0, 1
    )

    # Normalize cv_frac per day so it sums to 1
    day_frac_sums = df_aug.groupby("Day")["cv_frac"].transform("sum")
    df_aug["cv_frac_norm"] = np.where(day_frac_sums > 0,
                                      df_aug["cv_frac"] / day_frac_sums, 0)

    # Anchor to actual daily volume
    df_aug["cv_biased"] = df_aug["cv_frac_norm"] * df_aug["daily_cv"]

    # Peak-hour boosting
    df_aug["peak_factor"] = np.where(
        df_aug["hour"].between(9, 11), 1.03,
        np.where(df_aug["hour"].between(12, 16), 1.015, 1.0)
    )
    df_aug["cv_biased"] *= df_aug["peak_factor"]

    # Morning ramp reduction
    df_aug["ramp_factor"] = np.where(df_aug["hour"].between(6, 9), 0.97, 1.0)
    df_aug["cv_biased"] *= df_aug["ramp_factor"]

    # Smooth across intervals within each day (rolling 3-slot average)
    df_aug["cv_biased"] = df_aug.groupby("Day")["cv_biased"].transform(
        lambda x: x.rolling(3, center=True, min_periods=1).mean()
    )

    # Re-anchor to daily total and apply CV_BIAS
    df_aug["cv_biased"] = (
        df_aug["cv_biased"]
        / df_aug.groupby("Day")["cv_biased"].transform("sum")
        * df_aug["daily_cv"]
        * CV_BIAS
    )

    # Floor to int, then distribute residual to the slot with the highest remainder
    df_aug[f"Calls_Offered_{portfolio}"] = np.floor(
        df_aug["cv_biased"].clip(lower=0)
    ).astype(int)

    residual = df_aug["cv_biased"] - df_aug[f"Calls_Offered_{portfolio}"]
    residual_rank = residual.groupby(df_aug["Day"]).rank(method="first", ascending=False)
    df_aug.loc[residual_rank == 1, f"Calls_Offered_{portfolio}"] += 1

    # CCT and abandon rate with bias
    df_aug[f"CCT_{portfolio}"]            = (df_aug["cct_raw"] * CCT_BIAS).clip(lower=0)
    df_aug[f"Abandoned_Rate_{portfolio}"] = (df_aug["abd_raw"] * ABD_BIAS).clip(0, 1)
    df_aug[f"Abandoned_Calls_{portfolio}"] = (
        df_aug[f"Abandoned_Rate_{portfolio}"] * df_aug[f"Calls_Offered_{portfolio}"]
    ).round(0).astype(int)

    # Zero out all metrics on zero-volume slots
    zero_mask = df_aug[f"Calls_Offered_{portfolio}"] == 0
    df_aug.loc[zero_mask, f"Abandoned_Calls_{portfolio}"] = 0
    df_aug.loc[zero_mask, f"Abandoned_Rate_{portfolio}"]  = 0.0
    df_aug.loc[zero_mask, f"CCT_{portfolio}"]             = 0.0

    out_cols = ["date", "Day", "interval_str",
                f"Calls_Offered_{portfolio}",
                f"Abandoned_Calls_{portfolio}",
                f"Abandoned_Rate_{portfolio}",
                f"CCT_{portfolio}"]
    return df_aug[out_cols]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SAVE & LOAD
# ══════════════════════════════════════════════════════════════════════════════

def assemble_and_save(all_forecasts: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges per-portfolio forecast DataFrames into the submission-ready wide CSV.
    Column order matches the competition template exactly.
    """
    logger.info("Assembling final forecast CSV...")
    final = all_forecasts[0]
    for pf in all_forecasts[1:]:
        final = final.merge(pf, on=["date", "Day", "interval_str"], how="outer")

    final["Month"]    = TARGET_MONTH
    final["Interval"] = final["interval_str"].apply(
        lambda x: f"{int(x.split(':')[0])}:{x.split(':')[1]}"
    )
    final = final.sort_values(["Day", "interval_str"]).reset_index(drop=True)

    template_cols = [
        "Month", "Day", "Interval",
        "Calls_Offered_A",  "Abandoned_Calls_A",  "Abandoned_Rate_A",  "CCT_A",
        "Calls_Offered_B",  "Abandoned_Calls_B",  "Abandoned_Rate_B",  "CCT_B",
        "Calls_Offered_C",  "Abandoned_Calls_C",  "Abandoned_Rate_C",  "CCT_C",
        "Calls_Offered_D",  "Abandoned_Calls_D",  "Abandoned_Rate_D",  "CCT_D",
    ]
    for col in template_cols:
        if col not in ["Month", "Day", "Interval"]:
            final[col] = pd.to_numeric(final[col], errors="coerce").fillna(0).clip(lower=0)

    final = final[template_cols]

    os.makedirs(FORECAST_DIR, exist_ok=True)
    final.to_csv(OUTPUT_FILE, index=False)
    logger.info("✓ Saved forecast → %s  (%d rows × %d cols)",
                OUTPUT_FILE, len(final), len(final.columns))
    return final


def load_forecast_to_db(final: pd.DataFrame, engine) -> None:
    """
    Loads the forecast into Supabase in long format (one row per portfolio × interval).
    Requires a fact_forecast_calls table — see sql/create_tables.sql.
    """
    logger.info("Loading forecast to Supabase → %s", TABLE_FORECAST)
    rows = []
    for _, row in final.iterrows():
        for p in PORTFOLIOS:
            rows.append({
                "month":           row["Month"],
                "day":             int(row["Day"]),
                "interval_time":   row["Interval"],
                "portfolio":       p,
                "calls_offered":   int(row[f"Calls_Offered_{p}"]),
                "abandoned_calls": int(row[f"Abandoned_Calls_{p}"]),
                "abandon_rate":    float(row[f"Abandoned_Rate_{p}"]),
                "cct_seconds":     float(row[f"CCT_{p}"]),
            })
    df_long = pd.DataFrame(rows)

    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {TABLE_FORECAST} RESTART IDENTITY"))
    df_long.to_sql(TABLE_FORECAST, engine, if_exists="append", index=False, chunksize=1000)
    logger.info("✓ %s loaded — %d rows", TABLE_FORECAST, len(df_long))


# ══════════════════════════════════════════════════════════════════════════════
# SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def sanity_report(final: pd.DataFrame) -> None:
    """Prints a quick summary table and runs basic consistency checks."""
    logger.info("\n  %s", "─" * 58)
    logger.info("  %-12s %10s %8s %10s %10s", "Portfolio", "CV_total", "CV_max", "CCT_mean", "ABD_mean")
    logger.info("  %s", "─" * 58)
    for p in PORTFOLIOS:
        cv  = final[f"Calls_Offered_{p}"].sum()
        cvx = final[f"Calls_Offered_{p}"].max()
        cct = final[final[f"CCT_{p}"] > 0][f"CCT_{p}"].mean()
        abd = final[f"Abandoned_Rate_{p}"].mean()
        logger.info("  %-12s %10s %8s %10.1f %10.4f", p, f"{cv:,}", cvx, cct, abd)
    logger.info("  %s", "─" * 58)

    neg_count = (final.select_dtypes("number") < 0).any(axis=1).sum()
    logger.info("  Rows with any negative value : %d  (must be 0)", neg_count)

    mismatches = 0
    for p in PORTFOLIOS:
        expected = (final[f"Abandoned_Rate_{p}"] * final[f"Calls_Offered_{p}"]).round(0)
        mismatches += (final[f"Abandoned_Calls_{p}"] != expected).sum()
    logger.info("  Abandoned_Calls mismatches   : %d  (must be 0)", mismatches)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_forecast(load_to_db: bool = True) -> pd.DataFrame:
    """
    Full forecast run:
        1. Load clean CSVs (from ETL)
        2. Per portfolio: build profiles → train HGBR → forecast August
        3. Assemble + save wide CSV
        4. Load long-format to Supabase (optional)
        5. Sanity report
    """
    logger.info("=" * 60)
    logger.info("  FORECAST PIPELINE — August 2025")
    logger.info("=" * 60)

    all_daily, all_interval = load_clean_data()

    all_forecasts = []

    for portfolio in PORTFOLIOS:
        logger.info("─" * 60)
        logger.info("  Portfolio %s", portfolio)
        logger.info("─" * 60)

        df_d = all_daily[all_daily["portfolio"] == portfolio].copy()
        df_i = all_interval[all_interval["portfolio"] == portfolio].copy()

        profiles     = build_profiles(df_i)
        resid_models = build_residual_model(df_i, profiles)
        pf           = forecast_august(portfolio, profiles, resid_models, df_d)

        cv_col       = f"Calls_Offered_{portfolio}"
        aug_actual   = df_d[(df_d["year"] == 2025) & (df_d["month"] == 8)]["Call Volume"].sum()
        pred_total   = pf[cv_col].sum()
        logger.info(
            "  CV range: %d–%d | Monthly total: %d | Actual: %.0f | Ratio: %.3f",
            pf[cv_col].min(), pf[cv_col].max(), pred_total, aug_actual,
            pred_total / aug_actual if aug_actual > 0 else 0
        )
        all_forecasts.append(pf)

    final = assemble_and_save(all_forecasts)
    sanity_report(final)

    if load_to_db:
        engine = get_engine()
        load_forecast_to_db(final, engine)
    else:
        logger.info("Skipping DB load (load_to_db=False)")

    logger.info("=" * 60)
    logger.info("  FORECAST COMPLETE → %s", OUTPUT_FILE)
    logger.info("=" * 60)
    return final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the call center forecast pipeline")
    parser.add_argument(
        "--no-db", action="store_true",
        help="Skip loading forecast to Supabase"
    )
    args = parser.parse_args()
    run_forecast(load_to_db=not args.no_db)
