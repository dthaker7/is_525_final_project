"""
etl_pipeline.py — Extract → Transform → Load
---------------------------------------------
Reads raw Excel data (4 portfolios × daily + interval sheets),
cleans and standardizes it, saves clean CSVs to /cleaned,
and loads both fact tables into Supabase PostgreSQL.

Usage:
    python etl/etl_pipeline.py

GitHub Actions: triggered on push to main (see .github/workflows/etl.yml)
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np

# Allow imports from parent directory when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_FILE, CLEAN_DAILY_FILE, CLEAN_INTERVAL_FILE,
    PORTFOLIOS, MONTH_MAP, MONTH_WEIGHTS, ALL_INTERVALS,
    TABLE_DAILY, TABLE_INTERVAL, CLEANED_DIR
)
from db import get_engine

warnings.filterwarnings("ignore")

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT
# ══════════════════════════════════════════════════════════════════════════════

def extract_daily(portfolio: str) -> pd.DataFrame:
    """
    Read the '<portfolio> - Daily' sheet from the raw Excel file.
    Returns a DataFrame with parsed date, day-of-week, month, year, and day columns.
    """
    logger.info("  [extract] Daily sheet — Portfolio %s", portfolio)
    df = pd.read_excel(DATA_FILE, sheet_name=f"{portfolio} - Daily")
    df["date"]     = pd.to_datetime(df["Date"].astype(str).str[:8], format="%m/%d/%y")
    df["dow"]      = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["dow_name"] = df["date"].dt.day_name()
    df["month"]    = df["date"].dt.month
    df["year"]     = df["date"].dt.year
    df["day"]      = df["date"].dt.day
    return df.sort_values("date").reset_index(drop=True)


def extract_interval(portfolio: str) -> pd.DataFrame:
    """
    Read the '<portfolio> - Interval' sheet from the raw Excel file.
    Parses interval times and reconstructs full date from Month + Day columns.
    Expands to a complete 48-slot grid for every date (fills missing slots with NaN).
    """
    logger.info("  [extract] Interval sheet — Portfolio %s", portfolio)
    df = pd.read_excel(DATA_FILE, sheet_name=f"{portfolio} - Interval")

    # Parse interval time → HH:MM string
    df["interval_str"] = df["Interval"].apply(
        lambda x: f"{x.hour:02d}:{x.minute:02d}" if pd.notna(x) else None
    )
    df = df[df["interval_str"].notna()].copy()

    # Reconstruct date from Month name + Day number
    df["date"] = df.apply(
        lambda r: pd.Timestamp(year=2025, month=MONTH_MAP[r["Month"]], day=int(r["Day"]))
        if pd.notna(r["Day"]) else pd.NaT, axis=1
    )
    df = df[df["date"].notna()].copy()

    df["dow"]      = df["date"].dt.dayofweek
    df["dow_name"] = df["date"].dt.day_name()
    df["month_w"]  = df["Month"].map(MONTH_WEIGHTS)

    # Expand to full 48-slot grid — some intervals may be missing in raw data
    full_idx = pd.MultiIndex.from_product(
        [df["date"].unique(), ALL_INTERVALS], names=["date", "interval_str"]
    )
    df = (df.set_index(["date", "interval_str"])
            .reindex(full_idx)
            .reset_index())

    # Re-derive time dimension columns after reindex
    df["dow"]      = df["date"].dt.dayofweek
    df["dow_name"] = df["date"].dt.day_name()
    df["Month"]    = df["date"].dt.month.map({4: "April", 5: "May", 6: "June"})
    df["month_w"]  = df["Month"].map(MONTH_WEIGHTS)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════

def transform_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill null values in Call Volume, CCT, and Abandon Rate using same-DOW
    median within a ±8-week rolling window (falls back to global DOW median).
    Clips values to valid ranges and derives Abandoned Calls.
    Returns a clean, renamed DataFrame ready for loading.
    """
    logger.info("    [transform] Filling nulls in daily data (%d rows)", len(df))
    df = df.copy()

    for col in ["Call Volume", "CCT", "Abandon Rate"]:
        null_indices = df.index[df[col].isnull()].tolist()
        for i in null_indices:
            target_date = df.loc[i, "date"]
            dow         = df.loc[i, "dow"]
            window_mask = (
                (df["date"] >= target_date - pd.Timedelta(weeks=8)) &
                (df["date"] <= target_date + pd.Timedelta(weeks=8)) &
                (df["dow"] == dow) &
                df[col].notna() &
                (df.index != i)
            )
            neighbors = df.loc[window_mask, col]
            if len(neighbors) >= 2:
                df.loc[i, col] = neighbors.median()
            else:
                # Fall back to global DOW median
                df.loc[i, col] = df.loc[
                    (df["dow"] == dow) & df[col].notna(), col
                ].median()

    # Clip to valid ranges
    df["Call Volume"]  = df["Call Volume"].clip(lower=0).round(0)
    df["CCT"]          = df["CCT"].clip(lower=0)
    df["Abandon Rate"] = df["Abandon Rate"].clip(0, 1)

    # Derive abandoned_calls
    df["abandoned_calls"] = (df["Abandon Rate"] * df["Call Volume"]).round(0).astype(int)

    # Rename to snake_case for DB / CSV
    return df[["portfolio", "date", "dow", "dow_name", "month", "year", "day",
               "Call Volume", "CCT", "Abandon Rate", "abandoned_calls"]].rename(columns={
        "Call Volume":  "calls_offered",
        "CCT":          "cct_seconds",
        "Abandon Rate": "abandon_rate",
        "dow":          "day_of_week",
        "dow_name":     "day_of_week_name",
        "month":        "month_num",
        "day":          "day_num",
    })


def transform_interval(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute nulls using same-interval + same-DOW median lookup.
    Zero-volume slots get zero for all dependent metrics.
    Clips values, derives abandoned_calls, and renames to snake_case.
    Returns a clean DataFrame ready for loading.
    """
    logger.info("    [transform] Filling nulls in interval data (%d rows)", len(df))
    df = df.copy()

    # Zero-volume slots: zero out dependent metrics
    zero_mask = df["Call Volume"] == 0
    for col in ["CCT", "Abandoned Calls", "Abandoned Rate"]:
        if col in df.columns:
            df.loc[zero_mask, col] = df.loc[zero_mask, col].fillna(0)

    # Impute remaining nulls with interval+DOW median, then interval median
    for metric in ["Call Volume", "Abandoned Calls", "Abandoned Rate", "CCT"]:
        if metric not in df.columns:
            continue
        null_mask = df[metric].isnull()
        if null_mask.sum() == 0:
            continue

        # First pass: same interval + same DOW
        lkp = (df[df[metric].notna()]
               .groupby(["interval_str", "dow"])[metric].median())
        df.loc[null_mask, metric] = df[null_mask].apply(
            lambda r: lkp.get((r["interval_str"], r["dow"]), np.nan), axis=1
        )

        # Second pass: same interval only (fallback)
        still_null = df[metric].isnull()
        if still_null.sum() > 0:
            lkp2 = df[df[metric].notna()].groupby("interval_str")[metric].median()
            df.loc[still_null, metric] = df.loc[still_null, "interval_str"].map(lkp2)

    # Clip to valid ranges
    df["Call Volume"] = df["Call Volume"].clip(lower=0).fillna(0)
    if "Abandoned Rate" in df.columns:
        df["Abandoned Rate"] = df["Abandoned Rate"].clip(0, 1).fillna(0)
    df["CCT"] = df["CCT"].clip(lower=0).fillna(0)

    # Derive abandoned_calls consistently from rate × volume
    if "Abandoned Rate" in df.columns:
        df["abandoned_calls"] = (df["Abandoned Rate"] * df["Call Volume"]).round(0).astype(int)
    elif "Abandoned Calls" in df.columns:
        df["abandoned_calls"] = df["Abandoned Calls"].fillna(0).round(0).astype(int)
    else:
        df["abandoned_calls"] = 0

    # Rename to snake_case for DB / CSV
    return df[["portfolio", "date", "interval_str", "dow", "dow_name", "Month",
               "Call Volume", "abandoned_calls", "Abandoned Rate", "CCT"]].rename(columns={
        "Call Volume":    "calls_offered",
        "Abandoned Rate": "abandon_rate",
        "CCT":            "cct_seconds",
        "Month":          "month_name",
        "dow":            "day_of_week",
        "dow_name":       "day_of_week_name",
        "interval_str":   "interval",
    })


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_data(df: pd.DataFrame, table_name: str, engine) -> None:
    """
    Truncates the target Supabase table and loads the full DataFrame.
    Uses TRUNCATE + INSERT (full refresh) — safe for our dataset size.
    Logs row count and any errors.
    """
    logger.info("  [load] Writing %d rows → %s", len(df), table_name)
    try:
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {table_name} RESTART IDENTITY"))
        df.to_sql(table_name, engine, if_exists="append", index=False, chunksize=1000)
        logger.info("  ✓ %s loaded successfully", table_name)
    except Exception as e:
        logger.error("  ✗ Failed to load %s: %s", table_name, e)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_etl(load_to_db: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full ETL run for all portfolios.

    Steps:
        1. Extract daily + interval data from raw Excel (all 4 portfolios)
        2. Transform (clean, impute, clip, rename)
        3. Save clean CSVs to /cleaned
        4. Load into Supabase (fact_daily_calls + fact_interval_calls)

    Args:
        load_to_db: Set False to skip DB load (useful for local testing)

    Returns:
        (clean_daily_df, clean_interval_df) — the final concatenated DataFrames
    """
    logger.info("=" * 60)
    logger.info("  ETL PIPELINE — Call Center Analytics")
    logger.info("=" * 60)

    # Ensure output directories exist
    os.makedirs(CLEANED_DIR, exist_ok=True)

    daily_frames    = []
    interval_frames = []

    for portfolio in PORTFOLIOS:
        logger.info("─" * 60)
        logger.info("  Portfolio %s", portfolio)
        logger.info("─" * 60)

        # ── Extract ───────────────────────────────────────────────
        raw_daily    = extract_daily(portfolio)
        raw_interval = extract_interval(portfolio)

        # Tag with portfolio before transforming
        raw_daily["portfolio"]    = portfolio
        raw_interval["portfolio"] = portfolio

        # ── Transform ─────────────────────────────────────────────
        clean_daily    = transform_daily(raw_daily)
        clean_interval = transform_interval(raw_interval)

        daily_frames.append(clean_daily)
        interval_frames.append(clean_interval)

        logger.info(
            "  ✓ Portfolio %s — daily: %d rows | interval: %d rows",
            portfolio, len(clean_daily), len(clean_interval)
        )

    # ── Concatenate ───────────────────────────────────────────────
    all_daily    = pd.concat(daily_frames,    ignore_index=True)
    all_interval = pd.concat(interval_frames, ignore_index=True)

    # ── Null report ───────────────────────────────────────────────
    daily_nulls    = all_daily.isnull().sum().sum()
    interval_nulls = all_interval.isnull().sum().sum()
    logger.info("Null check — daily: %d  |  interval: %d", daily_nulls, interval_nulls)
    if daily_nulls > 0 or interval_nulls > 0:
        logger.warning("Non-zero nulls detected — check raw data")

    # ── Save CSVs ─────────────────────────────────────────────────
    all_daily.to_csv(CLEAN_DAILY_FILE,    index=False)
    all_interval.to_csv(CLEAN_INTERVAL_FILE, index=False)
    logger.info("✓ Saved: %s", CLEAN_DAILY_FILE)
    logger.info("✓ Saved: %s", CLEAN_INTERVAL_FILE)

    # ── Load to Supabase ──────────────────────────────────────────
    if load_to_db:
        logger.info("Connecting to Supabase...")
        engine = get_engine()
        load_data(all_daily,    TABLE_DAILY,    engine)
        load_data(all_interval, TABLE_INTERVAL, engine)
    else:
        logger.info("Skipping DB load (load_to_db=False)")

    logger.info("=" * 60)
    logger.info("  ETL COMPLETE")
    logger.info("=" * 60)

    return all_daily, all_interval


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the call center ETL pipeline")
    parser.add_argument(
        "--no-db", action="store_true",
        help="Skip loading to Supabase (output CSVs only)"
    )
    args = parser.parse_args()
    run_etl(load_to_db=not args.no_db)
