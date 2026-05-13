"""
config.py — Central configuration for all pipeline scripts
-----------------------------------------------------------
Edit this file to change paths, model parameters, or target months.
All other modules import from here so nothing is hardcoded elsewhere.
"""

import os

# ── File paths ────────────────────────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
RAW_DIR             = os.path.join(BASE_DIR, "raw")
CLEANED_DIR         = os.path.join(BASE_DIR, "cleaned")
FORECAST_DIR        = os.path.join(BASE_DIR, "forecast")

DATA_FILE           = os.path.join(RAW_DIR,      "Data_for_Datathon__Revised_.xlsx")
CLEAN_DAILY_FILE    = os.path.join(CLEANED_DIR,  "clean_daily.csv")
CLEAN_INTERVAL_FILE = os.path.join(CLEANED_DIR,  "clean_interval.csv")
OUTPUT_FILE         = os.path.join(FORECAST_DIR, "forecast_v24.csv")

# ── Portfolios & time ─────────────────────────────────────────────────────────
PORTFOLIOS   = ["A", "B", "C", "D"]
TARGET_MONTH = "August"
MONTH_MAP    = {"April": 4, "May": 5, "June": 6}

# ── 30-minute interval grid ───────────────────────────────────────────────────
ALL_INTERVALS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]

# ── Recency weights (higher = more recent = more influence on profiles) ───────
MONTH_WEIGHTS = {"April": 1.0, "May": 2.0, "June": 3.0}

# ── Forecast bias multipliers (intentional over-forecast per asymmetric penalty) ──
CV_BIAS  = 1.054   # call volume
ABD_BIAS = 1.04    # abandon rate
CCT_BIAS = 1.01    # call completion time

# ── Profile / HGBR blend (0.90 = 90% profile, 10% HGBR residual correction) ─
PROFILE_BLEND = 0.90

# ── Supabase table names ──────────────────────────────────────────────────────
TABLE_DAILY    = "fact_daily_calls"
TABLE_INTERVAL = "fact_interval_calls"
TABLE_FORECAST = "fact_forecast_calls"
