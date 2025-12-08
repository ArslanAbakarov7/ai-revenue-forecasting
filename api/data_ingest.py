"""
Data ingestion and consolidation utilities.
Normalizes schema variations and writes consolidated CSV.
"""

import glob
import json
import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
CONSOLIDATED_CSV = os.path.join(ARTIFACTS_DIR, "all_invoices_consolidated.csv")

def ingest_all_jsons(pattern=None, out_csv=CONSOLIDATED_CSV):
    """Load all JSON files matching pattern, unify columns, create date, and save CSV."""
    pattern = pattern or os.path.join(DATA_DIR, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No JSON files found using pattern: {pattern}")

    frames = []
    for f in files:
        with open(f, "r") as fh:
            data = json.load(fh)
        df = pd.DataFrame(data)
        # normalize column names
        df.columns = [c.lower() for c in df.columns]
        # unify stream id & times viewed names
        if "streamid" in df.columns and "stream_id" not in df.columns:
            df = df.rename(columns={"streamid": "stream_id"})
        if "timesviewed" in df.columns and "times_viewed" not in df.columns:
            df = df.rename(columns={"timesviewed": "times_viewed"})
        # ensure year/month/day -> date
        if {"year", "month", "day"}.issubset(df.columns):
            # coerce to numeric then build date
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
            df["day"] = pd.to_numeric(df["day"], errors="coerce").astype("Int64")
            df = df.dropna(subset=["year","month","day"])
            df["date"] = pd.to_datetime(dict(year=df["year"].astype(int),
                                             month=df["month"].astype(int),
                                             day=df["day"].astype(int)), errors="coerce")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            # skip problematic file
            continue

        # Normalize price/total_price
        if "total_price" in df.columns and "price" not in df.columns:
            df["price"] = pd.to_numeric(df["total_price"], errors="coerce")
        elif "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        else:
            df["price"] = np.nan

        # Use times_viewed if present
        if "times_viewed" in df.columns:
            df["times_viewed"] = pd.to_numeric(df["times_viewed"], errors="coerce")
        else:
            df["times_viewed"] = np.nan

        frames.append(df)

    if not frames:
        raise ValueError("No valid records found in given JSON files.")

    full = pd.concat(frames, ignore_index=True)

    # Attempt to repair price: if price exists but is unit price with times_viewed, compute total
    # If price seems very small and times_viewed present, but total_price field was missing, treat price as per-view.
    # Here, prefer existing 'price' as total if plausible; otherwise, if times_viewed present and price small, multiply.
    full["price"] = full["price"].abs()
    # If total_price column exists, fill price
    if "total_price" in full.columns:
        full["total_price"] = pd.to_numeric(full["total_price"], errors="coerce")
        full["price"] = full["price"].fillna(full["total_price"])

    # Drop rows without date or price
    full = full.dropna(subset=["date", "price"])

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    full.to_csv(out_csv, index=False)
    return out_csv
