"""
Feature engineering, model training, saving and prediction utilities.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
CONSOLIDATED_CSV = os.path.join(ARTIFACTS_DIR, "all_invoices_consolidated.csv")
FEATURES_CSV = os.path.join(ARTIFACTS_DIR, "monthly_features.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "final_model.joblib")

def build_monthly_features(consolidated_csv=CONSOLIDATED_CSV, out_csv=FEATURES_CSV):
    """Aggregate to monthly revenue and construct time-series features."""
    df = pd.read_csv(consolidated_csv, parse_dates=["date"])
    monthly = df.groupby(pd.Grouper(key="date", freq="ME"))["price"].sum()
    monthly.index = pd.to_datetime(monthly.index.to_period("M").to_timestamp())
    m = monthly.to_frame(name="monthly_revenue")

    m["lag_1"] = m["monthly_revenue"].shift(1)
    m["lag_3"] = m["monthly_revenue"].shift(3)
    m["lag_6"] = m["monthly_revenue"].shift(6)

    m["roll_mean_3"] = m["monthly_revenue"].shift(1).rolling(3, min_periods=1).mean()
    m["roll_mean_6"] = m["monthly_revenue"].shift(1).rolling(6, min_periods=1).mean()

    m["pct_change_1"] = m["monthly_revenue"].pct_change(1).fillna(0)
    m["pct_change_3"] = m["monthly_revenue"].pct_change(3).fillna(0)

    m["month"] = m.index.month
    m["month_sin"] = np.sin(2 * np.pi * m["month"] / 12.0)
    m["month_cos"] = np.cos(2 * np.pi * m["month"] / 12.0)

    m["target_next_month"] = m["monthly_revenue"].shift(-1)
    m.replace([np.inf, -np.inf], np.nan, inplace=True)
    m.dropna(inplace=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    m.to_csv(out_csv, index=True)
    return out_csv

def train_select_and_save(features_csv=FEATURES_CSV, model_path=MODEL_PATH):
    """Train baseline and models, select best by MAE, retrain on all data and save artifact."""
    m = pd.read_csv(features_csv, index_col=0, parse_dates=True)

    feature_cols = ["lag_1","lag_3","lag_6","roll_mean_3","roll_mean_6","pct_change_1","pct_change_3","month_sin","month_cos"]
    X = m[feature_cols].values
    y = m["target_next_month"].values

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Baseline
    baseline_pred = m["lag_1"].iloc[split:].values
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    # RandomForest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_mae = mean_absolute_error(y_test, rf.predict(X_test))

    # Try XGBoost (optional; may be slower)
    try:
        from xgboost import XGBRegressor
        xg = XGBRegressor(n_estimators=300, random_state=42, verbosity=0)
        xg.fit(X_train, y_train)
        xg_mae = mean_absolute_error(y_test, xg.predict(X_test))
    except Exception:
        xg = None
        xg_mae = np.inf

    maes = {"baseline": baseline_mae, "rf": rf_mae, "xg": xg_mae}
    best = min(maes, key=maes.get)

    # Retrain best on full data
    if best == "xg" and xg is not None:
        final_model = XGBRegressor(n_estimators=300, random_state=42, verbosity=0)
    else:
        final_model = RandomForestRegressor(n_estimators=200, random_state=42)

    final_model.fit(X, y)

    artifact = {
        "model": final_model,
        "feature_cols": feature_cols,
        "trained_at": datetime.utcnow().isoformat(),
        "mae": maes,
        "best": best
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(artifact, model_path)
    return model_path, maes, best

def load_artifact(path=MODEL_PATH):
    """Load saved artifact dictionary."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact missing: {path}")
    return joblib.load(path)

def predict_next_month_global(model_artifact=None, consolidated_csv=CONSOLIDATED_CSV):
    """Predict next-month revenue for global aggregated data."""
    if model_artifact is None:
        model_artifact = load_artifact()
    df = pd.read_csv(consolidated_csv, parse_dates=["date"])
    monthly = df.groupby(pd.Grouper(key="date", freq="ME"))["price"].sum()
    monthly.index = pd.to_datetime(monthly.index.to_period("M").to_timestamp())
    m = monthly.to_frame(name="monthly_revenue")

    # Build features same as training
    m["lag_1"] = m["monthly_revenue"].shift(1)
    m["lag_3"] = m["monthly_revenue"].shift(3)
    m["lag_6"] = m["monthly_revenue"].shift(6)
    m["roll_mean_3"] = m["monthly_revenue"].shift(1).rolling(3, min_periods=1).mean()
    m["roll_mean_6"] = m["monthly_revenue"].shift(1).rolling(6, min_periods=1).mean()
    m["pct_change_1"] = m["monthly_revenue"].pct_change(1).fillna(0)
    m["pct_change_3"] = m["monthly_revenue"].pct_change(3).fillna(0)
    m["month"] = m.index.month
    m["month_sin"] = np.sin(2 * np.pi * m["month"] / 12.0)
    m["month_cos"] = np.cos(2 * np.pi * m["month"] / 12.0)

    last = m.iloc[-1:]
    X = last[model_artifact["feature_cols"]].values
    pred = float(model_artifact["model"].predict(X)[0])
    return pred
