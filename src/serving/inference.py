# src/serving/inference.py
"""
Production-ready inference utilities for Telco Churn.
This file attempts to load a local pickle model first (models/final_xgb_model.pkl).
If not available, it tries to load the MLflow-logged model artifact automatically.
It also loads the saved feature column list (models/feature_columns.txt) produced by train.py.
"""

import os
import pickle
import pandas as pd
import glob
import mlflow

# LOCAL paths (used during development)
MODEL_DIR_LOCAL = "models"
FINAL_MODEL_PKL = os.path.join(MODEL_DIR_LOCAL, "final_xgb_model.pkl")
FEATURES_TXT = os.path.join(MODEL_DIR_LOCAL, "feature_columns.txt")

# Deterministic binary mapping used during training (keeps parity with build_features.py)
def _binary_map_for_series(s: pd.Series) -> pd.Series:
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # Yes/No
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
    # Gender
    if valset == {"Male", "Female"} or valset == {"Female", "Male"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")
    # generic two-value mapping (alphabetical)
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")
    return s

# Load model: prefer local pickle, fallback to MLflow artifact model
def _load_model():
    # 1) local pickle
    if os.path.exists(FINAL_MODEL_PKL):
        try:
            with open(FINAL_MODEL_PKL, "rb") as f:
                model = pickle.load(f)
            print(f"✅ Loaded local model from {FINAL_MODEL_PKL}")
            return model
        except Exception as e:
            print("⚠️ Failed to load local pickle model:", e)

    # 2) try MLflow artifacts (search mlruns for latest logged model)
    try:
        local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            print("ℹ️ Found MLflow artifact model at", latest_model)
            m = mlflow.pyfunc.load_model(latest_model)
            print("✅ Loaded MLflow model")
            return m
    except Exception as e:
        print("⚠️ MLflow model load failed:", e)

    raise FileNotFoundError("No model found. Place models/final_xgb_model.pkl or log a model to mlflow.")

# Load feature columns list (required)
def _load_feature_columns():
    if os.path.exists(FEATURES_TXT):
        with open(FEATURES_TXT, "r", encoding="utf-8") as f:
            cols = [ln.strip() for ln in f if ln.strip()]
        print(f"✅ Loaded {len(cols)} feature columns from {FEATURES_TXT}")
        return cols

    # try MLflow artifact copy if present
    try:
        local_feature_paths = glob.glob("./mlruns/*/*/artifacts/model/feature_columns.txt")
        if local_feature_paths:
            latest = max(local_feature_paths, key=os.path.getmtime)
            with open(latest, "r", encoding="utf-8") as f:
                cols = [ln.strip() for ln in f if ln.strip()]
            print("✅ Loaded feature columns from MLflow artifact:", latest)
            return cols
    except Exception as e:
        print("⚠️ Failed to load feature columns from MLflow:", e)

    raise FileNotFoundError("Feature columns file not found. Run train.py to generate models/feature_columns.txt")

# Serve-time transform (mirrors src/features/build_features.py)
def _serve_transform(df: pd.DataFrame, feature_cols):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Numeric coercion (same names as training)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Binary encoding: detect binary columns and apply deterministic map
    # We'll apply _binary_map_for_series to any column that is object and only two unique values
    for c in df.columns:
        if df[c].dtype == "object":
            unique_non_null = df[c].dropna().unique()
            if len(unique_non_null) == 2:
                df[c] = _binary_map_for_series(df[c])
    # Convert boolean types to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # One-hot encode remaining object columns with drop_first=True to match training
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Ensure all expected feature columns are present, fill missing with 0, drop extras
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df

# Top-level predict function: accepts dict -> returns dict with probability + label
_model = None
_FEATURE_COLS = None

def predict(input_dict: dict, return_proba: bool = True):
    global _model, _FEATURE_COLS
    if _model is None:
        _model = _load_model()
    if _FEATURE_COLS is None:
        _FEATURE_COLS = _load_feature_columns()

    # format input
    df = pd.DataFrame([input_dict])
    X = _serve_transform(df, _FEATURE_COLS)

    # model predict_proba when available
    try:
        if hasattr(_model, "predict_proba") and return_proba:
            proba = _model.predict_proba(X)
            # assume binary classifier: proba[:,1] is positive class
            score = float(proba[:, 1][0])
            pred = int((score >= 0.5))
            return {"prediction": pred, "probability": score}
        else:
            pred = _model.predict(X)
            if hasattr(pred, "tolist"):
                pred = int(pred.tolist()[0])
            return {"prediction": pred}
    except Exception as e:
        raise Exception(f"Inference failed: {e}")

# CLI test helper
if __name__ == "__main__":
    # small example using first row of processed dataset (if exists)
    try:
        df = pd.read_csv("data/processed/telco_churn_processed.csv")
        sample = df.drop(columns=["Churn"]).iloc[0].to_dict()
        print("Sample input keys:", list(sample.keys())[:10], "...")
        print("Running predict on sample (this uses feature columns saved during training)")
        print(predict(sample))
    except Exception as e:
        print("⚠️ Self-test failed:", e)
