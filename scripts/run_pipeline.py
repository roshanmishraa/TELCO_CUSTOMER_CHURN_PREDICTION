#!/usr/bin/env python3
"""
RUN COMPLETE ML PIPELINE:
1. Load raw data
2. Validate (Great Expectations)
3. Preprocess
4. Feature engineering
5. Train/Test split
6. Train model (XGBoost)
7. Evaluate
8. Log to MLflow
"""

import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

# --- Fix local import paths ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local ML pipeline modules
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data


def main(args):

    # -------------------------
    #  MLflow setup
    # -------------------------
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = args.mlflow_uri or f"file://{project_root}/mlruns"

    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():

        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        # -------------------------
        #  Load + Validate data
        # -------------------------
        print("📥 Loading data...")
        df = load_data(args.input)
        print(f"Loaded dataset: {df.shape}")

        print("🔍 Validating data with Great Expectations...")
        valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(valid))

        if not valid:
            raise ValueError(f"Data quality failed: {failed}")

        # -------------------------
        #  Preprocess
        # -------------------------
        print("🧹 Preprocessing...")
        df = preprocess_data(df, target_col=args.target)

        processed_path = os.path.join(project_root, "data/processed/telco_clean.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)

        # -------------------------
        #  Feature Engineering
        # -------------------------
        print("🛠 Building features...")
        df_enc = build_features(df, target_col=args.target)

        # Convert boolean → int
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)

        feature_cols = list(df_enc.drop(columns=[args.target]).columns)

        # Save schema for inference consistency
        schema_path = os.path.join(project_root, "artifacts/feature_columns.txt")
        os.makedirs(os.path.dirname(schema_path), exist_ok=True)
        with open(schema_path, "w") as f:
            f.write("\n".join(feature_cols))

        mlflow.log_artifact(schema_path)

        # -------------------------
        #  Train/Test Split
        # -------------------------
        print("📊 Splitting...")
        X = df_enc.drop(columns=[args.target])
        y = df_enc[args.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            stratify=y,
            random_state=42
        )

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Class imbalance ratio: {scale_pos_weight:.2f}")

        # -------------------------
        #  Train Model
        # -------------------------
        print("🤖 Training XGBoost...")
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        )

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)

        # -------------------------
        #  Evaluate
        # -------------------------
        print("📈 Evaluating...")
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= args.threshold).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(classification_report(y_test, y_pred, digits=3))

        # -------------------------
        #  Log Model
        # -------------------------
        mlflow.sklearn.log_model(model, "model")
        print("💾 Model saved to MLflow.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run full churn ML pipeline")
    p.add_argument("--input", required=True)
    p.add_argument("--target", default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", default="Telco Churn Pipeline")
    p.add_argument("--mlflow_uri", default=None)
    args = p.parse_args()

    main(args)
