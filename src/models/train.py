# src/models/train.py
import os
import pickle
import pandas as pd
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

# MLflow: use the same tracking URI you used earlier
mlflow.set_tracking_uri("file:///D:/TELCO_CUSTOMER_CHURN_PREDICTION/mlruns")
mlflow.set_experiment("Telco Churn - XGBoost")

DATA_PATH = "data/processed/telco_churn_processed.csv"
BEST_PARAMS_PKL = "models/best_params.pkl"
FINAL_MODEL_PKL = "models/final_xgb_model.pkl"
FEATURES_TXT = "models/feature_columns.txt"


def train_model(df: pd.DataFrame, target_col: str = "Churn"):
    # train/test split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # load best params if present, otherwise use defaults
    params = {}
    if os.path.exists(BEST_PARAMS_PKL):
        try:
            params = pickle.load(open(BEST_PARAMS_PKL, "rb"))
            print("📥 Loaded best params from", BEST_PARAMS_PKL)
        except Exception as e:
            print("⚠️ Failed to load best params, using defaults:", e)

    # ensure required fixed params
    params.update({
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss"
    })

    # train inside MLflow run
    with mlflow.start_run():
        print("🚀 Training XGBoost with params:", params)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        print(f"✅ Model Trained | Accuracy: {acc:.4f}, Recall: {rec:.4f}")

        # log params & metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)

        # Save final model locally (pickle) for quick inference
        os.makedirs("models", exist_ok=True)
        with open(FINAL_MODEL_PKL, "wb") as f:
            pickle.dump(model, f)
        print(f"📁 Saved final model to {FINAL_MODEL_PKL}")

        # Save the exact feature ordering (columns) used for training
        feature_cols = list(X_train.columns)
        with open(FEATURES_TXT, "w", encoding="utf-8") as f:
            for c in feature_cols:
                f.write(c + "\n")
        print(f"📁 Saved feature schema to {FEATURES_TXT} ({len(feature_cols)} cols)")

        # Log model to MLflow (xgboost flavor)
        try:
            mlflow.xgboost.log_model(model, artifact_path="model")
            # Also log the feature list as an artifact
            mlflow.log_artifact(FEATURES_TXT, artifact_path="model")
            print("🔗 Logged model + feature schema to MLflow artifacts")
        except Exception as e:
            print("⚠️ MLflow model logging failed:", e)

    return model


if __name__ == "__main__":
    print("📥 Loading processed dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"📊 Dataset loaded: {df.shape}")

    train_model(df, target_col="Churn")
