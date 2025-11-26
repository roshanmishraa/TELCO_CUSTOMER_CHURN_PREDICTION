import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import os

# Load processed data automatically
DATA_PATH = "data/processed/telco_churn_processed.csv"

def tune_model(X, y):
    """
    Tunes an XGBoost model using Optuna.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss"
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring="recall")
        return scores.mean()

    print("🔍 Starting hyperparameter tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("🎉 Best Params:", study.best_params)

    # Save best params
    os.makedirs("models", exist_ok=True)
    pd.to_pickle(study.best_params, "models/best_params.pkl")
    print("📁 Saved best parameters to models/best_params.pkl")

    return study.best_params


# --------------------------------------
# EXECUTION BLOCK
# --------------------------------------
if __name__ == "__main__":
    print("📥 Loading processed dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print(f"📊 Dataset loaded: {df.shape}")
    print("🚀 Starting Optuna tuning...\n")

    tune_model(X, y)
