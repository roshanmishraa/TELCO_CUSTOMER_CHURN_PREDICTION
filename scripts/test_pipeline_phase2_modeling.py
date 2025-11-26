import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
import optuna

print("\n=== Phase 2: Modeling with XGBoost + Optuna ===\n")

# Load processed dataset
df = pd.read_csv("data/processed/telco_churn_processed.csv")
print(f"Loaded dataset: {df.shape}")

# Ensure target is numeric 0/1
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1})

assert df["Churn"].isna().sum() == 0, "Churn has NaNs"
assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1"

# Split X and y
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Data split completed.")
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Classification threshold
THRESHOLD = 0.35  # same as your main pipeline

# === Optuna Objective Function ===
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)

    return recall_score(y_test, y_pred, pos_label=1)

# === Run Optuna ===
print("\nRunning Optuna hyperparameter tuning...\n")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# === Show Results ===
print("\n==============================")
print("Best Parameters Found:")
print(study.best_params)
print(f"Best Recall Score: {study.best_value:.4f}")
print("==============================\n")
