import os
import sys
import pandas as pd

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data

RAW = "data/raw/Telco-Customer-Churn.csv"
OUT = "data/processed/telco_churn_processed.csv"

# 1) LOAD RAW
print("📥 Loading raw dataset...")
df = pd.read_csv(RAW)
print(f"📊 Raw shape: {df.shape}")

# 🔧 PREPROCESS FIRST (so TotalCharges becomes numeric)
print("\n🛠️ Running preprocessing BEFORE validation...")
df = preprocess_data(df, target_col="Churn")

# Ensure Churn is 0/1
if "Churn" in df.columns and df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1}).astype("Int64")

print("🧹 Preprocessing complete. TotalCharges is now numeric.")

# 2) VALIDATE CLEANED DATA (industry standard)
print("\n🔍 Running Great Expectations validation on CLEANED data...")
valid, failed = validate_telco_data(df)

if not valid:
    raise ValueError(
        f"❌ DATA VALIDATION FAILED.\n"
        f"Failed checks: {failed}\n"
        "Fix your dataset before continuing."
    )
print("✅ Data validation PASSED! Proceeding to feature engineering.\n")

# 3) FEATURE ENGINEERING
df_processed = build_features(df, target_col="Churn")
print("🔧 Feature engineering complete.")

# 4) SAVE FINAL DATASET
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)

print(f"\n🎉 FINAL DATASET SAVED")
print(f"📁 {OUT}")
print(f"📏 Shape: {df_processed.shape}")
