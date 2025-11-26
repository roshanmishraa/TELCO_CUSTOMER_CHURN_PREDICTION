# test_pipeline_phase1.py
import os
import pandas as pd
import sys

# Add src folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# === FILE PATH FOR YOUR PROJECT ===
DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
TARGET_COL = "Churn"

def main():
    print("=== Testing Phase 1 Pipeline ===")
    print("Load → Preprocess → Feature Engineering\n")

    # 1. Load Raw Data
    print("[1] Loading raw data...")
    df = load_data(DATA_PATH)
    print(f"✔ Loaded: {df.shape}")
    print(df.head(3), "\n")

    # 2. Preprocess
    print("[2] Running preprocessing...")
    df_clean = preprocess_data(df, target_col=TARGET_COL)
    print(f"✔ After preprocessing: {df_clean.shape}")
    print(df_clean.head(3), "\n")

    # 3. Feature Engineering
    print("[3] Building features...")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"✔ After feature engineering: {df_features.shape}")
    print(df_features.head(3), "\n")

    print("🎉 PHASE 1 Pipeline successful!")

if __name__ == "__main__":
    main()
