import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

MODEL_PATH = "models/final_xgb_model.pkl"
DATA_PATH = "data/processed/telco_churn_processed.csv"


def evaluate_model(model, X_test, y_test):
    """
    Evaluates an XGBoost model on test data.
    """
    preds = model.predict(X_test)

    print("\n📊 CLASSIFICATION REPORT\n")
    print(classification_report(y_test, preds))

    print("\n🔢 CONFUSION MATRIX\n")
    print(confusion_matrix(y_test, preds))


if __name__ == "__main__":
    print("📥 Loading processed dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # split in the same way as training
    print("✂️ Splitting into train/test...")
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("📥 Loading trained model...")
    model = pickle.load(open(MODEL_PATH, "rb"))

    print("🚀 Evaluating model...\n")
    evaluate_model(model, X_test, y_test)
