import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import joblib

# === Config Paths (these must match your updated config.py) ===
from config import (
    TRAIN_EMB, TEST_EMB, VALID_EMB,
    TRAIN_TEXT, TEST_TEXT, VALID_TEXT,
    TEST_PRED_COMPARISON, MODEL_PATH
)

def load_data():
    train_x = pd.read_csv(TRAIN_EMB)
    test_x = pd.read_csv(TEST_EMB)
    val_x = pd.read_csv(VALID_EMB)

    train_y = pd.read_csv(TRAIN_TEXT)["truth_score"]
    test_y = pd.read_csv(TEST_TEXT)["truth_score"]
    val_y = pd.read_csv(VALID_TEXT)["truth_score"]

    return train_x, test_x, val_x, train_y, test_y, val_y

def train_random_forest(train_x, train_y):
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(train_x, train_y)
    return rf

def evaluate_model(model, x, y, name="Set"):
    predictions = model.predict(x)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    print(f"\n📊 {name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² : {r2:.4f}")
    return predictions, mae

def run_training_pipeline():
    train_x, test_x, val_x, train_y, test_y, val_y = load_data()
    print("✅ Data loaded.")

    print("\n🌲 Training Random Forest...")
    model = train_random_forest(train_x, train_y)

    evaluate_model(model, train_x, train_y, name="Training Set")
    test_preds, mae = evaluate_model(model, test_x, test_y, name="Test Set")
    evaluate_model(model, val_x, val_y, name="Validation Set")

    # Save test predictions comparison
    test_y = test_y.reset_index(drop=True)
    test_preds = pd.Series(test_preds)
    absolute_errors = np.abs(test_y - test_preds)
    within_mae_mask = absolute_errors <= mae

    comparison_df = pd.DataFrame({
        "Actual Truth Score": test_y,
        "Predicted Truth Score": test_preds,
        "Absolute Error": absolute_errors,
        "Correct (Within MAE)": within_mae_mask
    })
    comparison_df.to_csv(TEST_PRED_COMPARISON, index=False)
    print(f"\n📄 Saved test predictions comparison to: {TEST_PRED_COMPARISON}")

    correct_count = within_mae_mask.sum()
    accuracy_within_mae = correct_count / len(test_y) * 100
    print(f"✅ Accuracy Within MAE Range: {accuracy_within_mae:.2f}% (MAE = {mae:.4f})")

    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to: {MODEL_PATH}")

    return model, test_x, test_y

if __name__ == "__main__":
    run_training_pipeline()
