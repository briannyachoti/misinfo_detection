import os

def safe_run(script_path, label=""):
    print(f"\n🚀 Running: {label or script_path}")
    result = os.system(f"python {script_path}")
    if result != 0:
        print(f"❌ Failed: {script_path}")
    else:
        print(f"✅ Finished: {script_path}")

# === Paths to scripts ===
efficientnet_train = os.path.join("efficientnet", "train_effnet.py")
feature_extract = os.path.join("feature_based", "extract_features.py")
feature_train = os.path.join("feature_based", "train_xgboost.py")
feature_eval = os.path.join("feature_based", "evaluate_model.py")
hybrid_vote = os.path.join("hybrid", "hybrid_voting.py")

# === Pipeline Execution ===
print("📦 Starting full image pipeline...\n")

safe_run(efficientnet_train, "Train EfficientNet Model")
safe_run(feature_extract, "Extract Feature-Based Image Stats")
safe_run(feature_train, "Train Feature-Based Classifiers")
safe_run(feature_eval, "Evaluate Feature-Based Model")
safe_run(hybrid_vote, "Run Hybrid Voting Ensemble")

print("\n🎯 Full image pipeline completed successfully!")
