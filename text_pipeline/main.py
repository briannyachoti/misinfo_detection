import os

#print("📥 Generating enriched RoBERTa embeddings...")
#os.system("python generate_roberta_embeddings.py")

print("🌲 Training Random Forest...")
os.system("python train_rf_model.py")

#print("📊 Evaluating model (MAE, MSE, R²)...")
#os.system("python evaluate_model.py")

print("📈 Generating visualizations and MAE breakdowns...")
os.system("python visualize_rf_results.py")

print("✅ Text pipeline complete! Outputs saved in:")
print("   📁 embeddings/: RoBERTa features")
print("   📁 outputs/: predictions, plots, CSVs")
print("   📁 models/: saved model (.pkl)")
