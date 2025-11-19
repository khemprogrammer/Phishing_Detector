import json
from pathlib import Path
from joblib import load
import pandas as pd
from xgboost import XGBClassifier
from .features import FeatureExtractor

base = Path(__file__).resolve().parent
art_dir = base / "artifacts"

def analyze_feature_importance():
    meta_path = art_dir / "model_meta.json"
    if not meta_path.exists():
        print("Error: model_meta.json not found. Train models first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)["best"]

    if meta["name"] != "XGBoost":
        print(f"The best model is {meta['name']}, not XGBoost. Please adjust the script.")
        return

    fe_path = art_dir / "feature_extractor.pkl"
    if not fe_path.exists():
        print("Error: feature_extractor.pkl not found.")
        return
    fe = load(fe_path)

    model_path = art_dir / "best_model.pkl"
    if not model_path.exists():
        print("Error: best_model.pkl not found.")
        return
    model = load(model_path)

    if not isinstance(model, XGBClassifier):
        print("Error: Loaded model is not an XGBoost Classifier.")
        return

    # Get feature importances
    feature_importances = model.feature_importances_
    feature_names = [
        "url_len", "host_len", "path_len", "query_len", "num_dots", "num_hyphens",
        "num_digits", "num_params", "has_https", "has_at", "has_ip", "tld_len",
        "num_subdirs", "num_fragments", "tokens_count", "ratio_digits",
        "ratio_special", "starts_www"
    ]

    if len(feature_importances) != len(feature_names):
        print("Warning: Mismatch between number of feature importances and feature names.")
        print(f"Importances: {len(feature_importances)}, Names: {len(feature_names)}")
        return

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print("\nFeature Importances (XGBoost):")
    print(importance_df)

if __name__ == "__main__":
    analyze_feature_importance()