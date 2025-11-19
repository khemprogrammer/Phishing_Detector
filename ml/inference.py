import json
from pathlib import Path
from joblib import load
from .features import FeatureExtractor

base = Path(__file__).resolve().parent
art_dir = base / "artifacts"

_fe = None
_model = None
_meta = None

def _load_artifacts():
    global _fe, _model, _meta
    if _meta is None:
        meta_path = art_dir / "model_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError("model_meta.json not found. Train models first.")
        with open(meta_path) as f:
            _meta = json.load(f)["best"]
    if _fe is None:
        fe_path = art_dir / "feature_extractor.pkl"
        if not fe_path.exists():
            raise FileNotFoundError("feature_extractor.pkl not found.")
        _fe = load(fe_path)
    if _model is None:
        if _meta["model_type"] == "keras":
            import tensorflow as tf
            _model = tf.keras.models.load_model(art_dir / "best_model_keras.h5")
        else:
            _model = load(art_dir / "best_model.pkl")

def predict_url(url):
    _load_artifacts()
    X = _fe.transform([url])
    if _meta["model_type"] == "keras":
        prob = float(_model.predict(X, verbose=0).ravel()[0])
    else:
        prob = float(_model.predict_proba(X)[:,1][0])
    label = int(prob >= 0.5)
    return {"label": label, "probability": prob, "model": _meta["name"]}

_load_artifacts()