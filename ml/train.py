import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
from .features import FeatureExtractor

base = Path(__file__).resolve().parent
data_dir = base / "data"
art_dir = base / "artifacts"
plot_dir = base / "plots"
art_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)
TRAIN_MODE = os.getenv("TRAIN_MODE", "full")

def _normalize_url(u):
    if pd.isna(u):
        return None
    s = str(u).strip().strip('"').strip("'")
    if not s.lower().startswith(("http://", "https://")):
        s = "http://" + s
    return s

def load_data():
    phiusiil_path = data_dir / "PhiUSIIL_Phishing_URL_Dataset.csv"
    if phiusiil_path.exists():
        df_phiusiil = pd.read_csv(phiusiil_path)
        df_phiusiil = df_phiusiil.rename(columns={"URL": "url", "label": "original_label"})
        df_phiusiil["label"] = df_phiusiil["original_label"].map({1: 0, 0: 1}) # Invert labels: 1 (legitimate) -\u003e 0, 0 (phishing) -\u003e 1
        df_phiusiil = df_phiusiil.dropna(subset=["url", "label"]).loc[:, ["url", "label"]]
        
        ds_path = data_dir / "dataset_phishing.csv"
        if ds_path.exists():
            df_phishing = pd.read_csv(ds_path)
            df_phishing = df_phishing.drop_duplicates(subset=["url"]).copy()
            df_phishing["label"] = df_phishing["status"].map({"phishing": 1, "legitimate": 0})
            df_phishing = df_phishing.dropna(subset=["url","label"]).loc[:, ["url","label"]]
            
            df = pd.concat([df_phiusiil, df_phishing], ignore_index=True)
            df = df.drop_duplicates(subset=["url"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
            return df
        else:
            df = df_phiusiil.sample(frac=1.0, random_state=42).reset_index(drop=True)
            return df
    
    us_path = data_dir / "urlset.csv"
    if us_path.exists():
        try:
            df_raw = pd.read_csv(us_path)
        except Exception:
            df_raw = pd.read_csv(us_path, encoding="latin-1", engine="python", on_bad_lines="skip")
        url_col = None
        if "url" in df_raw.columns:
            url_col = "url"
        elif "domain" in df_raw.columns:
            url_col = "domain"
        else:
            raise ValueError("urlset.csv must contain 'url' or 'domain' column")
        if "label" in df_raw.columns:
            df_raw["label"] = pd.to_numeric(df_raw["label"], errors="coerce").round().astype("Int64")
        elif "status" in df_raw.columns:
            df_raw["label"] = df_raw["status"].map({"phishing": 1, "legitimate": 0})
        else:
            raise ValueError("urlset.csv must contain 'label' (0/1) or 'status' ('phishing'/'legitimate') column")
        df_raw[url_col] = df_raw[url_col].apply(_normalize_url)
        df = df_raw.drop_duplicates(subset=[url_col]).dropna(subset=[url_col, "label"]).rename(columns={url_col: "url"})
        df["label"] = df["label"].astype(int)
        if df["label"].nunique() != 2:
            raise ValueError("urlset.csv must include both classes (label 0 and 1)")
        df = df.loc[:, ["url", "label"]].sample(frac=1.0, random_state=42).reset_index(drop=True)
        return df
    ds_path = data_dir / "dataset_phishing.csv"
    if ds_path.exists():
        df_raw = pd.read_csv(ds_path)
        if "url" not in df_raw.columns:
            raise ValueError("dataset_phishing.csv must contain a 'url' column")
        if "status" not in df_raw.columns:
            raise ValueError("dataset_phishing.csv must contain a 'status' column with 'phishing'/'legitimate'")
        df = df_raw.drop_duplicates(subset=["url"]).copy()
        df["label"] = df_raw["status"].map({"phishing": 1, "legitimate": 0})
        df = df.dropna(subset=["url","label"]).loc[:, ["url","label"]]
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        return df
    phish_p = data_dir / "phish.csv"
    legit_p = data_dir / "legit.csv"
    if not phish_p.exists() or not legit_p.exists():
        raise FileNotFoundError("Provide dataset_phishing.csv or both phish.csv and legit.csv in ml/data/")
    phish = pd.read_csv(phish_p)
    legit = pd.read_csv(legit_p)
    phish = phish.drop_duplicates(subset=["url"]).assign(label=1)
    legit = legit.drop_duplicates(subset=["url"]).assign(label=0)
    df = pd.concat([phish[["url","label"]], legit[["url","label"]]], ignore_index=True)
    df = df.dropna(subset=["url"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def evaluate(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1, "roc_auc": auc}, cm

def plot_roc(y_true, y_prob, name):
    disp = RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"ROC: {name}")
    plt.savefig(plot_dir / f"roc_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_cm(cm, name):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(plot_dir / f"cm_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

def main():
    df = load_data()
    if TRAIN_MODE == "fast":
        df = df.sample(n=min(len(df), 5000), random_state=42).reset_index(drop=True)
    elif TRAIN_MODE == "medium":
        df = df.sample(n=min(len(df), 25000), random_state=42).reset_index(drop=True)
    X_urls = df["url"].tolist()
    y = df["label"].values.astype(int)
    fe = FeatureExtractor()
    X = fe.transform(X_urls)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    results = []
    if TRAIN_MODE == "fast":
        rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_prob_rf = rf.predict_proba(X_test)[:,1]
        metrics_rf, cm_rf = evaluate(y_test, y_prob_rf)
        plot_roc(y_test, y_prob_rf, "RandomForest")
        plot_cm(cm_rf, "RandomForest")
        results.append({"name": "RandomForest", "metrics": metrics_rf, "estimator": rf})
        xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, eval_metric="logloss", random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_prob_xgb = xgb.predict_proba(X_test)[:,1]
        metrics_xgb, cm_xgb = evaluate(y_test, y_prob_xgb)
        plot_roc(y_test, y_prob_xgb, "XGBoost")
        plot_cm(cm_xgb, "XGBoost")
        results.append({"name": "XGBoost", "metrics": metrics_xgb, "estimator": xgb})
    else:
        models = []
        pipe_lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
        grid_lr = {"clf__C": [1], "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]}
        models.append(("LogReg", pipe_lr, grid_lr))
        pipe_svm = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=42))])
        grid_svm = {"clf__C": [1], "clf__gamma": ["scale"], "clf__kernel": ["rbf"]}
        models.append(("SVM", pipe_svm, grid_svm))
        rf = RandomForestClassifier(random_state=42)
        grid_rf = {"n_estimators": [100], "max_depth": [None], "min_samples_split": [2]}
        models.append(("RandomForest", rf, grid_rf))
        gb = GradientBoostingClassifier(random_state=42)
        grid_gb = {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [3]}
        models.append(("GradBoost", gb, grid_gb))
        xgb = XGBClassifier(eval_metric="logloss", random_state=42)
        grid_xgb = {
            "n_estimators": [300],
            "max_depth": [5],
            "learning_rate": [0.1],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "gamma": [0],
            "reg_alpha": [0],
            "reg_lambda": [1]
        }
        models.append(("XGBoost", xgb, grid_xgb))
        for name, est, grid in models:
            cv_val = 2 if TRAIN_MODE == "medium" else 3
            gs = GridSearchCV(est, grid, scoring="roc_auc", cv=cv_val, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            y_prob = best.predict_proba(X_test)[:,1] if hasattr(best, "predict_proba") else best.decision_function(X_test)
            metrics, cm = evaluate(y_test, y_prob)
            plot_roc(y_test, y_prob, name)
            plot_cm(cm, name)
            results.append({"name": name, "metrics": metrics, "estimator": best})
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        dl = Sequential()
        dl.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
        dl.add(Dropout(0.3))
        dl.add(Dense(32, activation="relu"))
        dl.add(Dropout(0.2))
        dl.add(Dense(1, activation="sigmoid"))
        dl.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
        es = EarlyStopping(monitor="val_AUC", mode="max", patience=5, restore_best_weights=True)
        epochs = 20 if TRAIN_MODE == "fast" else (15 if TRAIN_MODE == "medium" else 50)
        dl.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=64, callbacks=[es], verbose=0)
        y_prob_dl = dl.predict(X_test, verbose=0).ravel()
        metrics_dl, cm_dl = evaluate(y_test, y_prob_dl)
        plot_roc(y_test, y_prob_dl, "DeepNN")
        plot_cm(cm_dl, "DeepNN")
        results.append({"name": "DeepNN", "metrics": metrics_dl, "estimator": dl})
    except Exception:
        pass
    from joblib import dump
    dump(fe, art_dir / "feature_extractor.pkl")
    best_item = max(results, key=lambda r: r["metrics"]["roc_auc"])
    if best_item["name"] == "DeepNN":
        best_item["estimator"].save(art_dir / "best_model_keras.h5")
        meta = {"model_type": "keras", "name": best_item["name"]}
    else:
        dump(best_item["estimator"], art_dir / "best_model.pkl")
        meta = {"model_type": "sklearn", "name": best_item["name"]}
    with open(art_dir / "model_meta.json", "w") as f:
        json.dump({"best": meta, "all": {r["name"]: r["metrics"] for r in results}}, f, indent=2)

if __name__ == "__main__":
    main()