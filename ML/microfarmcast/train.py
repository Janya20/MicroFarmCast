"""Train, evaluate, and persist the MicroFarmCast demonstration models."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from xgboost import XGBRegressor, XGBClassifier
from .data import load_csv
from .features import build_features, TARGETS


def run(config_path: str = "config.yaml") -> dict:
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_path = Path(cfg["data"]["input_csv"])
    if not input_path.is_absolute(): input_path = config_path.parent / input_path
    output = Path(cfg["data"]["artifact_dir"])
    if not output.is_absolute(): output = config_path.parent / output
    data = load_csv(input_path)
    if not data.get("synthetic_demo", pd.Series(False, index=data.index)).astype(str).str.lower().isin(["true", "1", "yes"]).all():
        synthetic = False
    else: synthetic = True
    data, features = build_features(data)
    times = np.array(sorted(data.timestamp.unique()))
    if len(times) < 20: raise ValueError("Need at least 20 distinct hourly timestamps for chronological splitting.")
    train_end = times[int(len(times)*.70)-1]
    valid_end = times[int(len(times)*.85)-1]
    # The next-hour label must remain inside the same partition; remove boundary rows.
    target_time = data.timestamp + pd.Timedelta(hours=1)
    parts = np.where((data.timestamp <= train_end) & (target_time <= train_end), "train",
             np.where((data.timestamp > train_end) & (data.timestamp <= valid_end) & (target_time <= valid_end), "validation",
             np.where(data.timestamp > valid_end, "test", "embargo")))
    data["_part"] = parts
    xraw = data[features].replace([np.inf, -np.inf], np.nan)
    output.mkdir(parents=True, exist_ok=True)
    (output/"models").mkdir(exist_ok=True)
    metrics = {"data_type": "synthetic-demo" if synthetic else "real-observations",
               "warning": "Synthetic-demo metrics demonstrate pipeline execution only; they are not evidence of real-world or farm-level accuracy." if synthetic else "Validate with local observations before operational use.",
               "input_rows": int(len(data)), "locations": int(data.location_id.nunique()),
               "train_rows": int((parts == "train").sum()), "validation_rows": int((parts == "validation").sum()),
               "test_rows": int((parts == "test").sum()), "embargo_rows": int((parts == "embargo").sum()),
               "split_train_end": str(train_end), "split_validation_end": str(valid_end), "models": {}}
    reg_params = dict(cfg["training"]["xgboost"])
    for col in TARGETS:
        label = f"target_{col}"
        mask = data[label].notna().to_numpy()
        train = (parts == "train") & mask
        valid = (parts == "validation") & mask
        test = (parts == "test") & mask
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(xraw.loc[train])
        x = pd.DataFrame(imputer.transform(xraw), columns=features, index=data.index)
        if col == "rainfall_mm":
            ybin = (data[label].fillna(0) > 0).astype(int)
            classifier = XGBClassifier(**cfg["training"]["classifier"], random_state=cfg["training"]["seed"], eval_metric="logloss")
            classifier.fit(x.loc[train], ybin.loc[train])
            wet_train = train & ybin.to_numpy().astype(bool)
            amount_model = XGBRegressor(**reg_params, random_state=cfg["training"]["seed"])
            amount_model.fit(x.loc[wet_train], data.loc[wet_train, label])
            bundle = {"kind":"rainfall", "classifier":classifier, "amount_model":amount_model, "imputer":imputer, "features":features}
            scores = {}
            for part, m in (("validation", valid), ("test", test)):
                if not m.any(): continue
                prob = classifier.predict_proba(x.loc[m])[:, 1]
                pred_wet = prob >= .5
                amount = np.maximum(0, amount_model.predict(x.loc[m]))
                pred = np.where(pred_wet, amount, 0)
                actual = data.loc[m, label].to_numpy()
                scores[part] = {"rain_accuracy": float(accuracy_score(actual > 0, pred_wet)),
                                "rain_precision": float(precision_score(actual > 0, pred_wet, zero_division=0)),
                                "rain_recall": float(recall_score(actual > 0, pred_wet, zero_division=0)),
                                "rain_f1": float(f1_score(actual > 0, pred_wet, zero_division=0)),
                                "rain_brier": float(brier_score_loss(actual > 0, prob)),
                                "amount_mae_all_cases": float(mean_absolute_error(actual, pred)),
                                "amount_rmse_all_cases": float(np.sqrt(mean_squared_error(actual, pred))),
                                "persistence_amount_mae": float(mean_absolute_error(actual, data.loc[m, col]))}
            metrics["models"][col] = {"status":"trained", "evaluation":scores}
        else:
            model = XGBRegressor(**reg_params, random_state=cfg["training"]["seed"])
            model.fit(x.loc[train], data.loc[train, label])
            bundle = {"kind":"regression", "model":model, "imputer":imputer, "features":features}
            scores = {}
            for part, m in (("validation", valid), ("test", test)):
                if not m.any(): continue
                actual = data.loc[m, label].to_numpy()
                pred = model.predict(x.loc[m])
                persistence = data.loc[m, col].to_numpy()
                scores[part] = {"xgboost_mae":float(mean_absolute_error(actual,pred)), "xgboost_rmse":float(np.sqrt(mean_squared_error(actual,pred))),
                                "persistence_mae":float(mean_absolute_error(actual,persistence)), "persistence_rmse":float(np.sqrt(mean_squared_error(actual,persistence)))}
            metrics["models"][col] = {"status":"trained", "evaluation":scores}
        joblib.dump(bundle, output/"models"/("rainfall.joblib" if col == "rainfall_mm" else f"{col}.joblib"))
    (output/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MicroFarmCast Model 1")
    parser.add_argument("--config", default="config.yaml")
    print(json.dumps(run(parser.parse_args().config), indent=2))
