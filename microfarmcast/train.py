"""Training loop for continuous targets and two-stage rainfall."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from .data import load_inputs, validate_targets
from .features import build_supervised
from .evaluate import chronological_masks, regression_metrics, classification_metrics


def _new_regressor(params):
    return XGBRegressor(objective="reg:squarederror", **params)


def _new_classifier(params):
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)


def train(config, config_path="config.json", synthetic_demo=False):
    required_coordinates = [config["latitude_column"], config["longitude_column"]]
    data, audit = load_inputs(config["input_paths"], config["timestamp_column"], config["location_id_column"], required_coordinates)
    data = validate_targets(data, config["target_columns"])
    supervised, features = build_supervised(data, config)
    ts, loc = config["timestamp_column"], config["location_id_column"]
    train_mask, val_mask, test_mask = chronological_masks(supervised, ts, config["validation_fraction"], config["test_fraction"], config["forecast_horizon_hours"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    params = dict(config["xgboost"])
    metrics, importance, medians = {}, [], {}
    summary = {**audit, "feature_count": len(features), "feature_columns": features, "rows_after_horizon_join": len(supervised), "forecast_horizon_hours": config["forecast_horizon_hours"], "synthetic_demo": bool(synthetic_demo), "label": "SYNTHETIC_DEMO_NOT_REAL_PERFORMANCE" if synthetic_demo else config.get("data_label", "REAL_INPUT_UNVERIFIED_UNLESS_SENSOR_LABELS_DOCUMENTED")}

    for target_name, target_col in config["target_columns"].items():
        y_col = f"__future__{target_name}"
        usable = supervised[y_col].notna()
        tr, va, te = train_mask & usable, val_mask & usable, test_mask & usable
        if tr.sum() < 10 or te.sum() < 1:
            metrics[target_name] = {"status": "skipped", "reason": "insufficient labeled train/test rows", "train_rows": int(tr.sum()), "test_rows": int(te.sum())}
            continue
        x_train = supervised.loc[tr, features].replace([np.inf, -np.inf], np.nan)
        med = x_train.median(numeric_only=True).fillna(0.0)
        medians[target_name] = {k: float(v) for k, v in med.items()}
        def matrix(mask):
            return supervised.loc[mask, features].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
        Xtr, Xte = matrix(tr), matrix(te)
        ytr = supervised.loc[tr, y_col].astype(float)
        yte = supervised.loc[te, y_col].astype(float)
        missing_counts = {c: int(supervised.loc[tr, c].isna().sum()) for c in features if supervised.loc[tr, c].isna().any()}

        if target_name != "rainfall":
            model = _new_regressor(params)
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            target_metrics = {"status": "trained", "validation_xgboost": regression_metrics(supervised.loc[va & usable, y_col], model.predict(matrix(va & usable))) if (va & usable).sum() else {"n": 0, "mae": None, "rmse": None}, "xgboost": regression_metrics(yte, pred), "persistence": regression_metrics(yte, supervised.loc[te, target_col].fillna(med.get(target_col, 0)).astype(float))}
            # Regional forecast baseline is only meaningful if explicitly supplied as a valid-time predictor.
            forecast_col = f"regional_forecast_{target_name}"
            if forecast_col in supervised.columns and supervised.loc[te, forecast_col].notna().any():
                valid = te & supervised[forecast_col].notna()
                target_metrics["regional_forecast"] = regression_metrics(supervised.loc[valid, y_col], supervised.loc[valid, forecast_col])
            model.save_model(str(output_dir / f"{target_name}.json"))
            gain = model.get_booster().get_score(importance_type="gain")
            importance.extend({"target": target_name, "feature": f, "gain": float(gain.get(f, 0.0))} for f in features)
            metrics[target_name] = target_metrics
            sites = sorted(supervised[loc].unique())
            if len(sites) >= 3:
                held = set(sites[::5]) or {sites[-1]}
                spatial_train = tr & ~supervised[loc].isin(held)
                spatial_test = te & supervised[loc].isin(held)
                if spatial_train.sum() >= 10 and spatial_test.sum() > 0:
                    spatial_model = _new_regressor(params)
                    spatial_model.fit(matrix(spatial_train), supervised.loc[spatial_train, y_col].astype(float))
                    metrics[target_name]["held_out_locations_xgboost"] = regression_metrics(
                        supervised.loc[spatial_test, y_col],
                        spatial_model.predict(matrix(spatial_test)),
                    )
        else:
            threshold = float(config["rain_threshold_mm"])
            ytr_rain = (ytr >= threshold).astype(int)
            if ytr_rain.nunique() < 2:
                metrics[target_name] = {"status": "skipped", "reason": "training period contains only one rain class"}
                continue
            classifier = _new_classifier(params)
            classifier.fit(Xtr, ytr_rain)
            rain_prob = classifier.predict_proba(Xte)[:, 1]
            rain_pred = rain_prob >= 0.5
            yte_rain = (yte >= threshold).to_numpy()
            val_supervised = va & usable
            if val_supervised.sum():
                val_truth = (supervised.loc[val_supervised, y_col] >= threshold).to_numpy()
                val_pred = classifier.predict_proba(matrix(val_supervised))[:, 1] >= 0.5
                validation_rain = classification_metrics(val_truth, val_pred)
            else:
                validation_rain = {"n": 0, "precision": None, "recall": None, "f1": None}
            amount_train = tr & (supervised[y_col] >= threshold)
            amount_model = None
            if amount_train.sum() >= 10:
                amount_model = _new_regressor(params)
                amount_model.fit(matrix(amount_train), supervised.loc[amount_train, y_col].astype(float))
                amount_model.save_model(str(output_dir / "rainfall_amount.json"))
                amount_pred = np.maximum(0, amount_model.predict(Xte))
            else:
                amount_pred = np.zeros(int(te.sum()))
            combined = np.where(rain_pred & (amount_model is not None), amount_pred, 0.0)
            rainfall_metrics = {"status": "trained", "validation_rain_classifier": validation_rain, "rain_classifier": classification_metrics(yte_rain, rain_pred), "rainfall_amount_on_observed_rain": regression_metrics(yte[yte >= threshold], amount_pred[yte >= threshold]), "combined_rainfall": regression_metrics(yte, combined), "amount_model_available": amount_model is not None}
            rainfall_metrics["persistence_rainfall"] = regression_metrics(yte, supervised.loc[te, target_col].fillna(0).astype(float))
            persistence_rain = supervised.loc[te, target_col].fillna(0).to_numpy() >= threshold
            rainfall_metrics["persistence_rain_classifier"] = classification_metrics(yte_rain, persistence_rain)
            classifier.save_model(str(output_dir / "rainfall_occurrence.json"))
            gain = classifier.get_booster().get_score(importance_type="gain")
            importance.extend({"target": "rainfall_occurrence", "feature": f, "gain": float(gain.get(f, 0.0))} for f in features)
            if amount_model is not None:
                gain = amount_model.get_booster().get_score(importance_type="gain")
                importance.extend({"target": "rainfall_amount", "feature": f, "gain": float(gain.get(f, 0.0))} for f in features)
            metrics[target_name] = rainfall_metrics
            sites = sorted(supervised[loc].unique())
            if len(sites) >= 3:
                held = set(sites[::5]) or {sites[-1]}
                spatial_train = tr & ~supervised[loc].isin(held)
                spatial_test = te & supervised[loc].isin(held)
                if spatial_train.sum() >= 10 and spatial_test.sum() > 0:
                    spatial_classes = (supervised.loc[spatial_train, y_col] >= threshold).astype(int)
                    if spatial_classes.nunique() == 2:
                        spatial_classifier = _new_classifier(params)
                        spatial_classifier.fit(matrix(spatial_train), spatial_classes)
                        spatial_prob = spatial_classifier.predict_proba(matrix(spatial_test))[:, 1]
                        spatial_truth = (supervised.loc[spatial_test, y_col] >= threshold).to_numpy()
                        spatial_pred = spatial_prob >= 0.5
                        metrics[target_name]["held_out_locations_rain_classifier"] = classification_metrics(spatial_truth, spatial_pred)
        metrics[target_name]["train_rows"] = int(tr.sum())
        metrics[target_name]["validation_rows_reserved"] = int(va.sum())
        metrics[target_name]["test_rows"] = int(te.sum())
        metrics[target_name]["training_missing_feature_counts"] = missing_counts

    summary["target_rows_missing_horizon_label"] = {name: int(supervised[f"__future__{name}"].isna().sum()) for name in config["target_columns"]}
    summary["validation_rows_total"] = int(val_mask.sum())
    summary["test_rows_total"] = int(test_mask.sum())
    summary["note"] = "Validation is reserved; model fitting uses training-period rows only. No tuning or real-world validation is implied."
    (output_dir / "training_medians.json").write_text(json.dumps(medians, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(importance).to_csv(output_dir / "feature_importance.csv", index=False)
    return metrics, summary
