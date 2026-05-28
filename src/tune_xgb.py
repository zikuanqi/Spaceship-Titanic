"""Optuna search for XGBoost hyperparameters.

Saves best params to output/xgb_params.json so train.py can load them.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from features import CATEGORICAL_COLS, build_features, encode_categoricals
from target_encoding import apply_target_encoding

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
PARAMS = ROOT / "params"
OUT.mkdir(exist_ok=True)
PARAMS.mkdir(exist_ok=True)

N_TRIALS = 25
N_SPLITS = 3
SEED = 42


def to_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes.astype("int32")
    return df


def cv_score(params: dict, train: pd.DataFrame, test: pd.DataFrame, target: pd.Series) -> float:
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    for tr_idx, va_idx in folds.split(train, target):
        x_tr, x_va = train.iloc[tr_idx], train.iloc[va_idx]
        y_tr, y_va = target.iloc[tr_idx], target.iloc[va_idx]
        x_tr_te, x_va_te, _ = apply_target_encoding(x_tr, y_tr, x_va, test)

        model = xgb.XGBClassifier(
            **params,
            tree_method="hist",
            eval_metric="logloss",
            early_stopping_rounds=120,
            random_state=SEED,
            verbosity=0,
        )
        model.fit(to_codes(x_tr_te), y_tr, eval_set=[(to_codes(x_va_te), y_va)], verbose=False)
        oof[va_idx] = model.predict_proba(to_codes(x_va_te))[:, 1]
    return accuracy_score(target, oof >= 0.5)


def objective(trial: optuna.Trial, train: pd.DataFrame, test: pd.DataFrame, target: pd.Series) -> float:
    params = {
        "n_estimators": 4000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
    }
    return cv_score(params, train, test, target)


def main() -> None:
    train_raw = pd.read_csv(DATA / "train.csv")
    test_raw = pd.read_csv(DATA / "test.csv")
    target = train_raw["Transported"].astype(int)

    train = build_features(train_raw.drop(columns=["Transported"]))
    test = build_features(test_raw)
    train, test = encode_categoricals(train, test)

    print(f"XGB Optuna: {N_TRIALS} trials, {N_SPLITS}-fold internal CV")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: objective(t, train, test, target), n_trials=N_TRIALS, show_progress_bar=False)

    print(f"\nbest CV: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    out_path = PARAMS / "xgb_params.json"
    out_path.write_text(json.dumps(study.best_params, indent=2))
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
