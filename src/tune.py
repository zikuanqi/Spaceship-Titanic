"""Optuna search for LightGBM hyperparameters.

Saves the best params to output/lgb_params.json so train.py can load them.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from features import build_features, encode_categoricals
from target_encoding import apply_target_encoding

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

N_TRIALS = 25
N_SPLITS = 3
SEED = 42


def cv_score(params: dict, train: pd.DataFrame, test: pd.DataFrame, target: pd.Series) -> float:
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    for tr_idx, va_idx in folds.split(train, target):
        x_tr, x_va = train.iloc[tr_idx], train.iloc[va_idx]
        y_tr, y_va = target.iloc[tr_idx], target.iloc[va_idx]

        x_tr_te, x_va_te, _ = apply_target_encoding(x_tr, y_tr, x_va, test)

        dtrain = lgb.Dataset(x_tr_te, y_tr, categorical_feature="auto")
        dvalid = lgb.Dataset(x_va_te, y_va, categorical_feature="auto", reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict(x_va_te, num_iteration=model.best_iteration)
    return accuracy_score(target, oof >= 0.5)


def objective(trial: optuna.Trial, train: pd.DataFrame, test: pd.DataFrame, target: pd.Series) -> float:
    params = {
        "objective": "binary",
        "metric": "binary_error",
        "verbose": -1,
        "seed": SEED,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 60),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
    }
    return cv_score(params, train, test, target)


def main() -> None:
    train_raw = pd.read_csv(DATA / "train.csv")
    test_raw = pd.read_csv(DATA / "test.csv")
    target = train_raw["Transported"].astype(int)

    train = build_features(train_raw.drop(columns=["Transported"]))
    test = build_features(test_raw)
    train, test = encode_categoricals(train, test)

    print(f"running Optuna: {N_TRIALS} trials, {N_SPLITS}-fold internal CV")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: objective(t, train, test, target), n_trials=N_TRIALS, show_progress_bar=False)

    print(f"\nbest CV: {study.best_value:.4f}")
    print("best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    saved = {
        "objective": "binary",
        "metric": "binary_error",
        "verbose": -1,
        **study.best_params,
    }
    out_path = OUT / "lgb_params.json"
    out_path.write_text(json.dumps(saved, indent=2))
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
