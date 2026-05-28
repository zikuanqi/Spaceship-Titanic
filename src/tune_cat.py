"""Optuna search for CatBoost hyperparameters.

Saves best params to params/cat_params.json so train.py can load them.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from features import CATEGORICAL_COLS, build_features, encode_categoricals
from target_encoding import apply_target_encoding

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PARAMS = ROOT / "params"
PARAMS.mkdir(exist_ok=True)

N_TRIALS = 25
N_SPLITS = 3
SEED = 42


def to_str_cats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("nan")
    return df


def cv_score(params: dict, train: pd.DataFrame, test: pd.DataFrame, target: pd.Series) -> float:
    cat_idx = [i for i, c in enumerate(train.columns) if c in CATEGORICAL_COLS]
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    for tr_idx, va_idx in folds.split(train, target):
        x_tr, x_va = train.iloc[tr_idx], train.iloc[va_idx]
        y_tr, y_va = target.iloc[tr_idx], target.iloc[va_idx]
        x_tr_te, x_va_te, _ = apply_target_encoding(x_tr, y_tr, x_va, test)
        x_tr_s = to_str_cats(x_tr_te); x_va_s = to_str_cats(x_va_te)

        model = CatBoostClassifier(
            **params,
            iterations=4000,
            early_stopping_rounds=120,
            eval_metric="Accuracy",
            random_seed=SEED,
            verbose=0,
            cat_features=cat_idx,
        )
        model.fit(x_tr_s, y_tr, eval_set=(x_va_s, y_va), use_best_model=True)
        oof[va_idx] = model.predict_proba(x_va_s)[:, 1]
    return accuracy_score(target, oof >= 0.5)


def objective(trial, train, test, target) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 9),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
        "random_strength": trial.suggest_float("random_strength", 0.5, 3.0),
        "border_count": trial.suggest_int("border_count", 64, 254),
    }
    return cv_score(params, train, test, target)


def main() -> None:
    train_raw = pd.read_csv(DATA / "train.csv")
    test_raw = pd.read_csv(DATA / "test.csv")
    target = train_raw["Transported"].astype(int)

    train = build_features(train_raw.drop(columns=["Transported"]))
    test = build_features(test_raw)
    train, test = encode_categoricals(train, test)

    print(f"CatBoost Optuna: {N_TRIALS} trials, {N_SPLITS}-fold internal CV")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: objective(t, train, test, target), n_trials=N_TRIALS, show_progress_bar=False)

    print(f"\nbest CV: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    out_path = PARAMS / "cat_params.json"
    out_path.write_text(json.dumps(study.best_params, indent=2))
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
