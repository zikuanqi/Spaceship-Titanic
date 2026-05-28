"""Train LightGBM with 5-fold stratified CV and write a submission file."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from features import build_features, encode_categoricals

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

N_SPLITS = 5
SEED = 42

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_error",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
}


def main() -> None:
    train_raw = pd.read_csv(DATA / "train.csv")
    test_raw = pd.read_csv(DATA / "test.csv")
    test_ids = test_raw["PassengerId"].copy()

    target = train_raw["Transported"].astype(int)
    train = build_features(train_raw.drop(columns=["Transported"]))
    test = build_features(test_raw)
    train, test = encode_categoricals(train, test)

    print(f"feature columns ({train.shape[1]}): {list(train.columns)}")

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(folds.split(train, target), 1):
        x_tr, x_va = train.iloc[tr_idx], train.iloc[va_idx]
        y_tr, y_va = target.iloc[tr_idx], target.iloc[va_idx]

        dtrain = lgb.Dataset(x_tr, y_tr, categorical_feature="auto")
        dvalid = lgb.Dataset(x_va, y_va, categorical_feature="auto", reference=dtrain)

        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict(x_va, num_iteration=model.best_iteration)
        test_pred += model.predict(test, num_iteration=model.best_iteration) / N_SPLITS
        print(f"fold {fold}: best_iter={model.best_iteration}  "
              f"val_acc={accuracy_score(y_va, oof[va_idx] >= 0.5):.4f}")

    oof_acc = accuracy_score(target, oof >= 0.5)
    print(f"\nCV accuracy: {oof_acc:.4f}")

    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Transported": (test_pred >= 0.5).astype(bool),
    })
    sub_path = OUT / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"wrote {sub_path}  ({len(submission)} rows)")

    pd.DataFrame({"PassengerId": train_raw["PassengerId"], "oof": oof}) \
        .to_csv(OUT / "oof.csv", index=False)


if __name__ == "__main__":
    main()
