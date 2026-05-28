"""Train an ensemble (LightGBM × multi-seed + XGBoost + CatBoost) with 5-fold CV.

Target encoding is applied inside each fold to avoid leakage. The final
submission averages all three model probabilities equally.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

from features import CATEGORICAL_COLS, build_features, encode_categoricals
from target_encoding import TARGET_ENCODE_COLS, apply_target_encoding

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
PARAMS = ROOT / "params"
OUT.mkdir(exist_ok=True)
PARAMS.mkdir(exist_ok=True)

N_SPLITS = 5
LGB_SEEDS = [42, 1337, 2024]


def load_lgb_params() -> dict:
    path = PARAMS / "lgb_params.json"
    if not path.exists():
        return {
            "objective": "binary", "metric": "binary_error", "verbose": -1,
            "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 30,
            "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
            "lambda_l1": 0.1, "lambda_l2": 1.0,
        }
    return json.loads(path.read_text())


def load_xgb_params() -> dict:
    path = PARAMS / "xgb_params.json"
    if not path.exists():
        return {
            "learning_rate": 0.03, "max_depth": 6, "min_child_weight": 5,
            "subsample": 0.85, "colsample_bytree": 0.85,
            "reg_lambda": 1.0, "reg_alpha": 0.1, "gamma": 0.0,
        }
    return json.loads(path.read_text())


def load_cat_params() -> dict:
    path = PARAMS / "cat_params.json"
    if not path.exists():
        return {
            "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3.0,
            "bagging_temperature": 0.5, "random_strength": 1.0, "border_count": 128,
        }
    return json.loads(path.read_text())


def to_categorical_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes.astype("int32")
    return df


def train_lgb_fold(x_tr, y_tr, x_va, y_va, x_test, params: dict, seed: int):
    p = {**params, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}
    dtrain = lgb.Dataset(x_tr, y_tr, categorical_feature="auto")
    dvalid = lgb.Dataset(x_va, y_va, categorical_feature="auto", reference=dtrain)
    model = lgb.train(
        p, dtrain, num_boost_round=4000, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)],
    )
    return (
        model.predict(x_va, num_iteration=model.best_iteration),
        model.predict(x_test, num_iteration=model.best_iteration),
    )


def train_xgb_fold(x_tr, y_tr, x_va, y_va, x_test, params: dict, seed: int = 42):
    x_tr_n = to_categorical_codes(x_tr)
    x_va_n = to_categorical_codes(x_va)
    x_test_n = to_categorical_codes(x_test)
    model = xgb.XGBClassifier(
        n_estimators=4000, **params,
        tree_method="hist", early_stopping_rounds=120, eval_metric="logloss",
        random_state=seed, verbosity=0,
    )
    model.fit(x_tr_n, y_tr, eval_set=[(x_va_n, y_va)], verbose=False)
    return model.predict_proba(x_va_n)[:, 1], model.predict_proba(x_test_n)[:, 1]


def train_cat_fold(x_tr, y_tr, x_va, y_va, x_test, params: dict, seed: int = 42):
    cat_idx = [i for i, c in enumerate(x_tr.columns) if c in CATEGORICAL_COLS]
    x_tr_s = x_tr.copy(); x_va_s = x_va.copy(); x_test_s = x_test.copy()
    for col in CATEGORICAL_COLS:
        if col in x_tr_s.columns:
            x_tr_s[col] = x_tr_s[col].astype(str).fillna("nan")
            x_va_s[col] = x_va_s[col].astype(str).fillna("nan")
            x_test_s[col] = x_test_s[col].astype(str).fillna("nan")
    model = CatBoostClassifier(
        **params,
        iterations=4000,
        early_stopping_rounds=120, eval_metric="Accuracy",
        random_seed=seed, verbose=0, cat_features=cat_idx,
    )
    model.fit(x_tr_s, y_tr, eval_set=(x_va_s, y_va), use_best_model=True)
    return model.predict_proba(x_va_s)[:, 1], model.predict_proba(x_test_s)[:, 1]


def main() -> None:
    train_raw = pd.read_csv(DATA / "train.csv")
    test_raw = pd.read_csv(DATA / "test.csv")
    test_ids = test_raw["PassengerId"].copy()
    target = train_raw["Transported"].astype(int)

    train_feats = build_features(train_raw.drop(columns=["Transported"]))
    test_feats = build_features(test_raw)
    train_feats, test_feats = encode_categoricals(train_feats, test_feats)

    print(f"features: {train_feats.shape[1]} cols + {len(TARGET_ENCODE_COLS)} TE features per fold")

    lgb_params = load_lgb_params()
    xgb_params = load_xgb_params()
    cat_params = load_cat_params()
    print(f"LGB params: {json.dumps({k: round(v,4) if isinstance(v,float) else v for k,v in lgb_params.items() if k not in ('objective','metric','verbose')}, indent=None)}")
    print(f"XGB params: {json.dumps({k: round(v,4) if isinstance(v,float) else v for k,v in xgb_params.items()}, indent=None)}")
    print(f"CAT params: {json.dumps({k: round(v,4) if isinstance(v,float) else v for k,v in cat_params.items()}, indent=None)}")

    oof_lgb = np.zeros(len(train_feats))
    oof_xgb = np.zeros(len(train_feats))
    oof_cat = np.zeros(len(train_feats))
    test_lgb = np.zeros(len(test_feats))
    test_xgb = np.zeros(len(test_feats))
    test_cat = np.zeros(len(test_feats))

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(folds.split(train_feats, target), 1):
        x_tr = train_feats.iloc[tr_idx]; x_va = train_feats.iloc[va_idx]
        y_tr = target.iloc[tr_idx]; y_va = target.iloc[va_idx]

        x_tr_te, x_va_te, x_test_te = apply_target_encoding(x_tr, y_tr, x_va, test_feats)

        # LightGBM × multiple seeds (averaged)
        lgb_va = np.zeros(len(x_va)); lgb_te = np.zeros(len(test_feats))
        for sd in LGB_SEEDS:
            va_p, te_p = train_lgb_fold(x_tr_te, y_tr, x_va_te, y_va, x_test_te, lgb_params, sd)
            lgb_va += va_p / len(LGB_SEEDS)
            lgb_te += te_p / len(LGB_SEEDS)
        oof_lgb[va_idx] = lgb_va
        test_lgb += lgb_te / N_SPLITS

        # XGBoost
        xgb_va, xgb_te = train_xgb_fold(x_tr_te, y_tr, x_va_te, y_va, x_test_te, xgb_params)
        oof_xgb[va_idx] = xgb_va
        test_xgb += xgb_te / N_SPLITS

        # CatBoost
        cat_va, cat_te = train_cat_fold(x_tr_te, y_tr, x_va_te, y_va, x_test_te, cat_params)
        oof_cat[va_idx] = cat_va
        test_cat += cat_te / N_SPLITS

        print(
            f"fold {fold}: "
            f"LGB={accuracy_score(y_va, lgb_va>=0.5):.4f}  "
            f"XGB={accuracy_score(y_va, xgb_va>=0.5):.4f}  "
            f"CAT={accuracy_score(y_va, cat_va>=0.5):.4f}  "
            f"BLEND={accuracy_score(y_va, ((lgb_va+xgb_va+cat_va)/3)>=0.5):.4f}"
        )

    print("\n=== overall CV accuracy ===")
    print(f"LGB   : {accuracy_score(target, oof_lgb>=0.5):.4f}  (logloss {log_loss(target, oof_lgb):.4f})")
    print(f"XGB   : {accuracy_score(target, oof_xgb>=0.5):.4f}  (logloss {log_loss(target, oof_xgb):.4f})")
    print(f"CAT   : {accuracy_score(target, oof_cat>=0.5):.4f}  (logloss {log_loss(target, oof_cat):.4f})")

    blend_eq = (oof_lgb + oof_xgb + oof_cat) / 3
    print(f"BLEND eq    : {accuracy_score(target, blend_eq>=0.5):.4f}  (logloss {log_loss(target, blend_eq):.4f})")

    # ---- search blend weights (simplex grid, step 0.05) -----------------
    best_w = (1/3, 1/3, 1/3); best_acc = accuracy_score(target, blend_eq >= 0.5)
    for a in np.arange(0, 1.001, 0.05):
        for b in np.arange(0, 1.001 - a, 0.05):
            c = 1.0 - a - b
            if c < 0:
                continue
            mix = a * oof_lgb + b * oof_xgb + c * oof_cat
            acc = accuracy_score(target, mix >= 0.5)
            if acc > best_acc:
                best_acc = acc; best_w = (a, b, c)
    print(f"BLEND weights: LGB={best_w[0]:.2f} XGB={best_w[1]:.2f} CAT={best_w[2]:.2f}  → CV {best_acc:.4f}")

    # ---- search threshold on the weighted blend -------------------------
    blend_w = best_w[0]*oof_lgb + best_w[1]*oof_xgb + best_w[2]*oof_cat
    best_thr = 0.5
    for thr in np.arange(0.30, 0.71, 0.01):
        acc = accuracy_score(target, blend_w >= thr)
        if acc > best_acc:
            best_acc = acc; best_thr = thr
    print(f"BLEND best  : threshold={best_thr:.2f}  → CV {best_acc:.4f}")

    # ---- rank averaging as an alternative blend -------------------------
    def to_rank(arr): return pd.Series(arr).rank(pct=True).to_numpy()
    rank_oof = (to_rank(oof_lgb) + to_rank(oof_xgb) + to_rank(oof_cat)) / 3
    rank_acc = max(
        accuracy_score(target, rank_oof >= thr) for thr in np.arange(0.30, 0.71, 0.01)
    )
    rank_thr = max(
        np.arange(0.30, 0.71, 0.01),
        key=lambda t: accuracy_score(target, rank_oof >= t),
    )
    print(f"BLEND rank  : threshold={rank_thr:.2f}  → CV {rank_acc:.4f}")

    if rank_acc > best_acc:
        print("→ using rank-averaging blend")
        final = (to_rank(test_lgb) + to_rank(test_xgb) + to_rank(test_cat)) / 3
        thr = rank_thr
    else:
        print(f"→ using weighted blend (LGB={best_w[0]:.2f} XGB={best_w[1]:.2f} CAT={best_w[2]:.2f})")
        final = best_w[0]*test_lgb + best_w[1]*test_xgb + best_w[2]*test_cat
        thr = best_thr

    submission = pd.DataFrame({"PassengerId": test_ids, "Transported": (final >= thr).astype(bool)})
    submission.to_csv(OUT / "submission.csv", index=False)
    print(f"\nwrote {OUT / 'submission.csv'}  ({len(submission)} rows)")

    pd.DataFrame({
        "PassengerId": train_raw["PassengerId"],
        "oof_lgb": oof_lgb, "oof_xgb": oof_xgb, "oof_cat": oof_cat,
        "oof_blend_weighted": blend_w, "oof_blend_rank": rank_oof,
    }).to_csv(OUT / "oof.csv", index=False)


if __name__ == "__main__":
    main()
