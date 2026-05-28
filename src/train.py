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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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


def train_hgb_fold(x_tr, y_tr, x_va, y_va, x_test, seed: int = 42):
    cat_mask = [c in CATEGORICAL_COLS for c in x_tr.columns]
    x_tr_n = to_categorical_codes(x_tr)
    x_va_n = to_categorical_codes(x_va)
    x_test_n = to_categorical_codes(x_test)
    model = HistGradientBoostingClassifier(
        max_iter=2000, learning_rate=0.04, max_depth=7,
        min_samples_leaf=25, l2_regularization=1.0,
        early_stopping=True, validation_fraction=None,
        n_iter_no_change=80, scoring="loss",
        categorical_features=cat_mask, random_state=seed,
    )
    model.fit(x_tr_n, y_tr)
    return model.predict_proba(x_va_n)[:, 1], model.predict_proba(x_test_n)[:, 1]


def train_lr_fold(x_tr, y_tr, x_va, y_va, x_test, seed: int = 42):
    """Logistic regression on dense, scaled features (codes for cats, median impute for nums)."""
    x_tr_n = to_categorical_codes(x_tr).astype("float64")
    x_va_n = to_categorical_codes(x_va).astype("float64")
    x_test_n = to_categorical_codes(x_test).astype("float64")
    medians = x_tr_n.median(numeric_only=True)
    x_tr_n = x_tr_n.fillna(medians)
    x_va_n = x_va_n.fillna(medians)
    x_test_n = x_test_n.fillna(medians)

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr_n)
    x_va_s = scaler.transform(x_va_n)
    x_test_s = scaler.transform(x_test_n)
    model = LogisticRegression(C=0.5, max_iter=3000, solver="lbfgs", random_state=seed)
    model.fit(x_tr_s, y_tr)
    return model.predict_proba(x_va_s)[:, 1], model.predict_proba(x_test_s)[:, 1]


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
    oof_hgb = np.zeros(len(train_feats))
    oof_lr  = np.zeros(len(train_feats))
    test_lgb = np.zeros(len(test_feats))
    test_xgb = np.zeros(len(test_feats))
    test_cat = np.zeros(len(test_feats))
    test_hgb = np.zeros(len(test_feats))
    test_lr  = np.zeros(len(test_feats))

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

        # HistGradientBoosting
        hgb_va, hgb_te = train_hgb_fold(x_tr_te, y_tr, x_va_te, y_va, x_test_te)
        oof_hgb[va_idx] = hgb_va
        test_hgb += hgb_te / N_SPLITS

        # Logistic Regression
        lr_va, lr_te = train_lr_fold(x_tr_te, y_tr, x_va_te, y_va, x_test_te)
        oof_lr[va_idx] = lr_va
        test_lr += lr_te / N_SPLITS

        print(
            f"fold {fold}: "
            f"LGB={accuracy_score(y_va, lgb_va>=0.5):.4f}  "
            f"XGB={accuracy_score(y_va, xgb_va>=0.5):.4f}  "
            f"CAT={accuracy_score(y_va, cat_va>=0.5):.4f}  "
            f"HGB={accuracy_score(y_va, hgb_va>=0.5):.4f}  "
            f"LR={accuracy_score(y_va, lr_va>=0.5):.4f}"
        )

    oofs = {"lgb": oof_lgb, "xgb": oof_xgb, "cat": oof_cat, "hgb": oof_hgb, "lr": oof_lr}
    tests = {"lgb": test_lgb, "xgb": test_xgb, "cat": test_cat, "hgb": test_hgb, "lr": test_lr}
    names = list(oofs.keys())

    print("\n=== overall CV accuracy ===")
    for n in names:
        print(f"{n.upper():5}: {accuracy_score(target, oofs[n]>=0.5):.4f}  (logloss {log_loss(target, oofs[n]):.4f})")

    oof_matrix = np.stack([oofs[n] for n in names], axis=1)
    test_matrix = np.stack([tests[n] for n in names], axis=1)
    y = target.to_numpy()

    blend_eq = oof_matrix.mean(axis=1)
    print(f"BLEND eq      : acc {accuracy_score(target, blend_eq>=0.5):.4f}  (logloss {log_loss(target, blend_eq):.4f})")

    # ---- logloss-optimal weights on the simplex (smooth, much less prone --
    # ---- to OOF overfit than tuning weights+threshold against accuracy). -
    from scipy.optimize import minimize

    def fit_logloss_weights(oof_M: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
        def neg_ll(w):
            w = np.clip(w, 1e-9, None); w = w / w.sum()
            p = np.clip(oof_M @ w, 1e-7, 1 - 1e-7)
            return -(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p)).mean()
        res = minimize(neg_ll, np.ones(oof_M.shape[1]) / oof_M.shape[1],
                       method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-7, "maxiter": 8000})
        w = np.clip(res.x, 0, None); return w / w.sum()

    w_opt = fit_logloss_weights(oof_matrix, y)
    blend_w = oof_matrix @ w_opt
    test_blend = test_matrix @ w_opt
    print(f"BLEND weights : {dict(zip(names, np.round(w_opt, 3)))}")
    print(f"BLEND ll-opt  : acc {accuracy_score(y, blend_w>=0.5):.4f}  (logloss {log_loss(y, blend_w):.4f})")

    # ---- "honest" blend CV: leave-one-fold-out for the weight fit --------
    # For each outer fold, fit weights on the other 4 folds' OOF only, then
    # score on the held-out fold. This estimates how the blending procedure
    # would do on truly unseen rows -- the CV/LB gap should follow this score.
    honest_preds = np.zeros(len(y))
    honest_folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for tr_idx, va_idx in honest_folds.split(np.zeros(len(y)), y):
        w_fold = fit_logloss_weights(oof_matrix[tr_idx], y[tr_idx])
        honest_preds[va_idx] = oof_matrix[va_idx] @ w_fold
    print(f"BLEND honest  : acc {accuracy_score(y, honest_preds>=0.5):.4f}  (logloss {log_loss(y, honest_preds):.4f})")
    print("  ^ this is a leak-free estimate of LB.")

    # ---- final submission: logloss weights, threshold 0.5 (no OOF tuning) -
    final = test_blend
    thr = 0.5
    submission = pd.DataFrame({"PassengerId": test_ids, "Transported": (final >= thr).astype(bool)})
    submission.to_csv(OUT / "submission.csv", index=False)
    print(f"\nwrote {OUT / 'submission.csv'}  ({len(submission)} rows, threshold {thr})")

    oof_df = pd.DataFrame({
        "PassengerId": train_raw["PassengerId"],
        **{f"oof_{n}": oofs[n] for n in names},
        "oof_blend": blend_w, "oof_honest": honest_preds,
    })
    oof_df.to_csv(OUT / "oof.csv", index=False)


if __name__ == "__main__":
    main()
