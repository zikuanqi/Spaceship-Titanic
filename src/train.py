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
LGB_SEEDS = [42, 1337, 2024, 7, 99, 314]

# Pseudo-labeling: pick test rows where the first-pass blend is very confident.
PSEUDO_HIGH = 0.92
PSEUDO_LOW = 0.08


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


def run_ensemble(
    train_feats: pd.DataFrame,
    target: pd.Series,
    test_feats: pd.DataFrame,
    lgb_params: dict,
    xgb_params: dict,
    cat_params: dict,
    extra_X: pd.DataFrame | None = None,
    extra_y: pd.Series | None = None,
    label: str = "",
) -> tuple[dict, dict, list]:
    """Run 5-fold ensemble over the original train rows.

    If `extra_X` / `extra_y` is given, they're appended to every fold's training
    set (used for pseudo-labeling). The held-out folds and test predictions
    remain on the original splits.
    """
    n_train = len(train_feats); n_test = len(test_feats)
    oof_lgb = np.zeros(n_train); test_lgb = np.zeros(n_test)
    oof_xgb = np.zeros(n_train); test_xgb = np.zeros(n_test)
    oof_cat = np.zeros(n_train); test_cat = np.zeros(n_test)
    oof_hgb = np.zeros(n_train); test_hgb = np.zeros(n_test)

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(folds.split(train_feats, target), 1):
        x_tr = train_feats.iloc[tr_idx]; x_va = train_feats.iloc[va_idx]
        y_tr = target.iloc[tr_idx]; y_va = target.iloc[va_idx]

        if extra_X is not None and len(extra_X) > 0:
            x_tr = pd.concat([x_tr, extra_X], axis=0, ignore_index=True)
            y_tr = pd.concat([y_tr, extra_y], axis=0, ignore_index=True)

        x_tr_te, x_va_te, x_test_te = apply_target_encoding(x_tr, y_tr, x_va, test_feats)

        # LightGBM × multiple seeds (averaged)
        lgb_va = np.zeros(len(x_va)); lgb_te = np.zeros(n_test)
        for sd in LGB_SEEDS:
            va_p, te_p = train_lgb_fold(x_tr_te, y_tr, x_va_te, y_va, x_test_te, lgb_params, sd)
            lgb_va += va_p / len(LGB_SEEDS)
            lgb_te += te_p / len(LGB_SEEDS)
        oof_lgb[va_idx] = lgb_va; test_lgb += lgb_te / N_SPLITS

        xgb_va, xgb_te = train_xgb_fold(x_tr_te, y_tr, x_va_te, y_va, x_test_te, xgb_params)
        oof_xgb[va_idx] = xgb_va; test_xgb += xgb_te / N_SPLITS

        cat_va, cat_te = train_cat_fold(x_tr_te, y_tr, x_va_te, y_va, x_test_te, cat_params)
        oof_cat[va_idx] = cat_va; test_cat += cat_te / N_SPLITS

        hgb_va, hgb_te = train_hgb_fold(x_tr_te, y_tr, x_va_te, y_va, x_test_te)
        oof_hgb[va_idx] = hgb_va; test_hgb += hgb_te / N_SPLITS

        print(
            f"[{label}] fold {fold}: "
            f"LGB={accuracy_score(y_va, lgb_va>=0.5):.4f}  "
            f"XGB={accuracy_score(y_va, xgb_va>=0.5):.4f}  "
            f"CAT={accuracy_score(y_va, cat_va>=0.5):.4f}  "
            f"HGB={accuracy_score(y_va, hgb_va>=0.5):.4f}"
        )

    oofs = {"lgb": oof_lgb, "xgb": oof_xgb, "cat": oof_cat, "hgb": oof_hgb}
    tests = {"lgb": test_lgb, "xgb": test_xgb, "cat": test_cat, "hgb": test_hgb}
    return oofs, tests, list(oofs.keys())


def fit_logloss_weights(oof_M: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize
    def neg_ll(w):
        w = np.clip(w, 1e-9, None); w = w / w.sum()
        p = np.clip(oof_M @ w, 1e-7, 1 - 1e-7)
        return -(y_arr * np.log(p) + (1 - y_arr) * np.log(1 - p)).mean()
    res = minimize(neg_ll, np.ones(oof_M.shape[1]) / oof_M.shape[1],
                   method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-7, "maxiter": 8000})
    w = np.clip(res.x, 0, None); return w / w.sum()


def honest_blend_cv(oof_M: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    """Leave-one-fold-out blend weight fit -> honest CV blend probabilities."""
    out = np.zeros(len(y_arr))
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    for tr_idx, va_idx in folds.split(np.zeros(len(y_arr)), y_arr):
        w = fit_logloss_weights(oof_M[tr_idx], y_arr[tr_idx])
        out[va_idx] = oof_M[va_idx] @ w
    return out


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

    oofs, tests, names = run_ensemble(
        train_feats, target, test_feats,
        lgb_params, xgb_params, cat_params,
        label="pass 1",
    )

    print("\n=== overall CV accuracy ===")
    for n in names:
        print(f"{n.upper():5}: {accuracy_score(target, oofs[n]>=0.5):.4f}  (logloss {log_loss(target, oofs[n]):.4f})")

    y = target.to_numpy()

    def report(oofs_d, tests_d, label):
        """Compute blend, stacking, honest CV; return picked (test_pred, oof_pred, honest_acc, method)."""
        M = np.stack([oofs_d[n] for n in names], axis=1)
        T = np.stack([tests_d[n] for n in names], axis=1)

        print(f"\n=== {label}: per-model OOF ===")
        for n in names:
            print(f"  {n.upper():5}: acc {accuracy_score(y, oofs_d[n]>=0.5):.4f}  "
                  f"(logloss {log_loss(y, oofs_d[n]):.4f})")

        w_opt = fit_logloss_weights(M, y)
        blend_w = M @ w_opt
        test_blend = T @ w_opt
        print(f"  BLEND weights : {dict(zip(names, np.round(w_opt, 3)))}")
        print(f"  BLEND ll-opt  : acc {accuracy_score(y, blend_w>=0.5):.4f}  "
              f"(logloss {log_loss(y, blend_w):.4f})")

        def to_logit(p):
            p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
        Z = to_logit(M); Z_test = to_logit(T)
        meta = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
        meta.fit(Z, y)
        stack_oof = meta.predict_proba(Z)[:, 1]
        stack_test = meta.predict_proba(Z_test)[:, 1]

        # honest CV: leave-one-fold-out re-fit of weights AND stacking
        honest_blend = honest_blend_cv(M, y)
        honest_stack = np.zeros(len(y))
        for tr_idx, va_idx in StratifiedKFold(N_SPLITS, shuffle=True, random_state=42).split(np.zeros(len(y)), y):
            mf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
            mf.fit(Z[tr_idx], y[tr_idx])
            honest_stack[va_idx] = mf.predict_proba(Z[va_idx])[:, 1]

        bh = accuracy_score(y, honest_blend >= 0.5)
        sh = accuracy_score(y, honest_stack >= 0.5)
        print(f"  BLEND honest  : acc {bh:.4f}  (logloss {log_loss(y, honest_blend):.4f})")
        print(f"  STACK honest  : acc {sh:.4f}  (logloss {log_loss(y, honest_stack):.4f})")

        if sh > bh:
            return test_blend if False else stack_test, stack_oof, sh, honest_stack, "stack"
        return test_blend, blend_w, bh, honest_blend, "blend"

    test1, oof1, honest1, honest1_arr, method1 = report(oofs, tests, "PASS 1 (no pseudo)")

    # ============== Pseudo-labeling round ==============
    mask_pseudo = (test1 >= PSEUDO_HIGH) | (test1 <= PSEUDO_LOW)
    n_pseudo = int(mask_pseudo.sum())
    print(f"\n=== Pseudo-labeling ===")
    print(f"  threshold [<{PSEUDO_LOW}, >{PSEUDO_HIGH}] → {n_pseudo}/{len(test_feats)} "
          f"test rows selected ({100*n_pseudo/len(test_feats):.1f}%)")

    pseudo_X = test_feats.iloc[mask_pseudo].reset_index(drop=True)
    pseudo_y = pd.Series((test1[mask_pseudo] >= 0.5).astype(int)).reset_index(drop=True)
    print(f"  pseudo label balance: {pseudo_y.mean():.3f} positive")

    oofs2, tests2, _ = run_ensemble(
        train_feats, target, test_feats,
        lgb_params, xgb_params, cat_params,
        extra_X=pseudo_X, extra_y=pseudo_y, label="pass 2",
    )
    test2, oof2, honest2, honest2_arr, method2 = report(oofs2, tests2, "PASS 2 (with pseudo)")

    # ============== Pick the better pass ==============
    if honest2 > honest1:
        print(f"\n→ Pseudo helped ({honest1:.4f} → {honest2:.4f}); using PASS 2.")
        final_test, final_oof, used_method, used_pass = test2, oof2, method2, "pass2"
    else:
        print(f"\n→ Pseudo did not help ({honest1:.4f} vs {honest2:.4f}); using PASS 1.")
        final_test, final_oof, used_method, used_pass = test1, oof1, method1, "pass1"

    thr = 0.5
    submission = pd.DataFrame({"PassengerId": test_ids, "Transported": (final_test >= thr).astype(bool)})
    submission.to_csv(OUT / "submission.csv", index=False)
    print(f"\nwrote {OUT / 'submission.csv'}  "
          f"({len(submission)} rows, {used_pass}/{used_method}, threshold={thr})")

    oof_df = pd.DataFrame({
        "PassengerId": train_raw["PassengerId"],
        **{f"oof1_{n}": oofs[n] for n in names},
        **{f"oof2_{n}": oofs2[n] for n in names},
        "honest1": honest1_arr, "honest2": honest2_arr,
        "oof_final": final_oof,
    })
    oof_df.to_csv(OUT / "oof.csv", index=False)


if __name__ == "__main__":
    main()
