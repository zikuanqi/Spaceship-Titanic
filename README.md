<h1 align="center">Spaceship Titanic</h1>

<p align="center">
  <em>A leak-free, multi-model ensemble for the Kaggle Spaceship Titanic competition.</em><br/>
  <em>Kaggle Spaceship Titanic 比赛的多模型集成方案,严格防止数据泄漏。</em>
</p>

<p align="center">
  <a href="https://www.kaggle.com/competitions/spaceship-titanic">
    <img src="https://img.shields.io/badge/Kaggle-Spaceship%20Titanic-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/CV%20Accuracy-0.8170-brightgreen" alt="CV"/>
  <img src="https://img.shields.io/badge/Models-5-blueviolet" alt="Models"/>
  <img src="https://img.shields.io/badge/LightGBM-4.6-success" alt="LightGBM"/>
  <img src="https://img.shields.io/badge/XGBoost-3.2-orange" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/CatBoost-1.2-yellow" alt="CatBoost"/>
  <img src="https://img.shields.io/badge/Optuna-4.8-7B61FF?logo=optuna&logoColor=white" alt="Optuna"/>
</p>

---

## Overview · 项目概述

**EN.** Predict which passengers of the Spaceship Titanic were transported to an alternate dimension. The pipeline combines heavy feature engineering, leak-free target encoding, Optuna-tuned gradient boosting models, and a log-loss optimized blend.

**中文.** 预测 Spaceship Titanic 上的哪些乘客被传送到了平行维度。流水线包含丰富的特征工程、无泄漏的目标编码、Optuna 调过参的三种梯度提升模型,以及基于 logloss 的最优加权融合。

---

## Results · 评测结果

| Stage · 阶段 | CV Accuracy |
|---|---|
| LightGBM baseline · 基线 | 0.8125 |
| + Optuna-tuned LGB & expanded features · 调参 + 新特征 | 0.8171 |
| + XGBoost + CatBoost weighted blend · 加权融合 | 0.8185 |
| + HistGradientBoosting + LogisticRegression · 模型多样化 | 0.8201 *(OOF-overfit)* |
| **Logloss-weighted blend, fixed threshold 0.5** · 当前版本 | **0.8170** *(honest)* |

> *Honest CV* uses leave-one-fold-out fitting of blend weights, so it estimates leaderboard performance without OOF re-fitting bias.
> "诚实 CV" 用留一折训练融合权重,避免在 OOF 上反复挑选导致的过拟合,更接近真实 LB 分数。

Top public-leaderboard scores typically sit at **0.81–0.83**.
公榜 top 选手通常在 **0.81–0.83** 区间。

---

## Approach · 技术方案

### 1. Feature Engineering · 特征工程
`src/features.py` — 45 columns / 45 列特征

- **PassengerId** → `Group`, `GroupPos`, `GroupSize`, `IsAlone`, `LargeGroup`
- **Cabin** → `Deck`, `CabinNum`, `Side`, `CabinRegion` (bucketed by 300 / 按 300 分桶)
- **Name** → `FamilySize` (by last name / 按姓氏分组)
- **Spending** (5 columns / 5 列消费):
  - aggregates: `TotalSpend`, `LogTotalSpend`, `NoSpend`, `NumSpendCats`, `TopSpendCat`
  - per-column: `Log{Col}` for each of the 5 services
  - groupings: `LuxurySpend` vs `EssentialSpend`, `LuxuryRatio`
- **Group-level / 组级统计**: `GroupTotalSpend`, `GroupMeanSpend`, `GroupCryoRatio`
- **Cross-imputation / 交叉填补**:
  - `CryoSleep=True` ⇒ all spending = 0
  - any spending recorded ⇒ `CryoSleep=False`
  - missing `CryoSleep` with zero spending ⇒ `True`
- **Interactions / 交叉特征**: `DeckPlanet` (Deck × HomePlanet)
- **Age / 年龄**: `IsChild` (<13), `IsTeen` (13–18), `AgeBin`, `AgeMissing`

### 2. Leak-free Target Encoding · 无泄漏目标编码
`src/target_encoding.py`

Smoothed target encoding for `Deck`, `HomePlanet`, `Destination`, `DeckPlanet`, `CabinRegion`, `TopSpendCat`. Encoding is recomputed **inside each CV fold** using only the training portion (smoothing = 20).
对 6 个类别特征做平滑目标编码,**在每折内只用训练折的目标值**计算,防止泄漏(平滑因子 = 20)。

### 3. Hyperparameter Tuning · 超参数搜索
Optuna TPE sampler · TPE 采样器, 25 trials, 3-fold internal CV.

| File · 文件 | Model · 模型 |
|---|---|
| `src/tune.py` | LightGBM |
| `src/tune_xgb.py` | XGBoost |
| `src/tune_cat.py` | CatBoost |

Best parameters are saved to `params/*.json` and loaded automatically by `train.py`.
调优结果保存到 `params/*.json`,`train.py` 自动加载。

### 4. Ensemble · 五模型集成
`src/train.py` — 5 base models, 5-fold stratified outer CV / 5 个基模型,5 折分层外层 CV:

1. **LightGBM** with tuned params, averaged over 3 seeds (42, 1337, 2024)
2. **XGBoost** with tuned params (`hist` tree method)
3. **CatBoost** with tuned params (native categorical support)
4. **HistGradientBoosting** (sklearn, native categorical via mask)
5. **LogisticRegression** (StandardScaler + median imputation)

### 5. Blending · 融合策略

- **Weights / 权重**: Nelder-Mead optimization on **log-loss** (smooth objective, robust against OOF overfit) over the simplex.
- **Threshold / 阈值**: fixed at **0.5** — never tuned on OOF.
- **Honest CV / 诚实评估**: leave-one-fold-out — for each outer fold, fit weights on the other 4 folds and score on the held-out one.

Current best weights (log-loss optimal):
当前 logloss 最优权重:

| Model | LGB | XGB | CAT | HGB | LR |
|---|---|---|---|---|---|
| Weight | 0.454 | 0.380 | 0.131 | 0.035 | 0.000 |

---

## Project Layout · 项目结构

```
.
├── data/                         # Kaggle CSVs (gitignored) · Kaggle 数据,已 gitignore
├── output/                       # submission.csv, oof.csv (gitignored) · 提交文件、OOF
├── params/                       # tuned hyperparameter JSONs · 调优好的超参数
│   ├── lgb_params.json
│   ├── xgb_params.json
│   └── cat_params.json
├── src/
│   ├── features.py               # feature engineering · 特征工程
│   ├── target_encoding.py        # smoothed target encoding · 平滑目标编码
│   ├── tune.py                   # Optuna for LightGBM
│   ├── tune_xgb.py               # Optuna for XGBoost
│   ├── tune_cat.py               # Optuna for CatBoost
│   └── train.py                  # 5-model ensemble + blending · 五模型集成与融合
├── requirements.txt
└── README.md
```

---

## Quick Start · 快速开始

### Prerequisites · 准备工作

- Python 3.10+
- A Kaggle account with API token configured (`~/.kaggle/access_token`)
  Kaggle 账号 + 已配置 API token

### Install dependencies · 安装依赖

```bash
pip install -r requirements.txt
```

### Download competition data · 下载比赛数据

```bash
python -c "import kagglehub; print(kagglehub.competition_download('spaceship-titanic'))"
```

Then copy `train.csv`, `test.csv`, `sample_submission.csv` into the `data/` directory.
然后把三个 CSV 复制到 `data/` 目录。

### (Optional) Re-tune hyperparameters · (可选)重新调参

`params/` 已经包含调好的参数。如要重跑:

```bash
python src/tune.py        # LightGBM
python src/tune_xgb.py    # XGBoost
python src/tune_cat.py    # CatBoost
```

### Train ensemble and generate submission · 训练并生成提交文件

```bash
python src/train.py
```

This writes `output/submission.csv`.
执行后会生成 `output/submission.csv`。

---

## Submit to Kaggle · 提交到 Kaggle

```bash
kaggle competitions submit \
    -c spaceship-titanic \
    -f output/submission.csv \
    -m "Logloss-weighted blend, honest CV 0.8170"
```

Check your submission score · 查看提交分数:
```bash
kaggle competitions submissions -c spaceship-titanic
```

---

## Tech Stack · 技术栈

| Category · 类别 | Libraries · 工具 |
|---|---|
| Gradient Boosting · 梯度提升 | LightGBM, XGBoost, CatBoost, scikit-learn HistGradientBoosting |
| Linear Model · 线性模型 | scikit-learn LogisticRegression |
| Hyperparameter Search · 调参 | Optuna (TPE sampler) |
| Data · 数据 | pandas, NumPy |
| Optimization · 优化 | SciPy (Nelder-Mead) |
| Data Acquisition · 数据获取 | kagglehub, kaggle CLI |

---

## Why The CV–LB Gap? · 为何 CV 与 LB 有差距?

A CV/LB gap of 1–2% is common on this dataset due to:
本数据集 1–2% 的 CV/LB 差距属于常见现象,主要来自:

1. **Distribution shift / 分布偏移** between train and test.
   训练集与测试集的分布差异。
2. **Fold variance / 折间方差** — only 8,693 training rows, so CV folds are small.
   训练集仅 8,693 行,折间方差大。
3. **OOF re-fitting bias / OOF 二次拟合偏差** — any procedure that picks weights/threshold from OOF inflates CV.
   任何在 OOF 上挑选权重或阈值的步骤都会让 CV 虚高。

The "honest CV" metric in this repo controls for #3 explicitly by leaving one fold out when fitting blend weights.
本仓库的"诚实 CV"通过在拟合权重时留出一折,显式控制掉第 3 项。

---

## License · 许可证

Not licensed. Code is shared for educational reference; please don't submit a verbatim copy in any active competition.
未指定开源许可。代码仅供学习参考,请勿在比赛进行中直接复制提交。
