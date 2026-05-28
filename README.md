# Spaceship Titanic

Solution for the Kaggle [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition — predict which passengers were transported to an alternate dimension.

## Result

| Stage | CV accuracy |
|---|---|
| LightGBM baseline | 0.8125 |
| + new features, Optuna-tuned LGB | 0.8171 |
| + XGBoost + CatBoost weighted blend | **0.8185** |

Top public-leaderboard scores typically sit at 0.81–0.82.

## Approach

### Feature engineering (`src/features.py`, 45 columns)
- `PassengerId` → `Group`, `GroupPos`, `GroupSize`, `IsAlone`, `LargeGroup`
- `Cabin` → `Deck`, `CabinNum`, `Side`, `CabinRegion` (bucketed)
- `Name` → `FamilySize` (by last name)
- Spending: `TotalSpend`, `LogTotalSpend`, `NoSpend`, per-column `log1p`, `LuxurySpend`, `EssentialSpend`, `LuxuryRatio`, `NumSpendCats`, `TopSpendCat`
- Group-level: `GroupTotalSpend`, `GroupMeanSpend`, `GroupCryoRatio`
- Cross-imputation: `CryoSleep=True` ⇒ spending = 0; any spending recorded ⇒ `CryoSleep=False`; `CryoSleep` missing with zero spending ⇒ `True`
- Interactions: `DeckPlanet` (Deck × HomePlanet)
- Age: `IsChild`, `IsTeen`, `AgeBin`, `AgeMissing`

### Leak-free target encoding (`src/target_encoding.py`)
Smoothed target encoding for `Deck`, `HomePlanet`, `Destination`, `DeckPlanet`, `CabinRegion`, `TopSpendCat` — computed **inside each CV fold** using only the training portion (smoothing = 20).

### Hyperparameter tuning (`src/tune.py`)
Optuna TPE sampler, 25 trials, 3-fold stratified internal CV. Saves best params to `output/lgb_params.json`.

### Ensemble (`src/train.py`)
For each of 5 outer folds:
1. **LightGBM** with tuned params, averaged over 3 seeds (42, 1337, 2024)
2. **XGBoost** (`hist`, depth 6, lr 0.03)
3. **CatBoost** (depth 6, lr 0.03, native categorical support)

Then on OOF predictions:
- **Weight search**: simplex grid over (LGB, XGB, CAT) ∈ ½ steps of 0.05 → winners `0.50 / 0.30 / 0.20`
- **Threshold search**: 0.30 → 0.70 step 0.01 → optimal 0.50
- **Rank-average comparison**: kept whichever scores higher OOF

## Reproduce

```bash
pip install -r requirements.txt

# 1. Download data
python -c "import kagglehub; print(kagglehub.competition_download('spaceship-titanic'))"
# Copy train.csv / test.csv / sample_submission.csv into data/

# 2. Tune LightGBM (saves output/lgb_params.json)
python src/tune.py

# 3. Train ensemble (writes output/submission.csv)
python src/train.py
```

## Submit to Kaggle

```bash
kaggle competitions submit -c spaceship-titanic \
    -f output/submission.csv \
    -m "LGB+XGB+CAT weighted blend, CV 0.8185"
```

## Project layout

```
.
├── data/                  # gitignored
├── output/                # submission.csv, oof.csv, lgb_params.json (gitignored)
├── src/
│   ├── features.py        # feature engineering
│   ├── target_encoding.py # leak-free smoothed target encoding
│   ├── tune.py            # Optuna search for LGB
│   └── train.py           # ensemble training + blend optimization
├── requirements.txt
└── README.md
```
