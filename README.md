# Spaceship Titanic

Solution for the Kaggle [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition — predict which passengers were transported to an alternate dimension.

## Result

| Metric | Score |
|---|---|
| 5-fold CV accuracy | **0.8125** |

LightGBM with feature engineering on the standard tabular features. Public-leaderboard top scores typically sit in the 0.81–0.82 range, so this is a solid baseline without ensembling or hyperparameter search.

## Approach

1. **Feature engineering** (`src/features.py`)
   - `PassengerId` → `Group`, `GroupPos`, `GroupSize`, `IsAlone`
   - `Cabin` → `Deck`, `CabinNum`, `Side`
   - `Name` → `FamilySize` (by last name)
   - 5 spending columns → `TotalSpend`, `NoSpend`, plus per-column `log1p` transforms
   - Cross-imputation: if `CryoSleep=True`, spending must be 0; if any spending recorded, `CryoSleep=False`
   - `IsChild` flag (age < 13)
2. **Model**: LightGBM, 5-fold stratified CV, early stopping on validation `binary_error`
3. **Prediction**: average of 5 fold predictions, threshold 0.5

## Reproduce

```bash
pip install -r requirements.txt
# Download competition data (uses your Kaggle credentials)
python -c "import kagglehub; print(kagglehub.competition_download('spaceship-titanic'))"
# Copy train.csv / test.csv / sample_submission.csv into data/
python src/train.py
```

Output:
- `output/submission.csv` — Kaggle-format predictions
- `output/oof.csv` — out-of-fold probabilities for analysis

## Submit to Kaggle

```bash
kaggle competitions submit -c spaceship-titanic \
    -f output/submission.csv \
    -m "LGB 5-fold, CV 0.8125"
```

## Project layout

```
.
├── data/                  # train.csv, test.csv, sample_submission.csv (gitignored)
├── output/                # submission.csv, oof.csv (gitignored)
├── src/
│   ├── features.py        # feature engineering
│   └── train.py           # training + CV + submission
├── requirements.txt
└── README.md
```
