"""Feature engineering for the Spaceship Titanic competition.

`build_features` creates all leak-free features. Target-encoded features are
computed separately inside the CV loop (see `target_encoding.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
LUXURY_COLS = ["RoomService", "Spa", "VRDeck"]   # transported less often
ESSENTIAL_COLS = ["FoodCourt", "ShoppingMall"]   # transported more often
CATEGORICAL_COLS = [
    "HomePlanet", "Destination", "Deck", "Side", "CryoSleep", "VIP",
    "DeckPlanet", "TopSpendCat", "AgeBin", "CabinRegion",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- PassengerId = "GGGG_PP" --------------------------------------------
    pid = df["PassengerId"].str.split("_", expand=True)
    df["Group"] = pid[0].astype(int)
    df["GroupPos"] = pid[1].astype(int)
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count")
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
    df["LargeGroup"] = (df["GroupSize"] >= 4).astype(int)

    # ---- Cabin = "Deck/Num/Side" --------------------------------------------
    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2]
    df["CabinRegion"] = pd.cut(
        df["CabinNum"],
        bins=[-1, 300, 600, 900, 1200, 1500, 1800, 2100],
        labels=["r0", "r1", "r2", "r3", "r4", "r5", "r6"],
    ).astype(object)

    # ---- Name → family grouping ---------------------------------------------
    df["LastName"] = df["Name"].fillna("Unknown_Unknown").str.split(" ").str[-1]
    df["FamilySize"] = df.groupby("LastName")["PassengerId"].transform("count")
    df.loc[df["LastName"] == "Unknown", "FamilySize"] = 1

    # ---- Spending sanity-imputation -----------------------------------------
    # People in CryoSleep cannot spend → 0.
    cryo_mask = df["CryoSleep"] == True
    for col in SPENDING_COLS:
        df.loc[cryo_mask, col] = df.loc[cryo_mask, col].fillna(0)
    spend_known = df[SPENDING_COLS].sum(axis=1, min_count=1)
    df.loc[df["CryoSleep"].isna() & (spend_known > 0), "CryoSleep"] = False
    df.loc[df["CryoSleep"].isna() & (spend_known == 0), "CryoSleep"] = True

    # ---- Spending aggregates ------------------------------------------------
    df["TotalSpend"] = df[SPENDING_COLS].sum(axis=1, min_count=1)
    df["NoSpend"] = (df["TotalSpend"].fillna(-1) == 0).astype(int)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"].fillna(0))
    for col in SPENDING_COLS:
        df[f"Log{col}"] = np.log1p(df[col].fillna(0))

    df["LuxurySpend"] = df[LUXURY_COLS].sum(axis=1, min_count=1).fillna(0)
    df["EssentialSpend"] = df[ESSENTIAL_COLS].sum(axis=1, min_count=1).fillna(0)
    df["LogLuxurySpend"] = np.log1p(df["LuxurySpend"])
    df["LogEssentialSpend"] = np.log1p(df["EssentialSpend"])
    df["LuxuryRatio"] = df["LuxurySpend"] / (df["TotalSpend"].fillna(0) + 1)
    df["NumSpendCats"] = (df[SPENDING_COLS].fillna(0) > 0).sum(axis=1)
    df["TopSpendCat"] = df[SPENDING_COLS].fillna(0).idxmax(axis=1)
    df.loc[df["TotalSpend"].fillna(0) == 0, "TopSpendCat"] = "none"

    # ---- Group-level statistics (computed across train+test together) ------
    group_total = df.groupby("Group")["TotalSpend"].transform("sum").fillna(0)
    df["GroupTotalSpend"] = group_total
    df["GroupLogTotalSpend"] = np.log1p(group_total)
    df["GroupMeanSpend"] = group_total / df["GroupSize"]
    df["GroupLogMeanSpend"] = np.log1p(df["GroupMeanSpend"])
    cryo_int = df["CryoSleep"].astype("boolean").astype("Int64").astype(float)
    df["GroupCryoRatio"] = cryo_int.groupby(df["Group"]).transform("mean")

    # ---- Age ----------------------------------------------------------------
    df["IsChild"] = (df["Age"].fillna(df["Age"].median()) < 13).astype(int)
    df["IsTeen"] = df["Age"].fillna(-1).between(13, 18).astype(int)
    df["AgeBin"] = pd.cut(
        df["Age"], bins=[-1, 12, 18, 30, 50, 200],
        labels=["child", "teen", "young", "mid", "senior"],
    ).astype(object)
    df["AgeMissing"] = df["Age"].isna().astype(int)

    # ---- Interactions -------------------------------------------------------
    df["DeckPlanet"] = df["Deck"].fillna("?").astype(str) + "_" + df["HomePlanet"].fillna("?").astype(str)

    # ---- Drop raw fields not used by the model ------------------------------
    df = df.drop(columns=["Name", "Cabin", "PassengerId", "LastName"])
    return df


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert categorical columns to pandas 'category' with shared categories."""
    for col in CATEGORICAL_COLS:
        if col not in train.columns:
            continue
        combined = pd.concat([train[col], test[col]], axis=0).astype("category")
        cats = combined.cat.categories
        train[col] = pd.Categorical(train[col], categories=cats)
        test[col] = pd.Categorical(test[col], categories=cats)
    return train, test
