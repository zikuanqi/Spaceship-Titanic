"""Feature engineering for the Spaceship Titanic competition."""

from __future__ import annotations

import numpy as np
import pandas as pd

SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL_COLS = ["HomePlanet", "Destination", "Deck", "Side", "CryoSleep", "VIP"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # PassengerId = "GGGG_PP" → group id + position in group
    pid = df["PassengerId"].str.split("_", expand=True)
    df["Group"] = pid[0].astype(int)
    df["GroupPos"] = pid[1].astype(int)
    df["GroupSize"] = df.groupby("Group")["PassengerId"].transform("count")
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

    # Cabin = "Deck/Num/Side"
    cabin = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2]

    # Last name → family grouping
    df["LastName"] = df["Name"].str.split(" ").str[-1]
    df["FamilySize"] = df.groupby("LastName")["PassengerId"].transform("count")

    # Spending features. People in CryoSleep cannot spend → 0 imputation is correct.
    for col in SPENDING_COLS:
        df.loc[df["CryoSleep"] == True, col] = df.loc[df["CryoSleep"] == True, col].fillna(0)
    df["TotalSpend"] = df[SPENDING_COLS].sum(axis=1, min_count=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"].fillna(0))
    for col in SPENDING_COLS:
        df[f"Log{col}"] = np.log1p(df[col].fillna(0))

    # If a passenger has any spending recorded, CryoSleep must be False.
    spend_known = df[SPENDING_COLS].sum(axis=1, min_count=1)
    df.loc[df["CryoSleep"].isna() & (spend_known > 0), "CryoSleep"] = False

    # Age bucket helps trees split cleanly on minors.
    df["IsChild"] = (df["Age"].fillna(df["Age"].median()) < 13).astype(int)

    df = df.drop(columns=["Name", "Cabin", "PassengerId", "LastName"])
    return df


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert object/bool columns to pandas 'category' dtype with shared categories."""
    for col in CATEGORICAL_COLS:
        combined = pd.concat([train[col], test[col]], axis=0).astype("category")
        cats = combined.cat.categories
        train[col] = pd.Categorical(train[col], categories=cats)
        test[col] = pd.Categorical(test[col], categories=cats)
    return train, test
