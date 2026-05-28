"""Leak-free target encoding computed inside each CV fold."""

from __future__ import annotations

import numpy as np
import pandas as pd


def smoothed_target_encode(
    train_col: pd.Series,
    target: pd.Series,
    valid_col: pd.Series,
    test_col: pd.Series,
    smoothing: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute smoothed target encoding using only the training fold.

    Encoding: enc(c) = (n_c * mean_c + smoothing * global_mean) / (n_c + smoothing)
    Unseen categories fall back to the global mean.
    """
    global_mean = target.mean()
    stats = target.groupby(train_col).agg(["mean", "count"])
    stats["enc"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    mapping = stats["enc"].to_dict()

    train_enc = train_col.map(mapping).fillna(global_mean).to_numpy()
    valid_enc = valid_col.map(mapping).fillna(global_mean).to_numpy()
    test_enc = test_col.map(mapping).fillna(global_mean).to_numpy()
    return train_enc, valid_enc, test_enc


TARGET_ENCODE_COLS = ["Deck", "HomePlanet", "Destination", "DeckPlanet", "CabinRegion", "TopSpendCat"]


def apply_target_encoding(
    train_df: pd.DataFrame,
    target: pd.Series,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str] = TARGET_ENCODE_COLS,
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add `<col>_te` columns to train/valid/test. Mutates copies."""
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()
    for col in cols:
        tr, va, te = smoothed_target_encode(
            train_df[col].astype(object),
            target,
            valid_df[col].astype(object),
            test_df[col].astype(object),
            smoothing=smoothing,
        )
        train_df[f"{col}_te"] = tr
        valid_df[f"{col}_te"] = va
        test_df[f"{col}_te"] = te
    return train_df, valid_df, test_df
