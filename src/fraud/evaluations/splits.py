import pandas as pd
from sklearn.model_selection import train_test_split


def temporal_split(df: pd.DataFrame, dt_col: str = "txn_ts",
                    train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Sorts by time, cuts train/val/test as contiguous chronological chunks.
    This is the split every model in this project actually gets trained
    and evaluated on, from here forward.
    """
    df_sorted = df.sort_values(dt_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]

    print(f"Temporal split — train: {len(train)} ({train[dt_col].min()} to {train[dt_col].max()}) | "
          f"val: {len(val)} | test: {len(test)}")
    return train, val, test


def random_split(df: pd.DataFrame, label_col: str = "is_fraud",
                  train_frac: float = 0.70, val_frac: float = 0.15,
                  random_state: int = 42):
    """
    Standard random stratified split — included ONLY to run the
    random-vs-temporal comparison experiment (see
    scripts/random_vs_temporal.py). Do not use this split for any model
    you intend to actually report or deploy.
    """
    train, rest = train_test_split(
        df, train_size=train_frac, stratify=df[label_col], random_state=random_state
    )
    val_frac_of_rest = val_frac / (1 - train_frac)
    val, test = train_test_split(
        rest, train_size=val_frac_of_rest, stratify=rest[label_col], random_state=random_state
    )
    print(f"Random split — train: {len(train)} | val: {len(val)} | test: {len(test)}")
    return train, val, test