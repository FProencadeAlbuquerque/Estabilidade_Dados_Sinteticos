"""Dataset loading and the fixed train/test partition."""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load(dataset: str) -> pd.DataFrame:
    return pd.read_csv(config.DATASETS[dataset])


def split(df: pd.DataFrame):
    """8,000 / 2,000 partition with the fixed seed used throughout the paper."""
    train, test = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def load_synthetic(dataset: str, model: str) -> pd.DataFrame:
    return pd.read_csv(config.out_dir(dataset) / "synthetic" / f"{model}.csv")
