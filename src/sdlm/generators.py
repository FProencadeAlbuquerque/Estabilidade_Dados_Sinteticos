"""Tabular generators (SDV): CTGAN, TVAE and Gaussian copula."""
import random

import numpy as np
import pandas as pd
import torch
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

from . import config


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def numerical_metadata(df: pd.DataFrame) -> Metadata:
    """All columns are continuous measurements; declare them explicitly instead of
    relying on auto-detection (constant set-point columns could be misread)."""
    cols = {c: {"sdtype": "numerical"} for c in df.columns}
    return Metadata.load_from_dict({"tables": {TABLE: {"columns": cols}}})


TABLE = "table"


def _fit(syn, train):
    """SDV >= 2 takes {table_name: DataFrame}; older releases take the DataFrame."""
    try:
        syn.fit({TABLE: train})
    except (TypeError, AttributeError, ValueError):
        syn.fit(train)


def _sample(syn, n):
    out = syn.sample(num_rows=n)
    return next(iter(out.values())) if isinstance(out, dict) else out


def build(model: str, train: pd.DataFrame, epochs: int | None = None, verbose: bool = True):
    metadata = numerical_metadata(train)
    if model == "ctgan":
        params = dict(config.CTGAN_PARAMS)
        if epochs:
            params["epochs"] = epochs
        return CTGANSynthesizer(metadata, verbose=verbose, **params)
    if model == "tvae":
        params = dict(config.TVAE_PARAMS)
        if epochs:
            params["epochs"] = epochs
        return TVAESynthesizer(metadata, verbose=verbose, **params)
    if model == "copula":
        return GaussianCopulaSynthesizer(metadata)
    raise ValueError(model)


def fit_sample(model: str, train: pd.DataFrame, n: int = config.N_SYNTH,
               epochs: int | None = None, seed: int = config.SEED):
    """Fit `model` on `train` and draw `n` synthetic rows. Returns (synth, synthesizer)."""
    set_seed(seed)
    syn = build(model, train, epochs)
    _fit(syn, train)
    set_seed(seed)
    synth = _sample(syn, n)
    return synth[train.columns], syn
