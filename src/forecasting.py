"""Forecast combination and persistence-benchmark helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .mbts import MBTS


def nrmse(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    scale = np.std(actual)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)) / scale) if scale else np.inf


def persistence_nrmse(series: pd.Series) -> float:
    series = pd.Series(series).dropna()
    return nrmse(series.iloc[1:].to_numpy(), series.shift(1).iloc[1:].to_numpy())


def shrinking_window_mbts_prediction(
    differenced_data: pd.DataFrame, *, n_windows: int = 5, pmax: int = 10
) -> float:
    """One-step prediction from several shrinking samples, inverse-NRMSE weighted.

    This mirrors the notebook's forecast-combination idea. It is a research
    helper, not a claim that inverse in-sample NRMSE produces optimal weights.
    """
    n = len(differenced_data)
    weighted_prediction = 0.0
    total_weight = 0.0
    for i in range(n_windows):
        start = i * (n // n_windows)
        train = differenced_data.iloc[start:]
        if len(train) <= pmax + 2:
            continue
        try:
            model = MBTS(pmax).fit(train)
            if not np.isfinite(model.nrmse_) or model.nrmse_ <= 0:
                continue
            pred = model.predict(train.iloc[-pmax:])
        except (ValueError, np.linalg.LinAlgError):
            continue
        weight = 1.0 / model.nrmse_
        weighted_prediction += weight * pred
        total_weight += weight
    if total_weight == 0:
        raise ValueError("no valid mBTS window produced a forecast")
    return float(weighted_prediction / total_weight)
