"""Rolling Granger-causality utilities used in the thesis research."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, grangercausalitytests


def rolling_pairwise_granger(
    target: pd.Series,
    candidate: pd.Series,
    *,
    max_lag: int = 8,
    window_sizes=(30, 45, 60, 90, 180, 360),
    step: int = 10,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Measure how often a candidate Granger-causes the target in rolling windows.

    As in the original notebook, a two-series window is differenced when the
    pair does not pass the Engle-Granger cointegration test at ``alpha``.
    """
    data = pd.concat([target.rename("target"), candidate.rename("candidate")], axis=1).dropna()
    rows = []
    for window in window_sizes:
        causal = total = 0
        for start in range(0, len(data) - window + 1, step):
            sample = data.iloc[start : start + window]
            if coint(sample.iloc[:, 0], sample.iloc[:, 1])[1] >= alpha:
                sample = sample.diff().dropna()
            if len(sample) <= max_lag + 1:
                continue
            result = grangercausalitytests(sample, max_lag, verbose=False)
            significant = any(res[0]["ssr_ftest"][1] < alpha for res in result.values())
            causal += int(significant)
            total += 1
        rows.append(
            {
                "window_size": int(window),
                "causal_windows": causal,
                "tested_windows": total,
                "causality_rate": causal / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("window_size")
