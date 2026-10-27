"""Rolling mBTS summaries and coefficient-stability diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .mbts import MBTS


def rolling_mbts_summary(
    df: pd.DataFrame,
    *,
    pmax_values=(5, 10, 15, 20),
    window_sizes=(30, 60, 90, 120, 150),
    step: int = 10,
) -> tuple[pd.Series, pd.DataFrame]:
    """Aggregate external-lag selection over rolling windows.

    Returns the percentage of windows with at least one non-target selected lag
    and a variable-level table with selection counts and total selected lags.
    The first column is the response, matching the thesis notebooks.
    """
    target = df.columns[0]
    causal_hits = {w: 0 for w in window_sizes}
    tested = {w: 0 for w in window_sizes}
    counts = {c: 0 for c in df.columns[1:]}
    strengths = {c: 0 for c in df.columns[1:]}

    for window in window_sizes:
        for pmax in pmax_values:
            if window <= pmax + 2:
                continue
            for start in range(0, len(df) - window + 1, step):
                sample = df.iloc[start : start + window].dropna()
                if len(sample) <= pmax + 2:
                    continue
                try:
                    model = MBTS(pmax).fit(sample)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                selected = model.selected_lags()
                external = selected.drop(columns=target)
                active = external.any(axis=0)
                if active.any():
                    causal_hits[window] += 1
                    for variable in active.index[active]:
                        counts[variable] += 1
                        strengths[variable] += int(external[variable].sum())
                tested[window] += 1

    rates = pd.Series(
        {w: (100 * causal_hits[w] / tested[w] if tested[w] else np.nan) for w in window_sizes},
        name="windows_with_external_lags_pct",
    )
    variable_table = pd.DataFrame(
        {"selection_count": counts, "total_selected_lags": strengths}
    )
    total = variable_table["selection_count"].sum()
    variable_table["share_of_selection_events_pct"] = (
        100 * variable_table["selection_count"] / total if total else 0.0
    )
    return rates, variable_table.sort_values("selection_count", ascending=False)


def lag_coefficient_stability(
    data: pd.DataFrame,
    *,
    response_index: int = 0,
    pmax_values=(5, 7, 10, 14),
    window_sizes=(30, 60, 90, 120),
    step: int = 10,
) -> pd.DataFrame:
    """Reproduce the notebook's coefficient dispersion diagnostic.

    For each window/pmax configuration, fit the unrestricted lag design with
    least squares and report abs(std / mean) * 100 by variable and lag.
    Lower values indicate more stable coefficients in this diagnostic.
    """
    records = []
    names = list(data.columns)
    for window in window_sizes:
        for pmax in pmax_values:
            if window <= pmax:
                continue
            for start in range(0, len(data) - window + 1, step):
                sample = data.iloc[start : start + window].dropna()
                if len(sample) < window:
                    continue
                y = sample.iloc[pmax:, response_index].to_numpy()
                columns = []
                labels = []
                for var_idx, name in enumerate(names):
                    for lag in range(1, pmax + 1):
                        columns.append(sample.iloc[pmax - lag : window - lag, var_idx].to_numpy())
                        labels.append((name, lag))
                design = np.vstack(columns).T
                beta, *_ = np.linalg.lstsq(design, y, rcond=None)
                records.extend(
                    {"variable": v, "lag": lag, "coefficient": float(b)}
                    for (v, lag), b in zip(labels, beta)
                )
    coeffs = pd.DataFrame(records)
    summary = coeffs.groupby(["variable", "lag"])["coefficient"].agg(["mean", "std"])
    summary["std_pct_of_abs_mean"] = 100 * summary["std"].abs() / summary["mean"].abs()
    return summary["std_pct_of_abs_mean"].unstack("lag")
