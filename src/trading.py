"""Exploratory one-day directional rule from the thesis."""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_directional_rule(
    price: pd.Series, forecast: pd.Series, *, threshold: float = 0.005
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Apply the thesis's +/-0.5% forecast threshold.

    +1 when forecast is > threshold above the previous price, -1 when it is
    below by at least threshold, and 0 otherwise. Positions last one period.
    """
    aligned = pd.concat([price.rename("price"), forecast.rename("forecast")], axis=1).dropna()
    aligned["previous_price"] = aligned["price"].shift(1)
    forecast_move = (aligned["forecast"] - aligned["previous_price"]) / aligned["previous_price"]
    aligned["signal"] = np.select(
        [forecast_move > threshold, forecast_move <= -threshold], [1, -1], default=0
    )
    aligned["market_return"] = aligned["price"].pct_change()
    aligned["strategy_return"] = aligned["signal"] * aligned["market_return"]
    aligned["equity"] = (1 + aligned["strategy_return"].fillna(0)).cumprod()

    trades = aligned.loc[aligned["signal"] != 0, "strategy_return"].dropna()
    wins = trades[trades > 0]
    losses = trades[trades < 0]
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan
    metrics = {
        "trades": float(len(trades)),
        "win_rate": float(len(wins) / len(trades)) if len(trades) else np.nan,
        "reward_to_risk": float(avg_win / abs(avg_loss)) if np.isfinite(avg_loss) and avg_loss != 0 else np.nan,
        "ending_equity": float(aligned["equity"].iloc[-1]),
    }
    return aligned, metrics
