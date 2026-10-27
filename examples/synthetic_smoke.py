"""Small deterministic smoke test; no claim of reproducing thesis results."""
import numpy as np
import pandas as pd

from src import MBTS, evaluate_directional_rule, persistence_nrmse

rng = np.random.default_rng(7)
n = 240
liquidity = np.cumsum(rng.normal(0, 1, n))
rate = np.cumsum(rng.normal(0, 0.6, n))
btc = np.empty(n)
btc[0] = 100
for t in range(1, n):
    lagged_liquidity_change = liquidity[t - 1] - liquidity[t - 2] if t > 1 else 0
    btc[t] = btc[t - 1] + 0.18 * lagged_liquidity_change + rng.normal(0, 1)

df = pd.DataFrame({"BTC": btc, "LIQ": liquidity, "RATE": rate})
diff = df.diff().dropna()
model = MBTS(pmax=4).fit(diff)
assert model.selected_lags().shape == (4, 3)
assert np.isfinite(persistence_nrmse(df["BTC"]))

forecast = df["BTC"].shift(1) * (1 + 0.006 * np.sign(df["BTC"].diff().fillna(0)))
_, metrics = evaluate_directional_rule(df["BTC"], forecast)
assert metrics["trades"] >= 0
print("thesis research utilities: smoke test passed")
