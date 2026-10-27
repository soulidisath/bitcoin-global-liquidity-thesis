"""Sparse lag selection used in the diploma-thesis experiments.

This module is a cleaned implementation of the mBTS notebook logic. mBTS
(modified Backward-in-Time Selection) is a published lag-selection method;
it was applied in the thesis, not invented there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MBTS:
    """Modified Backward-in-Time Selection for a multivariate time series.

    The first dataframe column is treated as the response, matching the thesis
    notebooks. Candidate lags are admitted when they improve the BIC criterion.
    """

    def __init__(self, pmax: int = 10):
        if pmax < 1:
            raise ValueError("pmax must be at least 1")
        self.pmax = int(pmax)
        self.index_: np.ndarray | None = None
        self.max_order_: int | None = None
        self.coef_: np.ndarray | None = None
        self.columns_: list[str] | None = None
        self.nrmse_: float | None = None
        self.means_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "MBTS":
        if len(df) <= self.pmax + 2:
            raise ValueError("not enough observations for the requested pmax")

        x = df.astype(float).to_numpy(copy=True)
        n, k = x.shape
        self.columns_ = list(df.columns)
        self.means_ = np.mean(x, axis=0)
        x -= self.means_

        response = 0
        index = np.zeros(k * self.pmax, dtype=int)
        orders = np.zeros(k, dtype=int)

        mse = self._fit_mse(x, response, orders, index)
        bic_old = self._bic(mse, n, orders, index)
        in_game = np.arange(k)
        increment = 1

        while len(in_game):
            in_game = in_game[orders[in_game] < self.pmax]
            if not len(in_game):
                break

            bics = np.full(len(in_game), np.inf)
            mses = np.full(len(in_game), np.inf)
            candidates: list[tuple[np.ndarray, np.ndarray]] = []

            for j, variable in enumerate(in_game):
                order_tmp = orders.copy()
                order_tmp[variable] = min(self.pmax, order_tmp[variable] + increment)
                index_tmp = index.copy()
                lag_position = order_tmp[variable] - 1
                index_tmp[variable * self.pmax + lag_position] = 1

                try:
                    mses[j] = self._fit_mse(x, response, order_tmp, index_tmp)
                    bics[j] = self._bic(mses[j], n, order_tmp, index_tmp)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    pass
                candidates.append((order_tmp, index_tmp))

            best = int(np.argmin(bics))
            bic_new = bics[best]

            if not np.isfinite(bic_new) or bic_old <= bic_new:
                increment += 1
                if increment > self.pmax - int(np.min(orders)):
                    break
                continue

            orders, index = candidates[best]
            bic_old = bic_new
            mse = mses[best]
            increment = 1

        # Normalize the notebook's variable-major search vector to lag-major
        # storage so selection, coefficients, and prediction use one ordering.
        self.index_ = np.reshape(index, (k, self.pmax)).T.reshape(-1)
        self.max_order_ = int(np.max(orders))

        if self.max_order_ == 0 or not np.any(self.index_):
            self.coef_ = np.array([], dtype=float)
            self.nrmse_ = 1.0
            return self

        lag_matrix = self._multilag_matrix(x, response, orders, self.index_)
        y = lag_matrix[:, 0]
        design = lag_matrix[:, 1:]
        self.coef_ = np.linalg.pinv(design) @ y
        residuals = y - design @ self.coef_
        scale = np.std(y)
        self.nrmse_ = float(np.sqrt(np.mean(residuals**2)) / scale) if scale else np.inf
        return self

    def selected_lags(self) -> pd.DataFrame:
        if self.index_ is None or self.columns_ is None:
            raise ValueError("fit the model first")
        k = len(self.columns_)
        matrix = self.index_.reshape(self.pmax, k)
        return pd.DataFrame(
            matrix.astype(bool),
            index=[f"lag_{i}" for i in range(1, self.pmax + 1)],
            columns=self.columns_,
        )

    def coefficients(self) -> pd.DataFrame:
        if self.index_ is None or self.columns_ is None or self.coef_ is None:
            raise ValueError("fit the model first")
        table = pd.DataFrame(
            0.0,
            index=self.columns_,
            columns=[f"lag_{i}" for i in range(1, self.pmax + 1)],
        )
        coords = np.argwhere(self.selected_lags().to_numpy())
        for (lag_idx, var_idx), coef in zip(coords, self.coef_):
            table.iat[var_idx, lag_idx] = float(coef)
        return table

    def predict(self, recent: pd.DataFrame) -> float:
        if self.coef_ is None or self.index_ is None or self.columns_ is None:
            raise ValueError("fit the model first")
        if self.max_order_ == 0 or len(self.coef_) == 0:
            return float(recent.iloc[-1, 0])
        if len(recent) < self.pmax:
            raise ValueError(f"recent must contain at least {self.pmax} rows")

        x = recent[self.columns_].astype(float).to_numpy(copy=True)
        x -= self.means_
        k = x.shape[1]
        features = []
        selected = self.selected_lags().to_numpy()
        for lag_idx in range(self.pmax):
            for var_idx in range(k):
                if selected[lag_idx, var_idx]:
                    features.append(x[-(lag_idx + 1), var_idx])
        centered_prediction = float(np.asarray(features) @ self.coef_)
        return centered_prediction + float(self.means_[0])

    @staticmethod
    def _bic(mse: float, n: int, orders: np.ndarray, index: np.ndarray) -> float:
        effective_n = n - int(np.max(orders))
        if mse <= 0 or effective_n <= 0:
            return np.inf
        return effective_n * np.log(mse) + np.sum(index) * np.log(effective_n)

    def _multilag_matrix(
        self, x: np.ndarray, response: int, orders: np.ndarray, index: np.ndarray
    ) -> np.ndarray:
        n, k = x.shape
        temp = np.full((n, k * self.pmax), np.nan)
        for lag in range(1, self.pmax + 1):
            for variable in range(k):
                if lag <= orders[variable]:
                    temp[lag:, (lag - 1) * k + variable] = x[:-lag, variable]
        out = np.column_stack((x[:, response], temp[:, index == 1]))
        return out[int(np.max(orders)) :]

    def _fit_mse(
        self, x: np.ndarray, response: int, orders: np.ndarray, search_index: np.ndarray
    ) -> float:
        k = x.shape[1]
        lag_major = np.reshape(search_index, (k, self.pmax)).T.reshape(-1)
        if not np.any(lag_major):
            start = int(np.max(orders))
            return float(np.var(x[start:, response]))
        matrix = self._multilag_matrix(x, response, orders, lag_major)
        y = matrix[:, 0]
        design = matrix[:, 1:]
        beta = np.linalg.pinv(design) @ y
        residuals = y - design @ beta
        return float(np.mean(residuals**2))
