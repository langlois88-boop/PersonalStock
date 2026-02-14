from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class PortfolioEnv(gym.Env):
    """Simple portfolio environment for historical price simulation.

    Observation: [cash, positions..., prices...]
    Action: weights vector in [-1, 1] for each asset (sell/buy)
    """

    def __init__(self, prices: pd.DataFrame, initial_cash: float = 10000.0):
        super().__init__()
        self.prices = prices.dropna().copy()
        self.assets = list(self.prices.columns)
        self.initial_cash = initial_cash

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.assets),), dtype=np.float32
        )

        obs_size = 1 + len(self.assets) * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_index = 0
        self.cash = self.initial_cash
        self.positions = np.zeros(len(self.assets), dtype=np.float32)
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        prices = self.prices.iloc[self.step_index].values.astype(np.float32)

        # Execute trades based on action weights
        for i, w in enumerate(action):
            if w > 0:
                buy_amount = self.cash * w
                if prices[i] > 0:
                    self.positions[i] += buy_amount / prices[i]
                    self.cash -= buy_amount
            elif w < 0:
                sell_amount = self.positions[i] * (-w)
                self.positions[i] -= sell_amount
                self.cash += sell_amount * prices[i]

        # Advance time
        self.step_index += 1
        done = self.step_index >= len(self.prices) - 1

        next_prices = self.prices.iloc[self.step_index].values.astype(np.float32)
        portfolio_value = self.cash + np.sum(self.positions * next_prices)
        reward = portfolio_value

        return self._get_observation(), reward, done, False, {
            "value": portfolio_value,
            "cash": self.cash,
        }

    def _get_observation(self) -> np.ndarray:
        prices = self.prices.iloc[self.step_index].values.astype(np.float32)
        return np.concatenate([[self.cash], self.positions, prices]).astype(np.float32)


def simulate_strategy(prices: pd.DataFrame) -> Dict[str, List[float]]:
    """Baseline strategy: buy-and-hold equally weighted on day 1."""
    if prices.empty:
        return {"values": []}

    cash = 10000.0
    assets = prices.columns
    first = prices.iloc[0]
    positions = {a: (cash / len(assets)) / first[a] for a in assets}

    values = []
    for _, row in prices.iterrows():
        value = sum(positions[a] * row[a] for a in assets)
        values.append(float(value))

    return {"values": values}


def simulate_drip(prices: pd.DataFrame, annual_yield: float = 0.03) -> Dict[str, List[float]]:
    """DRIP strategy: distribute dividends monthly and reinvest equally."""
    if prices.empty:
        return {"values": []}

    cash = 10000.0
    assets = prices.columns
    first = prices.iloc[0]
    positions = {a: (cash / len(assets)) / first[a] for a in assets}
    cash = 0.0

    values = []
    for i, row in prices.iterrows():
        value = sum(positions[a] * row[a] for a in assets)
        values.append(float(value))

        # Monthly reinvestment every ~21 trading days
        if len(values) % 21 == 0:
            dividend = value * (annual_yield / 12)
            cash += dividend
            for a in assets:
                if row[a] > 0:
                    positions[a] += (cash / len(assets)) / row[a]
            cash = 0.0

    return {"values": values}


def simulate_rebalance(
    prices: pd.DataFrame,
    rebalance_days: int = 21,
) -> Dict[str, List[float]]:
    """Rebalance to equal weights periodically."""
    if prices.empty:
        return {"values": []}

    cash = 10000.0
    assets = prices.columns
    first = prices.iloc[0]
    positions = {a: (cash / len(assets)) / first[a] for a in assets}
    cash = 0.0

    values = []
    for idx, row in enumerate(prices.itertuples(index=False), start=1):
        row_dict = {assets[i]: row[i] for i in range(len(assets))}
        value = sum(positions[a] * row_dict[a] for a in assets) + cash
        values.append(float(value))

        if idx % rebalance_days == 0:
            target_value = value / len(assets)
            for a in assets:
                price = row_dict[a]
                if price > 0:
                    positions[a] = target_value / price

    return {"values": values}
