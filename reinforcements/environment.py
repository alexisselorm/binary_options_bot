# reinforcements/environment.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TradingEnvironment(gym.Env):
    """
    Custom RL environment for binary options trade decisions.
    Observation: [confidence, lstm_pred] → shape=(2,)
    Action: Box(2,) → [trade_flag (0.0–1.0), stake_fraction (0.01–1.0)]
    Reward: profit value (can be positive or negative)
    """

    def __init__(self, dataset: np.ndarray = np.load("data/rl_training_data.npy")):
        super().__init__()
        self.dataset = dataset.astype(np.float32)
        self.index = 0

        # Observation: confidence + lstm_pred
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # Action space: [0 or 1 for trade/skip, 0.01–1.0 stake multiplier]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.01]), high=np.array([1.0, 1.0]), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        obs = self._get_state()
        return obs, {}

    def _get_state(self):
        return self.dataset[self.index][:2]  # confidence, lstm_pred

    def step(self, action):
        trade_raw, stake = action
        trade = int(np.round(trade_raw))
        confidence, lstm_pred, actual = self.dataset[self.index]

        self.index += 1
        done = self.index >= len(self.dataset)
        truncated = False

        if trade == 0:
            reward = 0.0
        else:
            pred = 1 if confidence > 0.5 else 0
            correct = pred == actual
            reward = float(stake * (1.0 if correct else -1.0))

        info = {}
        obs = self._get_state() if not done else np.zeros(2, dtype=np.float32)
        return obs, reward, done, truncated, info

    def render(self):
        pass  # No-op for now
