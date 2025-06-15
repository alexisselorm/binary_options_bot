# reinforcements/agent.py

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from reinforcements.environment import TradingEnvironment


class RLAgent:
    def __init__(
        self,
        model_path: Optional[str] = None,
        dataset: Optional[np.ndarray] = None,
        verbose: int = 0,
        default_stake: float = 1.0,  # NEW
    ):
        """
        Parameters
        ----------
        model_path : str | None
            Path to a saved PPO model (.zip). If None, a fresh model is created.
        dataset : np.ndarray | None
            Training data in shape (N, 3) = [confidence, lstm_pred, label].
            Required when training a new model; optional for inference.
        verbose : int
            Verbosity level for logging (0 = silent)
        default_stake : float
            Used when model does not output stake amount (e.g., Discrete-only model)
        """
        self.default_stake = float(np.clip(default_stake, 0.01, 1.0))

        if dataset is None:
            print("⚠️ No specific dataset provided; Loading rl_training_data.")
            dataset = np.load('data/rl_training_data.npy')
            print(f"✅ Loaded training data with shape {dataset.shape}")

        self.env = DummyVecEnv([lambda: TradingEnvironment(dataset)])

        if model_path and model_path.endswith(".zip"):
            self.model: PPO = PPO.load(model_path, env=None)
            self.model.set_env(self.env)
            if verbose:
                print(f"✅ RL model loaded from {model_path}")
        else:
            self.model = PPO("MlpPolicy", self.env, verbose=verbose)
            if verbose:
                print("🆕 New PPO model initialised")

    # ------------------------------------------------------------------
    def predict(self, state_vec: np.ndarray) -> Tuple[int, float]:
        """
        Parameters
        ----------
        state_vec : np.ndarray
            Current observation [confidence, lstm_pred]  shape=(2,)

        Returns
        -------
        (trade_action, stake_multiplier)
            trade_action     : 0 = skip, 1 = trade
            stake_multiplier : 0.01‒1.0
        """
        obs = state_vec.astype(np.float32).reshape(1, -1)
        raw_action, _ = self.model.predict(obs, deterministic=True)

        # Ensure raw_action is 2D then flatten to (2,)
        if isinstance(raw_action, np.ndarray) and raw_action.shape == (1, 2):
            trade, stake = raw_action[0]  # Extract from batch
        else:
            raise ValueError(f"Unexpected action shape: {raw_action.shape}")

        trade = int(np.round(trade))
        stake = float(np.clip(stake, 0.01, 1.0))

        return trade, stake

    # ------------------------------------------------------------------

    def train(self, timesteps: int = 100_000):
        """Train the PPO agent on its current env."""
        self.model.learn(total_timesteps=timesteps)

    def save(self, path: str = "models/ppo_agent.zip"):
        """Save model to disk."""
        self.model.save(path)
        print(f"✅ RL model saved to {path}")
