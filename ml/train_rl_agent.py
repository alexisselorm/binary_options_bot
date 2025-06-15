# ml/train_rl_agent.py
"""
Train a PPO agent that learns when (and how much) to trade.
The environment offers a two-action space:
    0 = SKIP trade
    1 = TAKE trade
Observation = [confidence, lstm_prediction]
Reward      = trade PnL  (or small −0.01 penalty when skipping)

Data file expected at  data/rl_training_data.npy
Each row: [confidence, lstm_pred, actual]
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from reinforcements.environment import TradingEnvironment

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_PATH = "models/ppo_agent.zip"
DATA_PATH = "data/rl_training_data.npy"
TIMESTEPS = 100_000

# ------------------------------------------------------------------
# Load data (or create dummy if none yet)
# ------------------------------------------------------------------
if os.path.exists(DATA_PATH):
    dataset = np.load(DATA_PATH).astype(np.float32)
    print(f"\U0001F4CA Loaded RL dataset: {dataset.shape} rows")
else:
    print("⚠️ RL data file not found. Using 1 dummy sample.")
    dataset = np.array([[0.5, 0.0, 1.0]], dtype=np.float32)

# ------------------------------------------------------------------
# Build & validate environment
# ------------------------------------------------------------------
env = DummyVecEnv([lambda: TradingEnvironment(dataset)])

# ------------------------------------------------------------------
# Train PPO model
# ------------------------------------------------------------------
model = PPO("MlpPolicy", env, verbose=1)
reward_log = model.learn(total_timesteps=TIMESTEPS)

# ------------------------------------------------------------------
# Save model
# ------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
print(f"✅ RL agent saved to {MODEL_PATH}")

# ------------------------------------------------------------------
# Plot learning curve
# ------------------------------------------------------------------
try:
    rewards = reward_log.episode_rewards if hasattr(
        reward_log, 'episode_rewards') else []
    if rewards:
        plt.plot(rewards)
        plt.title("Training Reward Curve")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig("models/training_curve.png")
        print("📉 Training curve saved to models/training_curve.png")
except Exception as e:
    print(f"⚠️ Could not save reward curve: {e}")
