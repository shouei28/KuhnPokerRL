"""
Training script for Masked DQN on Texas Hold'em No-Limit.

Trains a DQN agent (player_0) against a random opponent (player_1).
Logs training progress and saves the trained model.
"""

import warnings
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from poker_env import TexasHoldemEnv
from masked_dqn import MaskedDQN

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
TOTAL_TIMESTEPS = 200_000      # Total training steps
LEARNING_RATE = 1e-4           # Learning rate for the Q-network
BUFFER_SIZE = 50_000           # Replay buffer capacity
BATCH_SIZE = 64                # Mini-batch size for updates
GAMMA = 0.99                   # Discount factor
EXPLORATION_FRACTION = 0.3     # Fraction of training for epsilon decay
EXPLORATION_FINAL_EPS = 0.05   # Final epsilon after decay
TARGET_UPDATE_INTERVAL = 1000  # Steps between target network syncs
LEARNING_STARTS = 5000         # Steps before first gradient update
TRAIN_FREQ = 4                 # Train every N steps
MODEL_SAVE_PATH = "models/dqn_poker"
LOG_DIR = "logs/"


# ──────────────────────────────────────────────
# Evaluation Callback
# ──────────────────────────────────────────────
class EvalCallback(BaseCallback):
    """Evaluates the agent every `eval_freq` steps by playing test games."""

    def __init__(self, eval_freq=10_000, n_eval_games=100, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_games = n_eval_games

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self):
        env = TexasHoldemEnv()
        wins, total_reward = 0, 0.0

        for _ in range(self.n_eval_games):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_reward += episode_reward
            if episode_reward > 0:
                wins += 1

        avg_reward = total_reward / self.n_eval_games
        win_rate = wins / self.n_eval_games * 100

        print(f"\n{'='*50}")
        print(f"  Eval @ step {self.n_calls:,}")
        print(f"  Win rate: {win_rate:.1f}% | Avg reward: {avg_reward:.2f}")
        print(f"  Epsilon: {self.model.exploration_rate:.3f}")
        print(f"{'='*50}\n")


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train():
    print("Creating environment...")
    env = TexasHoldemEnv()

    print("Initializing Masked DQN...")
    model = MaskedDQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        learning_starts=LEARNING_STARTS,
        train_freq=TRAIN_FREQ,
        policy_kwargs=dict(net_arch=[256, 256]),  # Two hidden layers
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"  Observation dim: {env.observation_space.shape[0]}")
    print(f"  Action space: {env.action_space.n} actions")
    print(f"  Net architecture: [256, 256]")
    print()

    eval_callback = EvalCallback(eval_freq=10_000, n_eval_games=100)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        log_interval=100,
    )

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
