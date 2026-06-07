"""
Masked DQN for Texas Hold'em No-Limit.

During SB3's internal training rollouts, action masking is handled by the
environment wrapper (illegal actions get remapped to legal ones).

During inference/evaluation, we override predict() to:
- Exploration: sample uniformly from legal actions only
- Exploitation: mask illegal Q-values to -inf before argmax
"""

import numpy as np
import torch as th
from stable_baselines3 import DQN


class MaskedDQN(DQN):
    """DQN with action masking during prediction."""

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Override predict to apply action masking."""
        action_mask = self._get_action_mask()

        if not deterministic and np.random.random() < self.exploration_rate:
            # Epsilon-greedy: sample from legal actions only
            legal_actions = np.where(action_mask == 1)[0]
            if len(legal_actions) == 0:
                legal_actions = np.arange(self.action_space.n)
            action = np.array([np.random.choice(legal_actions)])
        else:
            # Greedy: compute Q-values and mask illegal ones
            if not isinstance(observation, np.ndarray):
                observation = np.array(observation)
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)

            obs_tensor, _ = self.policy.obs_to_tensor(observation)
            with th.no_grad():
                q_values = self.policy.q_net(obs_tensor)

            # Mask illegal actions with -inf
            mask_tensor = th.tensor(
                action_mask, dtype=th.float32, device=q_values.device
            ).unsqueeze(0)
            q_values = q_values + (1 - mask_tensor) * (-1e8)
            action = q_values.argmax(dim=1).cpu().numpy()

        return action, state

    def _get_action_mask(self):
        """Get the action mask from the current environment."""
        try:
            env = self.get_env()
            base_env = env.envs[0].unwrapped
            if hasattr(base_env, "action_masks"):
                return base_env.action_masks()
        except Exception:
            pass
        return np.ones(self.action_space.n, dtype=np.int8)
