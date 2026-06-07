"""
Single-agent Gymnasium wrapper for PettingZoo's Texas Hold'em No-Limit.

Converts the multi-agent environment into a single-agent one where:
- player_0 is the learning agent
- player_1 plays randomly (uniform over legal actions)

Action masking is handled internally — any illegal action is automatically
remapped to a random legal action. The mask is also exposed via action_masks()
for use during inference.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo.classic import texas_holdem_no_limit_v6


class TexasHoldemEnv(gym.Env):
    """Single-agent wrapper for Texas Hold'em No-Limit."""

    metadata = {"render_modes": ["human", None]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.agent_id = "player_0"

        # Create a temporary env to get space dimensions
        tmp = texas_holdem_no_limit_v6.env()
        tmp.reset()
        obs_space = tmp.observation_space(self.agent_id)

        self.observation_space = spaces.Box(
            low=obs_space["observation"].low,
            high=obs_space["observation"].high,
            dtype=np.float32,
        )
        self.action_space = tmp.action_space(self.agent_id)
        self.n_actions = self.action_space.n

        # Action mask: 1 = legal, 0 = illegal
        self.current_action_mask = np.ones(self.n_actions, dtype=np.int8)
        self._game_over = False
        self._cached_reward = 0.0
        self.pz_env = None

    def reset(self, seed=None, options=None):
        self._game_over = False
        self._cached_reward = 0.0
        self.pz_env = texas_holdem_no_limit_v6.env(render_mode=self.render_mode)
        self.pz_env.reset(seed=seed)

        # Play through any initial opponent turns
        self._play_opponents()

        # Check if game already ended (e.g. opponent folded immediately)
        if self._check_game_over():
            self._game_over = True
            self.current_action_mask = np.ones(self.n_actions, dtype=np.int8)
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        obs = self._get_agent_obs()
        return obs, {}

    def step(self, action):
        # Handle numpy arrays from SB3 predict
        if hasattr(action, '__len__'):
            action = action.item() if hasattr(action, 'item') else action[0]
        action = int(action)

        # If game already ended, return terminal
        if self._game_over:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            self.current_action_mask = np.ones(self.n_actions, dtype=np.int8)
            return obs, 0.0, True, False, {}

        # Remap illegal actions to random legal ones
        if self.current_action_mask[action] == 0:
            legal = np.where(self.current_action_mask == 1)[0]
            if len(legal) == 0:
                self._game_over = True
                return (np.zeros(self.observation_space.shape, dtype=np.float32),
                        0.0, True, False, {})
            action = int(np.random.choice(legal))

        # Take our action
        self.pz_env.step(action)

        # Play through all opponent turns
        self._play_opponents()

        # Check if game ended
        if self._check_game_over():
            self._game_over = True
            self.current_action_mask = np.ones(self.n_actions, dtype=np.int8)
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, self._cached_reward, True, False, {}

        # Game continues
        obs = self._get_agent_obs()
        return obs, 0.0, False, False, {}

    def _play_opponents(self):
        """Play random legal actions for all non-learning agents."""
        while (self.pz_env.agents and
               self.pz_env.agent_selection != self.agent_id):
            obs, _, term, trunc, _ = self.pz_env.last()
            if term or trunc:
                self.pz_env.step(None)
            else:
                mask = np.array(obs["action_mask"])
                legal = np.where(mask == 1)[0]
                if len(legal) == 0:
                    self.pz_env.step(None)
                else:
                    self.pz_env.step(int(np.random.choice(legal)))

    def _check_game_over(self):
        """Check if game ended and capture reward BEFORE clearing agents."""
        # All agents already cleared
        if not self.pz_env.agents:
            return True

        # Our agent already removed
        if self.agent_id not in self.pz_env.agents:
            return True

        # Check if it's our turn but we're terminated
        if self.pz_env.agent_selection == self.agent_id:
            _, reward, term, trunc, _ = self.pz_env.last()
            if term or trunc:
                # CAPTURE REWARD BEFORE acknowledging termination
                self._cached_reward = float(reward)
                self.pz_env.step(None)
                self._play_opponents()
                return True

        return False

    def _get_agent_obs(self):
        """Get the learning agent's observation and update mask."""
        full_obs, _, _, _, _ = self.pz_env.last()
        self.current_action_mask = np.array(full_obs["action_mask"], dtype=np.int8)
        return np.array(full_obs["observation"], dtype=np.float32)

    def action_masks(self):
        """Return the current action mask."""
        return self.current_action_mask.copy()
