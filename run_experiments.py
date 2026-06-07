"""
Full experiment runner for reward shaping research.

Trains DQN agents under 5 reward shaping conditions, collects metrics
at regular intervals, and generates publication-ready charts.

Usage:
    python run_experiments.py                   # full run (200k steps, 3 seeds)
    python run_experiments.py --steps 50000     # quick test run
    python run_experiments.py --seeds 1         # single seed (faster)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import argparse
import numpy as np
from collections import Counter
from poker_env import TexasHoldemEnv
from masked_dqn import MaskedDQN
from stable_baselines3.common.callbacks import BaseCallback

ACTION_NAMES = ["Fold", "Check", "Call", "Raise ½", "Raise Full"]

# ──────────────────────────────────────────────
# Reward Shaping Environment
# ──────────────────────────────────────────────
class Shaped_TexasHoldemEnv(TexasHoldemEnv):
    """Env wrapper that applies reward shaping on top of game rewards."""

    def __init__(self, fold_penalty=0.0, raise_bonus=0.0, survival_bonus=0.0, **kwargs):
        super().__init__(**kwargs)
        self.fold_penalty = fold_penalty
        self.raise_bonus = raise_bonus
        self.survival_bonus = survival_bonus
        self.action_log = []

    def step(self, action):
        # Get raw action int
        act = action
        if hasattr(act, '__len__'):
            act = act.item() if hasattr(act, 'item') else act[0]
        act = int(act)

        obs, reward, term, trunc, info = super().step(action)

        # Log action
        if act < 5:
            self.action_log.append(act)

        # Apply shaping (only on intermediate steps)
        if not (term or trunc):
            if act == 0:
                reward += self.fold_penalty
            elif act in (3, 4):
                reward += self.raise_bonus
            reward += self.survival_bonus

        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        return super().reset(**kwargs)


# ──────────────────────────────────────────────
# Metrics Callback
# ──────────────────────────────────────────────
class MetricsCallback(BaseCallback):
    """Collects win rate, avg reward, epsilon at regular intervals."""

    def __init__(self, eval_freq=2000, n_eval_games=50):
        super().__init__(verbose=0)
        self.eval_freq = eval_freq
        self.n_eval_games = n_eval_games
        self.results = {
            "steps": [],
            "win_rates": [],
            "avg_rewards": [],
            "epsilons": [],
        }

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self):
        eval_env = TexasHoldemEnv()  # No shaping during eval — raw performance
        wins, total_reward = 0, 0.0

        for _ in range(self.n_eval_games):
            obs, _ = eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = eval_env.step(action)
                ep_reward += reward
                done = term or trunc
            total_reward += ep_reward
            if ep_reward > 0:
                wins += 1

        win_rate = wins / self.n_eval_games * 100
        avg_reward = total_reward / self.n_eval_games

        self.results["steps"].append(self.num_timesteps)
        self.results["win_rates"].append(round(win_rate, 2))
        self.results["avg_rewards"].append(round(avg_reward, 2))
        self.results["epsilons"].append(round(self.model.exploration_rate, 4))

        print(f"    Step {self.num_timesteps:>7,d} | "
              f"Win: {win_rate:5.1f}% | "
              f"Reward: {avg_reward:>+7.1f} | "
              f"ε: {self.model.exploration_rate:.3f}")


# ──────────────────────────────────────────────
# Experiment Conditions
# ──────────────────────────────────────────────
CONDITIONS = {
    "Baseline": {"fold_penalty": 0.0, "raise_bonus": 0.0, "survival_bonus": 0.0},
    "Fold Penalty": {"fold_penalty": -5.0, "raise_bonus": 0.0, "survival_bonus": 0.0},
    "Raise Bonus": {"fold_penalty": 0.0, "raise_bonus": 2.0, "survival_bonus": 0.0},
    "Survival Bonus": {"fold_penalty": 0.0, "raise_bonus": 0.0, "survival_bonus": 0.1},
    "Combined": {"fold_penalty": -5.0, "raise_bonus": 2.0, "survival_bonus": 0.1},
}


def get_final_action_distribution(model, n_games=200):
    """Evaluate action distribution over test games."""
    env = TexasHoldemEnv()
    action_counts = Counter()
    total_actions = 0

    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            act = int(action[0]) if hasattr(action, '__len__') else int(action)
            if act < 5:
                action_counts[act] += 1
                total_actions += 1
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

    dist = {ACTION_NAMES[i]: action_counts.get(i, 0) for i in range(5)}
    return dist, total_actions


def run_single_experiment(condition_name, shaping_params, total_timesteps, seed):
    """Train one agent under one condition with one seed."""
    print(f"\n  Seed {seed}:")

    env = Shaped_TexasHoldemEnv(**shaping_params)

    model = MaskedDQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-5,
        buffer_size=200_000,
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=1000,
        learning_starts=10000,
        train_freq=4,
        policy_kwargs=dict(net_arch=[512, 512]),
        verbose=0,
        seed=seed,
    )

    callback = MetricsCallback(eval_freq=5000, n_eval_games=100)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Final evaluation
    action_dist, total_actions = get_final_action_distribution(model, n_games=200)

    # Get action log from training
    training_action_counts = Counter(env.action_log)
    training_action_dist = {ACTION_NAMES[i]: training_action_counts.get(i, 0) for i in range(5)}

    # Save model
    model_path = f"models/experiment_{condition_name.replace(' ', '_')}_seed{seed}"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)

    return {
        "condition": condition_name,
        "seed": seed,
        "shaping": shaping_params,
        "learning_curve": callback.results,
        "final_action_dist": action_dist,
        "final_action_total": total_actions,
        "training_action_dist": training_action_dist,
        "model_path": model_path,
    }


def run_all_experiments(total_timesteps=200_000, n_seeds=3):
    """Run all conditions across all seeds."""
    all_results = {}

    for cond_name, shaping in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"  CONDITION: {cond_name}")
        print(f"  Shaping: {shaping}")
        print(f"{'='*60}")

        cond_results = []
        for seed in range(1, n_seeds + 1):
            result = run_single_experiment(cond_name, shaping, total_timesteps, seed)
            cond_results.append(result)

        all_results[cond_name] = cond_results

    # Also run random baseline evaluation
    print(f"\n{'='*60}")
    print(f"  RANDOM BASELINE")
    print(f"{'='*60}")
    env = TexasHoldemEnv()
    rand_wins, rand_total, rand_actions = 0, 0.0, Counter()
    n_games = 200
    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            mask = env.action_masks()
            legal = np.where(mask == 1)[0]
            act = int(np.random.choice(legal))
            rand_actions[act] += 1
            obs, reward, term, trunc, _ = env.step(act)
            ep_r += reward
            done = term or trunc
        rand_total += ep_r
        if ep_r > 0:
            rand_wins += 1

    all_results["_random_baseline"] = {
        "win_rate": round(rand_wins / n_games * 100, 2),
        "avg_reward": round(rand_total / n_games, 2),
        "action_dist": {ACTION_NAMES[i]: rand_actions.get(i, 0) for i in range(5)},
    }
    print(f"  Win rate: {all_results['_random_baseline']['win_rate']}%")
    print(f"  Avg reward: {all_results['_random_baseline']['avg_reward']}")

    # Save all results
    os.makedirs("results", exist_ok=True)
    save_path = "results/experiment_results.json"

    # Convert to serializable format
    serializable = {}
    for k, v in all_results.items():
        if k == "_random_baseline":
            serializable[k] = v
        else:
            serializable[k] = []
            for run in v:
                serializable[k].append({
                    "condition": run["condition"],
                    "seed": run["seed"],
                    "shaping": run["shaping"],
                    "learning_curve": run["learning_curve"],
                    "final_action_dist": run["final_action_dist"],
                    "final_action_total": run["final_action_total"],
                    "training_action_dist": run["training_action_dist"],
                    "model_path": run["model_path"],
                })

    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n📁 Results saved to {save_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reward shaping experiments")
    parser.add_argument("--steps", type=int, default=500_000, help="Training steps per condition")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds per condition")
    args = parser.parse_args()

    print("🃏 Reward Shaping Experiment Runner")
    print(f"   Steps per condition: {args.steps:,}")
    print(f"   Seeds per condition: {args.seeds}")
    print(f"   Total training runs: {len(CONDITIONS) * args.seeds}")

    results = run_all_experiments(total_timesteps=args.steps, n_seeds=args.seeds)

    print("\n✅ All experiments complete! Run 'python plot_results.py' to generate charts.")
