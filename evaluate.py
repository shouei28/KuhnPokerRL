"""
Evaluation script for trained DQN poker agent.

Loads a saved model and runs test games against a random opponent.
Reports win rate, avg reward, action distribution, and game-by-game results.

Usage:
    python evaluate.py                          # uses default model path
    python evaluate.py --model models/dqn_poker # specify model
    python evaluate.py --games 500              # number of test games
    python evaluate.py --verbose                # show each game's result
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
from collections import Counter
from poker_env import TexasHoldemEnv
from masked_dqn import MaskedDQN

ACTION_NAMES = ["Fold", "Check", "Call", "Raise ½", "Raise Full"]


def evaluate(model_path, n_games=200, verbose=False):
    # Load model
    print(f"\n📂 Loading model from: {model_path}")
    try:
        model = MaskedDQN.load(model_path)
    except FileNotFoundError:
        print(f"\n❌ Model not found at '{model_path}'")
        print("   Train a model first:  python train.py")
        print("   Or specify a path:    python evaluate.py --model <path>")
        return

    env = TexasHoldemEnv()
    model.set_env(env)

    print(f"🎮 Running {n_games} test games against random opponent...\n")

    # Track results
    wins, losses, draws = 0, 0, 0
    total_reward = 0.0
    rewards_list = []
    action_counts = Counter()
    game_lengths = []

    for i in range(n_games):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_actions = []
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            act = int(action[0]) if hasattr(action, '__len__') else int(action)
            episode_actions.append(ACTION_NAMES[act])
            action_counts[act] += 1

            obs, reward, term, trunc, _ = env.step(action)
            episode_reward += reward
            steps += 1
            done = term or trunc

        total_reward += episode_reward
        rewards_list.append(episode_reward)
        game_lengths.append(steps)

        if episode_reward > 0:
            wins += 1
            result = "✅ Win"
        elif episode_reward < 0:
            losses += 1
            result = "❌ Loss"
        else:
            draws += 1
            result = "➖ Draw"

        if verbose:
            print(f"  Game {i+1:>4d}: {result}  reward={episode_reward:>+8.1f}  "
                  f"actions=[{' → '.join(episode_actions)}]")

    # ──────────────────────────────────────────────
    # Results Summary
    # ──────────────────────────────────────────────
    rewards_arr = np.array(rewards_list)
    total_actions = sum(action_counts.values())

    print("=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Games played:    {n_games}")
    print(f"  Wins:            {wins}  ({wins/n_games*100:.1f}%)")
    print(f"  Losses:          {losses}  ({losses/n_games*100:.1f}%)")
    print(f"  Draws:           {draws}  ({draws/n_games*100:.1f}%)")
    print("-" * 55)
    print(f"  Avg reward:      {rewards_arr.mean():+.2f}")
    print(f"  Median reward:   {np.median(rewards_arr):+.2f}")
    print(f"  Std reward:      {rewards_arr.std():.2f}")
    print(f"  Max reward:      {rewards_arr.max():+.1f}")
    print(f"  Min reward:      {rewards_arr.min():+.1f}")
    print(f"  Avg game length: {np.mean(game_lengths):.1f} steps")
    print("-" * 55)
    print("  Action Distribution:")
    for act_id in range(5):
        count = action_counts[act_id]
        pct = count / total_actions * 100 if total_actions > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {ACTION_NAMES[act_id]:>10s}: {count:>5d}  ({pct:>5.1f}%)  {bar}")
    print("=" * 55)

    # Compare to random baseline
    print("\n🎲 Running random baseline for comparison...")
    rand_wins = 0
    rand_total = 0.0
    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            mask = env.action_masks()
            legal = np.where(mask == 1)[0]
            action = np.random.choice(legal)
            obs, reward, term, trunc, _ = env.step(action)
            ep_r += reward
            done = term or trunc
        rand_total += ep_r
        if ep_r > 0:
            rand_wins += 1

    print(f"  Random win rate:  {rand_wins/n_games*100:.1f}%")
    print(f"  Random avg reward: {rand_total/n_games:+.2f}")
    improvement = (wins/n_games - rand_wins/n_games) * 100
    print(f"\n  📊 DQN improvement over random: {improvement:+.1f} percentage points")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DQN poker agent")
    parser.add_argument("--model", type=str, default="models/dqn_poker",
                        help="Path to saved model (default: models/dqn_poker)")
    parser.add_argument("--games", type=int, default=200,
                        help="Number of test games (default: 200)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show each game's result")
    args = parser.parse_args()

    evaluate(args.model, n_games=args.games, verbose=args.verbose)
