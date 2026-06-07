"""
Generate clean, readable charts from experiment results.
Uses smoothing and simplified styling for poster/paper readability.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 13

RESULTS_FILE = "results/experiment_results.json"
OUTPUT_DIR = "results/figures"
SMOOTH_WINDOW = 10  # Rolling average window

COLORS = {
    "Baseline": "#6366f1",
    "Fold Penalty": "#ef4444",
    "Raise Bonus": "#f59e0b",
    "Survival Bonus": "#10b981",
    "Combined": "#8b5cf6",
}

ACTION_NAMES = ["Fold", "Check", "Call", "Raise ½", "Raise Full"]
ACTION_COLORS = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]
CONDITIONS = ["Baseline", "Fold Penalty", "Raise Bonus", "Survival Bonus", "Combined"]


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def smooth(data, window=SMOOTH_WINDOW):
    """Simple moving average."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def average_across_seeds(condition_runs):
    all_steps = [r["learning_curve"]["steps"] for r in condition_runs]
    all_wins = [r["learning_curve"]["win_rates"] for r in condition_runs]
    all_rewards = [r["learning_curve"]["avg_rewards"] for r in condition_runs]

    min_len = min(len(s) for s in all_steps)
    steps = all_steps[0][:min_len]
    win_mean = np.mean([w[:min_len] for w in all_wins], axis=0)
    rew_mean = np.mean([r[:min_len] for r in all_rewards], axis=0)

    return steps, win_mean, rew_mean


def plot_win_rate_curves(results):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('white')

    for cond in CONDITIONS:
        if cond not in results:
            continue
        steps, win_mean, _ = average_across_seeds(results[cond])

        # Smooth
        win_smooth = smooth(win_mean)
        n = len(win_smooth)
        steps_smooth = [s / 1000 for s in steps[:n]]

        ax.plot(steps_smooth, win_smooth, color=COLORS[cond],
                linewidth=2.5, label=cond)

    # Random baseline
    if "_random_baseline" in results:
        rand_wr = results["_random_baseline"]["win_rate"]
        ax.axhline(y=rand_wr, color="#94a3b8", linestyle="--", linewidth=1.5,
                    label=f"Random ({rand_wr}%)")

    ax.set_xlabel("Training Steps (×1000)")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate Over Training", fontsize=16, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(20, 75)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_win_rate_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✅ fig1_win_rate_curves.png")


def plot_reward_curves(results):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('white')

    for cond in CONDITIONS:
        if cond not in results:
            continue
        steps, _, rew_mean = average_across_seeds(results[cond])

        rew_smooth = smooth(rew_mean)
        n = len(rew_smooth)
        steps_smooth = [s / 1000 for s in steps[:n]]

        ax.plot(steps_smooth, rew_smooth, color=COLORS[cond],
                linewidth=2.5, label=cond)

    ax.axhline(y=0, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Training Steps (×1000)")
    ax.set_ylabel("Average Reward (chips)")
    ax.set_title("Average Reward Over Training", fontsize=16, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_reward_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✅ fig2_reward_curves.png")


def plot_action_distribution(results):
    """Horizontal stacked bar chart — much easier to read."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('white')

    # Compute distributions
    data = {}
    for cond in CONDITIONS:
        if cond not in results:
            continue
        totals = {a: 0 for a in ACTION_NAMES}
        grand_total = 0
        for run in results[cond]:
            dist = run["final_action_dist"]
            total = run["final_action_total"]
            grand_total += total
            for a in ACTION_NAMES:
                totals[a] += dist.get(a, 0)
        if grand_total > 0:
            data[cond] = {a: totals[a] / grand_total * 100 for a in ACTION_NAMES}
        else:
            data[cond] = {a: 0 for a in ACTION_NAMES}

    y_pos = np.arange(len(CONDITIONS))
    left = np.zeros(len(CONDITIONS))

    for i, action in enumerate(ACTION_NAMES):
        values = [data.get(c, {}).get(action, 0) for c in CONDITIONS]
        bars = ax.barh(y_pos, values, left=left, height=0.6,
                       label=action, color=ACTION_COLORS[i], edgecolor="white", linewidth=0.5)
        # Add percentage labels for segments > 8%
        for j, (v, l) in enumerate(zip(values, left)):
            if v > 8:
                ax.text(l + v / 2, j, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
        left += values

    ax.set_yticks(y_pos)
    ax.set_yticklabels(CONDITIONS, fontsize=12)
    ax.set_xlabel("Action Frequency (%)")
    ax.set_title("Action Distribution by Condition", fontsize=16, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10, ncol=3)
    ax.set_xlim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_action_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✅ fig3_action_distribution.png")


def plot_final_comparison(results):
    """Simple grouped bar chart: win rate + random baseline side by side."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('white')

    # Final win rates (averaged across seeds)
    win_rates = []
    for cond in CONDITIONS:
        if cond not in results:
            win_rates.append(0)
            continue
        final_wrs = [r["learning_curve"]["win_rates"][-1] for r in results[cond]
                      if r["learning_curve"]["win_rates"]]
        win_rates.append(np.mean(final_wrs) if final_wrs else 0)

    x = np.arange(len(CONDITIONS))
    colors_list = [COLORS[c] for c in CONDITIONS]
    bars = ax.bar(x, win_rates, width=0.6, color=colors_list, edgecolor="white", linewidth=0.5)

    # Random baseline line
    if "_random_baseline" in results:
        rand_wr = results["_random_baseline"]["win_rate"]
        ax.axhline(y=rand_wr, color="#94a3b8", linestyle="--", linewidth=2,
                    label=f"Random Baseline ({rand_wr}%)")

    # Value labels on bars
    for bar, val in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=11)
    ax.set_ylabel("Final Win Rate (%)")
    ax.set_title("Final Win Rate Comparison", fontsize=16, fontweight="bold", pad=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(win_rates) + 15)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_final_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✅ fig4_final_comparison.png")


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n📊 Loading results from {RESULTS_FILE}...")
    results = load_results()

    print(f"\n📈 Generating clean figures...\n")
    plot_win_rate_curves(results)
    plot_reward_curves(results)
    plot_action_distribution(results)
    plot_final_comparison(results)

    print(f"\n📁 All figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
