"""
Web dashboard for Texas Hold'em DQN training.

Provides real-time visualization of training metrics, game replays,
and controls (start/pause/save) via Flask + SocketIO.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import time
import threading
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO

from poker_env import TexasHoldemEnv
from masked_dqn import MaskedDQN
from stable_baselines3.common.callbacks import BaseCallback

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ──────────────────────────────────────────────
# Global Training State
# ──────────────────────────────────────────────
state = {
    "is_running": False,
    "is_paused": False,
    "step": 0,
    "episode": 0,
    "epsilon": 1.0,
    "win_rate": 0.0,
    "avg_reward": 0.0,
    "total_timesteps": 200_000,
    "action_counts": [0, 0, 0, 0, 0],
    "history": {"steps": [], "win_rates": [], "avg_rewards": [], "epsilons": []},
    "recent_games": [],
    "model": None,
    "env": None,
}

ACTION_NAMES = ["Fold", "Check", "Call", "Raise ½", "Raise Full"]

pause_event = threading.Event()
pause_event.set()  # Not paused initially
stop_event = threading.Event()
model_lock = threading.Lock()

# ──────────────────────────────────────────────
# Reward Shaping Config (modifiable from frontend)
# ──────────────────────────────────────────────
reward_config = {
    "fold_penalty": 0.0,
    "raise_bonus": 0.0,
    "survival_bonus": 0.0,
}


# ──────────────────────────────────────────────
# Logging Environment Wrapper
# ──────────────────────────────────────────────
class LoggingTexasHoldemEnv(TexasHoldemEnv):
    """Extends TexasHoldemEnv with action logging and reward shaping."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.game_log = []
        self.current_game_actions = []
        self.step_count = 0

    def reset(self, **kwargs):
        self.current_game_actions = []
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)

        # Handle array actions
        act = action
        if hasattr(act, '__len__'):
            act = int(act[0]) if len(act) > 0 else 0
        act = int(act)

        # Log action
        self.current_game_actions.append(ACTION_NAMES[min(act, 4)])
        self.step_count += 1

        # Track action distribution
        if act < 5:
            state["action_counts"][act] += 1

        # Apply reward shaping
        shaped = reward
        if not (term or trunc):
            if act == 0:
                shaped += reward_config["fold_penalty"]
            elif act in (3, 4):
                shaped += reward_config["raise_bonus"]
            shaped += reward_config["survival_bonus"]

        # On game end, log the game
        if term or trunc:
            game = {
                "actions": self.current_game_actions.copy(),
                "reward": float(reward),
                "result": "Win" if reward > 0 else ("Loss" if reward < 0 else "Draw"),
            }
            state["recent_games"].insert(0, game)
            state["recent_games"] = state["recent_games"][:20]
            self.current_game_actions = []

        return obs, shaped, term, trunc, info


# ──────────────────────────────────────────────
# Dashboard Callback
# ──────────────────────────────────────────────
class DashboardCallback(BaseCallback):
    """Sends real-time updates to the frontend during training."""

    def __init__(self, eval_freq=2000, n_eval_games=50):
        super().__init__(verbose=0)
        self.eval_freq = eval_freq
        self.n_eval_games = n_eval_games

    def _on_step(self):
        # Check for stop signal
        if stop_event.is_set():
            return False

        # Handle pause
        while not pause_event.is_set():
            if stop_event.is_set():
                return False
            time.sleep(0.1)

        state["step"] = self.num_timesteps
        state["epsilon"] = self.model.exploration_rate

        # Emit live stats every 200 steps
        if self.n_calls % 200 == 0:
            socketio.emit("training_update", {
                "step": self.num_timesteps,
                "epsilon": round(self.model.exploration_rate, 4),
                "action_counts": state["action_counts"],
                "recent_games": state["recent_games"][:5],
            })

        # Full evaluation periodically
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()

        return True

    def _evaluate(self):
        eval_env = TexasHoldemEnv()
        wins, total_reward = 0, 0.0

        for _ in range(self.n_eval_games):
            obs, _ = eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                with model_lock:
                    action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = eval_env.step(action)
                ep_reward += reward
                done = term or trunc
            total_reward += ep_reward
            if ep_reward > 0:
                wins += 1

        win_rate = wins / self.n_eval_games * 100
        avg_reward = total_reward / self.n_eval_games

        state["win_rate"] = round(win_rate, 1)
        state["avg_reward"] = round(avg_reward, 2)
        state["history"]["steps"].append(self.num_timesteps)
        state["history"]["win_rates"].append(round(win_rate, 1))
        state["history"]["avg_rewards"].append(round(avg_reward, 2))
        state["history"]["epsilons"].append(round(self.model.exploration_rate, 4))

        socketio.emit("eval_update", {
            "step": self.num_timesteps,
            "win_rate": state["win_rate"],
            "avg_reward": state["avg_reward"],
            "epsilon": round(self.model.exploration_rate, 4),
            "history": state["history"],
        })


# ──────────────────────────────────────────────
# Training Thread
# ──────────────────────────────────────────────
def training_worker(total_timesteps, hyperparams):
    """Runs training in a background thread."""
    try:
        state["is_running"] = True
        state["action_counts"] = [0, 0, 0, 0, 0]
        state["recent_games"] = []
        state["history"] = {"steps": [], "win_rates": [], "avg_rewards": [], "epsilons": []}
        stop_event.clear()
        pause_event.set()

        env = LoggingTexasHoldemEnv()
        state["env"] = env

        model = MaskedDQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=hyperparams.get("lr", 1e-4),
            buffer_size=hyperparams.get("buffer_size", 50000),
            batch_size=hyperparams.get("batch_size", 64),
            gamma=hyperparams.get("gamma", 0.99),
            exploration_fraction=hyperparams.get("exploration_fraction", 0.3),
            exploration_final_eps=hyperparams.get("exploration_final_eps", 0.05),
            target_update_interval=hyperparams.get("target_update_interval", 1000),
            learning_starts=hyperparams.get("learning_starts", 5000),
            train_freq=4,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
        )

        with model_lock:
            state["model"] = model

        callback = DashboardCallback(
            eval_freq=hyperparams.get("eval_freq", 2000),
            n_eval_games=hyperparams.get("n_eval_games", 50),
        )

        socketio.emit("status", {"status": "training", "message": "Training started"})

        model.learn(total_timesteps=total_timesteps, callback=callback)

        # Save final model
        os.makedirs("models", exist_ok=True)
        model.save("models/dqn_poker_final")

        socketio.emit("status", {"status": "completed", "message": "Training completed!"})

    except Exception as e:
        socketio.emit("status", {"status": "error", "message": str(e)})
    finally:
        state["is_running"] = False


# ──────────────────────────────────────────────
# Routes & Socket Events
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("start_training")
def handle_start(data):
    if state["is_running"]:
        socketio.emit("status", {"status": "error", "message": "Training already running"})
        return

    total = data.get("total_timesteps", 200000)
    hyperparams = data.get("hyperparams", {})

    # Update reward shaping config
    rs = data.get("reward_shaping", {})
    reward_config["fold_penalty"] = float(rs.get("fold_penalty", 0))
    reward_config["raise_bonus"] = float(rs.get("raise_bonus", 0))
    reward_config["survival_bonus"] = float(rs.get("survival_bonus", 0))

    thread = threading.Thread(target=training_worker, args=(total, hyperparams), daemon=True)
    thread.start()


@socketio.on("pause_training")
def handle_pause():
    if not state["is_running"]:
        return
    if pause_event.is_set():
        pause_event.clear()
        state["is_paused"] = True
        socketio.emit("status", {"status": "paused", "message": "Training paused"})
    else:
        pause_event.set()
        state["is_paused"] = False
        socketio.emit("status", {"status": "training", "message": "Training resumed"})


@socketio.on("stop_training")
def handle_stop():
    stop_event.set()
    pause_event.set()
    state["is_running"] = False
    state["is_paused"] = False
    socketio.emit("status", {"status": "stopped", "message": "Training stopped"})


@socketio.on("save_model")
def handle_save():
    if state["model"] is None:
        socketio.emit("status", {"status": "error", "message": "No model to save"})
        return
    try:
        os.makedirs("models", exist_ok=True)
        path = f"models/dqn_poker_step_{state['step']}"
        with model_lock:
            state["model"].save(path)
        socketio.emit("status", {"status": "saved", "message": f"Model saved to {path}"})
    except Exception as e:
        socketio.emit("status", {"status": "error", "message": str(e)})


@socketio.on("get_state")
def handle_get_state():
    socketio.emit("full_state", {
        "is_running": state["is_running"],
        "is_paused": state["is_paused"],
        "step": state["step"],
        "epsilon": state["epsilon"],
        "win_rate": state["win_rate"],
        "avg_reward": state["avg_reward"],
        "action_counts": state["action_counts"],
        "history": state["history"],
        "recent_games": state["recent_games"][:10],
    })


if __name__ == "__main__":
    print("\n🃏 Texas Hold'em DQN Training Dashboard")
    print("   Open http://localhost:5050 in your browser\n")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False, allow_unsafe_werkzeug=True)
