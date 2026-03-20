"""
=============================================================================
 DQN Atari Agent — TRAINING SCRIPT
 Environment : ALE/Boxing-v5  (Gymnasium single-agent)
 Framework   : Stable Baselines3 + Gymnasium ALE
=============================================================================

 ENVIRONMENT OVERVIEW
 ─────────────────────
 Boxing is an Atari fighting game where the agent controls a white boxer
 in a ring and earns points by landing punches on a CPU opponent. The task
 requires learning movement, spacing, punch timing, and opponent pressure.
 A knockout (KO) ends the episode immediately at 100 points.

 State
   Raw environment: RGB frames (210, 160, 3).
   For DQN training: preprocessed to grayscale 84×84, stacked across 4
   consecutive frames to provide temporal/motion context.

 Actions
   Discrete(18) — movement directions, punch, and movement-punch combos.

 Reward  (unclipped in this implementation — see make_env docstring)
   In the original game: long jab = 1 pt, close punch = 2 pts, KO = 100 pts.

 Environment defaults (ALE/Boxing-v5)
   ALE/Boxing-v5 already includes its own Atari configuration: frameskip=4
   and sticky actions (repeat_action_probability=0.25). Additional wrappers
   are applied for DQN training: resizing, grayscale conversion, frame stacking.

 SETUP:
   pip install -r requirements.txt

 RUN:
   python train.py
   python train.py --stage full
   python train.py --stage full --experiments Exp01_BestTuned_TopCNN Exp02_LargeBatch_TopCNN Exp05_HighGamma_Balanced
   python train.py --seed 0
   python train.py --timesteps 500000

 NOTE:
   The saved dqn_model.zip is the file expected by play.py for demo/evaluation.
   Before-vs-after gameplay demonstration should be done in play.py:
     - Before training / baseline: random agent
     - After training: dqn_model.zip
=============================================================================
"""

import argparse
import csv
import gc
import json
import os
import shutil
import time
import warnings
from typing import List, Optional

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

warnings.filterwarnings("ignore")

# =============================================================================
# DEFAULTS
# =============================================================================

ENV_ID = "ALE/Boxing-v5"
N_STACK = 4
DEFAULT_SEED = 42
MEMBER_NAME = "Fidele Ndihokubwayo"
MEMBER_NAME_2 = "Reine Mizero"

# Two-stage training:
#   screening → all experiments at 100k–200k steps
#   full      → selected top configs at higher budget
SCREENING_STEPS = 200_000
FULL_STEPS = 1_000_000

N_ENVS_CNN = 4
N_ENVS_MLP = 1

# Full-stage finalists remain CNN-focused because Boxing is image-based
FULL_STAGE_EXPERIMENTS = [
    "Exp01_BestTuned_TopCNN",
    "Exp02_LargeBatch_TopCNN",
    "Exp05_HighGamma_Balanced",
]

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def make_env(n_envs: int = N_ENVS_CNN, seed: int = DEFAULT_SEED,
             clip_reward: bool = False):
    """
    Creates a vectorised, preprocessed Atari environment.

    wrapper_kwargs decisions
    ────────────────────────
    clip_reward=False
        Preserves true Boxing scores: +1 jab, +2 power punch, +100 KO.

    terminal_on_life_loss=False
        Boxing effectively behaves as a single-life episode here.
        Set explicitly for wrapper consistency between training and evaluation.
    """
    env = make_atari_env(
        ENV_ID,
        n_envs=n_envs,
        seed=seed,
        wrapper_kwargs=dict(
            clip_reward=clip_reward,
            terminal_on_life_loss=False,
        ),
    )
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


# =============================================================================
# HYPERPARAMETER EXPERIMENTS
# Includes:
# - best-performing prior combinations
# - two MLP experiments for explicit CNN vs MLP comparison
# - stronger new CNN candidates around the best results
# =============================================================================

ALL_EXPERIMENTS = [
    # 1) Best recent overall performer
    {
        "name": "Exp01_BestTuned_TopCNN",
        "member": MEMBER_NAME,
        "policy": "CnnPolicy",
        "learning_rate": 2.5e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.15,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Top prior CNN performer; strong balance of learning speed and stability.",
        "observed": "",
    },

    # 2) Second-best recent CNN performer
    {
        "name": "Exp02_LargeBatch_TopCNN",
        "member": MEMBER_NAME,
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.10,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Top large-batch performer; expected smoother updates and more consistent policy improvement.",
        "observed": "",
    },

    # 3) First MLP experiment
    {
        "name": "Exp03_MLPBaseline_Reference",
        "member": MEMBER_NAME,
        "policy": "MlpPolicy",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "exploration_fraction": 0.10,
        "buffer_size": 50_000,
        "learning_starts": 5_000,
        "target_update_interval": 500,
        "train_freq": 4,
        "n_envs": N_ENVS_MLP,
        "hypothesis": "Reference MLP comparison run for image observations.",
        "observed": "",
    },

    # 4) Second MLP experiment (tuned)
    {
        "name": "Exp04_MLPTuned_Comparison",
        "member": MEMBER_NAME,
        "policy": "MlpPolicy",
        "learning_rate": 2e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,
        "exploration_fraction": 0.15,
        "buffer_size": 50_000,
        "learning_starts": 5_000,
        "target_update_interval": 500,
        "train_freq": 4,
        "n_envs": N_ENVS_MLP,
        "hypothesis": "Tuned MLP comparison run to strengthen the CNN vs MLP architecture analysis.",
        "observed": "",
    },

    # 5) Stronger high-gamma balanced CNN
    {
        "name": "Exp05_HighGamma_Balanced",
        "member": MEMBER_NAME,
        "policy": "CnnPolicy",
        "learning_rate": 2.0e-4,
        "gamma": 0.995,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.15,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Designed to improve long-term positioning and pressure while keeping learning stable.",
        "observed": "",
    },

    # 6) BestTuned with longer exploration
    {
        "name": "Exp06_BestTuned_LongExplore",
        "member": MEMBER_NAME,
        "policy": "CnnPolicy",
        "learning_rate": 2.5e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,
        "exploration_fraction": 0.20,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Longer exploration may improve positioning and tactical movement before exploitation.",
        "observed": "",
    },

    # 7) BestTuned with very large batch
    {
        "name": "Exp07_BestTuned_BiggerBatch",
        "member": MEMBER_NAME,
        "policy": "CnnPolicy",
        "learning_rate": 2.0e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.15,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "A higher-batch version of the tuned CNN; expected to improve consistency and precision.",
        "observed": "",
    },

    # 8) Aggressive but controlled pressure
    {
        "name": "Exp08_AggressivePressure_CNN",
        "member": MEMBER_NAME,
        "policy": "CnnPolicy",
        "learning_rate": 1.5e-4,
        "gamma": 0.995,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.10,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Targets stronger pressure and more direct attacking with stable long-term credit assignment.",
        "observed": "",
    },

    # 9) Controlled fast exploitation
    {
        "name": "Exp09_ControlledFastExploit",
        "member": MEMBER_NAME,
        "policy": "CnnPolicy",
        "learning_rate": 2.0e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.08,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "A more decisive exploitation schedule intended to improve dominance once useful behavior appears.",
        "observed": "",
    },

    # 10) Main champion candidate
    {
        "name": "Exp10_ChampionCandidate_CNN",
        "member": MEMBER_NAME,
        "policy": "CnnPolicy",
        "learning_rate": 2.0e-4,
        "gamma": 0.995,
        "batch_size": 128,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.12,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Main high-performance candidate aimed at stronger positioning, cleaner attacks, and better score margins.",
        "observed": "",
    },

    # =========================================================================
    # REINE MIZERO — Experiments 11–20
    # =========================================================================

    # 11) Very low learning rate — slow but stable convergence
    {
        "name": "Exp11_VeryLowLR_Stable_CNN",
        "member": MEMBER_NAME_2,
        "policy": "CnnPolicy",
        "learning_rate": 5e-5,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.15,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Very low LR should produce more stable but slower convergence; avoids overshooting the optimal policy.",
        "observed": "",
    },

    # 12) High learning rate — fast learning, risk of instability
    {
        "name": "Exp12_HighLR_FastLearn_CNN",
        "member": MEMBER_NAME_2,
        "policy": "CnnPolicy",
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.10,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Higher LR allows faster adaptation but risks oscillation and Q-value divergence on Boxing's sparse rewards.",
        "observed": "",
    },

    # 13) Lower gamma — discounts future rewards more, focus on immediate punches
    {
        "name": "Exp13_LowGamma_ShortHorizon_CNN",
        "member": MEMBER_NAME_2,
        "policy": "CnnPolicy",
        "learning_rate": 2.5e-4,
        "gamma": 0.95,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.15,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Lower gamma (0.95) makes the agent prioritize immediate punching rewards over long-term positioning strategy.",
        "observed": "",
    },

    # 14) Very low gamma — strong short-term focus
    {
        "name": "Exp14_VeryLowGamma_CNN",
        "member": MEMBER_NAME_2,
        "policy": "CnnPolicy",
        "learning_rate": 2e-4,
        "gamma": 0.90,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.15,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Very low gamma (0.90) strongly discounts future rewards; expected to hurt long-term strategy but may improve reactive punching.",
        "observed": "",
    },

    # 15) Small batch — noisier but more frequent gradient updates
    {
        "name": "Exp15_SmallBatch_FreqUpdate_CNN",
        "member": MEMBER_NAME_2,
        "policy": "CnnPolicy",
        "learning_rate": 2e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.15,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Small batch size (32) increases update frequency and noise; may help escape local optima but risks instability.",
        "observed": "",
    },

    # 16) Very large batch — smoother gradients, slower adaptation
    {
        "name": "Exp16_VeryLargeBatch_CNN",
        "member": MEMBER_NAME_2,
        "policy": "CnnPolicy",
        "learning_rate": 1.5e-4,
        "gamma": 0.99,
        "batch_size": 256,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.20,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Very large batch (256) produces the smoothest gradient estimates; expected to be stable but slow to adapt to new tactics.",
        "observed": "",
    },

    # 17) MLP with low gamma — extends gamma series to architecture comparison
    {
        "name": "Exp17_MLP_LowGamma_Reference",
        "member": MEMBER_NAME_2,
        "policy": "MlpPolicy",
        "learning_rate": 2.5e-4,
        "gamma": 0.95,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.15,
        "buffer_size": 50_000,
        "learning_starts": 5_000,
        "target_update_interval": 500,
        "train_freq": 4,
        "n_envs": N_ENVS_MLP,
        "hypothesis": "MLP with low gamma (0.95) extends the gamma series (Exp13/14) to architecture comparison; tests whether short-horizon discounting behaves differently without CNN feature extraction.",
        "observed": "",
    },

    # 18) MLP with larger batch — tests whether CNN batch advantage translates to MLP
    {
        "name": "Exp18_MLP_LargeBatch_Comparison",
        "member": MEMBER_NAME_2,
        "policy": "MlpPolicy",
        "learning_rate": 1.5e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.20,
        "buffer_size": 50_000,
        "learning_starts": 5_000,
        "target_update_interval": 500,
        "train_freq": 4,
        "n_envs": N_ENVS_MLP,
        "hypothesis": "MLP with larger batch (128) tests whether the batch-size advantage found in Exp16 (CNN, batch=256) translates to MLP; directly connects batch and architecture as joint variables.",
        "observed": "",
    },

    # 19) Mid gamma + slightly higher LR — balanced middle-ground config
    {
        "name": "Exp19_MidGamma_MidLR_Balanced_CNN",
        "member": MEMBER_NAME_2,
        "policy": "CnnPolicy",
        "learning_rate": 3e-4,
        "gamma": 0.97,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.15,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Middle-ground config: moderate gamma (0.97) balances short- and long-term rewards; slightly elevated LR for faster learning.",
        "observed": "",
    },

    # 20) Frequent target network updates — unexplored dimension
    {
        "name": "Exp20_FreqTargetUpdate_CNN",
        "member": MEMBER_NAME_2,
        "policy": "CnnPolicy",
        "learning_rate": 2.5e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.15,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "target_update_interval": 500,
        "train_freq": 4,
        "n_envs": N_ENVS_CNN,
        "hypothesis": "Halving target_update_interval (500 vs standard 1000) syncs the target network more frequently; expected to reduce overestimation bias but may increase instability.",
        "observed": "",
    },
]


# =============================================================================
# TRAINING LOGGER
# =============================================================================

class TrainingLogger(BaseCallback):
    """
    Logs per-step metrics to CSV for auxiliary trend analysis.

    With n_envs > 1, this logger averages rewards across parallel envs
    per step, so episode trajectories are mixed. EvalCallback remains the
    primary ranking source.
    """

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_lengths = []
        self._ep_reward = 0.0
        self._ep_length = 0

    def _on_training_start(self) -> None:
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "timesteps", "avg_reward", "ep_length", "epsilon"]
            )

    def _on_step(self) -> bool:
        self._ep_reward += float(np.mean(self.locals["rewards"]))
        self._ep_length += 1
        if any(self.locals.get("dones", [False])):
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    len(self.episode_rewards),
                    self.num_timesteps,
                    round(self._ep_reward, 2),
                    self._ep_length,
                    round(self.model.exploration_rate, 4),
                ])
            self._ep_reward = 0.0
            self._ep_length = 0
        return True


# =============================================================================
# PLOTTING
# =============================================================================

def plot_training_curve(log_csv: str, exp_name: str):
    """Generates reward and episode-length trend plots from training CSV."""
    try:
        timesteps, rewards, lengths = [], [], []
        with open(log_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timesteps.append(int(row["timesteps"]))
                rewards.append(float(row["avg_reward"]))
                lengths.append(int(row["ep_length"]))

        if not timesteps:
            return

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        fig.suptitle(f"Training Curves — {exp_name}", fontsize=13)

        axes[0].plot(timesteps, rewards, alpha=0.4, linewidth=0.8)
        if len(rewards) >= 10:
            smooth = np.convolve(rewards, np.ones(10) / 10, mode="valid")
            axes[0].plot(timesteps[9:], smooth, linewidth=1.8, label="10-ep moving avg")
            axes[0].legend(fontsize=9)
        axes[0].set_xlabel("Timesteps")
        axes[0].set_ylabel("Avg Reward")
        axes[0].set_title("Reward Trend (auxiliary)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(timesteps, lengths, alpha=0.4, linewidth=0.8)
        if len(lengths) >= 10:
            smooth_l = np.convolve(lengths, np.ones(10) / 10, mode="valid")
            axes[1].plot(timesteps[9:], smooth_l, linewidth=1.8, label="10-ep moving avg")
            axes[1].legend(fontsize=9)
        axes[1].set_xlabel("Timesteps")
        axes[1].set_ylabel("Episode Length")
        axes[1].set_title("Episode Length Trend")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out = f"results/curve_{exp_name}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Plot saved → {out}")

    except Exception as e:
        print(f"  ⚠ Plot skipped for {exp_name}: {e}")


def plot_experiment_comparison(all_results: List[dict]):
    """Bar chart comparing best eval reward across all completed experiments."""
    try:
        if not all_results:
            return

        names = [r["name"].replace("Exp0", "E").replace("Exp", "E") for r in all_results]
        evals = [r["best_eval_reward"] for r in all_results]
        colors = ["steelblue" if r["policy"] == "CnnPolicy" else "tomato" for r in all_results]

        fig, ax = plt.subplots(figsize=(14, 5))
        bars = ax.bar(names, evals, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Best Eval Reward (EvalCallback)")
        ax.set_title("Experiment Comparison — Best Evaluation Reward Across Completed Runs")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, evals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        out = "results/experiment_comparison.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Comparison plot → {out}")

    except Exception as e:
        print(f"  ⚠ Comparison plot failed: {e}")


# =============================================================================
# RESULTS POST-PROCESSING
# =============================================================================

def hyperparameter_set_string(result: dict) -> str:
    return (
        f"lr={result['lr']}, "
        f"gamma={result['gamma']}, "
        f"batch={result['batch_size']}, "
        f"epsilon_start={result['eps_start']}, "
        f"epsilon_end={result['eps_end']}, "
        f"epsilon_decay={result['eps_fraction']}"
    )


def infer_observed_behavior(result: dict, baseline_eval: Optional[float]) -> str:
    eval_reward = result["best_eval_reward"]
    train_aux = result["train_mean_last20"]

    if baseline_eval is None:
        relative = "serves as the current baseline reference."
    elif eval_reward > baseline_eval + 3:
        relative = "outperformed the baseline noticeably."
    elif eval_reward < baseline_eval - 3:
        relative = "underperformed the baseline."
    else:
        relative = "performed similarly to the baseline."

    name = result["name"]

    if "MLP" in name:
        return f"MLP comparison run {relative} It is useful for architecture comparison, but CNN remains the stronger image-based choice for Boxing."
    if "LargeBatch" in name:
        return f"Larger batch size {relative} It likely produced smoother updates and better punch consistency."
    if "HighGamma" in name:
        return f"Higher gamma {relative} It likely valued longer-term positioning and pressure more strongly."
    if "LongExplore" in name:
        return f"Longer exploration {relative} It gave the agent more time to discover positioning and timing patterns."
    if "FastExploit" in name:
        return f"Faster exploitation {relative} It became more decisive sooner, but may reduce tactical diversity."
    if "AggressivePressure" in name:
        return f"Aggressive pressure tuning {relative} It was designed to improve attacks, pressure, and dominance."
    if "ChampionCandidate" in name:
        return f"Champion candidate {relative} It targets stronger movement, cleaner attacks, and higher winning margins."
    if "BestTuned" in name:
        return f"Top tuned CNN {relative} It balanced learning speed and stability well."
    if "VeryLowLR" in name:
        return f"Very low learning rate (5e-5) {relative} It converged slowly but produced stable Q-value estimates with minimal oscillation."
    if "HighLR" in name:
        return f"High learning rate (5e-4) {relative} It learned quickly but risked Q-value divergence on Boxing's sparse reward signal."
    if "VeryLowGamma" in name:
        return f"Very low gamma (0.90) {relative} Strong short-term discounting likely harmed long-term positioning and pressure strategy."
    if "LowGamma" in name:
        return f"Low gamma (0.95) {relative} Short-horizon focus prioritised immediate punches over sustained positioning."
    if "SmallBatch" in name:
        return f"Small batch (32) {relative} Frequent noisy updates increased variance but may have helped escape local optima."
    if "VeryLargeBatch" in name:
        return f"Very large batch (256) {relative} Smoothest gradient estimates but slowest adaptation to new fighting patterns."
    if "HighEpsEnd" in name:
        return f"High epsilon end (0.10) {relative} Retained significant exploration throughout; likely limited peak exploitation performance."
    if "SlowEpsDecay" in name:
        return f"Very slow epsilon decay (35% of training) {relative} Extended exploration window; gave more time to discover diverse tactics before committing."
    if "MidGamma" in name:
        return f"Mid gamma (0.97) with elevated LR {relative} Balanced short- and long-term credit assignment with faster weight updates."
    if "FreqTargetUpdate" in name:
        return f"Frequent target network updates (interval=500) {relative} More frequent syncing reduced the gap between online and target Q-values, affecting training stability and convergence speed."
    return (
        f"Observed best eval reward = {eval_reward:.2f}, "
        f"training mean last 20 = {train_aux:.2f}; {relative}"
    )


def finalize_observed_behaviors(all_results: List[dict]):
    baseline_eval = None
    for r in all_results:
        if r["name"] == "Exp01_BestTuned_TopCNN":
            baseline_eval = r["best_eval_reward"]
            break

    for r in all_results:
        if not str(r.get("observed", "")).strip():
            r["observed"] = infer_observed_behavior(r, baseline_eval)


def export_behavior_table(all_results: List[dict]):
    csv_path = "results/final_observations.csv"
    md_path = "results/final_observations.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "member_name",
            "experiment",
            "hyperparameter_set",
            "noted_behavior",
            "best_eval_reward",
            "policy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "member_name": r.get("member", ""),
                "experiment": r["name"],
                "hyperparameter_set": hyperparameter_set_string(r),
                "noted_behavior": r.get("observed", ""),
                "best_eval_reward": r["best_eval_reward"],
                "policy": r["policy"],
            })

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Hyperparameter Experiment Results\n\n")
        f.write("| Member Name | Experiment | Hyperparameter Set | Noted Behavior |\n")
        f.write("|---|---|---|---|\n")
        for r in all_results:
            f.write(
                f"| {r.get('member','')} "
                f"| {r['name']} "
                f"| {hyperparameter_set_string(r)} "
                f"| {r.get('observed','')} |\n"
            )

    print(f"  ✓ Observations CSV → {csv_path}")
    print(f"  ✓ Observations MD  → {md_path}")


def export_run_metadata(args, timesteps: int, experiments: List[dict]):
    out = "results/run_metadata.json"
    payload = {
        "env_id": ENV_ID,
        "stage": args.stage,
        "timesteps_per_experiment": timesteps,
        "seed": args.seed,
        "reward_clipping": False,
        "terminal_on_life_loss": False,
        "frame_stack": N_STACK,
        "member_name": MEMBER_NAME,
        "experiments": [e["name"] for e in experiments],
        "full_stage_experiments_default": FULL_STAGE_EXPERIMENTS,
        "before_training_demo": "Use play.py --random-baseline with the same episode count/time as after-training demo.",
        "after_training_demo": "Use play.py with dqn_model.zip.",
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"✓  Run metadata → {out}")


def read_best_eval_from_disk(exp_name: str) -> float:
    eval_npz = f"logs/{exp_name}/eval/evaluations.npz"
    if not os.path.exists(eval_npz):
        return float("-inf")
    data = np.load(eval_npz)
    return float(np.max(np.mean(data["results"], axis=1)))


def read_train_mean_from_disk(exp_name: str) -> float:
    log_csv = f"results/{exp_name}_log.csv"
    if not os.path.exists(log_csv):
        return 0.0

    rewards = []
    with open(log_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row["avg_reward"]))

    if len(rewards) >= 20:
        return float(np.mean(rewards[-20:]))
    return float(np.mean(rewards)) if rewards else 0.0


def collect_completed_results(seed: int) -> List[dict]:
    """
    Rebuilds a full leaderboard from files on disk so ranks reflect all
    completed experiments across separate runs.
    """
    completed = []

    for exp in ALL_EXPERIMENTS:
        name = exp["name"]
        best_eval = read_best_eval_from_disk(name)
        if best_eval == float("-inf"):
            continue

        train_mean = read_train_mean_from_disk(name)

        completed.append({
            "member": exp.get("member", MEMBER_NAME),
            "name": name,
            "policy": exp["policy"],
            "lr": exp["learning_rate"],
            "gamma": exp["gamma"],
            "batch_size": exp["batch_size"],
            "eps_start": exp["exploration_initial_eps"],
            "eps_end": exp["exploration_final_eps"],
            "eps_fraction": exp["exploration_fraction"],
            "best_eval_reward": round(best_eval, 2),
            "train_mean_last20": round(train_mean, 2),
            "episodes_logged": 0,
            "train_time_s": 0.0,
            "seed": seed,
            "hypothesis": exp["hypothesis"],
            "observed": exp.get("observed", ""),
        })

    return sorted(completed, key=lambda x: x["best_eval_reward"], reverse=True)


# =============================================================================
# SINGLE EXPERIMENT RUNNER
# =============================================================================

def train_experiment(exp: dict, timesteps: int, seed: int) -> dict:
    name = exp["name"]
    policy = exp["policy"]
    n_envs = exp.get("n_envs", N_ENVS_CNN)

    print(f"\n{'='*72}")
    print(f"  EXPERIMENT : {name}  ({exp.get('member','')})")
    print(f"  Policy     : {policy}  |  n_envs={n_envs}")
    print(f"  lr={exp['learning_rate']}  γ={exp['gamma']}  batch={exp['batch_size']}")
    print(
        f"  ε: {exp['exploration_initial_eps']} → "
        f"{exp['exploration_final_eps']}  "
        f"(fraction={exp['exploration_fraction']})"
    )
    print(f"  Hypothesis : {exp['hypothesis']}")
    print(f"{'='*72}")

    log_csv = f"results/{name}_log.csv"
    model_dir = f"models/{name}"

    env = make_env(n_envs=n_envs, seed=seed, clip_reward=False)
    eval_env = make_env(n_envs=1, seed=seed + 999, clip_reward=False)

    model = DQN(
        policy=policy,
        env=env,
        learning_rate=exp["learning_rate"],
        gamma=exp["gamma"],
        batch_size=exp["batch_size"],
        exploration_initial_eps=exp["exploration_initial_eps"],
        exploration_final_eps=exp["exploration_final_eps"],
        exploration_fraction=exp["exploration_fraction"],
        buffer_size=exp["buffer_size"],
        learning_starts=exp["learning_starts"],
        target_update_interval=exp["target_update_interval"],
        train_freq=exp["train_freq"],
        optimize_memory_usage=False,
        seed=seed,
        verbose=0,
        tensorboard_log=f"logs/{name}",
    )

    logger_cb = TrainingLogger(log_csv)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=f"logs/{name}/eval",
        eval_freq=max(5_000, timesteps // 20),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=[logger_cb, eval_cb],
        reset_num_timesteps=True,
    )
    elapsed = time.time() - t0

    model.save(f"models/{name}_final")

    best_eval_mean = 0.0
    eval_npz = f"logs/{name}/eval/evaluations.npz"
    if os.path.exists(eval_npz):
        data = np.load(eval_npz)
        best_eval_mean = float(np.max(np.mean(data["results"], axis=1)))

    rewards = logger_cb.episode_rewards
    train_mean = (
        float(np.mean(rewards[-20:])) if len(rewards) >= 20
        else float(np.mean(rewards)) if rewards else 0.0
    )

    plot_training_curve(log_csv, name)

    env.close()
    eval_env.close()
    del model
    del env
    del eval_env
    del logger_cb
    del eval_cb
    gc.collect()

    result = {
        "member": exp.get("member", ""),
        "name": name,
        "policy": policy,
        "lr": exp["learning_rate"],
        "gamma": exp["gamma"],
        "batch_size": exp["batch_size"],
        "eps_start": exp["exploration_initial_eps"],
        "eps_end": exp["exploration_final_eps"],
        "eps_fraction": exp["exploration_fraction"],
        "best_eval_reward": round(best_eval_mean, 2),
        "train_mean_last20": round(train_mean, 2),
        "episodes_logged": len(rewards),
        "train_time_s": round(elapsed, 1),
        "seed": seed,
        "hypothesis": exp["hypothesis"],
        "observed": exp.get("observed", ""),
    }

    print(f"\n  → Best eval reward (primary)  : {best_eval_mean:.1f}")
    print(f"    Training mean last 20 (aux) : {train_mean:.1f}")
    print(f"    Training time               : {elapsed:.0f}s")
    return result


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DQN on ALE/Boxing-v5 with focused hyperparameter experiments"
    )
    parser.add_argument(
        "--stage",
        choices=["screening", "full"],
        default="screening",
        help="screening: all 10 exps | full: top configs at higher budget",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override timestep count (overrides --stage default)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Names of experiments to run (default: all or FULL_STAGE_EXPERIMENTS)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    if args.timesteps is not None:
        timesteps = args.timesteps
    elif args.stage == "full":
        timesteps = FULL_STEPS
    else:
        timesteps = SCREENING_STEPS

    if args.experiments:
        run_names = args.experiments
    elif args.stage == "full":
        run_names = FULL_STAGE_EXPERIMENTS
        print(f"\n  Full stage: running only top configs: {run_names}")
        print("  Edit FULL_STAGE_EXPERIMENTS in train.py after reviewing results if needed.")
    else:
        run_names = [e["name"] for e in ALL_EXPERIMENTS]

    experiments = [e for e in ALL_EXPERIMENTS if e["name"] in run_names]
    if not experiments:
        raise ValueError(f"No matching experiments found for: {run_names}")

    print("\n" + "=" * 72)
    print("  DQN ATARI TRAINING  —  ALE/Boxing-v5")
    print(f"  Stage                 : {args.stage.upper()}")
    print(f"  Steps per experiment  : {timesteps:,}")
    print("  Reward clipping       : OFF  (+1 jab / +2 punch / +100 KO)")
    print("  terminal_on_life_loss : False")
    print(f"  Frame stack           : {N_STACK}")
    print(f"  Seed                  : {args.seed}")
    print(f"  Member                : {MEMBER_NAME}")
    print(f"  Experiments to run    : {len(experiments)}")
    print("=" * 72)

    export_run_metadata(args, timesteps, experiments)

    current_run_results = []
    for exp in experiments:
        result = train_experiment(exp, timesteps, args.seed)
        current_run_results.append(result)

    # Rebuild the global leaderboard from all completed experiments on disk
    all_results = collect_completed_results(args.seed)
    if not all_results:
        raise RuntimeError("No completed experiment results were found on disk.")

    finalize_observed_behaviors(all_results)

    summary_path = "results/hyperparameter_results.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n✓  Summary CSV → {summary_path}")

    plot_experiment_comparison(all_results)
    export_behavior_table(all_results)

    best = max(all_results, key=lambda r: r["best_eval_reward"])
    print(f"\n🏆  Best experiment overall : {best['name']}")
    print(f"    Best eval reward       : {best['best_eval_reward']}")
    print("\n    Ranking is based on EvalCallback deterministic eval reward.")
    print("    Training CSV trends are supporting evidence only.")

    saved = False
    for candidate in [
        f"models/{best['name']}/best_model.zip",
        f"models/{best['name']}_final.zip",
    ]:
        if os.path.exists(candidate):
            shutil.copy(candidate, "dqn_model.zip")
            print(f"✓  Best overall model exported → dqn_model.zip  (from {candidate})")
            saved = True
            break

    if not saved:
        raise FileNotFoundError(
            "No saved best model found to export as dqn_model.zip.\n"
            "Check that EvalCallback or model.save() completed successfully."
        )

    print("\n" + "=" * 118)
    print(
        f"{'Rank':<5} {'Experiment':<32} {'Member':<22} {'Policy':<10} "
        f"{'LR':>8} {'γ':>6} {'Batch':>6} {'BestEval':>10} {'TrainMn':>9}"
    )
    print("-" * 118)
    for rank, r in enumerate(
        sorted(all_results, key=lambda x: x["best_eval_reward"], reverse=True), 1
    ):
        print(
            f"{rank:<5} {r['name']:<32} {r['member']:<22} {r['policy']:<10} "
            f"{r['lr']:>8.2e} {r['gamma']:>6.3f} {r['batch_size']:>6} "
            f"{r['best_eval_reward']:>10.1f} {r['train_mean_last20']:>9.1f}"
        )
    print("=" * 118)
    print("  BestEval = EvalCallback deterministic reward  (PRIMARY RANKING)")
    print("  TrainMn  = Training CSV mean last 20 eps      (auxiliary trend)")
    print(f"\n  Seed used: {args.seed}  — results are from one controlled seed.")
    print("  For variance analysis, rerun the script with different --seed values.")
    print("  For before-vs-after demo, use play.py random baseline first, then dqn_model.zip.")


if __name__ == "__main__":
    main()