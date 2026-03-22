"""
=============================================================================
 DQN Atari Agent — PLAYING / EVALUATION SCRIPT
 Environment : ALE/Boxing-v5  (Gymnasium single-agent)
 Policy      : Greedy evaluation via deterministic=True
 Framework   : Stable Baselines3 + Gymnasium ALE
=============================================================================

 ENVIRONMENT
   Boxing is an Atari fighting game. The agent controls a white boxer
   and must earn points by landing punches on a CPU opponent. The task
   requires learning movement, spacing, punch timing, and pressure.
   A KO ends the episode at 100 points.

   Raw environment: RGB frames (210, 160, 3).
   For DQN evaluation: preprocessed to grayscale 84×84, stacked across
   4 consecutive frames for temporal/motion context.

 IMPORTANT — wrapper settings must mirror train.py exactly:
   clip_reward           = False      (preserves true Boxing scores)
   terminal_on_life_loss = False
   N_STACK               = 4
   ENV_ID                = "ALE/Boxing-v5"

 RUN:
   python play.py                               # trained agent, 5 episodes, GUI
   python play.py --episodes 10                 # trained agent, custom count
   python play.py --model dqn_model             # explicit model path (no .zip)
   python play.py --no-render                   # headless mode
   python play.py --record                      # save .mp4 to ./videos/
   python play.py --demo                        # single trained-agent demo
   python play.py --random-baseline             # random agent only
   python play.py --compare                     # random baseline + trained agent
   python play.py --compare --episodes 3        # same-conditions before/after

 NOTE:
   Gymnasium 1.x requires ALE environment registration in code:
     import ale_py
     gym.register_envs(ale_py)
=============================================================================
"""

import argparse
import json
import os
import time
import warnings
from typing import Dict, List

import numpy as np
import gymnasium as gym
import ale_py

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

warnings.filterwarnings("ignore")

# Register ALE environments for Gymnasium 1.x
gym.register_envs(ale_py)

# =============================================================================
# CONSTANTS — must match train.py exactly
# =============================================================================

ENV_ID = "ALE/Boxing-v5"
N_STACK = 4
SEED = 42


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def make_eval_env(render_mode: str = None):
    """
    Builds a single evaluation environment.

    Wrapper settings deliberately mirror train.py:
      clip_reward=False           preserve true Boxing scores (+1/+2/+100)
      terminal_on_life_loss=False consistent with training setup
      n_envs=1                    single env for clean per-episode tracking

    ALE/Boxing-v5 already includes its own Atari configuration
    (frameskip=4, sticky actions). Additional preprocessing wrappers are
    applied here: resizing, grayscale conversion, and frame stacking.
    """
    env_kwargs = {}
    if render_mode:
        env_kwargs["render_mode"] = render_mode

    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=SEED,
        env_kwargs=env_kwargs,
        wrapper_kwargs=dict(
            clip_reward=False,
            terminal_on_life_loss=False,
        ),
    )
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


# =============================================================================
# HELPERS
# =============================================================================

def outcome_label(ep_reward: float) -> str:
    if ep_reward > 0:
        return "WIN   🥊"
    elif ep_reward == 0:
        return "DRAW  🤝"
    else:
        return "LOSS  😞"


def save_json(payload: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def summarize_scores(agent_name: str, rewards: List[float], lengths: List[int]) -> Dict:
    return {
        "agent": agent_name,
        "env": ENV_ID,
        "episodes": len(rewards),
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "min_reward": round(float(np.min(rewards)), 2),
        "max_reward": round(float(np.max(rewards)), 2),
        "mean_ep_length": round(float(np.mean(lengths)), 1),
        "wins": sum(1 for r in rewards if r > 0),
        "draws": sum(1 for r in rewards if r == 0),
        "losses": sum(1 for r in rewards if r < 0),
        "per_episode": [
            {
                "episode": i + 1,
                "reward": round(rewards[i], 2),
                "length": lengths[i],
            }
            for i in range(len(rewards))
        ],
    }


def export_comparison(trained_stats: dict, random_stats: dict):
    comparison = {
        "environment": ENV_ID,
        "same_conditions": trained_stats.get("episodes") == random_stats.get("episodes"),
        "trained_agent": {
            "mean_reward": trained_stats.get("mean_reward"),
            "std_reward": trained_stats.get("std_reward"),
            "wins": trained_stats.get("wins"),
            "draws": trained_stats.get("draws"),
            "losses": trained_stats.get("losses"),
            "mean_ep_length": trained_stats.get("mean_ep_length"),
        },
        "random_baseline": {
            "mean_reward": random_stats.get("mean_reward"),
            "std_reward": random_stats.get("std_reward"),
            "wins": random_stats.get("wins"),
            "draws": random_stats.get("draws"),
            "losses": random_stats.get("losses"),
            "mean_ep_length": random_stats.get("mean_ep_length"),
        },
        "difference_mean_reward": round(
            float(trained_stats.get("mean_reward", 0.0)) - float(random_stats.get("mean_reward", 0.0)),
            2,
        ),
        "difference_wins": int(trained_stats.get("wins", 0)) - int(random_stats.get("wins", 0)),
    }
    out = "results/agent_comparison.json"
    save_json(comparison, out)
    print(f"  Comparison saved → {out}")


# =============================================================================
# TRAINED AGENT EVALUATION
# =============================================================================

def evaluate_agent(
    model_path: str,
    n_episodes: int = 5,
    render: bool = True,
    record_video: bool = False,
    video_folder: str = "videos",
) -> dict:
    """
    Loads the trained DQN model and evaluates it for n_episodes.

    Greedy evaluation in SB3 is implemented via:
        action, _ = model.predict(obs, deterministic=True)

    With deterministic=True, DQN selects the action with the highest
    predicted Q-value at each step.
    """
    model_path_zip = model_path if model_path.endswith(".zip") else model_path + ".zip"
    if not os.path.exists(model_path_zip):
        raise FileNotFoundError(
            f"\n[ERROR] Model not found: {model_path_zip}\n"
            "Run train.py first to generate dqn_model.zip"
        )

    print(f"\n{'='*64}")
    print("  DQN BOXING EVALUATION  —  ALE/Boxing-v5")
    print(f"{'='*64}")
    print(f"  Model      : {model_path_zip}")
    print(f"  Episodes   : {n_episodes}")
    print(f"  Render     : {render}")
    print(f"  Record     : {record_video}")
    print("  Policy     : Greedy evaluation via deterministic=True")
    print("               a = argmax_a Q(s, a; θ)  |  18 actions")
    print("  Rewards    : unclipped  (+1 jab / +2 power punch / +100 KO)")
    print(f"{'='*64}\n")

    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        from stable_baselines3.common.vec_env import VecVideoRecorder
        env = make_eval_env(render_mode="rgb_array")
        env = VecVideoRecorder(
            env,
            video_folder=video_folder,
            record_video_trigger=lambda step: step == 0,
            video_length=20_000,
            name_prefix="boxing_dqn",
        )
    else:
        try:
            env = make_eval_env(render_mode="human" if render else None)
        except Exception as e:
            print(f"  ⚠ Render mode failed ({e}). Falling back to headless.")
            print("    Use --no-render to suppress this warning.")
            env = make_eval_env(render_mode=None)
            render = False

    model = DQN.load(model_path_zip, env=env)
    model.exploration_rate = 0.0

    print(f"  ✓ Model loaded from {model_path_zip}")
    print(f"  ✓ Greedy evaluation: deterministic=True | ε={model.exploration_rate}\n")

    episode_rewards = []
    episode_lengths = []

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        ep_length = 0
        done = False
        t_start = time.time()

        print(f"  Episode {ep}/{n_episodes} ", end="", flush=True)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            ep_reward += float(reward[0])
            ep_length += 1

            if render and not record_video:
                time.sleep(0.016)

            done = bool(done[0]) if hasattr(done, "__len__") else bool(done)

        elapsed = time.time() - t_start
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        print(
            f"→ Score: {ep_reward:+7.1f}  |  "
            f"Steps: {ep_length:5d}  |  "
            f"Time: {elapsed:.1f}s  |  {outcome_label(ep_reward)}"
        )

    env.close()

    if record_video:
        print(f"\n  ✓ Video saved → {video_folder}/")
        saved = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
        for vf in saved:
            print(f"    {os.path.join(video_folder, vf)}")

    stats = summarize_scores("trained_dqn", episode_rewards, episode_lengths)
    stats["model"] = model_path_zip
    stats["reward_clipping"] = False
    stats["greedy_policy"] = "deterministic=True"

    print(f"\n{'─'*62}")
    print("  EVALUATION SUMMARY  —  Greedy (deterministic=True)")
    print(f"{'─'*62}")
    print(f"  Episodes   : {stats['episodes']}")
    print(f"  Mean Score : {stats['mean_reward']:>+8.2f}  ±{stats['std_reward']:.2f}")
    print(f"  Min / Max  : {stats['min_reward']:>+8.2f}  /  {stats['max_reward']:>+.2f}")
    print(f"  Avg Length : {stats['mean_ep_length']:>8.1f} steps")
    print(f"  Record     : {stats['wins']}W  {stats['draws']}D  {stats['losses']}L")
    print(f"{'─'*62}\n")

    return stats


# =============================================================================
# RANDOM BASELINE
# =============================================================================

def run_random_baseline(n_episodes: int = 5, render: bool = True) -> dict:
    """
    Runs a random agent for comparison against the trained agent.
    Useful for before/after demonstration in presentation.
    """
    print(f"\n{'='*64}")
    print("  RANDOM BASELINE  —  ALE/Boxing-v5")
    print("  Action selection: uniform random over Discrete(18)")
    print(f"{'='*64}\n")

    try:
        env = make_eval_env(render_mode="human" if render else None)
    except Exception as e:
        print(f"  ⚠ Render mode failed ({e}). Falling back to headless.")
        env = make_eval_env(render_mode=None)
        render = False

    episode_rewards = []
    episode_lengths = []

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        ep_length = 0
        done = False
        t_start = time.time()

        print(f"  Episode {ep}/{n_episodes} ", end="", flush=True)

        while not done:
            action = np.array([env.action_space.sample()])
            obs, reward, done, info = env.step(action)

            ep_reward += float(reward[0])
            ep_length += 1

            if render:
                time.sleep(0.016)

            done = bool(done[0]) if hasattr(done, "__len__") else bool(done)

        elapsed = time.time() - t_start
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        print(
            f"→ Score: {ep_reward:+7.1f}  |  "
            f"Steps: {ep_length:5d}  |  "
            f"Time: {elapsed:.1f}s  |  {outcome_label(ep_reward)}"
        )

    env.close()

    stats = summarize_scores("random_baseline", episode_rewards, episode_lengths)

    print(f"\n{'─'*62}")
    print("  RANDOM BASELINE SUMMARY")
    print(f"{'─'*62}")
    print(f"  Episodes   : {stats['episodes']}")
    print(f"  Mean Score : {stats['mean_reward']:>+8.2f}  ±{stats['std_reward']:.2f}")
    print(f"  Min / Max  : {stats['min_reward']:>+8.2f}  /  {stats['max_reward']:>+.2f}")
    print(f"  Avg Length : {stats['mean_ep_length']:>8.1f} steps")
    print(f"  Record     : {stats['wins']}W  {stats['draws']}D  {stats['losses']}L")
    print(f"{'─'*62}\n")

    return stats


# =============================================================================
# COMPARE BEFORE VS AFTER
# =============================================================================

def run_compare(model_path: str, n_episodes: int, render: bool):
    """
    Runs the random baseline first (before training proxy), then the trained
    model under the same episode count and rendering conditions.
    """
    print("\n" + "=" * 72)
    print("  BEFORE VS AFTER COMPARISON  —  SAME CONDITIONS")
    print(f"  Episodes per agent : {n_episodes}")
    print(f"  Render             : {render}")
    print("=" * 72)

    random_stats = run_random_baseline(n_episodes=n_episodes, render=render)
    save_json(random_stats, "results/random_baseline_results.json")
    print("  Results saved → results/random_baseline_results.json\n")

    trained_stats = evaluate_agent(
        model_path=model_path,
        n_episodes=n_episodes,
        render=render,
        record_video=False,
    )
    save_json(trained_stats, "results/play_results.json")
    print("  Results saved → results/play_results.json")

    export_comparison(trained_stats, random_stats)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent on ALE/Boxing-v5"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dqn_model",
        help="Model path without .zip (default: dqn_model)",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Headless mode — no GUI window",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record gameplay to ./videos/ (records first rollout)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Single trained-agent demo episode",
    )
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help="Run random agent only",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run same-condition before-vs-after comparison: random baseline then trained model",
    )
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    if args.compare:
        run_compare(
            model_path=args.model,
            n_episodes=1 if args.demo else args.episodes,
            render=not args.no_render,
        )
        return

    if args.random_baseline:
        stats = run_random_baseline(
            n_episodes=args.episodes,
            render=not args.no_render,
        )
        out = "results/random_baseline_results.json"
        save_json(stats, out)
        print(f"  Results saved → {out}")
        return

    stats = evaluate_agent(
        model_path=args.model,
        n_episodes=1 if args.demo else args.episodes,
        render=not args.no_render,
        record_video=args.record,
    )

    out = "results/play_results.json"
    save_json(stats, out)
    print(f"  Results saved → {out}")


if __name__ == "__main__":
    main()