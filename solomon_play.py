import gymnasium as gym
import ale_py
from stable_baselines3 import DQN

gym.register_envs(ale_py)

env = gym.make("ALE/Boxing-v5", render_mode="human")

model = DQN.load("dqn_boxing_model")

obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    if done or truncated:
        obs, _ = env.reset()

env.close()