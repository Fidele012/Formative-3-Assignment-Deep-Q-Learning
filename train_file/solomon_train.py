import gymnasium as gym
import ale_py
from stable_baselines3 import DQN

gym.register_envs(ale_py)

env = gym.make("ALE/Boxing-v5")

model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
)

print("Train file is ready")

model.learn(total_timesteps=10000)
model.save("dqn_boxing_model")