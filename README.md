# Formative-3-Assignment-Deep-Q-Learning
Using Stable Baselines 3 and Gymnasium to train and evaluate a Deep Q-network (DQN) agent_Reinforcement Learning_Assignment

---

## Reine Mizero — Experiment Results

**10 experiments** were run on `ALE/Boxing-v5` (200,000 steps each, screening stage).

### Best Experiment: Exp16\_VeryLargeBatch\_CNN

| Parameter | Value |
|---|---|
| Policy | CnnPolicy |
| Learning rate | 0.00015 |
| Gamma | 0.99 |
| Batch size | 256 |
| Epsilon start / end | 1.0 → 0.01 |
| Epsilon decay fraction | 0.20 |
| Buffer size | 100,000 |
| Learning starts | 10,000 |
| Target update interval | 1,000 |
| Parallel envs | 4 |

**Why it performed best:** The large batch size (256) produced smoother gradient updates and better punch consistency, reducing Q-value variance compared to smaller-batch runs.

### Evaluation Results (Best Model — 3 Episodes)

| Metric | Value |
|---|---|
| Mean reward | 22.33 |
| Std reward | ±6.34 |
| Min / Max reward | 16.0 / 31.0 |
| Wins | 3 / 3 |
| Draws | 0 |
| Losses | 0 |

The trained agent won all 3 evaluation episodes against the Atari CPU opponent, scoring between 16 and 31 points per episode.
