# Formative-3-Assignment-Deep-Q-Learning
Using Stable Baselines 3 and Gymnasium to train and evaluate a Deep Q-network (DQN) agent_Reinforcement Learning_Assignment

---

## Experiment table and Results

** Each Group member performed 10 experiments** were run on `ALE/Boxing-v5` (200,000 steps each, screening stage).
| Member | Experiment ID | LR | Gamma | Batch | ε Start | ε End | ε Decay | Target Update | Best Eval Reward |
|---|---|---|---|---|---|---|---|---|---|
| Reine Mizero | Exp11_VeryLowLR_Stable_CNN | 5e-5 | 0.99 | 64 | 1.0 | 0.01 | 0.15 | 1000 | -2.3 |
| Reine Mizero | Exp12_HighLR_FastLearn_CNN | 5e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.10 | 1000 | -12.9 |
| Reine Mizero | Exp13_LowGamma_ShortHorizon_CNN | 2.5e-4 | 0.95 | 64 | 1.0 | 0.01 | 0.15 | 1000 | 14 |
| Reine Mizero | Exp14_VeryLowGamma_CNN | 2e-4 | 0.90 | 64 | 1.0 | 0.01 | 0.15 | 1000 | 23.6 |
| Reine Mizero | Exp15_SmallBatch_FreqUpdate_CNN | 2e-4 | 0.99 | 32 | 1.0 | 0.01 | 0.15 | 1000 | -19.4 |
| Reine Mizero | Exp16_VeryLargeBatch_CNN ★ | 1.5e-4 | 0.99 | 256 | 1.0 | 0.01 | 0.20 | 1000 | **32.0** |
| Reine Mizero | Exp17_MLP_LowGamma_Reference | 2.5e-4 | 0.95 | 64 | 1.0 | 0.01 | 0.15 | 500 | -21.4 |
| Reine Mizero | Exp18_MLP_LargeBatch_Comparison | 1.5e-4 | 0.99 | 128 | 1.0 | 0.01 | 0.20 | 500 | 0.4 |
| Reine Mizero | Exp19_MidGamma_MidLR_Balanced_CNN | 3e-4 | 0.97 | 64 | 1.0 | 0.01 | 0.15 | 1000 | 0.5 |
| Reine Mizero | Exp20_FreqTargetUpdate_CNN | 2.5e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.15 | 500 | -16.8 |
| Fidele Ndihokubwayo | Exp01_BestTuned_TopCNN | 2.5e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.15 | 1000 | 12.2 |
| Fidele Ndihokubwayo | Exp02_LargeBatch_TopCNN | 1e-4 | 0.99 | 128 | 1.0 | 0.01 | 0.10 | 1000 | 6.8 |
| Fidele Ndihokubwayo | Exp03_MLPBaseline_Reference | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.10 | 500 | 1.8 |
| Fidele Ndihokubwayo | Exp04_MLPTuned_Comparison | 2e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.15 | 500 | -0.3 |
| Fidele Ndihokubwayo | Exp05_HighGamma_Balanced | 2.0e-4 | 0.995 | 64 | 1.0 | 0.01 | 0.15 | 1000 | -3.3 |
| Fidele Ndihokubwayo | Exp06_BestTuned_LongExplore | 2.5e-4 | 0.99 | 64 | 1.0 | 0.02 | 0.20 | 1000 | -5.9 |
| Fidele Ndihokubwayo | Exp07_BestTuned_BiggerBatch | 2.0e-4 | 0.99 | 128 | 1.0 | 0.01 | 0.15 | 1000 | -7.4 |
| Fidele Ndihokubwayo | Exp08_AggressivePressure_CNN | 1.5e-4 | 0.995 | 64 | 1.0 | 0.01 | 0.10 | 1000 | -7.4 |
| Fidele Ndihokubwayo | Exp09_ControlledFastExploit | 2.0e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.08 | 1000 | -7.6 |
| Fidele Ndihokubwayo | Exp10_ChampionCandidate_CNN | 2.0e-4 | 0.995 | 128 | 1.0 | 0.01 | 0.12 | 1000 | -12.6 |
| Wengelawit Solomon | Exp01_Baseline_CNN | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | 1000 | -10.5 |
| Wengelawit Solomon | Exp02_HighLR_Aggressive_CNN | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 0.1 | 1000 | -11.0 |
| Wengelawit Solomon | Exp03_LowGamma_Repetitive_CNN | 5e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.1 | 1000 | -11.0 |
| Wengelawit Solomon | Exp04_LargeBatch_Stable_CNN | 5e-4 | 0.95 | 64 | 1.0 | 0.05 | 0.1 | 1000 | -5.25 |
| Wengelawit Solomon | Exp05_HighGamma_BetterAttack_CNN | 5e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.1 | 1000 | -4.0 |
| Wengelawit Solomon | Exp06_HigherExploration_SlowStart_CNN | 5e-4 | 0.99 | 64 | 1.0 | 0.10 | 0.1 | 1000 | -6.0 |
| Wengelawit Solomon | Exp07_LowExploration_Rigid_CNN | 5e-4 | 0.99 | 64 | 1.0 | 0.01 | 0.1 | 1000 | 0.0 |
| Wengelawit Solomon | Exp08_LowLR_Controlled_CNN | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 0.1 | 1000 | -6.0 |
| Wengelawit Solomon | Exp09_MidLR_Stable_CNN | 2e-4 | 0.95 | 64 | 1.0 | 0.05 | 0.1 | 1000 | -8.0 |
| Wengelawit Solomon | Exp10_LargeBatch_Smoother_CNN | 5e-4 | 0.95 | 128 | 1.0 | 0.05 | 0.1 | 1000 | -6.5 |
| Liliane Umwanankabandi | Exp1_Baseline_CNN | 5e-5 | 0.98 | 32 | 1.0 | 0.05 | 0.20 | 1000 | -12.0 |
| Liliane Umwanankabandi | Exp2_HigherLR_CNN | 2e-4 | 0.98 | 32 | 1.0 | 0.05 | 0.20 | 1000 | -10.5 |
| Liliane Umwanankabandi | Exp3_HighLR_CNN | 5e-4 | 0.98 | 32 | 1.0 | 0.05 | 0.20 | 1000 | -9.75 |
| Liliane Umwanankabandi | Exp4_LowGamma_StrongerShortTerm_CNN | 5e-4 | 0.95 | 32 | 1.0 | 0.05 | 0.20 | 1000 | -2.0 |
| Liliane Umwanankabandi | Exp5_MidGamma_LowExplore_CNN | 5e-4 | 0.97 | 32 | 1.0 | 0.01 | 0.20 | 1000 | -12.2 |
| Liliane Umwanankabandi | Exp6_LargeBatch_CNN | 5e-4 | 0.95 | 64 | 1.0 | 0.05 | 0.20 | 1000 | -25.2 |
| Liliane Umwanankabandi | Exp7_BestLowExplore_CNN | 5e-4 | 0.95 | 32 | 1.0 | 0.01 | 0.20 | 1000 | 3.4 |
| Liliane Umwanankabandi | Exp8_HighExplore_CNN | 5e-4 | 0.95 | 32 | 1.0 | 0.10 | 0.20 | 1000 | -64.4 |
| Liliane Umwanankabandi | Exp9_FastExploit_CNN | 5e-4 | 0.95 | 32 | 1.0 | 0.01 | 0.10 | 1000 | -25.0 |
| Liliane Umwanankabandi | Exp10_LongExplore_CNN | 5e-4 | 0.95 | 32 | 1.0 | 0.01 | 0.30 | 1000 | -33.0 |

### Best Experiment: Exp16\_VeryLargeBatch\_CNN


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
## 🎮 Demo Video
[▶ Watch the Game Demo](https://drive.google.com/file/d/1XPUJP5MhZ-W6HMvjEz343fseFMxCQWM_/view?usp=sharing)
