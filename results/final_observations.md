# Hyperparameter Experiment Results

| Member Name | Experiment | Hyperparameter Set | Noted Behavior |
|---|---|---|---|
| Fidele Ndihokubwayo | Exp07_BestTuned_BiggerBatch | lr=0.0002, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.15 | Top tuned CNN outperformed the baseline noticeably. It balanced learning speed and stability well. |
| Fidele Ndihokubwayo | Exp10_ChampionCandidate_CNN | lr=0.0002, gamma=0.995, batch=128, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.12 | Champion candidate outperformed the baseline noticeably. It targets stronger movement, cleaner attacks, and higher winning margins. |
| Fidele Ndihokubwayo | Exp01_BestTuned_TopCNN | lr=0.00025, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.15 | Top tuned CNN performed similarly to the baseline. It balanced learning speed and stability well. |
| Fidele Ndihokubwayo | Exp02_LargeBatch_TopCNN | lr=0.0001, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | Larger batch size performed similarly to the baseline. It likely produced smoother updates and better punch consistency. |
| Fidele Ndihokubwayo | Exp03_MLPBaseline_Reference | lr=0.0001, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | MLP comparison run underperformed the baseline. It is useful for architecture comparison, but CNN remains the stronger image-based choice for Boxing. |
| Fidele Ndihokubwayo | Exp06_BestTuned_LongExplore | lr=0.00025, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.2 | Longer exploration underperformed the baseline. It gave the agent more time to discover positioning and timing patterns. |
| Fidele Ndihokubwayo | Exp08_AggressivePressure_CNN | lr=0.00015, gamma=0.995, batch=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | Aggressive pressure tuning underperformed the baseline. It was designed to improve attacks, pressure, and dominance. |
| Fidele Ndihokubwayo | Exp09_ControlledFastExploit | lr=0.0002, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.08 | Faster exploitation underperformed the baseline. It became more decisive sooner, but may reduce tactical diversity. |
| Fidele Ndihokubwayo | Exp05_HighGamma_Balanced | lr=0.0002, gamma=0.995, batch=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.15 | Higher gamma underperformed the baseline. It likely valued longer-term positioning and pressure more strongly. |
| Fidele Ndihokubwayo | Exp04_MLPTuned_Comparison | lr=0.0002, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.15 | MLP comparison run underperformed the baseline. It is useful for architecture comparison, but CNN remains the stronger image-based choice for Boxing. |
