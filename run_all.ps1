$ErrorActionPreference = "Stop"

python train.py --experiments Exp01_BestTuned_TopCNN --timesteps 100000
python train.py --experiments Exp02_LargeBatch_TopCNN --timesteps 100000
python train.py --experiments Exp03_MLPBaseline_Reference --timesteps 100000
python train.py --experiments Exp04_MLPTuned_Comparison --timesteps 100000
python train.py --experiments Exp05_HighGamma_Balanced --timesteps 100000
python train.py --experiments Exp06_BestTuned_LongExplore --timesteps 100000
python train.py --experiments Exp07_BestTuned_BiggerBatch --timesteps 100000
python train.py --experiments Exp08_AggressivePressure_CNN --timesteps 100000
python train.py --experiments Exp09_ControlledFastExploit --timesteps 100000
python train.py --experiments Exp10_ChampionCandidate_CNN --timesteps 100000