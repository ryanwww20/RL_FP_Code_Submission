# RL for Waveguide Optimization using Meep

This project uses Reinforcement Learning (PPO and SAC) to optimize waveguide structures simulated with Meep.

## Folder Structure
```bash
ppo_model_log_<start_time>/
├── img/
│   ├── flux.gif
│   └── design.gif
├── plot/              (real-time update)
│   ├── transmission.png
│   ├── balance.png
│   └── score.png
└── result.csv

eval_result/
├── distribution.gif
├── design.gif
├── material.txt
└── score.csv

models/
└── ppo_model_<timestamp>.zip

job_log/
└── rl_log_<pid>.txt     (added manually)

ppo_tensorboard/

archive/                 (personal trash)
playground/              (personal experiment)
```

