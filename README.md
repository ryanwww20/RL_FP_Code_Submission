# RL for Waveguide Optimization

## Environments & Installation

```bash
# 1. Create environment with Python
conda create --name rl_waveguide python=3.9

# 2. Activate it
conda activate rl_waveguide

# 3. Install Meep Python bindings (pymeep) from conda-forge
conda install -c conda-forge pymeep

# 4. Install other dependencies
pip install -r requirements.txt
pip install ./baseline
```

## Code Structure

```
RL_FP_Code_Submission/
├── train_ppo.py              # Main PPO training script
├── eval.py                    # Model evaluation script
├── config.yaml                # Configuration file for training and simulation parameters
├── config.py                  # Configuration loader
├── goos_sim.py                # GOOS simulation utilities
├── requirements.txt           # Python dependencies
│
├── envs/                      # Reinforcement learning environment
│   ├── Discrete_gym.py        # Custom Gym environment for waveguide optimization
│   ├── meep_simulation.py     # Meep simulation wrapper
│   └── custom_feature_extractor.py  # Custom feature extractor for PPO
│
└── baseline/                  # Baseline optimization methods
    ├── power_splitter/        # Power splitter baseline implementation
    │   ├── src/
    │   │   ├── core/          # Core optimization code
    │   │   │   └── power_splitter_cont_opt.py  # Continuous optimization
    │   │   ├── tools/         # Utility tools
    │   │   │   └── pickle_discretizer.py  # Discretization tool
    │   │   └── debug/         # Debug and testing scripts
    │   │       └── quick_sim_test.py  # FDFD simulation test
    │   └── ckpt/              # Checkpoint files
    └── spins/                 # SPINS library for inverse design
```

## Execution

### Training PPO Model

Run the training script:
```bash
python train_ppo.py
```

You can customize the following parameters in `config.yaml`:
- `target_ratio`: Target ratio for output 1 (default: 0.5)
- `training.ppo.learning_rate`: Learning rate for PPO training
- `simulation.simulation_time`: Simulation time parameter

### Running Baseline

The baseline consists of three steps:

**Step 01: Optimize**
```bash
python /storage/undergrad/b12901074/RL_FP_Code_Submission/baseline/power_splitter/src/core/power_splitter_cont_opt.py --max-iters 200 --target-ratio 0.7
```

**Step 02: Discretization**
```bash
python /storage/undergrad/b12901074/RL_FP_Code_Submission/baseline/power_splitter/src/tools/pickle_discretizer.py <path_to_pkl_file_generated_by_step_01>
```

**Step 03: FDFD Simulation**
```bash
python /storage/undergrad/b12901074/RL_FP_Code_Submission/baseline/power_splitter/src/debug/quick_sim_test.py <path_to_pkl_file_generated_by_step_02>
```
