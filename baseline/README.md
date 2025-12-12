# Power Splitter Optimization

Inverse design optimization for a power splitter that couples input power into two output arms with a specified ratio (default: 60/40 split).

## Overview

This project implements a GOOS-based inverse design optimization for a power splitter device. The optimization uses continuous design variables and progressively discretizes them using sigmoid sharpening to achieve binary (silicon/silica) structures.

## Project Structure

```
power_splitter/
├── src/
│   ├── core/                    # Core optimization modules
│   │   ├── config.py           # Configuration dataclasses
│   │   ├── geometry.py         # Geometry calculation and validation
│   │   ├── utils.py            # Utility functions (config loading, file handling)
│   │   ├── visualization.py    # Visualization functions
│   │   └── power_splitter_cont_opt.py  # Main optimization script
│   │
│   ├── debug/                  # Debug and testing scripts
│   │   ├── quick_sim_test.py   # Quick evaluation of saved designs
│   │   ├── debug_sim.py        # Reproduce simulation results for debugging
│   │   └── straight_waveguide_check.py  # Baseline validation (straight waveguide)
│   │
│   ├── tools/                  # Utility tools
│   │   └── pickle_discretizer.py  # Discretize continuous designs to binary
│   │
│   └── utils/                  # Shared utilities
│       └── visualization_utils.py  # Shared visualization functions
│
└── ckpt/                       # Checkpoint files (saved optimization results)
```

## Installation

Ensure you have the required dependencies:

```bash
pip install ./baseline
# Install spins package (see main project README)
```

## Usage

### Running Optimization

Run the main optimization script:

```bash
cd power_splitter/src/core
python power_splitter_cont_opt.py run path/to/save_folder
```

**Options:**
- `--target-ratio FLOAT`: Desired power ratio for upper arm (default: 0.6 for 60/40 splitter)
- `--max-iters INT`: Maximum optimization iterations (default: 60)
- `--config PATH`: Optional JSON/YAML config file with nested overrides
- `--plot-geometry`: Plot geometry layout before optimization
- `--visualize`: Render permittivity after optimization

**Example:**
```bash
python power_splitter_cont_opt.py run ../results --target-ratio 0.5 --max-iters 100
```

### Viewing Results

Inspect a saved optimization step:

```bash
python power_splitter_cont_opt.py view path/to/save_folder --step 10
```

**Options:**
- `--step INT`: Checkpoint step number (required)
- `--components`: Also plot Ey/Ez field components to inspect quasi-TE/TM modes
- `--config PATH`: Optional config file (if different from default)

**Example:**
```bash
python power_splitter_cont_opt.py view ../results/20240101-120000 --step 50 --components
```

## Configuration

The optimization uses a hierarchical configuration system with the following components:

- **DesignConfig**: Design region dimensions, pixel size, resolution
- **WaveguideConfig**: Waveguide geometry (width, lengths, offsets)
- **MaterialConfig**: Refractive indices (air, background/silica, core/silicon)
- **SimulationConfig**: FDFD simulation parameters (wavelength, region size, PML, monitors)
- **OptimizationConfig**: Optimization settings (iterations, target ratio, sigmoid factors)

Default values are provided, but you can override them via:
1. Command-line arguments (`--target-ratio`, `--max-iters`)
2. JSON/YAML config file (`--config`)

**Example config file (`config.yaml`):**
```yaml
design:
  width: 2000
  height: 2000
  pixel_size: 100
simulation:
  wavelength: 1550
  region: 6000
optimization:
  target_ratio: 0.6
  max_iters: 100
```

## Debug Tools

### Quick Simulation Test

Evaluate a saved design checkpoint:

```bash
cd power_splitter/src/debug
python quick_sim_test.py step50.pkl --stage auto
```

**Options:**
- `--stage {auto,cont,sig}`: Optimization stage (auto-detect by default)
- `--save-path PATH`: Temporary GOOS plan save folder
- `--no-show`: Don't display plots, only print powers
- `--fig-prefix PREFIX`: Prefix for saved figures

### Debug Simulation

Reproduce simulation results for debugging:

```bash
python debug_sim.py step160.pkl
```

This script compares sigmoid and linear design evaluations.

### Straight Waveguide Check

Validate simulation setup with a baseline straight waveguide:

```bash
python straight_waveguide_check.py --save-folder ../results/validation
```

This checks if the simulation stack (mesh, sources, monitors) is working correctly before running inverse design.

## Tools

### Discretizer

Convert continuous design variables to binary (silicon/silica):

```bash
cd power_splitter/src/tools
python pickle_discretizer.py step50.pkl --out-pkl step50_discrete.pkl
```

**Options:**
- `--out-pkl PATH`: Output path for discretized pickle
- `--out-binary PATH`: Export binary mask as text file
- `--threshold FLOAT`: Manual threshold (0-1), default: auto-detected via k-means
- `--optimize-threshold`: Run binary search to find optimal threshold
- `--target-ratio FLOAT`: Target power ratio (must match original optimization)

## Output Files

Optimization runs create the following files in the save folder:

- `step{N}.pkl`: Checkpoint files with optimization state
- `step{N}.png`: Visualization of permittivity, field, and design variables
- `step{N}_components.png`: Field component plots (if `--components` used)
- `geometry_sanity_check.png`: Geometry layout (if `--plot-geometry` used)
- `spins.log`: Optimization log file

## Key Features

1. **Two-Phase Optimization**:
   - Continuous phase: Optimizes continuous design variables
   - Discrete phase: Progressively sharpens design using sigmoid factors

2. **Power Ratio Objective**:
   - Targets specific power split ratio (default: 60/40)
   - Includes power loss penalty to maintain high transmission

3. **Modular Architecture**:
   - Clean separation of concerns (config, geometry, simulation, visualization)
   - Easy to extend and modify

4. **Comprehensive Debugging**:
   - Multiple debug scripts for validation and troubleshooting
   - Visualization tools for result inspection

## Troubleshooting

### Import Errors

If you encounter import errors when running scripts directly:

1. **Run from the correct directory:**
   ```bash
   cd power_splitter/src/core
   python power_splitter_cont_opt.py ...
   ```

2. **Or use module syntax:**
   ```bash
   cd power_splitter/src
   python -m core.power_splitter_cont_opt ...
   ```

### Simulation Issues

If optimization fails or produces unexpected results:

1. **Run baseline check first:**
   ```bash
   python debug/straight_waveguide_check.py
   ```
   This validates your simulation setup.

2. **Check geometry warnings:**
   The optimizer prints geometry validation warnings. Address any issues before running.

3. **Verify configuration:**
   Ensure your config parameters (especially mesh resolution and PML thickness) are appropriate for your system.

## Citation

If you use this code in your research, please cite the original SPINS-B project and GOOS framework.

## License

See the main project LICENSE file for details.

