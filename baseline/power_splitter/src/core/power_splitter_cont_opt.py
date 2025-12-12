"""GOOS power splitter optimization targeting 60% / 40% split.

This module implements an inverse design optimization for a power splitter
that couples input power into two output arms with a specified ratio.

Usage
-----
Run optimization (default 60/40 split, configurable via --target-ratio):

    $ python power_splitter_cont_opt.py run path/to/save_folder

Inspect a saved step:

    $ python power_splitter_cont_opt.py view path/to/save_folder --step 10
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spins import goos
from spins.goos_sim import maxwell

# Handle both relative imports (when used as module) and absolute imports (when run as script)
if __name__ == "__main__":
    # Add src directory to path when running as script
    # File is at: power_splitter/src/core/power_splitter_cont_opt.py
    # Need to add: power_splitter/src to sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/core
    src_dir = os.path.dirname(script_dir)  # .../src
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from core.config import SplitterConfig
    from core.geometry import geometry_summary, validate_geometry
    from core.utils import (
        build_config_from_file,
        resolve_save_folder,
        scalar_from_any,
    )
    from core.visualization import plot_geometry_layout
else:
    # Relative imports when used as module
    from .config import SplitterConfig
    from .geometry import geometry_summary, validate_geometry
    from .utils import (
        build_config_from_file,
        resolve_save_folder,
        scalar_from_any,
    )
    from .visualization import plot_geometry_layout

# Baseline power for normalization (from straight waveguide check)
BASELINE_POWER = 1.000000


def create_design(config: SplitterConfig) -> Tuple[
    goos.Variable,
    goos.Shape,
    goos.Shape,
    goos.Shape,
    goos.Shape,
    goos.Flow,
]:
    """Create design variables and static waveguide structures.
    
    Args:
        config: Power splitter configuration
        
    Returns:
        Tuple of (design_var, wg_in, wg_up, wg_down, design_shape, eps_render)
    """
    design_cfg = config.design
    wg_cfg = config.waveguide
    mat_cfg = config.material

    def initializer(size):
        """Initialize design variables with random values."""
        np.random.seed(42)
        return np.random.random(size) * 0.2 + 0.5

    # Create continuous design variable
    var, design = goos.pixelated_cont_shape(
        initializer=initializer,
        pos=goos.Constant([0, 0, 0]),
        extents=[design_cfg.width, design_cfg.height, design_cfg.thickness],
        material=goos.material.Material(index=mat_cfg.background_index),
        material2=goos.material.Material(index=mat_cfg.core_index),
        pixel_size=[design_cfg.pixel_size, design_cfg.pixel_size, design_cfg.thickness],
        var_name="design_var",
    )

    # Create input waveguide
    geom = geometry_summary(config)
    input_center_x = geom["input_start"] + wg_cfg.input_length / 2

    wg_in = goos.Cuboid(
        pos=goos.Constant([input_center_x, 0, 0]),
        extents=goos.Constant([wg_cfg.input_length, wg_cfg.width, design_cfg.thickness]),
        material=goos.material.Material(index=mat_cfg.core_index),
    )

    # Create output waveguides
    wg_up = goos.Cuboid(
        pos=goos.Constant([wg_cfg.output_center, wg_cfg.offset, 0]),
        extents=goos.Constant([wg_cfg.output_length, wg_cfg.width, design_cfg.thickness]),
        material=goos.material.Material(index=mat_cfg.core_index),
    )

    wg_down = goos.Cuboid(
        pos=goos.Constant([wg_cfg.output_center, -wg_cfg.offset, 0]),
        extents=goos.Constant([wg_cfg.output_length, wg_cfg.width, design_cfg.thickness]),
        material=goos.material.Material(index=mat_cfg.core_index),
    )

    # Create epsilon renderer for visualization
    eps_render = maxwell.RenderShape(
        design,
        region=goos.Box3d(
            center=[0, 0, 0],
            extents=[design_cfg.width, design_cfg.height, 0],
        ),
        mesh=maxwell.UniformMesh(dx=design_cfg.pixel_size),
        wavelength=config.simulation.wavelength,
        name="eps_rendered",
    )

    return var, wg_in, wg_up, wg_down, design, eps_render


def create_simulation(
    eps: goos.Shape,
    config: SplitterConfig,
    name: str = "sim_splitter"
) -> goos.Flow:
    """Set up the FDFD simulation and monitors.
    
    Args:
        eps: Permittivity structure (Shape)
        config: Power splitter configuration
        name: Simulation name
        
    Returns:
        Simulation flow with monitors
    """
    geom = geometry_summary(config)
    design = config.design
    wg = config.waveguide
    sim_cfg = config.simulation
    mat_cfg = config.material

    sim = maxwell.fdfd_simulation(
        name=name,
        wavelength=sim_cfg.wavelength,
        eps=eps,
        solver="local_direct",
        sources=[
            maxwell.WaveguideModeSource(
                center=[geom["source_x"], 0, 0],
                extents=[0, wg.width * 3, design.thickness * 4],
                normal=[1, 0, 0],
                mode_num=1,
                power=1,
            )
        ],
        simulation_space=maxwell.SimulationSpace(
            mesh=maxwell.UniformMesh(dx=design.resolution),
            sim_region=goos.Box3d(
                center=[0, 0, 0],
                extents=[sim_cfg.region, sim_cfg.region, sim_cfg.z_extent],
            ),
            pml_thickness=[
                sim_cfg.pml_thickness,
                sim_cfg.pml_thickness,
                sim_cfg.pml_thickness,
                sim_cfg.pml_thickness,
                0,
                0,
            ],
        ),
        # Use air as the simulation background; the design region still maps
        # value 0 → background_index (silica) and 1 → core_index (silicon).
        background=goos.material.Material(index=mat_cfg.air_index),
        outputs=[
            maxwell.Epsilon(name="eps"),
            maxwell.ElectricField(name="field"),
            maxwell.WaveguideModeOverlap(
                name="overlap_up",
                center=[sim_cfg.monitor_position, wg.offset, 0],
                extents=[0, wg.width * 1.5, design.thickness * 2],
                normal=[1, 0, 0],
                mode_num=1,
                power=1,
            ),
            maxwell.WaveguideModeOverlap(
                name="overlap_down",
                center=[sim_cfg.monitor_position, -wg.offset, 0],
                extents=[0, wg.width * 1.5, design.thickness * 2],
                normal=[1, 0, 0],
                mode_num=1,
                power=1,
            ),
        ],
    )

    return sim


def create_objective(
    sim: goos.Flow,
    config: SplitterConfig,
    name_prefix: str = "obj_splitter"
) -> Tuple[goos.Flow, ...]:
    """Construct the power-ratio objective function.
    
    Args:
        sim: Simulation flow with overlap monitors
        config: Power splitter configuration
        name_prefix: Prefix for objective term names
        
    Returns:
        Tuple of (obj, ratio_term, penalty_term, total_power_term, power_up, power_down)
    """
    def named(expr, suffix):
        """Helper to name objective terms."""
        return goos.rename(expr, name=f"{name_prefix}.{suffix}")

    # Extract raw powers from overlap monitors
    power_up_raw = goos.abs(sim["overlap_up"]) ** 2
    power_down_raw = goos.abs(sim["overlap_down"]) ** 2

    # Normalize powers
    power_up = named(power_up_raw / BASELINE_POWER, "power_up")
    power_down = named(power_down_raw / BASELINE_POWER, "power_down")
    total_power = power_up + power_down + 1e-12

    # Calculate power ratios
    ratio_up = power_up / total_power
    ratio_down = power_down / total_power
    target_ratio = goos.Constant(config.optimization.target_ratio)
    
    # Objective terms
    ratio_mse = (ratio_up - target_ratio) ** 2 + (ratio_down - (1 - target_ratio)) ** 2
    power_penalty = config.optimization.power_loss_weight * (1 - total_power) ** 2

    ratio_term = named(ratio_mse, "ratio_mse")
    penalty_term = named(power_penalty, "power_penalty")
    total_power_term = named(total_power, "total_power")
    obj = goos.rename(ratio_term + penalty_term, name=name_prefix)
    
    return obj, ratio_term, penalty_term, total_power_term, power_up, power_down


def run(
    save_folder: str,
    config: SplitterConfig,
    visualize: bool = False,
    plot_geometry: bool = False,
) -> None:
    """Run power splitter optimization.
    
    Args:
        save_folder: Directory to save optimization results
        config: Power splitter configuration
        visualize: Whether to visualize final permittivity
        plot_geometry: Whether to plot geometry layout before optimization
    """
    goos.util.setup_logging(save_folder)

    # Validate geometry
    issues = validate_geometry(config)
    if issues:
        print("Geometry sanity warnings:")
        for issue in issues:
            print(f"  - {issue}")

    if plot_geometry:
        geometry_plot_path = os.path.join(save_folder, "geometry_sanity_check.png")
        plot_geometry_layout(config, save_path=geometry_plot_path)

    with goos.OptimizationPlan(save_path=save_folder) as plan:
        # Create design structure
        var, wg_in, wg_up, wg_down, design, eps_render = create_design(config)

        # Continuous optimization phase
        eps_struct = goos.GroupShape([wg_in, wg_up, wg_down, design])
        sim = create_simulation(eps_struct, config, name="sim_splitter_cont")
        obj, ratio_term, penalty_term, total_power_term, power_up, power_down = (
            create_objective(sim, config, name_prefix="obj_splitter_cont")
        )

        goos.opt.scipy_minimize(
            obj,
            "L-BFGS-B",
            max_iters=config.optimization.max_iters,
            monitor_list=[
                sim["eps"],
                sim["field"],
                power_up,
                power_down,
                total_power_term,
                ratio_term,
                penalty_term,
                obj,
            ],
            name="optimize_splitter_cont",
        )

        # Discrete optimization phase with sigmoid sharpening
        sigmoid_factor = goos.Variable(4, parameter=True, name="discr_factor")
        design_sig = goos.cast(
            goos.Sigmoid(sigmoid_factor * (2 * design - 1)),
            goos.Shape
        )

        eps_struct_sig = goos.GroupShape([wg_in, wg_up, wg_down, design_sig])
        sim_sig = create_simulation(eps_struct_sig, config, name="sim_splitter_sig")
        (
            obj_sig,
            ratio_term_sig,
            penalty_term_sig,
            total_power_sig,
            power_up_sig,
            power_down_sig,
        ) = create_objective(sim_sig, config, name_prefix="obj_splitter_sig")

        # Iterate through sigmoid factors for progressive discretization
        for factor in config.optimization.sigmoid_factors:
            sigmoid_factor.set(factor)
            goos.opt.scipy_minimize(
                obj_sig,
                "L-BFGS-B",
                max_iters=min(20, config.optimization.max_iters),
                monitor_list=[
                    sim_sig["eps"],
                    sim_sig["field"],
                    power_up_sig,
                    power_down_sig,
                    total_power_sig,
                    ratio_term_sig,
                    penalty_term_sig,
                    obj_sig,
                ],
                name=f"optimize_splitter_discrete_{factor}",
            )

        plan.save()
        plan.run()

        if visualize:
            goos.util.visualize_eps(sim_sig["eps"].get().array[2])


def _extract_monitor_data(
    monitor_data: dict,
    candidates: List[str]
) -> any:
    """Extract monitor data by trying multiple candidate keys.
    
    Args:
        monitor_data: Dictionary of monitor data
        candidates: List of candidate keys to try
        
    Returns:
        Monitor data value or None if not found
    """
    for candidate in candidates:
        if candidate in monitor_data:
            return monitor_data[candidate]
    return None


def _prepare_2d_plot_data(data: np.ndarray, z_slice_idx: int = None) -> np.ndarray:
    """Extract and transpose 2D slice from 3D data for plotting.
    
    Args:
        data: 3D array (Nx, Ny, Nz) or 2D array (Nx, Ny)
        z_slice_idx: Index for z-slice (if None, uses middle slice for 3D)
        
    Returns:
        2D array transposed for imshow (y, x)
    """
    if data.ndim == 3:
        if z_slice_idx is None:
            z_slice_idx = data.shape[2] // 2
        data = data[:, :, z_slice_idx]
    return data.T


def view(
    save_folder: str,
    step: int,
    config: SplitterConfig | None = None,
    components: bool = False,
) -> None:
    """View and visualize optimization results from a saved checkpoint.
    
    Args:
        save_folder: Directory containing checkpoint files
        step: Checkpoint step number
        config: Optional configuration (uses default if None)
        components: Whether to plot Ey/Ez field components
    """
    if step is None:
        raise ValueError("Must specify --step when viewing results.")

    # Load checkpoint data
    checkpoint_path = os.path.join(save_folder, f"step{step}.pkl")
    with open(checkpoint_path, "rb") as fp:
        data = pickle.load(fp)

    monitor_data = data.get("monitor_data", {})

    # Extract epsilon and field data
    eps_raw = _extract_monitor_data(
        monitor_data,
        ["sim_splitter_cont.eps", "sim_splitter_sig.eps"]
    )
    field_raw = _extract_monitor_data(
        monitor_data,
        ["sim_splitter_cont.field", "sim_splitter_sig.field"]
    )
    
    if eps_raw is None or field_raw is None:
        raise KeyError("Could not find epsilon/field monitors in the selected step.")

    # Process data for visualization
    eps = np.real(eps_raw[2])
    field = np.linalg.norm(field_raw, axis=0)
    
    z_slice_idx = eps.shape[2] // 2
    eps_plot = _prepare_2d_plot_data(eps, z_slice_idx)
    field_plot = _prepare_2d_plot_data(field, z_slice_idx)
    
    if config is None:
        config = SplitterConfig()
    
    # Calculate plot extent
    nm_to_um = 1e-3
    half_region = config.simulation.region * nm_to_um / 2
    extent = [-half_region, half_region, -half_region, half_region]

    # Extract and print metrics
    power_up_val = scalar_from_any(
        _extract_monitor_data(
            monitor_data,
            ["obj_splitter_cont.power_up", "obj_splitter_sig.power_up"]
        )
    )
    power_down_val = scalar_from_any(
        _extract_monitor_data(
            monitor_data,
            ["obj_splitter_cont.power_down", "obj_splitter_sig.power_down"]
        )
    )
    total_power_val = scalar_from_any(
        _extract_monitor_data(
            monitor_data,
            ["obj_splitter_cont.total_power", "obj_splitter_sig.total_power"]
        )
    )
    ratio_val = scalar_from_any(
        _extract_monitor_data(
            monitor_data,
            ["obj_splitter_cont.ratio_mse", "obj_splitter_sig.ratio_mse"]
        )
    )
    penalty_val = scalar_from_any(
        _extract_monitor_data(
            monitor_data,
            ["obj_splitter_cont.power_penalty", "obj_splitter_sig.power_penalty"]
        )
    )

    # Print metrics
    if any(val is not None for val in (total_power_val, power_up_val, power_down_val)):
        metrics_text = []
        if power_up_val is not None:
            metrics_text.append(f"up={power_up_val:.4f}")
        if power_down_val is not None:
            metrics_text.append(f"down={power_down_val:.4f}")
        if total_power_val is not None:
            metrics_text.append(f"total={total_power_val:.4f}")
        print(f"[Step {step}] Waveguide powers: {', '.join(metrics_text)}")

    if ratio_val is not None or penalty_val is not None:
        ratio_text = f"ratio_mse={ratio_val:.4e}" if ratio_val is not None else None
        penalty_text = (
            f"power_penalty={penalty_val:.4e}" if penalty_val is not None else None
        )
        text_items = [t for t in (ratio_text, penalty_text) if t]
        if text_items:
            print(f"[Step {step}] Objective terms: {', '.join(text_items)}")

    # Extract design variables if available
    design_vals = None
    if "variable_data" in data and "design_var" in data["variable_data"]:
        design_vals = np.array(data["variable_data"]["design_var"]["value"])

    design_plot = None
    if design_vals is not None:
        design_plot = _prepare_2d_plot_data(design_vals, z_slice_idx)

    # Create main visualization
    cols = 3
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))

    # Permittivity plot
    im1 = axes[0].imshow(
        eps_plot, cmap="viridis", aspect="equal", extent=extent, origin="lower"
    )
    axes[0].set_title(f"Permittivity (Step {step})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("x (μm)")
    axes[0].set_ylabel("y (μm)")
    axes[0].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    fig.colorbar(im1, ax=axes[0], label="|ε|")

    # Field magnitude plot
    im2 = axes[1].imshow(
        field_plot, cmap="hot", aspect="equal", extent=extent, origin="lower"
    )
    axes[1].set_title(f"Field Magnitude (Step {step})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("x (μm)")
    axes[1].set_ylabel("y (μm)")
    axes[1].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    fig.colorbar(im2, ax=axes[1], label="|E|")

    # Design variables plot
    if design_plot is not None:
        design_extent = [
            config.design.width * nm_to_um / -2,
            config.design.width * nm_to_um / 2,
            config.design.height * nm_to_um / -2,
            config.design.height * nm_to_um / 2,
        ]
        im3 = axes[2].imshow(
            design_plot,
            cmap="Greys",
            aspect="equal",
            extent=design_extent,
            origin="lower",
            vmin=0,
            vmax=1
        )
        axes[2].set_title(f"Design Variables (Step {step})", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("x (μm)")
        axes[2].set_ylabel("y (μm)")
        axes[2].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        fig.colorbar(im3, ax=axes[2], label="Value")
    else:
        axes[2].text(0.5, 0.5, "No design data found", ha="center", va="center")
        axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    if save_folder:
        save_path = os.path.join(save_folder, f"step{step}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")

    # Optional: plot field components (Ey, Ez) for quasi-TE/TM inspection
    if components:
        Ex = field_raw[0]
        Ey = field_raw[1]
        Ez = field_raw[2]

        Ey_plot = _prepare_2d_plot_data(np.abs(Ey) ** 2, z_slice_idx)
        Ez_plot = _prepare_2d_plot_data(np.abs(Ez) ** 2, z_slice_idx)

        fig_comp, axes_comp = plt.subplots(1, 2, figsize=(12, 5))

        im_ey = axes_comp[0].imshow(
            Ey_plot, cmap="plasma", aspect="equal", extent=extent, origin="lower"
        )
        axes_comp[0].set_title(f"|Ey|^2 (Step {step})", fontsize=12, fontweight="bold")
        axes_comp[0].set_xlabel("x (μm)")
        axes_comp[0].set_ylabel("y (μm)")
        axes_comp[0].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        fig_comp.colorbar(im_ey, ax=axes_comp[0], label="|Ey|^2")

        im_ez = axes_comp[1].imshow(
            Ez_plot, cmap="plasma", aspect="equal", extent=extent, origin="lower"
        )
        axes_comp[1].set_title(f"|Ez|^2 (Step {step})", fontsize=12, fontweight="bold")
        axes_comp[1].set_xlabel("x (μm)")
        axes_comp[1].set_ylabel("y (μm)")
        axes_comp[1].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
        fig_comp.colorbar(im_ez, ax=axes_comp[1], label="|Ez|^2")

        plt.tight_layout()
        plt.show()

        if save_folder:
            comp_path = os.path.join(save_folder, f"step{step}_components.png")
            fig_comp.savefig(comp_path, dpi=150, bbox_inches="tight")
            print(f"Saved field components visualization to: {comp_path}")


def main() -> None:
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(description="60/40 power splitter optimizer.")
    parser.add_argument("action", choices=("run", "view"), help="Action to perform")
    parser.add_argument("save_folder", help="Save folder for optimization or checkpoint location")
    parser.add_argument("--step", type=int, help="Checkpoint step for view action")
    parser.add_argument("--visualize", action="store_true", help="Render permittivity after optimization")
    parser.add_argument(
        "--plot-geometry",
        action="store_true",
        help="Plot geometry sanity check before optimization",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=None,
        help="Desired power ratio for the upper arm (default: 0.6 for 60/40 splitter)",
    )
    parser.add_argument("--max-iters", type=int, default=60, help="Maximum optimization iterations")
    parser.add_argument(
        "--config",
        help="Optional JSON/YAML config file with nested overrides",
        default=None,
    )
    parser.add_argument(
        "--components",
        action="store_true",
        help="In view mode, also plot Ey/Ez components to inspect quasi-TE/TM",
    )

    args = parser.parse_args()
    save_folder = resolve_save_folder(args.save_folder, args.action)
    config = build_config_from_file(args.config, args.target_ratio, args.max_iters)

    if args.action == "run":
        run(
            save_folder,
            config,
            visualize=args.visualize,
            plot_geometry=args.plot_geometry,
        )
    elif args.action == "view":
        view(
            save_folder,
            args.step,
            config if args.config else None,
            components=args.components,
        )


if __name__ == "__main__":
    main()
