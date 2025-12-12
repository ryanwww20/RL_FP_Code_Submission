"""Visualization functions for power splitter optimization."""

import os
import sys
from typing import Dict, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Handle both relative imports (when used as module) and absolute imports (when run as script)
if __name__ == "__main__" or not __package__:
    # Add src directory to path
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/core
    src_dir = os.path.dirname(script_dir)  # .../src
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from core.config import SplitterConfig
    from core.geometry import geometry_summary
else:
    from .config import SplitterConfig
    from .geometry import geometry_summary


def plot_geometry_layout(
    config: SplitterConfig,
    summary: Dict[str, float] | None = None,
    save_path: str | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot geometry layout with design region, waveguides, and monitors.
    
    Args:
        config: Power splitter configuration
        summary: Optional pre-computed geometry summary
        save_path: Optional path to save figure
        
    Returns:
        Tuple of (figure, axes) objects
    """
    summary = summary or geometry_summary(config)
    design = config.design
    wg = config.waveguide
    sim = config.simulation

    nm_to_um = 1e-3
    fig, ax = plt.subplots(figsize=(9, 6))

    # Simulation boundary
    sim_bounds = summary["sim_bounds"]
    sim_rect = patches.Rectangle(
        (sim_bounds[0] * nm_to_um, sim_bounds[2] * nm_to_um),
        sim.region * nm_to_um,
        sim.region * nm_to_um,
        linewidth=1.5,
        edgecolor="gray",
        facecolor="none",
        linestyle="--",
        label="Simulation",
    )
    ax.add_patch(sim_rect)

    # Design region
    design_rect = patches.Rectangle(
        (summary["design_bounds"][0] * nm_to_um, summary["design_bounds"][2] * nm_to_um),
        design.width * nm_to_um,
        design.height * nm_to_um,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
        label="Design",
    )
    ax.add_patch(design_rect)

    # Input waveguide
    ax.add_patch(
        patches.Rectangle(
            (summary["input_start"] * nm_to_um, -wg.width / 2 * nm_to_um),
            wg.input_length * nm_to_um,
            wg.width * nm_to_um,
            color="royalblue",
            alpha=0.4,
            label="Input WG",
        )
    )

    # Output waveguides
    ax.add_patch(
        patches.Rectangle(
            (summary["output_start"] * nm_to_um, (wg.offset - wg.width / 2) * nm_to_um),
            wg.output_length * nm_to_um,
            wg.width * nm_to_um,
            color="seagreen",
            alpha=0.4,
            label="Output WG",
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (summary["output_start"] * nm_to_um, (-wg.offset - wg.width / 2) * nm_to_um),
            wg.output_length * nm_to_um,
            wg.width * nm_to_um,
            color="seagreen",
            alpha=0.4,
        )
    )

    # Source / monitor markers
    ax.axvline(summary["source_x"] * nm_to_um, color="red", linestyle=":", label="Source")
    ax.axvline(summary["monitor_x"] * nm_to_um, color="purple", linestyle="-.", label="Monitor")

    ax.set_aspect("equal")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title("Power Splitter Geometry Overview")

    span_x = max(abs(sim_bounds[0]), abs(sim_bounds[1])) * nm_to_um
    span_y = max(abs(sim_bounds[2]), abs(sim_bounds[3])) * nm_to_um
    ax.set_xlim(-span_x, span_x)
    ax.set_ylim(-span_y, span_y)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    
    return fig, ax

