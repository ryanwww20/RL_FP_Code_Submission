"""Geometry calculation and validation utilities."""

import os
import sys
from typing import Dict, List

# Handle both relative imports (when used as module) and absolute imports (when run as script)
if __name__ == "__main__" or not __package__:
    # Add src directory to path
    script_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/core
    src_dir = os.path.dirname(script_dir)  # .../src
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from core.config import SplitterConfig
else:
    from .config import SplitterConfig


def geometry_summary(config: SplitterConfig) -> Dict[str, float]:
    """Calculate convenience geometry values shared across helpers.
    
    Args:
        config: Power splitter configuration
        
    Returns:
        Dictionary containing computed geometry values:
        - input_start: X-coordinate of input waveguide start
        - input_center: X-coordinate of input waveguide center
        - source_x: X-coordinate of source position
        - design_bounds: (xmin, xmax, ymin, ymax) tuple for design region
        - sim_bounds: (xmin, xmax, ymin, ymax) tuple for simulation region
        - output_start: X-coordinate of output waveguide start
        - monitor_x: X-coordinate of monitor plane
    """
    design = config.design
    waveguide = config.waveguide
    sim = config.simulation

    input_start = -design.width / 2 - waveguide.input_length
    input_center = input_start + waveguide.input_length / 2
    source_x = input_start * sim.source_shift
    
    design_bounds = (
        -design.width / 2,
        design.width / 2,
        -design.height / 2,
        design.height / 2,
    )
    
    sim_bounds = (
        -sim.region / 2,
        sim.region / 2,
        -sim.region / 2,
        sim.region / 2,
    )
    
    return {
        "input_start": input_start,
        "input_center": input_center,
        "source_x": source_x,
        "design_bounds": design_bounds,
        "sim_bounds": sim_bounds,
        "output_start": design.width / 2,
        "monitor_x": sim.monitor_position,
    }


def validate_geometry(config: SplitterConfig) -> List[str]:
    """Validate geometry configuration and return warnings.
    
    Args:
        config: Power splitter configuration
        
    Returns:
        List of human-readable warning messages for suspicious geometry
    """
    issues: List[str] = []
    design = config.design
    wg = config.waveguide
    summary = geometry_summary(config)

    if summary["monitor_x"] < design.width / 2:
        issues.append(
            "Monitor plane lies inside the design region; consider moving it "
            "further to the right so modal overlaps are well-defined."
        )
    
    if wg.output_center < design.width / 2:
        issues.append(
            "Output waveguides do not fully clear the design area "
            "(output_center < design.width / 2)."
        )
    
    if wg.offset + wg.width / 2 > design.height / 2:
        issues.append(
            "Waveguide offset pushes outputs outside the design window "
            "(offset too large)."
        )
    
    if design.width <= 0 or design.height <= 0:
        issues.append("Design region dimensions must be positive.")
    
    if wg.width <= 0:
        issues.append("Waveguide width must be positive.")
    
    if not (0 < config.optimization.target_ratio < 1):
        issues.append("Target ratio should be in (0, 1).")
    
    return issues

