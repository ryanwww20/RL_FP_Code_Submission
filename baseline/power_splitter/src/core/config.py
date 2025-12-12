"""Configuration dataclasses for power splitter optimization."""

import dataclasses
from typing import Any, Dict, Tuple


@dataclasses.dataclass
class DesignConfig:
    """Design region configuration.
    
    Attributes:
        width: Design region width in nanometers
        height: Design region height in nanometers
        thickness: Design region thickness in nanometers
        pixel_size: Pixel size for design parameterization in nanometers
        resolution: Simulation mesh resolution in nanometers
    """
    width: float = 2000
    height: float = 2000
    thickness: float = 220
    pixel_size: float = 100
    resolution: float = 50


@dataclasses.dataclass
class WaveguideConfig:
    """Waveguide geometry configuration.
    
    Attributes:
        width: Waveguide width in nanometers
        input_length: Input waveguide length in nanometers
        output_length: Output waveguide length in nanometers
        offset: Vertical offset between output waveguides in nanometers
        output_center: X-coordinate of output waveguide center in nanometers
    """
    width: float = 400
    input_length: float = 1500
    output_length: float = 1500
    offset: float = 600
    output_center: float = 1750


@dataclasses.dataclass
class MaterialConfig:
    """Material refractive index configuration.
    
    Attributes:
        air_index: Background index outside the design region
        background_index: Index for value 0 inside design (e.g., silica)
        core_index: Index for value 1 inside design (e.g., silicon)
    """
    air_index: float = 1.0
    background_index: float = 1.45  # Silica
    core_index: float = 3.45  # Silicon


@dataclasses.dataclass
class SimulationConfig:
    """FDFD simulation configuration.
    
    Attributes:
        wavelength: Operating wavelength in nanometers
        region: Simulation region size in nanometers
        z_extent: Z-direction extent in nanometers
        pml_thickness: PML boundary thickness in nanometers
        source_shift: Source position shift factor (0-1)
        monitor_position: X-coordinate of monitor plane in nanometers
    """
    wavelength: float = 1550
    region: float = 6000
    z_extent: float = 40
    pml_thickness: float = 200
    source_shift: float = 0.95
    monitor_position: float = 1800


@dataclasses.dataclass
class OptimizationConfig:
    """Optimization algorithm configuration.
    
    Attributes:
        max_iters: Maximum optimization iterations
        target_ratio: Target power ratio for upper arm (0-1)
        power_loss_weight: Weight for power loss penalty term
        sigmoid_factors: Sigmoid sharpening factors for discrete optimization
    """
    max_iters: int = 100
    target_ratio: float = 0.6  # 60% power in upper arm (60/40 splitter)
    power_loss_weight: float = 0.1
    sigmoid_factors: Tuple[int, ...] = (
        4, 8, 16, 24,
        32, 48, 64, 96,
        128, 256, 384, 512,
        768, 1024, 1536, 2048,
        3072, 4096, 6144, 8192
    )


@dataclasses.dataclass
class SplitterConfig:
    """Complete power splitter configuration.
    
    Combines all sub-configurations into a single structure.
    """
    design: DesignConfig = dataclasses.field(default_factory=DesignConfig)
    waveguide: WaveguideConfig = dataclasses.field(default_factory=WaveguideConfig)
    material: MaterialConfig = dataclasses.field(default_factory=MaterialConfig)
    simulation: SimulationConfig = dataclasses.field(default_factory=SimulationConfig)
    optimization: OptimizationConfig = dataclasses.field(default_factory=OptimizationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return dataclasses.asdict(self)

