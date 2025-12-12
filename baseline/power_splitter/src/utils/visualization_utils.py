"""Shared visualization utilities for power splitter analysis scripts."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def prepare_2d_slice(data: np.ndarray, z_slice_idx: int = None) -> np.ndarray:
    """Extract and transpose a 2D slice from 3D data for plotting.
    
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


def get_extent(region_size: float) -> Tuple[float, float, float, float]:
    """Get extent tuple for imshow from simulation region size (in nm).
    
    Args:
        region_size: Simulation region size in nanometers
    
    Returns:
        (xmin, xmax, ymin, ymax) in micrometers
    """
    nm_to_um = 1e-3
    half_size = region_size * nm_to_um / 2
    return (-half_size, half_size, -half_size, half_size)


def plot_eps_field(
    eps: np.ndarray,
    field: np.ndarray,
    extent: Tuple[float, float, float, float],
    save_path: str = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot permittivity and field magnitude side by side.
    
    Args:
        eps: Permittivity array (2D or 3D)
        field: Field magnitude array (2D or 3D)
        extent: (xmin, xmax, ymin, ymax) for imshow
        save_path: Optional path to save figure
        show: Whether to call plt.show()
    
    Returns:
        (figure, axes) tuple
    """
    eps_plot = prepare_2d_slice(eps)
    field_plot = prepare_2d_slice(field)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(eps_plot, cmap="viridis", aspect="equal", extent=extent, origin="lower")
    axes[0].set_title("Permittivity (ε)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("x (μm)")
    axes[0].set_ylabel("y (μm)")
    axes[0].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    fig.colorbar(im1, ax=axes[0], label="|ε|")
    
    im2 = axes[1].imshow(field_plot, cmap="hot", aspect="equal", extent=extent, origin="lower")
    axes[1].set_title("|E| Field Magnitude", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("x (μm)")
    axes[1].set_ylabel("y (μm)")
    axes[1].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    fig.colorbar(im2, ax=axes[1], label="|E|")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig, axes


def plot_field_components(
    Ey: np.ndarray,
    Ez: np.ndarray,
    extent: Tuple[float, float, float, float],
    save_path: str = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot Ey and Ez field components side by side.
    
    Args:
        Ey: Ey field component (2D or 3D)
        Ez: Ez field component (2D or 3D)
        extent: (xmin, xmax, ymin, ymax) for imshow
        save_path: Optional path to save figure
        show: Whether to call plt.show()
    
    Returns:
        (figure, axes) tuple
    """
    Ey_plot = prepare_2d_slice(np.abs(Ey) ** 2)
    Ez_plot = prepare_2d_slice(np.abs(Ez) ** 2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im_ey = axes[0].imshow(Ey_plot, cmap="plasma", aspect="equal", extent=extent, origin="lower")
    axes[0].set_title("|Ey|^2 (central z-slice)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("x (μm)")
    axes[0].set_ylabel("y (μm)")
    axes[0].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    fig.colorbar(im_ey, ax=axes[0], label="|Ey|^2")
    
    im_ez = axes[1].imshow(Ez_plot, cmap="plasma", aspect="equal", extent=extent, origin="lower")
    axes[1].set_title("|Ez|^2 (central z-slice)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("x (μm)")
    axes[1].set_ylabel("y (μm)")
    axes[1].grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    fig.colorbar(im_ez, ax=axes[1], label="|Ez|^2")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig, axes

