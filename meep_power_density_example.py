"""
Example: Get power density at a specific point in Meep
"""

import meep as mp
import numpy as np
import matplotlib.pyplot as plt


def get_power_density_at_point(sim, position, component='x'):
    """
    Get power density (Poynting vector) at a specific point.
    Calculates from E and H fields using get_field_point.

    Args:
        sim: Meep simulation object
        position: mp.Vector3 position where to measure
        component: 'x', 'y', 'z', or 'total' for total power density

    Returns:
        power_density: Power density value at the point (real part)
    """
    # Get fields for 2D TM mode (Ez, Hx, Hy)
    Ez = sim.get_field_point(mp.Ez, position)
    Hx = sim.get_field_point(mp.Hx, position)
    Hy = sim.get_field_point(mp.Hy, position)

    # Calculate Poynting vector: S = E × H
    # For 2D TM mode: Sx = -Ez * Hy, Sy = Ez * Hx
    Sx = -Ez * Hy  # Power flow in x-direction
    Sy = Ez * Hx   # Power flow in y-direction
    Sz = 0.0        # No z-component in 2D

    if component == 'x':
        return np.real(Sx)
    elif component == 'y':
        return np.real(Sy)
    elif component == 'z':
        return np.real(Sz)
    elif component == 'total':
        # Total power density magnitude
        return np.real(np.sqrt(Sx**2 + Sy**2 + Sz**2))
    else:
        raise ValueError("component must be 'x', 'y', 'z', or 'total'")


def get_power_density_2d_tm(sim, position):
    """
    Get power density in x and y directions for 2D TM mode.

    Args:
        sim: Meep simulation object
        position: mp.Vector3 position where to measure

    Returns:
        (Sx, Sy): Power density in x and y directions (real parts)
    """
    # Get fields and calculate Poynting vector
    Ez = sim.get_field_point(mp.Ez, position)
    Hx = sim.get_field_point(mp.Hx, position)
    Hy = sim.get_field_point(mp.Hy, position)

    Sx = -Ez * Hy
    Sy = Ez * Hx

    return np.real(Sx), np.real(Sy)


# Example usage
if __name__ == "__main__":
    # Create a simple 2D simulation
    resolution = 50
    cell_size = mp.Vector3(4, 2, 0)
    pml_layers = [mp.PML(0.2)]

    # Simple source
    sources = [mp.Source(
        mp.ContinuousSource(wavelength=1.55, width=20),
        component=mp.Ez,
        center=mp.Vector3(-1.5, 0, 0),
        size=mp.Vector3(0, 0.5, 0)
    )]

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        resolution=resolution,
        dimensions=2
    )

    # Run simulation
    sim.run(until=20)

    # plot the power density at x = 2.0
    # from y = -1 to y = 1
    y_positions = np.linspace(-1, 1, 100)
    power_densities = []
    for y in y_positions:
        position = mp.Vector3(2.0, y, 0)
        power_density = get_power_density_at_point(sim, position)
        power_densities.append(power_density)
    plt.figure(figsize=(10, 6))
    plt.plot(y_positions, power_densities, 'b-',
             linewidth=2, label='Power Density')
    plt.xlabel('y (microns)')
    plt.ylabel('Power Density')
    plt.title(f'Power Density at x = 2.0μm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # plot the flux distribution at x = 2.0
    flux_distribution = calculator.calculate_flux(x_position=2.0)
    print(f"Flux distribution at {position}: {flux_distribution}")
    plt.figure(figsize=(10, 6))
    plt.plot(flux_distribution, 'b-', linewidth=2, label='Flux Distribution')
    plt.xlabel('Detector Index')
    plt.ylabel('Flux')
    plt.title(f'Flux Distribution at x = {position.x}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
