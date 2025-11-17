"""GOOS power splitter optimization targeting 60% / 40% split.

This example follows the `bend90` workflow but tailors the objective so that
an input excitation from the left couples 60% of the power into the upper arm
and 40% into the lower arm.

Usage
-----
Run optimization:

    $ python power_splitter_60_40.py run path/to/save_folder

Inspect a saved step:

    $ python power_splitter_60_40.py view path/to/save_folder --step 10
"""
import argparse
import dataclasses
import os
import pickle

import numpy as np

from spins import goos
from spins.goos_sim import maxwell


@dataclasses.dataclass
class Options:
    """Simulation / design hyper-parameters."""

    design_width: float = 2000
    design_height: float = 2000
    thickness: float = 220
    pixel_size: float = 40

    wg_width: float = 400
    wg_len_in: float = 2500
    wg_len_out: float = 2500
    wg_offset: float = 600
    wg_output_center: float = 1200

    wlen: float = 1550
    background_index: float = 1.0
    core_index: float = 3.45

    sim_region: float = 4000
    sim_z_extent: float = 40
    pml_thickness: float = 400
    source_pos: float = -1500
    monitor_pos: float = 1800

    max_iters: int = 60
    target_ratio: float = 0.6  # power in upper arm
    power_loss_weight: float = 0.1


def create_design(params: Options):
    """Creates design variables and static structures."""

    def initializer(size):
        np.random.seed(42)
        return np.random.random(size) * 0.2 + 0.5

    var, design = goos.pixelated_cont_shape(
        initializer=initializer,
        pos=goos.Constant([0, 0, 0]),
        extents=[params.design_width, params.design_height, params.thickness],
        material=goos.material.Material(index=params.background_index),
        material2=goos.material.Material(index=params.core_index),
        pixel_size=[params.pixel_size, params.pixel_size, params.thickness],
        var_name="design_var",
    )

    wg_in = goos.Cuboid(
        pos=goos.Constant([-params.wg_len_in / 2 - 100, 0, 0]),
        extents=goos.Constant([params.wg_len_in, params.wg_width, params.thickness]),
        material=goos.material.Material(index=params.core_index),
    )

    wg_up = goos.Cuboid(
        pos=goos.Constant([params.wg_output_center, params.wg_offset, 0]),
        extents=goos.Constant([params.wg_len_out, params.wg_width, params.thickness]),
        material=goos.material.Material(index=params.core_index),
    )

    wg_down = goos.Cuboid(
        pos=goos.Constant([params.wg_output_center, -params.wg_offset, 0]),
        extents=goos.Constant([params.wg_len_out, params.wg_width, params.thickness]),
        material=goos.material.Material(index=params.core_index),
    )

    eps_struct = goos.GroupShape([wg_in, wg_up, wg_down, design])

    eps_render = maxwell.RenderShape(
        design,
        region=goos.Box3d(
            center=[0, 0, 0],
            extents=[params.design_width, params.design_height, 0],
        ),
        mesh=maxwell.UniformMesh(dx=params.pixel_size),
        wavelength=params.wlen,
        name="eps_rendered",
    )

    return var, eps_struct, eps_render


def create_simulation(eps: goos.Shape, params: Options):
    """Sets up the FDFD simulation and monitors."""

    sim = maxwell.fdfd_simulation(
        name="sim_splitter",
        wavelength=params.wlen,
        eps=eps,
        solver="local_direct",
        sources=[
            maxwell.WaveguideModeSource(
                center=[params.source_pos, 0, 0],
                extents=[0, params.wg_width * 6, params.thickness * 4],
                normal=[1, 0, 0],
                mode_num=0,
                power=1,
            )
        ],
        simulation_space=maxwell.SimulationSpace(
            mesh=maxwell.UniformMesh(dx=params.pixel_size),
            sim_region=goos.Box3d(
                center=[0, 0, 0],
                extents=[params.sim_region, params.sim_region, params.sim_z_extent],
            ),
            pml_thickness=[
                params.pml_thickness,
                params.pml_thickness,
                params.pml_thickness,
                params.pml_thickness,
                0,
                0,
            ],
        ),
        background=goos.material.Material(index=params.background_index),
        outputs=[
            maxwell.Epsilon(name="eps"),
            maxwell.ElectricField(name="field"),
            maxwell.WaveguideModeOverlap(
                name="overlap_up",
                center=[params.monitor_pos, params.wg_offset, 0],
                extents=[0, params.wg_width, params.thickness * 2],
                normal=[1, 0, 0],
                mode_num=0,
                power=1,
            ),
            maxwell.WaveguideModeOverlap(
                name="overlap_down",
                center=[params.monitor_pos, -params.wg_offset, 0],
                extents=[0, params.wg_width, params.thickness * 2],
                normal=[1, 0, 0],
                mode_num=0,
                power=1,
            ),
        ],
    )

    return sim


def create_objective(sim, params: Options):
    """Constructs the power-ratio objective."""

    power_up = goos.abs(sim["overlap_up"])
    power_down = goos.abs(sim["overlap_down"])
    total_power = power_up + power_down + 1e-12

    ratio = power_up / total_power
    target_ratio = goos.Constant(params.target_ratio)
    ratio_error = (ratio - target_ratio) ** 2
    power_penalty = params.power_loss_weight * (1 - total_power)

    obj = goos.rename(ratio_error + power_penalty, name="obj_splitter")
    return obj, power_up, power_down


def run(save_folder: str, params: Options, visualize: bool = False):
    goos.util.setup_logging(save_folder)

    with goos.OptimizationPlan(save_path=save_folder) as plan:
        var, eps_struct, eps_render = create_design(params)
        sim = create_simulation(eps_struct, params)
        obj, power_up, power_down = create_objective(sim, params)

        goos.opt.scipy_minimize(
            obj,
            "L-BFGS-B",
            max_iters=params.max_iters,
            monitor_list=[sim["eps"], sim["field"], power_up, power_down, obj],
            name="optimize_splitter",
        )

        plan.save()
        plan.run()

        if visualize:
            goos.util.visualize_eps(sim["eps"].get().array[2])


def view(save_folder: str, step: int):
    if step is None:
        raise ValueError("Must specify --step when viewing results.")

    with open(os.path.join(save_folder, f"step{step}.pkl"), "rb") as fp:
        data = pickle.load(fp)

    eps = np.linalg.norm(data["monitor_data"]["sim_splitter.eps"], axis=0)
    field = np.linalg.norm(data["monitor_data"]["sim_splitter.field"], axis=0)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Permittivity slice")
    plt.imshow(eps[:, :, eps.shape[2] // 2])
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Field magnitude slice")
    plt.imshow(field[:, :, field.shape[2] // 2])
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="60/40 power splitter optimizer.")
    parser.add_argument("action", choices=("run", "view"))
    parser.add_argument("save_folder")
    parser.add_argument("--step", type=int, help="Checkpoint step for view.")
    parser.add_argument("--visualize", action="store_true", help="Render permittivity.")
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.6,
        help="Desired power ratio for the upper arm.",
    )
    parser.add_argument("--max-iters", type=int, default=60)

    args = parser.parse_args()
    params = Options(target_ratio=args.target_ratio, max_iters=args.max_iters)

    if args.action == "run":
        run(args.save_folder, params, visualize=args.visualize)
    else:
        view(args.save_folder, args.step)


if __name__ == "__main__":
    main()

