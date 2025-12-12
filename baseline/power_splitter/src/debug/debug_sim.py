"""Debug script to reproduce simulation results from a saved checkpoint."""

import argparse
import os
import pickle
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spins import goos
from core.config import SplitterConfig
from core.power_splitter_cont_opt import create_design, create_simulation, create_objective


def debug_reproduction(pkl_path: str = None):
    """Reproduce simulation results from a checkpoint for debugging."""
    if pkl_path is None:
        pkl_path = 'step160.pkl'
    
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Checkpoint file not found: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print("Variable keys:", data.get('variable_data', {}).keys())
    
    design_vals = np.array(data['variable_data']['design_var']['value'], dtype=float)
    discr_factor_val = (
        float(np.asarray(data['variable_data']['discr_factor']['value']).item())
        if 'discr_factor' in data.get('variable_data', {})
        else 4.0
    )
    
    print(f"Loaded discr_factor: {discr_factor_val}")

    with goos.OptimizationPlan(save_path='.') as plan:
        config = SplitterConfig()
        
        # 1. Create base design
        var, wg_in, wg_up, wg_down, design, eps_render = create_design(config)
        
        # 2. Replicate the Sigmoid transformation used in training
        # Note: The Variable needs to be created exactly as in the script to match the graph if we were loading full graph, 
        # but here we are rebuilding.
        sigmoid_factor = goos.Variable(discr_factor_val, parameter=True, name="discr_factor")
        design_sig = goos.cast(goos.Sigmoid(sigmoid_factor * (2 * design - 1)), goos.Shape)
        
        eps_struct_sig = goos.GroupShape([wg_in, wg_up, wg_down, design_sig])
        
        # 3. Create simulation for Sigmoid case
        sim_sig = create_simulation(eps_struct_sig, config, name="sim_splitter_sig")
        
        # 4. Create objective to match the logs
        # Note: create_objective returns (obj, ratio_term, penalty_term, total_power_term, power_up, power_down)
        obj_sig, ratio_term_sig, penalty_term_sig, total_power_sig, power_up_sig, power_down_sig = \
            create_objective(sim_sig, config, name_prefix="obj_splitter_sig")

        # 5. Set values
        var.set(design_vals)
        sigmoid_factor.set(discr_factor_val)
        
        # 6. Evaluate
        print("\n--- Running Simulation with SIGMOID Design ---")
        # We can evaluate the objective terms directly
        # power_up_sig is the normalized power
        
        flow_results = plan.eval_nodes([
            sim_sig["overlap_up"], 
            sim_sig["overlap_down"],
            power_up_sig,
            power_down_sig,
            total_power_sig
        ])
        
        # Unpack flows
        results = [f.array for f in flow_results]
        
        ov_up_raw = results[0]
        ov_down_raw = results[1]
        p_up_norm = results[2]
        p_down_norm = results[3]
        p_total_norm = results[4]
        
        print(f"Raw Overlap Up (Amplitude): {ov_up_raw}")
        print(f"Raw Overlap Up Power (|Amp|^2): {np.abs(ov_up_raw)**2}")
        print(f"Raw Overlap Down Power (|Amp|^2): {np.abs(ov_down_raw)**2}")
        print(f"Normalized Power Up (Log equivalent): {p_up_norm}")
        print(f"Normalized Power Down (Log equivalent): {p_down_norm}")
        print(f"Normalized Total Power: {p_total_norm}")
        
        # 7. Compare with Linear Design (what quick_sim_test likely ran)
        print("\n--- Running Simulation with LINEAR Design (Quick Sim Check) ---")
        eps_struct_lin = goos.GroupShape([wg_in, wg_up, wg_down, design])
        sim_lin = create_simulation(eps_struct_lin, config, name="sim_splitter_lin")
        
        results_lin = sim_lin.get()
        # [eps, field, overlap_up, overlap_down]
        ov_up_lin = results_lin[2].array
        ov_down_lin = results_lin[3].array
        
        print(f"Linear Overlap Up Power: {np.abs(ov_up_lin)**2}")
        print(f"Linear Overlap Down Power: {np.abs(ov_down_lin)**2}")


def main():
    parser = argparse.ArgumentParser(description="Debug script to reproduce simulation results")
    parser.add_argument("pkl", nargs="?", default="step160.pkl", help="Path to checkpoint .pkl file")
    args = parser.parse_args()
    debug_reproduction(args.pkl)


if __name__ == "__main__":
    main()

