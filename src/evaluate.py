# filepath: c:\Users\abdul\OneDrive\Belgeler\GitHub\lnn-advanced\Modular LNN\src\evaluate.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import SymLogNorm
from torch.func import vmap, hessian
from torch.utils.data import TensorDataset, DataLoader

from dataset_creation import create_test_data, normalize_testing_data, create_training_data, normalize_training_data, create_validation_data
from ode_solve import solve_with_lnn_robust, solve_with_lnn_torchode
from file_exports import save_trajectory_test_results

# =============================================================================
# TESTING FUNCTIONS (from test.py)
# =============================================================================
    
def check_out_of_distribution(particle, q_qdot_test_list):

  scale_factor = particle.scale
  position_bounds = particle.train_hyperparams['position_bounds']
  velocity_bounds = particle.train_hyperparams['velocity_bounds']
  for i in range(len(q_qdot_test_list)):
    pr = q_qdot_test_list[i]
    
    # Convert to numpy array if it's a tensor
    if hasattr(pr, 'cpu'):
      pr = pr.cpu().numpy()
    elif not isinstance(pr, np.ndarray):
      pr = np.array(pr)
    
    # Create a copy to avoid modifying the original data
    pr = pr.copy()
    
    # Unnormalize data for out-of-distribution check by multiplying by scale factors  
    num_dims = len(scale_factor)
    for j in range(num_dims):
      pr[:, j] *= scale_factor[j]
      pr[:, j + num_dims] *= scale_factor[j]
      if j in particle.angle_indices:
        pr[:, j] = np.mod(pr[:, j], 2 * np.pi)
    for j in range(num_dims):
      small = pr[:, j] < position_bounds[j][0]
      big = pr[:, j] > position_bounds[j][1]
      vel_small = pr[:, j + num_dims] < velocity_bounds[j][0]
      vel_big = pr[:, j + num_dims] > velocity_bounds[j][1]
      if np.any(small):
        print(f"⚠️  OUT-OF-DISTRIBUTION WARNING: Test case {i+1}, coordinate {j+1} position values are BELOW training bounds!")
        print(f"   This can cause numerical instability and crash the NN solver.")
        print(f"   Found at indices: {np.where(small)[0]}")
        print(f"   Values: {pr[small, j]} < {position_bounds[j][0]} (training lower bound)")
      if np.any(big):
        print(f"⚠️  OUT-OF-DISTRIBUTION WARNING: Test case {i+1}, coordinate {j+1} position values are ABOVE training bounds!")
        print(f"   This can cause numerical instability and crash the NN solver.")
        print(f"   Found at indices: {np.where(big)[0]}")
        print(f"   Values: {pr[big, j]} > {position_bounds[j][1]} (training upper bound)")
      if np.any(vel_small):
        print(f"⚠️  OUT-OF-DISTRIBUTION WARNING: Test case {i+1}, coordinate {j+1} velocity values are BELOW training bounds!")
        print(f"   This can cause numerical instability and crash the NN solver.")
        print(f"   Found at indices: {np.where(vel_small)[0]}")
        print(f"   Values: {pr[vel_small, j + num_dims]} < {velocity_bounds[j][0]} (training lower bound)")
      if np.any(vel_big):
        print(f"⚠️  OUT-OF-DISTRIBUTION WARNING: Test case {i+1}, coordinate {j+1} velocity values are ABOVE training bounds!")
        print(f"   This can cause numerical instability and crash the NN solver.")
        print(f"   Found at indices: {np.where(vel_big)[0]}")
        print(f"   Values: {pr[vel_big, j + num_dims]} > {velocity_bounds[j][1]} (training upper bound)")


def create_trajectory_tests(particle, dtype=torch.float32, device=None):
    num_tests = particle.test_params['num_tests']
    time_bounds = particle.test_params['time_bounds']
    time_step = particle.test_params['time_step']
    test_seed = particle.test_params['test_seed']
    position_start_end_test = particle.test_params['position_bounds']
    velocity_start_end_test = particle.test_params['velocity_bounds']
    scale_factor = particle.scale
    particle.scale_constants([1.0 for _ in range(particle.dof//2)])

    testing_data = create_test_data(particle, num_tests, time_bounds, 
                                        time_step, particle.dof//2, 
                                        position_start_end_test, velocity_start_end_test,
                                        seed = test_seed, device = device)

    testing_data = normalize_testing_data(testing_data, particle.dof//2, torch.tensor(scale_factor, device=device, dtype=dtype))
    q_qdot_test_list, qdot_qdotdot_test_list, t_test = testing_data

    particle.scale_constants(scale_factor)

    check_out_of_distribution(particle, q_qdot_test_list)
    return q_qdot_test_list, qdot_qdotdot_test_list, t_test

def trajectory_test(particle, model, q_qdot_test_list, qdot_qdotdot_test_list, t_test, dtype, device, save_path, ode_dtype=torch.float64, max_batch_size=None, num_time_splits=None, verbose=True):
    """
    Run neural network tests and compute trajectory comparisons using batch processing.
    
    Args:
        max_batch_size: Maximum number of test cases to process in one batch. 
                       If None, tries to process all at once. Use this to control GPU memory usage.
                       Example: max_batch_size=2 will process 4 tests as 2 chunks of 2 tests each.
        num_time_splits: Number of time segments to split the time domain into.
                        If None, processes the full time domain at once. Use this to reduce memory usage.
                        Example: num_time_splits=2 will solve 0 to t/2, then t/2 to t.
    """
    t_span = (t_test[0], t_test[-1])
    
    # Stack all test cases for batch processing
    x0_batch = torch.stack([test_data[0] for test_data in q_qdot_test_list])  # Shape: (num_tests, state_dim)
    
    if verbose:
        print(f"Processing {x0_batch.shape[0]} test cases in batch...")
        print(f"Initial conditions shape: {x0_batch.shape}")
        if max_batch_size is not None:
            print(f"Max batch size set to: {max_batch_size}")
        if num_time_splits is not None:
            print(f"Time splits set to: {num_time_splits}")
    
    # Solve all test cases in parallel (or chunked/time-split)
    nn_test_batch = solve_with_lnn_torchode(particle, model, x0_batch, t_span, t_test, 
                                          dtype=ode_dtype, device=device, 
                                          max_batch_size=max_batch_size, num_time_splits=num_time_splits)
    
    # Convert batch result back to list format for compatibility with existing code
    nn_test_list = []
    for i in range(x0_batch.shape[0]):
        # Convert numpy array back to torch tensor for compatibility with plotting functions
        # Ensure tensor is on CPU for plotting compatibility
        tensor_result = torch.tensor(nn_test_batch[i], dtype=dtype, device='cpu')
        nn_test_list.append(tensor_result)
        if verbose:
            print(f"Test case {i+1} completed")

    # Save trajectory test results
    save_trajectory_test_results(q_qdot_test_list[0], nn_test_list, q_qdot_test_list, t_test, save_path)
    
    return nn_test_list

def plot_trajectory_test_comparisons(particle, q_qdot_test_list, nn_test_list, t_test, save_path,  animate=False, verbose=True):
    """
    Plot comparison between analytical and neural network predictions.
    """
    # Create test_results directory
    test_dir = None
    if save_path:
        test_dir = os.path.join(save_path, "test_results")
        try:
            os.makedirs(test_dir, exist_ok=True)
            if verbose:
                print(f"Test comparison plots will be saved to: {test_dir}")
        except Exception as e:
            print(f"Warning: Could not create test results directory {test_dir}: {e}")
            test_dir = None
    elif verbose:
        print("No save_path provided - test plots will not be saved to file")

    for i in range(len(nn_test_list)):
        if verbose:
            print(f"\n{'='*50}")
            print(f"TEST CASE {i+1}")
            print(f"{'='*50}")
            print(f"Analytical Solution for Test Case {i+1}:")
        
        # Plot analytical solution - handle both numpy and torch
        analytical_data = q_qdot_test_list[i]
        if hasattr(analytical_data, 'cpu'):
            analytical_data = analytical_data.cpu()
        analytical_fig = particle.plot_solved_dynamics(t_test, analytical_data, 
                                                       labelstr=f"Analytical Test {i+1}", color='blue')
        
        # Save analytical plot
        if test_dir:
            analytical_fig.savefig(os.path.join(test_dir, f"analytical_test_{i+1}.png"), dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Analytical test {i+1} plot saved to {test_dir}/analytical_test_{i+1}.png")
        
        if verbose:
            analytical_fig.show()
        else:
            plt.close(analytical_fig)
        
        if verbose:
            print(f"Neural Network Prediction for Test Case {i+1}:")
        
        # Plot neural network prediction
        nn_fig = particle.plot_solved_dynamics(t_test, nn_test_list[i], 
                                               labelstr=f"NN Predicted Test {i+1}", color='red')
        
        # Save NN plot
        if test_dir:
            nn_fig.savefig(os.path.join(test_dir, f"nn_prediction_test_{i+1}.png"), dpi=300, bbox_inches='tight')
            if verbose:
                print(f"NN prediction test {i+1} plot saved to {test_dir}/nn_prediction_test_{i+1}.png")
        
        if verbose:
            nn_fig.show()
        else:
            plt.close(nn_fig)

        # Direct comparison plot
        # Determine coordinate names based on system
        coord_names = []
        if hasattr(particle, 'angle_indices') and particle.angle_indices:
            for j in range(particle.dof//2):
                if j in particle.angle_indices:
                    coord_names.append(f'θ{j+1}')
                else:
                    coord_names.append(f'q{j+1}')
        else:
            for j in range(particle.dof//2):
                coord_names.append(f'q{j+1}')

        # Dynamic subplot layout based on number of coordinates
        num_coords = particle.dof // 2
        num_cols = min(num_coords, 3)  # Max 3 columns
        num_rows = (num_coords + num_cols - 1) // num_cols  # Ceiling division
        
        # Add extra row for error and energy plots
        total_rows = num_rows + 1
        fig, axes = plt.subplots(total_rows, num_cols, figsize=(4*num_cols, 4*total_rows))
        fig.suptitle(f'Direct Comparison - Test Case {i+1}', fontsize=14)
        
        # Handle different subplot configurations
        if total_rows == 1 and num_cols == 1:
            axes = np.array([[axes]])
        elif total_rows == 1:
            axes = axes.reshape(1, -1)
        elif num_cols == 1:
            axes = axes.reshape(-1, 1)
        elif axes.ndim == 1:
            axes = axes.reshape(total_rows, num_cols)
        
        # Plot coordinate comparisons
        for coord_idx in range(num_coords):
            row = coord_idx // num_cols
            col = coord_idx % num_cols
            
            ax = axes[row, col]
            # Handle both torch tensor and numpy array inputs and unnormalize
            analytical_coord = q_qdot_test_list[i].cpu().numpy() if torch.is_tensor(q_qdot_test_list[i]) else q_qdot_test_list[i]
            predicted_coord = nn_test_list[i].cpu().numpy() if torch.is_tensor(nn_test_list[i]) else nn_test_list[i]
            
            # Unnormalize the data by multiplying by scale factors
            scale_factors = particle.scale
            analytical_coord_unnorm = analytical_coord.copy()
            predicted_coord_unnorm = predicted_coord.copy()
            
            # Unnormalize positions and velocities
            for j in range(len(scale_factors)):
                analytical_coord_unnorm[:, j] *= scale_factors[j]  # positions
                analytical_coord_unnorm[:, j + len(scale_factors)] *= scale_factors[j]  # velocities
                predicted_coord_unnorm[:, j] *= scale_factors[j]  # positions  
                predicted_coord_unnorm[:, j + len(scale_factors)] *= scale_factors[j]  # velocities
            
            ax.plot(t_test, analytical_coord_unnorm[:, coord_idx], 'b-', 
                   label=f'Analytical {coord_names[coord_idx]}', linewidth=2)
            ax.plot(t_test, predicted_coord_unnorm[:, coord_idx], 'r--', 
                   label=f'NN Predicted {coord_names[coord_idx]}', linewidth=2)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{coord_names[coord_idx]}')
            ax.set_title(f'{coord_names[coord_idx]} Comparison')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused coordinate subplots
        for coord_idx in range(num_coords, num_rows * num_cols):
            row = coord_idx // num_cols
            col = coord_idx % num_cols
            if row < num_rows:  # Only hide if within the coordinate rows
                axes[row, col].set_visible(False)

        # Error analysis (bottom row, first subplot)
        # Handle both torch tensor and numpy array inputs
        analytical_data = q_qdot_test_list[i]
        predicted_data = nn_test_list[i]
        
        # Convert to numpy for consistent processing
        if hasattr(analytical_data, 'cpu'):
            analytical_data_np = analytical_data.cpu().numpy()
        else:
            analytical_data_np = analytical_data if isinstance(analytical_data, np.ndarray) else np.array(analytical_data)
        
        if hasattr(predicted_data, 'cpu'):
            predicted_data_np = predicted_data.cpu().numpy() 
        else:
            predicted_data_np = predicted_data if isinstance(predicted_data, np.ndarray) else np.array(predicted_data)
        
        # Unnormalize data for error calculation
        scale_factors = particle.scale
        analytical_data_unnorm = analytical_data_np.copy()
        predicted_data_unnorm = predicted_data_np.copy()
        
        # Unnormalize positions and velocities
        for j in range(len(scale_factors)):
            analytical_data_unnorm[:, j] *= scale_factors[j]  # positions
            analytical_data_unnorm[:, j + len(scale_factors)] *= scale_factors[j]  # velocities
            predicted_data_unnorm[:, j] *= scale_factors[j]  # positions  
            predicted_data_unnorm[:, j + len(scale_factors)] *= scale_factors[j]  # velocities
        
        position_error = np.abs(analytical_data_unnorm[:, :particle.dof//2] - 
                               predicted_data_unnorm[:, :particle.dof//2])
        total_position_error = np.sum(position_error, axis=1)

        error_ax = axes[num_rows, 0]
        error_ax.plot(t_test, total_position_error, 'g-', linewidth=2)
        error_ax.set_xlabel('Time (s)')
        error_ax.set_ylabel('Total Position Error')
        error_ax.set_title('Position Error vs Time')
        error_ax.grid(True, alpha=0.3)
        error_ax.set_yscale('log')

        # Energy comparison and normalized energy error (bottom row, second subplot)
        # Handle both torch tensor and numpy array inputs for energy calculation
        if hasattr(analytical_data, 'cpu'):
            analytical_states = analytical_data.cpu()
        elif isinstance(analytical_data, np.ndarray):
            analytical_states = torch.tensor(analytical_data)
        else:
            analytical_states = analytical_data
        
        if hasattr(predicted_data, 'cpu'):
            predicted_states = predicted_data.cpu()
        elif isinstance(predicted_data, np.ndarray):
            predicted_states = torch.tensor(predicted_data)
        else:
            predicted_states = predicted_data
        
        E_analytical = [particle.energy(x) for x in analytical_states]
        E_predicted = [particle.energy(x) for x in predicted_states]

        # Calculate normalized energy error - properly convert tensors to numpy
        E_analytical_np = np.array([float(e.cpu()) if hasattr(e, 'cpu') else float(e) for e in E_analytical])
        E_predicted_np = np.array([float(e.cpu()) if hasattr(e, 'cpu') else float(e) for e in E_predicted])
        E_initial = E_analytical_np[0]  # Initial energy for normalization (now a float)
        energy_error_normalized = np.abs(E_predicted_np - E_analytical_np) / np.abs(E_initial)

        if num_cols > 1:
            energy_ax = axes[num_rows, 1]
        else:
            # If only one column, use next available row
            if num_rows + 1 < total_rows:
                energy_ax = axes[num_rows + 1, 0] if total_rows > 2 else plt.gca()
            else:
                # Create additional subplot if needed
                fig.add_subplot(total_rows, 1, total_rows)
                energy_ax = plt.gca()
        
        energy_ax.plot(t_test, E_analytical, 'b-', label='Analytical Energy', linewidth=2)
        energy_ax.plot(t_test, E_predicted, 'r--', label='NN Predicted Energy', linewidth=2)
        energy_ax.set_xlabel('Time (s)')
        energy_ax.set_ylabel('Energy (J)')
        energy_ax.set_title('Energy Conservation Comparison')
        energy_ax.grid(True, alpha=0.3)
        energy_ax.legend()
        
        # Add normalized energy error plot if there's space
        if num_cols > 2:
            energy_error_ax = axes[num_rows, 2]
            energy_error_ax.plot(t_test, energy_error_normalized, 'purple', linewidth=2)
            energy_error_ax.set_xlabel('Time (s)')
            energy_error_ax.set_ylabel('Normalized Energy Error')
            energy_error_ax.set_title('Energy Error (Normalized)')
            energy_error_ax.grid(True, alpha=0.3)
            energy_error_ax.set_yscale('log')

        # Hide unused subplots in bottom row
        start_col = 3 if num_cols > 2 else 2
        for col_idx in range(start_col, num_cols):
            axes[num_rows, col_idx].set_visible(False)

        plt.tight_layout()
        
        # Save plot
        if test_dir:
            plt.savefig(os.path.join(test_dir, f"test_case_{i+1}_comparison.png"), dpi=300, bbox_inches='tight')
            if verbose:
                print(f"Test case {i+1} comparison saved to {test_dir}/test_case_{i+1}_comparison.png")
        
        if verbose:
            plt.show()
        else:
            plt.close()

        # Print numerical error metrics
        max_pos_error = np.max(total_position_error)
        mean_pos_error = np.mean(total_position_error)
        energy_drift_analytical = np.std(E_analytical_np)
        energy_drift_predicted = np.std(E_predicted_np)
        max_energy_error = np.max(energy_error_normalized)
        mean_energy_error = np.mean(energy_error_normalized)
        final_energy_error = energy_error_normalized[-1]
        
        # Ensure all values are scalars for printing
        max_pos_error = float(max_pos_error)
        mean_pos_error = float(mean_pos_error)
        energy_drift_analytical = float(energy_drift_analytical)
        energy_drift_predicted = float(energy_drift_predicted)
        max_energy_error = float(max_energy_error)
        mean_energy_error = float(mean_energy_error)
        final_energy_error = float(final_energy_error)

        if verbose:
            print(f"Error Metrics for Test Case {i+1}:")
            print(f"  Max Position Error: {max_pos_error:.6f}")
            print(f"  Mean Position Error: {mean_pos_error:.6f}")
            print(f"  Energy Drift (Analytical): {energy_drift_analytical:.6f}")
            print(f"  Energy Drift (Predicted): {energy_drift_predicted:.6f}")
            print(f"  Max Normalized Energy Error: {max_energy_error:.6f}")
            print(f"  Mean Normalized Energy Error: {mean_energy_error:.6f}")
            print(f"  Final Normalized Energy Error: {final_energy_error:.6f}")

        # Save individual trajectory plots for publication
        if test_dir:
            save_individual_trajectory_plots(particle, q_qdot_test_list[i], nn_test_list[i], t_test, 
                                           test_dir, i+1, coord_names, verbose)
            
            # Save individual error analysis plots
            save_individual_error_plots(t_test, total_position_error, E_analytical, E_predicted, 
                                       energy_error_normalized, test_dir, i+1, verbose)
            
            # Create cartesian trajectory animations with fading trails
            if animate:
                create_cartesian_animations(particle, q_qdot_test_list[i], nn_test_list[i], t_test, 
                                        test_dir, i+1, verbose)

def save_individual_trajectory_plots(particle, analytical_trajectory, predicted_trajectory, t_test, 
                                   save_dir, test_case_num, coord_names, verbose=True):
    """
    Save individual trajectory plots for each coordinate for research paper purposes.
    
    Args:
        particle: Physical system instance
        analytical_trajectory: Ground truth trajectory data
        predicted_trajectory: Neural network predicted trajectory
        t_test: Time array
        save_dir: Directory to save plots
        test_case_num: Test case number for naming
        coord_names: List of coordinate names
        verbose: Whether to print save messages
    """
    # Create individual trajectories directory
    individual_dir = os.path.join(save_dir, "individual_trajectories", f"test_case_{test_case_num}")
    try:
        os.makedirs(individual_dir, exist_ok=True)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not create individual trajectories directory: {e}")
        return
    
    # Convert to numpy if needed and unnormalize
    if hasattr(analytical_trajectory, 'cpu'):
        analytical_data = analytical_trajectory.cpu().numpy()
    elif isinstance(analytical_trajectory, np.ndarray):
        analytical_data = analytical_trajectory
    else:
        analytical_data = np.array(analytical_trajectory)
        
    if hasattr(predicted_trajectory, 'cpu'):
        predicted_data = predicted_trajectory.cpu().numpy()
    elif isinstance(predicted_trajectory, np.ndarray):
        predicted_data = predicted_trajectory  
    else:
        predicted_data = np.array(predicted_trajectory)
    
    # Unnormalize the data
    scale_factors = particle.scale
    analytical_data_unnorm = analytical_data.copy()
    predicted_data_unnorm = predicted_data.copy()
    
    # Unnormalize positions and velocities
    for j in range(len(scale_factors)):
        analytical_data_unnorm[:, j] *= scale_factors[j]  # positions
        analytical_data_unnorm[:, j + len(scale_factors)] *= scale_factors[j]  # velocities
        predicted_data_unnorm[:, j] *= scale_factors[j]  # positions  
        predicted_data_unnorm[:, j + len(scale_factors)] *= scale_factors[j]  # velocities
    
    num_coords = particle.dof // 2
    
    # Save individual coordinate plots
    for coord_idx in range(num_coords):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Position plot
        ax.plot(t_test, analytical_data_unnorm[:, coord_idx], 'b-', 
               label=f'Analytical {coord_names[coord_idx]}', linewidth=2.5)
        ax.plot(t_test, predicted_data_unnorm[:, coord_idx], 'r--', 
               label=f'NN Predicted {coord_names[coord_idx]}', linewidth=2.5)
        
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel(f'{coord_names[coord_idx]}', fontsize=14)
        ax.set_title(f'{coord_names[coord_idx]} Trajectory - Test Case {test_case_num}', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        filename = f"{coord_names[coord_idx].lower()}_trajectory_test_{test_case_num}.png"
        plt.savefig(os.path.join(individual_dir, filename), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Velocity plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        vel_idx = coord_idx + num_coords
        ax.plot(t_test, analytical_data_unnorm[:, vel_idx], 'b-', 
               label=f'Analytical {coord_names[coord_idx]}̇', linewidth=2.5)
        ax.plot(t_test, predicted_data_unnorm[:, vel_idx], 'r--', 
               label=f'NN Predicted {coord_names[coord_idx]}̇', linewidth=2.5)
        
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel(f'{coord_names[coord_idx]}̇', fontsize=14)
        ax.set_title(f'{coord_names[coord_idx]} Velocity - Test Case {test_case_num}', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        filename = f"{coord_names[coord_idx].lower()}_velocity_test_{test_case_num}.png"
        plt.savefig(os.path.join(individual_dir, filename), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Error plot for this coordinate
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        position_error = np.abs(analytical_data_unnorm[:, coord_idx] - predicted_data_unnorm[:, coord_idx])
        velocity_error = np.abs(analytical_data_unnorm[:, vel_idx] - predicted_data_unnorm[:, vel_idx])
        
        ax.plot(t_test, position_error, 'g-', label=f'{coord_names[coord_idx]} Error', linewidth=2.5)
        ax.plot(t_test, velocity_error, 'm-', label=f'{coord_names[coord_idx]}̇ Error', linewidth=2.5)
        
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel('Absolute Error', fontsize=14)
        ax.set_title(f'{coord_names[coord_idx]} Error Analysis - Test Case {test_case_num}', 
                    fontsize=16, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        filename = f"{coord_names[coord_idx].lower()}_error_test_{test_case_num}.png"
        plt.savefig(os.path.join(individual_dir, filename), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    if verbose:
        print(f"Individual trajectory plots saved to: {individual_dir}")

def save_individual_error_plots(t_test, position_error, E_analytical, E_predicted, 
                               energy_error_normalized, save_dir, test_case_num, verbose=True):
    """
    Save individual error analysis plots for research paper purposes.
    
    Args:
        t_test: Time array
        position_error: Total position error over time
        E_analytical: Analytical energy over time
        E_predicted: Predicted energy over time
        energy_error_normalized: Normalized energy error over time
        save_dir: Directory to save plots
        test_case_num: Test case number for naming
        verbose: Whether to print save messages
    """
    # Create individual errors directory
    error_dir = os.path.join(save_dir, "individual_errors", f"test_case_{test_case_num}")
    try:
        os.makedirs(error_dir, exist_ok=True)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not create individual errors directory: {e}")
        return
    
    # Position error plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(t_test, position_error, 'g-', linewidth=2.5)
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Total Position Error', fontsize=14)
    ax.set_title(f'Position Error vs Time - Test Case {test_case_num}', 
                fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(error_dir, f"position_error_test_{test_case_num}.png"), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Energy conservation plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(t_test, E_analytical, 'b-', label='Analytical Energy', linewidth=2.5)
    ax.plot(t_test, E_predicted, 'r--', label='NN Predicted Energy', linewidth=2.5)
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Energy (J)', fontsize=14)
    ax.set_title(f'Energy Conservation - Test Case {test_case_num}', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(error_dir, f"energy_conservation_test_{test_case_num}.png"), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Normalized energy error plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(t_test, energy_error_normalized, 'purple', linewidth=2.5)
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Normalized Energy Error', fontsize=14)
    ax.set_title(f'Normalized Energy Error vs Time - Test Case {test_case_num}', 
                fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    # Add horizontal lines for reference
    ax.axhline(y=1e-2, color='orange', linestyle='--', alpha=0.7, label='1% Error')
    ax.axhline(y=1e-3, color='red', linestyle='--', alpha=0.7, label='0.1% Error')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(error_dir, f"normalized_energy_error_test_{test_case_num}.png"), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Combined error summary plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Position error subplot
    axes[0].plot(t_test, position_error, 'g-', linewidth=2.5)
    axes[0].set_xlabel('Time (s)', fontsize=14)
    axes[0].set_ylabel('Total Position Error', fontsize=14)
    axes[0].set_title('Position Error', fontsize=14, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=12)
    
    # Energy error subplot
    axes[1].plot(t_test, energy_error_normalized, 'purple', linewidth=2.5)
    axes[1].set_xlabel('Time (s)', fontsize=14)
    axes[1].set_ylabel('Normalized Energy Error', fontsize=14)
    axes[1].set_title('Normalized Energy Error', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=12)
    
    # Add reference lines
    axes[1].axhline(y=1e-2, color='orange', linestyle='--', alpha=0.7, label='1% Error')
    axes[1].axhline(y=1e-3, color='red', linestyle='--', alpha=0.7, label='0.1% Error')
    axes[1].legend(fontsize=12)
    
    plt.suptitle(f'Error Summary - Test Case {test_case_num}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(error_dir, f"error_summary_test_{test_case_num}.png"), 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    if verbose:
        print(f"Individual error plots saved to: {error_dir}")

def get_cartesian_data(particle, trajectory_data):
    """
    Convert trajectory data to cartesian coordinates.
    
    Args:
        particle: Physical system instance
        trajectory_data: State trajectory data [time, state_dim]
    
    Returns:
        cartesian_positions: Cartesian positions [time, spatial_dims]
    """
    # Ensure data is on CPU and converted to numpy for animation compatibility
    if hasattr(trajectory_data, 'cpu'):
        trajectory_data = trajectory_data.cpu().numpy()
    elif isinstance(trajectory_data, torch.Tensor):
        trajectory_data = trajectory_data.detach().cpu().numpy()
    elif not isinstance(trajectory_data, np.ndarray):
        trajectory_data = np.array(trajectory_data)
    
    if hasattr(particle, 'to_cartesian') and callable(getattr(particle, 'to_cartesian')):
        # Use particle's conversion method - it expects full state
        try:
            return particle.to_cartesian(trajectory_data)
        except Exception as e:
            print(f"Warning: particle.to_cartesian failed: {e}")
            # Fall back to position extraction
            pass
    
    # Assume already cartesian, extract position coordinates only
    num_pos_coords = particle.dof // 2
    if hasattr(trajectory_data, 'shape'):
        if len(trajectory_data.shape) >= 2 and trajectory_data.shape[1] >= num_pos_coords:
            return trajectory_data[:, :num_pos_coords]
        else:
            print(f"Warning: trajectory_data shape {trajectory_data.shape} insufficient for {num_pos_coords} coordinates")
            return trajectory_data
    else:
        print(f"Warning: trajectory_data has no shape attribute, returning as-is")
        return trajectory_data

def create_spring_zigzag(x0, y0, x1, y1, n_coils=8, coil_width=0.1):
    """
    Create a zigzag spring pattern from (x0, y0) to (x1, y1).
    
    Args:
        x0, y0: Start coordinates
        x1, y1: End coordinates
        n_coils: Number of coils in the spring
        coil_width: Width of each coil relative to spring length
    
    Returns:
        spring_x, spring_y: Arrays of x and y coordinates for the spring
    """
    # Vector from start to end
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return np.array([x0, x1]), np.array([y0, y1])
    
    # Unit vectors along and perpendicular to spring
    unit_x = dx / length
    unit_y = dy / length
    perp_x = -unit_y  # Perpendicular unit vector
    perp_y = unit_x
    
    # Create zigzag pattern
    n_points = n_coils * 4 + 1  # 4 points per coil + start point
    t = np.linspace(0, 1, n_points)
    
    # Zigzag amplitude (width of coil)
    amplitude = coil_width * length
    
    # Create zigzag pattern in local coordinates
    zigzag = np.zeros_like(t)
    for i in range(1, n_points - 1):
        coil_phase = (t[i] * n_coils) % 1
        if coil_phase < 0.25:
            zigzag[i] = amplitude * (4 * coil_phase)
        elif coil_phase < 0.5:
            zigzag[i] = amplitude * (2 - 4 * coil_phase)
        elif coil_phase < 0.75:
            zigzag[i] = amplitude * (-4 * coil_phase + 2)
        else:
            zigzag[i] = amplitude * (4 * coil_phase - 4)
    
    # Transform to world coordinates
    spring_x = x0 + t * dx + zigzag * perp_x
    spring_y = y0 + t * dy + zigzag * perp_y
    
    return spring_x, spring_y

def detect_system_structure(particle, cartesian_data):
    """
    Detect the physical structure of the system for visualization.
    
    Returns:
        dict: Structure information for plotting
    """
    # Get the class name of the particle object
    class_name = particle.__class__.__name__.lower()
    
    structure = {
        'type': 'simple',
        'num_bodies': 1,
        'connections': [],
        'labels': []
    }
    
    # Detect system type based on class name
    if 'triple_pendulum' in class_name:
        structure.update({
            'type': 'triple_pendulum',
            'num_bodies': 3,
            'connections': [(0, 1), (1, 2), (2, 3)],  # origin->mass1->mass2->mass3
            'labels': ['Origin', 'Mass 1', 'Mass 2', 'Mass 3']
        })
    elif 'double_pendulum' in class_name:
        structure.update({
            'type': 'double_pendulum',
            'num_bodies': 2,
            'connections': [(0, 1), (1, 2)],  # origin->mass1->mass2
            'labels': ['Origin', 'Mass 1', 'Mass 2']
        })
    elif 'spring_pendulum' in class_name:
        structure.update({
            'type': 'spring_pendulum', 
            'num_bodies': 1,
            'connections': [(0, 1)],  # origin->mass
            'labels': ['Origin', 'Mass']
        })
    elif 'sphere_geodesic' in class_name:
        structure.update({
            'type': 'sphere_geodesic',
            'num_bodies': 1,
            'connections': [],
            'labels': ['Particle on Sphere']
        })
    elif 'harmonic_oscillator' in class_name:
        structure.update({
            'type': 'harmonic_oscillator',
            'num_bodies': 1,
            'connections': [],
            'labels': ['Mass']
        })
    elif 'constant_force' in class_name:
        structure.update({
            'type': 'constant_force',
            'num_bodies': 1,
            'connections': [],
            'labels': ['Particle']
        })
    else:
        # Default case - try to infer from dimensions
        if cartesian_data.shape[1] <= 2:
            structure.update({
                'type': 'simple',
                'num_bodies': 1,
                'connections': [],
                'labels': ['Particle']
            })
        else:
            # Unknown multi-body system - make a guess based on dimensions
            n_bodies = cartesian_data.shape[1] // 2  # Assume 2D positions for each body
            structure.update({
                'type': f'multi_body_{n_bodies}',
                'num_bodies': n_bodies,
                'connections': [(i, i+1) for i in range(n_bodies)],  # Chain connection
                'labels': [f'Body {i+1}' for i in range(n_bodies+1)]  # Include origin
            })
    
    return structure

def create_cartesian_animations(particle, analytical_trajectory, predicted_trajectory, t_test, 
                               save_dir, test_case_num, verbose=True):
    """
    Create animated cartesian trajectory plots with fading trails.
    """
    try:
        import matplotlib.animation as animation
    except ImportError:
        if verbose:
            print("Cartesian animation skipped: matplotlib.animation not available")
        return
    
    # Create cartesian animations directory
    anim_dir = os.path.join(save_dir, "cartesian_animations", f"test_case_{test_case_num}")
    try:
        os.makedirs(anim_dir, exist_ok=True)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not create animations directory: {e}")
        return
    
    # Convert trajectories to CPU for animation compatibility (get_cartesian_data handles this)
    analytical_cart = get_cartesian_data(particle, analytical_trajectory)
    predicted_cart = get_cartesian_data(particle, predicted_trajectory)
    
    # Detect system structure
    structure = detect_system_structure(particle, analytical_cart)
    
    try:
        # Create individual animations
        create_single_trajectory_animation(analytical_cart, t_test, structure, 
                                         anim_dir, f"analytical_test_{test_case_num}", 
                                         "Analytical Trajectory", 'blue', verbose)
        
        create_single_trajectory_animation(predicted_cart, t_test, structure,
                                         anim_dir, f"predicted_test_{test_case_num}",
                                         "LNN Predicted Trajectory", 'red', verbose)
        
        # Create side-by-side comparison animation
        create_comparison_animation(analytical_cart, predicted_cart, t_test, structure,
                                   anim_dir, f"comparison_test_{test_case_num}", verbose)
    except Exception as e:
        if verbose:
            print(f"Animation creation failed: {e}")
    
    # Create static plots with fading effect
    create_static_faded_trajectories_improved(analytical_cart, predicted_cart, t_test, structure,
                                            anim_dir, f"static_traces_test_{test_case_num}", verbose)

def create_single_trajectory_animation(cartesian_data, t_test, structure, save_dir, 
                                     filename, title, color, verbose=True, trail_length=50):
    """Create animation for a single trajectory with fading trail."""
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title(f'{title}', fontsize=16, fontweight='bold')
    
    # Set up data for plotting - use end effector mass for multi-pendulum systems
    if structure['type'] == 'triple_pendulum' and cartesian_data.shape[1] >= 6:
        # For triple pendulum, track the third mass (end effector)
        x_data = cartesian_data[:, 4]  # x3 - third mass
        y_data = cartesian_data[:, 5]  # y3 - third mass
        # Get bounds that include all three masses
        all_x = np.concatenate([cartesian_data[:, 0], cartesian_data[:, 2], cartesian_data[:, 4]])  # x1, x2, x3
        all_y = np.concatenate([cartesian_data[:, 1], cartesian_data[:, 3], cartesian_data[:, 5]])  # y1, y2, y3
    elif structure['type'] == 'double_pendulum' and cartesian_data.shape[1] >= 4:
        # For double pendulum, track the second mass (end effector)
        x_data = cartesian_data[:, 2]  # x2 - second mass
        y_data = cartesian_data[:, 3]  # y2 - second mass
        # Get bounds that include both masses
        all_x = np.concatenate([cartesian_data[:, 0], cartesian_data[:, 2]])  # x1 and x2
        all_y = np.concatenate([cartesian_data[:, 1], cartesian_data[:, 3]])  # y1 and y2
    else:
        # For other systems, use first position
        if cartesian_data.shape[1] > 1:
            x_data = cartesian_data[:, 0]
            y_data = cartesian_data[:, 1] 
            all_x, all_y = x_data, y_data
        else:
            x_data = cartesian_data.flatten()
            y_data = np.zeros_like(x_data)
            all_x, all_y = x_data, y_data
    
    # Set axis limits to show full pendulum range
    x_range = np.max(all_x) - np.min(all_x)
    y_range = np.max(all_y) - np.min(all_y)
    padding = max(x_range, y_range) * 0.15 + 0.1
    
    ax.set_xlim(np.min(all_x) - padding, np.max(all_x) + padding)
    ax.set_ylim(np.min(all_y) - padding, np.max(all_y) + padding)
    ax.set_xlabel('X Position (m)', fontsize=14)
    ax.set_ylabel('Y Position (m)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Initialize plot elements
    trail_lines = []
    structure_lines = []
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame):
        # Clear previous frame
        for line in trail_lines + structure_lines:
            line.remove()
        trail_lines.clear()
        structure_lines.clear()
        
        # Draw smooth fading trail with slower fade (longer trail)
        trail_length_long = min(200, frame)  # Much longer trail
        start_idx = max(0, frame - trail_length_long)
        for i in range(start_idx, frame):
            alpha = max(0.05, 1.0 - (frame - i) / trail_length_long)  # Slower fade
            if i + 1 < len(x_data):
                line = ax.plot([x_data[i], x_data[i+1]], [y_data[i], y_data[i+1]], 
                             color=color, alpha=alpha, linewidth=1.5)[0]
                trail_lines.append(line)
        
        # Draw structural elements for multi-pendulum systems (no dots on trail)
        if structure['type'] == 'triple_pendulum' and cartesian_data.shape[1] >= 6:
            mass1_x, mass1_y = cartesian_data[frame, 0], cartesian_data[frame, 1]
            mass2_x, mass2_y = cartesian_data[frame, 2], cartesian_data[frame, 3]
            mass3_x, mass3_y = cartesian_data[frame, 4], cartesian_data[frame, 5]
            
            # Draw pendulum structure
            rod1 = ax.plot([0, mass1_x], [0, mass1_y], 'k-', linewidth=3, alpha=0.8)[0]
            rod2 = ax.plot([mass1_x, mass2_x], [mass1_y, mass2_y], 'k-', linewidth=3, alpha=0.8)[0]
            rod3 = ax.plot([mass2_x, mass3_x], [mass2_y, mass3_y], 'k-', linewidth=3, alpha=0.8)[0]
            
            # Mark masses (these are the physical masses, not trail markers)
            mass1_marker = ax.plot(mass1_x, mass1_y, 'o', color='darkblue', markersize=12, 
                                 markeredgecolor='black', markeredgewidth=1)[0]
            mass2_marker = ax.plot(mass2_x, mass2_y, 'o', color='darkorange', markersize=12, 
                                 markeredgecolor='black', markeredgewidth=1)[0]
            mass3_marker = ax.plot(mass3_x, mass3_y, 'o', color='darkred', markersize=12, 
                                 markeredgecolor='black', markeredgewidth=1)[0]
            
            structure_lines.extend([rod1, rod2, rod3, mass1_marker, mass2_marker, mass3_marker])
        
        elif structure['type'] == 'double_pendulum' and cartesian_data.shape[1] >= 4:
            mass1_x, mass1_y = cartesian_data[frame, 0], cartesian_data[frame, 1]
            mass2_x, mass2_y = cartesian_data[frame, 2], cartesian_data[frame, 3]
            
            # Draw pendulum structure
            rod1 = ax.plot([0, mass1_x], [0, mass1_y], 'k-', linewidth=3, alpha=0.8)[0]
            rod2 = ax.plot([mass1_x, mass2_x], [mass1_y, mass2_y], 'k-', linewidth=3, alpha=0.8)[0]
            
            # Mark masses (these are the physical masses, not trail markers)
            mass1_marker = ax.plot(mass1_x, mass1_y, 'o', color='darkblue', markersize=12, 
                                 markeredgecolor='black', markeredgewidth=1)[0]
            mass2_marker = ax.plot(mass2_x, mass2_y, 'o', color='darkred', markersize=12, 
                                 markeredgecolor='black', markeredgewidth=1)[0]
            
            structure_lines.extend([rod1, rod2, mass1_marker, mass2_marker])
        
        elif structure['type'] == 'spring_pendulum':
            # Draw realistic zigzag spring
            spring_x, spring_y = create_spring_zigzag(0, 0, x_data[frame], y_data[frame], n_coils=8, coil_width=0.1)
            spring_line = ax.plot(spring_x, spring_y, 'k-', linewidth=2, alpha=0.8)[0]
            
            # Draw mass at end
            mass_marker = ax.plot(x_data[frame], y_data[frame], 'o', color='darkred', markersize=10, 
                                markeredgecolor='black', markeredgewidth=1)[0]
            
            structure_lines.extend([spring_line, mass_marker])
        
        time_text.set_text(f'Time: {t_test[frame]:.2f} s')
        
        return trail_lines + structure_lines + [time_text]
    
    # Create and save animation
    anim = animation.FuncAnimation(fig, animate, frames=len(t_test), 
                                  interval=50, blit=False, repeat=True)
    
    try:
        anim.save(os.path.join(save_dir, f"{filename}.mp4"), writer='ffmpeg', fps=20, dpi=150)
        if verbose:
            print(f"Animation saved: {filename}.mp4")
    except Exception as e:
        try:
            anim.save(os.path.join(save_dir, f"{filename}.gif"), writer='pillow', fps=10, dpi=100)
            if verbose:
                print(f"Animation saved as GIF: {filename}.gif")
        except Exception as e2:
            if verbose:
                print(f"Animation save failed: {e2}")
    
    plt.close(fig)

def create_comparison_animation(analytical_cart, predicted_cart, t_test, structure, 
                               save_dir, filename, verbose=True):
    """Create side-by-side comparison animation."""
    import matplotlib.animation as animation
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Trajectory Comparison: Analytical vs LNN Predicted', fontsize=16, fontweight='bold')
    
    # Setup both subplots with proper scaling for multi-pendulum systems
    for ax, data, title, color in [(ax1, analytical_cart, 'Analytical', 'blue'), 
                                   (ax2, predicted_cart, 'LNN Predicted', 'red')]:
        
        # Determine bounds based on system type
        if structure['type'] == 'triple_pendulum' and data.shape[1] >= 6:
            all_x = np.concatenate([data[:, 0], data[:, 2], data[:, 4]])  # x1, x2, x3
            all_y = np.concatenate([data[:, 1], data[:, 3], data[:, 5]])  # y1, y2, y3
        elif structure['type'] == 'double_pendulum' and data.shape[1] >= 4:
            all_x = np.concatenate([data[:, 0], data[:, 2]])  # x1 and x2
            all_y = np.concatenate([data[:, 1], data[:, 3]])  # y1 and y2
        else:
            if data.shape[1] > 1:
                all_x = data[:, 0]
                all_y = data[:, 1]
            else:
                all_x = data.flatten()
                all_y = np.zeros_like(all_x)
        
        x_range = np.max(all_x) - np.min(all_x)
        y_range = np.max(all_y) - np.min(all_y)
        padding = max(x_range, y_range) * 0.15 + 0.1
        
        ax.set_xlim(np.min(all_x) - padding, np.max(all_x) + padding)
        ax.set_ylim(np.min(all_y) - padding, np.max(all_y) + padding)
        ax.set_xlabel('X Position (m)', fontsize=14)
        ax.set_ylabel('Y Position (m)', fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Animation elements
    trail_lines1, trail_lines2 = [], []
    struct_lines1, struct_lines2 = [], []
    
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate_comparison(frame):
        # Clear previous frame
        for lines in [trail_lines1, trail_lines2, struct_lines1, struct_lines2]:
            for line in lines:
                line.remove()
            lines.clear()
        
        # Animate both trajectories
        for ax, data, trail_lines, struct_lines, color in [
            (ax1, analytical_cart, trail_lines1, struct_lines1, 'blue'),
            (ax2, predicted_cart, trail_lines2, struct_lines2, 'red')
        ]:
            # Determine what to track based on system type
            if structure['type'] == 'triple_pendulum' and data.shape[1] >= 6:
                # Track third mass (end effector)
                x_data = data[:, 4]
                y_data = data[:, 5]
            elif structure['type'] == 'double_pendulum' and data.shape[1] >= 4:
                # Track second mass (end effector)
                x_data = data[:, 2]
                y_data = data[:, 3]
            else:
                if data.shape[1] > 1:
                    x_data = data[:, 0]
                    y_data = data[:, 1]
                else:
                    x_data = data.flatten()
                    y_data = np.zeros_like(x_data)
            
            # Smooth fading trail with slower fade
            trail_length_long = min(200, frame)  # Much longer trail
            start_idx = max(0, frame - trail_length_long)
            for i in range(start_idx, frame):
                if i + 1 < len(x_data):
                    alpha = max(0.05, 1.0 - (frame - i) / trail_length_long)  # Slower fade
                    line = ax.plot([x_data[i], x_data[i+1]], [y_data[i], y_data[i+1]], 
                                 color=color, alpha=alpha, linewidth=1.5)[0]
                    trail_lines.append(line)
            
            # Draw structural elements (no trail dots)
            if structure['type'] == 'triple_pendulum' and data.shape[1] >= 6:
                mass1_x, mass1_y = data[frame, 0], data[frame, 1]
                mass2_x, mass2_y = data[frame, 2], data[frame, 3]
                mass3_x, mass3_y = data[frame, 4], data[frame, 5]
                
                # Draw pendulum structure
                rod1 = ax.plot([0, mass1_x], [0, mass1_y], 'k-', linewidth=3, alpha=0.8)[0]
                rod2 = ax.plot([mass1_x, mass2_x], [mass1_y, mass2_y], 'k-', linewidth=3, alpha=0.8)[0]
                rod3 = ax.plot([mass2_x, mass3_x], [mass2_y, mass3_y], 'k-', linewidth=3, alpha=0.8)[0]
                
                # Mark masses
                mass1_marker = ax.plot(mass1_x, mass1_y, 'o', color='darkblue', markersize=10)[0]
                mass2_marker = ax.plot(mass2_x, mass2_y, 'o', color='darkorange', markersize=10)[0]
                mass3_marker = ax.plot(mass3_x, mass3_y, 'o', color='darkred', markersize=10)[0]
                
                struct_lines.extend([rod1, rod2, rod3, mass1_marker, mass2_marker, mass3_marker])
            
            elif structure['type'] == 'double_pendulum' and data.shape[1] >= 4:
                mass1_x, mass1_y = data[frame, 0], data[frame, 1]
                mass2_x, mass2_y = data[frame, 2], data[frame, 3]
                
                # Draw pendulum structure
                rod1 = ax.plot([0, mass1_x], [0, mass1_y], 'k-', linewidth=3, alpha=0.8)[0]
                rod2 = ax.plot([mass1_x, mass2_x], [mass1_y, mass2_y], 'k-', linewidth=3, alpha=0.8)[0]
                
                # Mark masses
                mass1_marker = ax.plot(mass1_x, mass1_y, 'o', color='darkblue', markersize=10)[0]
                mass2_marker = ax.plot(mass2_x, mass2_y, 'o', color='darkred', markersize=10)[0]
                
                struct_lines.extend([rod1, rod2, mass1_marker, mass2_marker])
            
            elif structure['type'] == 'spring_pendulum':
                # Draw realistic zigzag spring
                spring_x, spring_y = create_spring_zigzag(0, 0, x_data[frame], y_data[frame], n_coils=8, coil_width=0.1)
                spring_line = ax.plot(spring_x, spring_y, 'k-', linewidth=2, alpha=0.8)[0]
                
                # Draw mass at end
                mass_marker = ax.plot(x_data[frame], y_data[frame], 'o', color='darkred', markersize=10, 
                                    markeredgecolor='black', markeredgewidth=1)[0]
                
                struct_lines.extend([spring_line, mass_marker])
        
        time_text.set_text(f'Time: {t_test[frame]:.2f} s')
        
        return trail_lines1 + trail_lines2 + struct_lines1 + struct_lines2 + [time_text]
    
    anim = animation.FuncAnimation(fig, animate_comparison, frames=len(t_test),
                                  interval=50, blit=False, repeat=True)
    
    # Save animation
    try:
        anim.save(os.path.join(save_dir, f"{filename}.mp4"), writer='ffmpeg', fps=20, dpi=150)
        if verbose:
            print(f"Comparison animation saved: {filename}.mp4")
    except Exception as e:
        try:
            anim.save(os.path.join(save_dir, f"{filename}.gif"), writer='pillow', fps=10, dpi=100)
            if verbose:
                print(f"Comparison animation saved as GIF: {filename}.gif")
        except Exception as e2:
            if verbose:
                print(f"Comparison animation save failed: {e2}")
    
    plt.close(fig)

def create_static_faded_trajectories(analytical_cart, predicted_cart, t_test, structure,
                                   save_dir, filename, verbose=True):
    """Create static plots with faded trajectory traces."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_title('Trajectory Comparison with Faded Traces', fontsize=16, fontweight='bold')
    
    # Plot both trajectories with alpha gradient
    for data, label, color in [(analytical_cart, 'Analytical', 'blue'), 
                               (predicted_cart, 'NN Predicted', 'red')]:
        x_data = data[:, 0] if data.shape[1] > 0 else data.flatten()
        y_data = data[:, 1] if data.shape[1] > 1 else np.zeros_like(x_data)
        
        # Create faded trail effect
        n_points = len(x_data)
        for i in range(n_points - 1):
            alpha = max(0.1, i / n_points)  # Fade from start to end
            ax.plot([x_data[i], x_data[i+1]], [y_data[i], y_data[i+1]], 
                   color=color, alpha=alpha, linewidth=1.5)
        
        # Mark start and end points
        ax.plot(x_data[0], y_data[0], 'o', color=color, markersize=10, 
               markeredgecolor='green', markeredgewidth=2, label=f'{label} Start')
        ax.plot(x_data[-1], y_data[-1], 's', color=color, markersize=10,
               markeredgecolor='red', markeredgewidth=2, label=f'{label} End')
    
    # Add structural elements for start and end configurations
    if structure['type'] == 'double_pendulum' and analytical_cart.shape[1] >= 4:
        # Start configuration
        ax.plot([0, analytical_cart[0,0], analytical_cart[0,2]], 
                [0, analytical_cart[0,1], analytical_cart[0,3]], 
                'go-', alpha=0.7, linewidth=3, markersize=8, label='Start Config')
        # End configuration  
        ax.plot([0, analytical_cart[-1,0], analytical_cart[-1,2]], 
                [0, analytical_cart[-1,1], analytical_cart[-1,3]], 
                'ro-', alpha=0.7, linewidth=3, markersize=8, label='End Config')
    
    ax.set_xlabel('X Position (m)', fontsize=14)
    ax.set_ylabel('Y Position (m)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), 
               dpi=300, bbox_inches='tight', facecolor='white')
    
    if verbose:
        print(f"Static faded trajectory plot saved: {filename}.png")
    
    plt.close()

def create_static_faded_trajectories_improved(analytical_cart, predicted_cart, t_test, structure,
                                            save_dir, filename, verbose=True):
    """Create separate static plots for analytical and LNN predicted trajectories with density gradient."""
    
    # Convert to numpy if needed
    if hasattr(analytical_cart, 'cpu'):
        analytical_data = analytical_cart.cpu().numpy()
    else:
        analytical_data = analytical_cart if isinstance(analytical_cart, np.ndarray) else np.array(analytical_cart)
        
    if hasattr(predicted_cart, 'cpu'):
        predicted_data = predicted_cart.cpu().numpy()
    else:
        predicted_data = predicted_cart if isinstance(predicted_cart, np.ndarray) else np.array(predicted_cart)
    
    # Determine what to track - prioritize second mass for double pendulum
    def get_trace_data(data):
        if structure['type'] == 'double_pendulum' and data.shape[1] >= 4:
            return data[:, 2], data[:, 3]  # Track second mass (end effector)
        elif data.shape[1] > 1:
            return data[:, 0], data[:, 1]  # Track first position for other systems
        else:
            x_data = data.flatten()
            return x_data, np.zeros_like(x_data)
    
    analytical_x, analytical_y = get_trace_data(analytical_data)
    predicted_x, predicted_y = get_trace_data(predicted_data)
    
    # Get common axis limits for both plots
    all_x = np.concatenate([analytical_x, predicted_x])
    all_y = np.concatenate([analytical_y, predicted_y])
    x_range = np.max(all_x) - np.min(all_x)
    y_range = np.max(all_y) - np.min(all_y)
    padding = max(x_range, y_range) * 0.15 + 0.1
    xlim = (np.min(all_x) - padding, np.max(all_x) + padding)
    ylim = (np.min(all_y) - padding, np.max(all_y) + padding)
    
    n_points = len(analytical_x)
    segment_length = max(1, n_points // 100)  # ~100 segments for smooth gradient
    
    # Create individual plots for each trajectory
    for data_x, data_y, data_full, color, label, file_suffix in [
        (analytical_x, analytical_y, analytical_data, 'blue', 'Analytical', 'analytical'),
        (predicted_x, predicted_y, predicted_data, 'red', 'LNN Predicted', 'lnn_predicted')
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_title(f'{label} Trajectory - Complete Trace with Density Gradient', 
                    fontsize=16, fontweight='bold')
        
        # Plot trajectory in segments with density gradient
        for i in range(0, n_points - segment_length, segment_length):
            end_idx = min(i + segment_length, n_points - 1)
            alpha = 0.15 + 0.85 * (i / n_points)  # Range 0.15 to 1.0
            
            ax.plot(data_x[i:end_idx+1], data_y[i:end_idx+1], 
                   color=color, alpha=alpha, linewidth=2.5)
        
        # Mark start and end
        ax.scatter(data_x[0], data_y[0], color=color, s=120, marker='o', 
                  edgecolor='darkgreen', linewidth=2, label='Start', zorder=5)
        ax.scatter(data_x[-1], data_y[-1], color=color, s=120, marker='s', 
                  edgecolor='darkred', linewidth=2, label='End', zorder=5)
    
        # Add pendulum structure if double pendulum
        if structure['type'] == 'double_pendulum' and data_full.shape[1] >= 4:
            # Start configuration (green)
            ax.plot([0, data_full[0,0], data_full[0,2]], [0, data_full[0,1], data_full[0,3]], 
                   'g-', alpha=0.8, linewidth=3, label='Initial Structure')
            # End configuration (red)
            ax.plot([0, data_full[-1,0], data_full[-1,2]], [0, data_full[-1,1], data_full[-1,3]], 
                   'r-', alpha=0.8, linewidth=3, label='Final Structure')
        
        ax.set_xlabel('X Position (m)', fontsize=14)
        ax.set_ylabel('Y Position (m)', fontsize=14)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_aspect('equal')
        
        # Save individual plot
        individual_filename = f"{filename}_{file_suffix}"
        try:
            plt.savefig(os.path.join(save_dir, f"{individual_filename}.png"), dpi=150, bbox_inches='tight')
            if verbose:
                print(f"Static trajectory plot saved: {individual_filename}.png")
        except Exception as e:
            if verbose:
                print(f"Static plot save failed for {label}: {e}")
        
        plt.close(fig)

def plot_lagrangian_1d(particle, model, q_qdot_test_list, nn_test_list, t_test, dtype, device, save_path, verbose=True):
    """
    Plot 1D Lagrangian comparisons for time series data.
    """
    if particle.dof != 2:
        if verbose:
            print("Lagrangian 1D plots only available for 1D systems (dof=2)")
        return

    # Create debug_results directory
    debug_dir = None
    if save_path:
        debug_dir = os.path.join(save_path, "debug_results")
        try:
            os.makedirs(debug_dir, exist_ok=True)
            if verbose:
                print(f"1D Lagrangian plots will be saved to: {debug_dir}")
        except Exception as e:
            print(f"Warning: Could not create debug directory {debug_dir}: {e}")
            debug_dir = None
    elif verbose:
        print("No save_path provided - 1D Lagrangian plots will not be saved to file")

    for i in range(len(nn_test_list)):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        particle.plot_lagrangian(t_test, q_qdot_test_list[i].cpu(), particle.lagrangian, 
                                f'Analytical lagrangian of test {i+1}')
        particle.plot_lagrangian(t_test, nn_test_list[i], particle.lagrangian, 
                                f'Analytical lagrangian of predicted {i+1}')
        particle.plot_lagrangian(t_test, q_qdot_test_list[i], model.plot_lagrangian, 
                                f'Learned lagrangian of test {i+1}')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Lagrangian')
        plt.title(f'Lagrangian Comparison - Test Case {i+1}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        if debug_dir:
            plt.savefig(os.path.join(debug_dir, f"lagrangian_1d_test_{i+1}.png"), dpi=300, bbox_inches='tight')
            if verbose:
                print(f"1D Lagrangian test {i+1} saved to {debug_dir}/lagrangian_1d_test_{i+1}.png")
        
        if verbose:
            plt.show()
        else:
            plt.close()

def plot_lagrangian_3d(particle, model, dtype, device, save_path, verbose=True):
    """
    Plot 3D and contour plots of the Lagrangian landscape for 1D systems.
    """
    if particle.dof != 2:
        if verbose:
            print("Lagrangian 3D plots only available for 1D systems (dof=2)")
        return

    # Create debug_results directory
    debug_dir = None
    if save_path:
        debug_dir = os.path.join(save_path, "debug_results")
        try:
            os.makedirs(debug_dir, exist_ok=True)
            if verbose:
                print(f"3D Lagrangian plots will be saved to: {debug_dir}")
        except Exception as e:
            print(f"Warning: Could not create debug directory {debug_dir}: {e}")
            debug_dir = None
    elif verbose:
        print("No save_path provided - 3D Lagrangian plots will not be saved to file")

    from mpl_toolkits.mplot3d import Axes3D
    
    def Lagr(x, y):
        xy_stacked = (x, y)
        return particle.lagrangian(xy_stacked)

    plot_n = 100
    x = 2*torch.rand(plot_n)-1
    y = 2*torch.rand(plot_n)-1
    X, Y = torch.meshgrid(x, y, indexing='xy')
    Z = Lagr(X,Y)
    
    # Analytical Lagrangian 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('q')
    ax.set_ylabel('q̇')
    ax.set_zlabel('Analytical Lagrangian')
    ax.set_title('Analytical Lagrangian 3D Plot')
    fig.colorbar(surf)
    
    if debug_dir:
        plt.savefig(os.path.join(debug_dir, "analytical_lagrangian_3d.png"), dpi=300, bbox_inches='tight')
    if verbose:
        plt.show()
    else:
        plt.close()

    # Learned Lagrangian 3D plot
    def learned_lagr(x, y, device=None):
        if device is None:
            device = torch.device('cpu')
        xy_stacked = torch.stack([x.flatten(), y.flatten()], dim=1).to(device, dtype)
        return model.plot_lagrangian(xy_stacked)

    Z_learned = torch.reshape(learned_lagr(X, Y, device), (plot_n, plot_n)).cpu()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(X, Y, Z_learned, cmap='viridis')
    ax.set_xlabel('q')
    ax.set_ylabel('q̇')
    ax.set_zlabel('Learned Lagrangian')
    ax.set_title('Learned Lagrangian 3D Plot')
    fig.colorbar(surf, ax=ax)
    
    if debug_dir:
        plt.savefig(os.path.join(debug_dir, "learned_lagrangian_3d.png"), dpi=300, bbox_inches='tight')
    if verbose:
        plt.show()
    else:
        plt.close()

    # Difference plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(X, Y, Z-2*Z_learned, cmap='viridis')

    ax.set_xlabel('q')
    ax.set_ylabel('q̇')
    ax.set_zlabel('Difference')
    ax.set_title('Difference of Lagrangians 3D Plot')
    fig.colorbar(surf, ax=ax)
    
    if debug_dir:
        plt.savefig(os.path.join(debug_dir, "lagrangian_difference_3d.png"), dpi=300, bbox_inches='tight')
    if verbose:
        plt.show()
    else:
        plt.close()

    # Contour plots
    fig = plt.figure(figsize=(15, 5))
    
    # Analytical contour
    ax1 = fig.add_subplot(1, 3, 1)
    contour1 = ax1.contour(X, Y, Z, cmap='viridis')
    ax1.set_xlabel('q')
    ax1.set_ylabel('q̇')
    ax1.set_title('Analytical Lagrangian Contour')
    fig.colorbar(contour1, ax=ax1)

    # Learned contour
    ax2 = fig.add_subplot(1, 3, 2)
    contour2 = ax2.contour(X, Y, Z_learned, cmap='viridis')
    ax2.set_xlabel('q')
    ax2.set_ylabel('q̇')
    ax2.set_title('Learned Lagrangian Contour')
    fig.colorbar(contour2, ax=ax2)

    # Difference contour
    ax3 = fig.add_subplot(1, 3, 3)
    contour3 = ax3.contour(X, Y, Z-Z_learned, cmap='viridis')
    ax3.set_xlabel('q')
    ax3.set_ylabel('q̇')
    ax3.set_title('Difference of Lagrangians')
    fig.colorbar(contour3, ax=ax3)

    plt.tight_layout()
    
    if debug_dir:
        plt.savefig(os.path.join(debug_dir, "lagrangian_contour_comparison.png"), dpi=300, bbox_inches='tight')
    if verbose:
        plt.show()
    else:
        plt.close()

# =============================================================================
# VALIDATION ANALYSIS FUNCTIONS
# =============================================================================

def plot_training_validation_curves(history, save_path=None, verbose=True):
    """
    Plot training and validation loss curves over epochs.
    
    Args:
        history: Dictionary containing 'train_loss', 'val_loss', 'lr' lists
        save_path: Path to save plots
        verbose: Whether to display plots and print messages
    """
    # Create validation_results directory
    val_dir = None
    if save_path:
        val_dir = os.path.join(save_path, "validation_results")
        try:
            os.makedirs(val_dir, exist_ok=True)
            if verbose:
                print(f"Validation plots will be saved to: {val_dir}")
        except Exception as e:
            print(f"Warning: Could not create validation directory {val_dir}: {e}")
            val_dir = None
    elif verbose:
        print("No save_path provided - validation plots will not be saved to file")
    
    # Check if validation data exists
    has_validation = 'val_loss' in history and len(history['val_loss']) > 0
    
    if not has_validation:
        if verbose:
            print("No validation data found in history - skipping validation plots")
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Highlight potential overfitting
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    
    # Find where validation loss starts consistently increasing while training decreases
    window = 5  # Look at 5-epoch windows
    if len(epochs) > window:
        overfitting_detected = False
        for i in range(window, len(epochs)):
            train_trend = np.mean(train_loss[i-window:i]) - np.mean(train_loss[i-window//2:i-window//2+window//2])
            val_trend = np.mean(val_loss[i-window:i]) - np.mean(val_loss[i-window//2:i-window//2+window//2])
            
            if train_trend < -0.001 and val_trend > 0.001:  # Training decreasing, validation increasing
                ax1.axvline(x=epochs[i], color='orange', linestyle='--', alpha=0.7, label='Potential Overfitting')
                overfitting_detected = True
                break
        
        if overfitting_detected:
            ax1.legend()
    
    # Plot 2: Learning rate
    ax2.plot(epochs, history['lr'], 'g-', label='Learning Rate', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    if val_dir:
        plt.savefig(os.path.join(val_dir, "training_validation_curves.png"), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        if verbose:
            print("Training/validation curves saved")
    
    if verbose:
        plt.show()
    else:
        plt.close()
    
    # Print summary statistics
    min_train_loss = min(history['train_loss'])
    min_val_loss = min(history['val_loss'])
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    best_val_epoch = np.argmin(history['val_loss']) + 1
    
    summary = [
        "=== VALIDATION SUMMARY ===",
        f"Minimum Training Loss: {min_train_loss:.6f}",
        f"Minimum Validation Loss: {min_val_loss:.6f}",
        f"Final Training Loss: {final_train_loss:.6f}",
        f"Final Validation Loss: {final_val_loss:.6f}",
        f"Best Validation Epoch: {best_val_epoch}",
        f"Validation Gap (Final): {abs(final_val_loss - final_train_loss):.6f}"
    ]
    
    if verbose:
        for line in summary:
            print(line)
    
    # Save summary to file
    if val_dir:
        try:
            with open(os.path.join(val_dir, "validation_summary.txt"), 'w') as f:
                f.write('\n'.join(summary))
            if verbose:
                print(f"Validation summary saved to {val_dir}/validation_summary.txt")
        except Exception as e:
            print(f"Warning: Could not save validation summary: {e}")

def compute_validation_metrics(particle, model, save_path=None, dtype=torch.float32, device=None, 
                              position_bounds_override=None, velocity_bounds_override=None,
                              validation_samples=None, verbose=True):
    """
    Compute validation metrics on fresh validation data for debug analysis.
    
    Args:
        particle: Physical system instance
        model: Trained model
        save_path: Path to save analysis
        dtype: Data type for tensors
        device: Device to place tensors on  
        position_bounds_override: Custom position bounds for validation
        velocity_bounds_override: Custom velocity bounds for validation
        validation_samples: Custom number of validation samples
        verbose: Whether to print results and save files
    """
    debug_dir = None
    if save_path:
        debug_dir = os.path.join(save_path, "debug_results")
        try:
            os.makedirs(debug_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create debug directory {debug_dir}: {e}")
            debug_dir = None
    elif verbose:
        print("No save_path provided - validation metrics will not be saved to file")
    
    # Create fresh validation data
    validation_data = create_validation_data(
        particle, dtype, device,
        position_bounds_override=position_bounds_override,
        velocity_bounds_override=velocity_bounds_override,
        num_samples_override=validation_samples
    )
    
    # Apply same normalization as training
    scale_factor = particle.scale
    _, validation_data_normalized = normalize_training_data(validation_data, particle.dof//2, scale_factor=scale_factor)
    q_qdot_val, qdot_qdotdot_val = validation_data_normalized
    
    # Create data loader
    batch_size = particle.train_hyperparams['minibatch_size']
    val_ds = TensorDataset(q_qdot_val, qdot_qdotdot_val)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Compute validation loss
    model.eval()
    total_val_loss = 0.0
    total_samples = 0
    
    # Clear any existing gradients but don't disable gradient computation
    # We need gradients for forward pass derivatives, but not for backpropagation
    for param in model.parameters():
        param.grad = None
    
    for val_xi, val_yi in val_loader:
        val_xi, val_yi = val_xi.to(device, dtype), val_yi.to(device, dtype)
        
        # Forward pass needs gradients for derivative computation  
        val_pred = model(val_xi)
        
        # Use same loss function as training
        from LNN import loss
        batch_loss = loss(val_pred, val_yi)
        
        batch_size = val_xi.size(0)
        total_val_loss += batch_loss.item() * batch_size
        total_samples += batch_size
        
        # Clear gradients after each batch to prevent accumulation
        for param in model.parameters():
            param.grad = None
    
    avg_val_loss = total_val_loss / total_samples
    
    # Prepare output
    bound_info = ""
    if position_bounds_override or velocity_bounds_override:
        bound_info = " (Custom Bounds)"
    
    output = [
        f"=== VALIDATION METRICS{bound_info} ===",
        f"Validation Samples: {len(val_ds)}",
        f"Validation Loss: {avg_val_loss:.6f}",
        f"Scale Factor Used: {scale_factor}",
    ]
    
    if position_bounds_override:
        output.append(f"Custom Position Bounds: {position_bounds_override}")
    if velocity_bounds_override:
        output.append(f"Custom Velocity Bounds: {velocity_bounds_override}")
    
    if verbose:
        for line in output:
            print(line)
    
    # Save to file
    if debug_dir:
        try:
            with open(os.path.join(debug_dir, "validation_metrics.txt"), 'w') as f:
                f.write('\n'.join(output))
            if verbose:
                print(f"Validation metrics saved to {debug_dir}/validation_metrics.txt")
        except Exception as e:
            print(f"Warning: Could not save validation metrics: {e}")
    
    return avg_val_loss

# =============================================================================
# MODEL INSPECTION FUNCTIONS (from inspect_model.py)
# =============================================================================

def debug_model(particle, model, save_path, dtype, device, verbose=True, 
                position_bounds_override=None, velocity_bounds_override=None):
    """
    Run all diagnostics on the model, creating its own training data loader.
    """
    # Create training data similar to train.py
    total_data_points = particle.train_hyperparams['num_samples']
    position_start_end = particle.train_hyperparams['position_bounds']
    velocity_start_end = particle.train_hyperparams['velocity_bounds']
    train_seed = particle.train_hyperparams['train_seed']

    scale_factor = particle.scale
    particle.scale_constants([1.0 for _ in range(particle.dof//2)])
    training_data = create_training_data(particle, total_data_points, particle.dof//2, position_start_end, velocity_start_end, seed=4*train_seed+4355, dtype=dtype, device=device)
    particle.scale_constants(scale_factor)

    _, training_data = normalize_training_data(training_data, particle.dof//2, scale_factor)
    q_qdot_train, qdot_qdotdot_train = training_data
    
    batch_size = particle.train_hyperparams['minibatch_size']
    train_ds = TensorDataset(q_qdot_train, qdot_qdotdot_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Get a batch for diagnostics
    x_batch = next(iter(train_loader))[0]
    x_batch = x_batch.to(device, dtype)
    run_all_diagnostics(model, x_batch, check_grads=False, save_path=save_path, verbose=verbose)

    inspect_problematic_batch(particle, model, train_loader, save_path, dtype, device, verbose=verbose)
    plot_lagrangian_systematic(particle, model, save_path, dtype, device, verbose=verbose,
                               position_bounds_override=position_bounds_override, velocity_bounds_override=velocity_bounds_override)
    plot_mass_matrix_determinant_systematic(particle, model, save_path, dtype, device, verbose=verbose,
                                           position_bounds_override=position_bounds_override, velocity_bounds_override=velocity_bounds_override)
    inspect_network_vitals(model, save_path, verbose=verbose)
    
    # Add validation metrics analysis (use validation bounds from particle if not overridden)
    val_pos_bounds = position_bounds_override or particle.test_params.get('validation_position_bounds')
    val_vel_bounds = velocity_bounds_override or particle.test_params.get('validation_velocity_bounds')
    
    compute_validation_metrics(particle, model, save_path, dtype, device, 
                              val_pos_bounds, val_vel_bounds, verbose=verbose)

# ...rest of inspect_model.py functions...

# =============================================================================
# MODEL INSPECTION FUNCTIONS
# =============================================================================

def test_forward_pass(model, x_batch, verbose=True):
    model.eval()
    with torch.no_grad():
        out = model(x_batch)
    
    output = []
    output.append(f"Forward output shape: {out.shape}")
    output.append(f"Sample outputs:\n{out[:5]}")
    
    if verbose:
        for line in output:
            print(line)
    
    return output

def test_jacobian_hessian_shapes(model, x_batch, verbose=True):
    jac, hess = model.batch_jac_and_hes(model.network, x_batch)
    
    if verbose:
        print("Jacobian shape:", jac.shape)
        print("Hessian shape:", hess.shape)
    
    return jac, hess

def test_hessian_condition_numbers(hess, input_dim, verbose=True):
    n = input_dim // 2
    h_blocks = hess[:, n:, n:]  # [B, n, n]
    output = []
    
    for i in range(h_blocks.shape[0]):
        block = h_blocks[i]
        s = torch.linalg.svdvals(block)
        cond = (s[0] / s[-1]).item() if s[-1] != 0 else float('inf')
        line = f"Sample {i} Hessian condition number: {cond:.2e}"
        output.append(line)
        if verbose:
            print(line)
    
    return output

def check_gradients(model, verbose=True):
    output = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                line = f"NaNs in gradient of {name}"
                output.append(line)
                if verbose:
                    print(line)
            if torch.isinf(param.grad).any():
                line = f"Infs in gradient of {name}"
                output.append(line)
                if verbose:
                    print(line)
    
    if not output:
        output.append("No NaN/Inf gradients detected")
        if verbose:
            print("No NaN/Inf gradients detected")
    
    return output

def test_hessian_inversion(hess, input_dim, verbose=True):
    n = input_dim // 2
    h_blocks = hess[:, n:, n:]
    output = []
    
    for i, H in enumerate(h_blocks):
        try:
            Hinv = torch.linalg.pinv(H)
            line = f"Sample {i}: Inversion OK. Max abs val: {Hinv.abs().max().item():.2e}"
            output.append(line)
            if verbose:
                print(line)
        except RuntimeError as e:
            line = f"Sample {i}: Inversion failed! {e}"
            output.append(line)
            if verbose:
                print(line)
    
    return output

def add_nan_hooks(model):
    def forward_hook(module, input, output):
        if isinstance(output, torch.Tensor) and (torch.isnan(output).any() or torch.isinf(output).any()):
            print(f"⚠️ NaN or Inf detected in {module.__class__.__name__} forward!")

    def backward_hook(module, grad_input, grad_output):
        if any(torch.isnan(g).any() or torch.isinf(g).any() for g in grad_output if g is not None):
            print(f"⚠️ NaN or Inf detected in {module.__class__.__name__} backward!")

def run_all_diagnostics(model, x_batch, check_grads=False, save_path=None, verbose=True):
    debug_dir = None
    if save_path:
        debug_dir = os.path.join(save_path, "debug_results")
        try:
            os.makedirs(debug_dir, exist_ok=True)
            if verbose:
                print(f"Debug output will be saved to: {debug_dir}")
        except Exception as e:
            print(f"Warning: Could not create debug directory {debug_dir}: {e}")
            debug_dir = None
    elif verbose:
        print("No save_path provided - debug output will not be saved to file")
    
    output = []
    
    if verbose:
        print("===== FORWARD CHECK =====")
    output.append("===== FORWARD CHECK =====")
    forward_output = test_forward_pass(model, x_batch, verbose)
    output.extend(forward_output)

    if verbose:
        print("\n===== JACOBIAN/HESSIAN SHAPES =====")
    output.append("\n===== JACOBIAN/HESSIAN SHAPES =====")
    jac, hess = test_jacobian_hessian_shapes(model, x_batch, verbose)
    output.append(f"Jacobian shape: {jac.shape}")
    output.append(f"Hessian shape: {hess.shape}")

    if verbose:
        print("\n===== HESSIAN CONDITION NUMBERS =====")
    output.append("\n===== HESSIAN CONDITION NUMBERS =====")
    cond_output = test_hessian_condition_numbers(hess, x_batch.shape[1], verbose)
    output.extend(cond_output)

    if verbose:
        print("\n===== HESSIAN INVERSION CHECK =====")
    output.append("\n===== HESSIAN INVERSION CHECK =====")
    inv_output = test_hessian_inversion(hess, x_batch.shape[1], verbose)
    output.extend(inv_output)

    if check_grads:
        if verbose:
            print("\n===== GRADIENT CHECK =====")
        output.append("\n===== GRADIENT CHECK =====")
        grad_output = check_gradients(model, verbose)
        output.extend(grad_output)

    if verbose:
        print("\n===== REGISTERED HOOKS FOR NaNs/INFs =====")
    output.append("\n===== REGISTERED HOOKS FOR NaNs/INFs =====")
    add_nan_hooks(model)
    output.append("NaN/Inf hooks registered")

    # Save output to file
    if debug_dir:
        with open(os.path.join(debug_dir, "diagnostics_output.txt"), 'w') as f:
            f.write('\n'.join(output))

def inspect_problematic_batch(particle, model, train_loader, save_path=None, dtype=torch.float32, device=None, verbose=True):
    """
    Grabs one batch, finds the input with the highest condition number,
    and prints its details.
    """
    debug_dir = None
    if save_path:
        debug_dir = os.path.join(save_path, "debug_results")
        try:
            os.makedirs(debug_dir, exist_ok=True)
            if verbose:
                print(f"Debug output will be saved to: {debug_dir}")
        except Exception as e:
            print(f"Warning: Could not create debug directory {debug_dir}: {e}")
            debug_dir = None
    elif verbose:
        print("No save_path provided - debug output will not be saved to file")
    
    model.eval() # Set model to evaluation mode
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(device, dtype)

    output = []
    output.append("--- Inspecting a single batch ---")
    
    with torch.no_grad():
        # Manually perform the forward pass steps to get the Hessian
        B, D = x_batch.shape
        n = D // 2
        
        # We need gradients for the Hessian, so we'll use the vmap approach
        jac, hess = model.batch_jac_and_hes(model.network, x_batch)
        A_batch = hess[:, n:, n:] # The mass matrix for each sample in the batch
        
        # Calculate condition numbers for the whole batch
        cond_nums = torch.linalg.cond(A_batch)
        
        # Find the worst offender
        worst_idx = torch.argmax(cond_nums)
        max_cond = cond_nums[worst_idx]
        problem_input = x_batch[worst_idx]
        
        output.append(f"Max condition number in batch: {max_cond:.4e} at index {worst_idx}")
        output.append(f"Problematic Input (normalized) [q1, q2, q_dot1, q_dot2]:\n{problem_input.cpu().numpy()}")
        
        # Denormalize to see the physical values
        # Assumes 'scale_factor' is a list or tensor of your normalization constants
        scale_tensor = torch.tensor(particle.scale, device=problem_input.device, dtype=dtype)

        # Get the number of position coordinates dynamically
        n_pos = len(particle.scale)
        unscaled_input = torch.zeros_like(problem_input)
        
        # Unscale positions: multiply by corresponding scale factor
        for i in range(n_pos):
            unscaled_input[i] = problem_input[i] * scale_tensor[i]
        
        # Unscale velocities: multiply by corresponding scale factor
        for i in range(n_pos):
            unscaled_input[i + n_pos] = problem_input[i + n_pos] * scale_tensor[i]
        
        output.append(f"Problematic Input (unscaled):\n{unscaled_input.cpu().numpy()}")
        
        # Let's look at its Hessian
        A_problem = A_batch[worst_idx]
        A_det = torch.linalg.det(A_problem)
        output.append(f"\nHessian (Mass Matrix) for this input:\n{A_problem.cpu().numpy()}")
        output.append(f"Determinant: {A_det:.4e}")

    if verbose:
        for line in output:
            print(line)
    
    # Save output to file
    if debug_dir:
        with open(os.path.join(debug_dir, "problematic_batch_analysis.txt"), 'w') as f:
            f.write('\n'.join(output))

# =============================================================================
# SYSTEMATIC VISUALIZATION FUNCTIONS  
# =============================================================================

def _extract_velocity_bounds(particle, velocity_bounds_override=None):
    """Extract and normalize velocity bounds from particle parameters."""
    if velocity_bounds_override is not None:
        # Use manually provided bounds
        v_mins = [bound[0] for bound in velocity_bounds_override]
        v_maxs = [bound[1] for bound in velocity_bounds_override]
    else:
        # Use training bounds by default, fallback to test bounds
        if hasattr(particle, 'train_hyperparams') and 'velocity_bounds' in particle.train_hyperparams:
            velocity_bounds = particle.train_hyperparams['velocity_bounds']
        else:
            velocity_bounds = particle.test_params['velocity_bounds']
        v_mins = [bound[0] for bound in velocity_bounds]
        v_maxs = [bound[1] for bound in velocity_bounds]
    return v_mins, v_maxs

def _get_velocity_combinations(particle, v_values=[-1, 0, 1], velocity_bounds_override=None):
    """Generate velocity combinations for systematic visualization."""
    v_mins, v_maxs = _extract_velocity_bounds(particle, velocity_bounds_override)
    n_vel_dims = len(v_mins)
    
    # For Option B: fix higher dimensions, vary first two
    if n_vel_dims <= 2:
        # For 1D or 2D systems, use direct grid
        combinations = []
        if n_vel_dims == 1:
            for v1 in v_values:
                actual_v1 = v_mins[0] + (v1 + 1) * 0.5 * (v_maxs[0] - v_mins[0])
                combinations.append([actual_v1])
        else:  # n_vel_dims == 2
            for v1 in v_values:
                for v2 in v_values:
                    actual_v1 = v_mins[0] + (v1 + 1) * 0.5 * (v_maxs[0] - v_mins[0])
                    actual_v2 = v_mins[1] + (v2 + 1) * 0.5 * (v_maxs[1] - v_mins[1])
                    combinations.append([actual_v1, actual_v2])
        return combinations
    else:
        # For higher dimensions: Option B - 3 grids with different fixed values for qdot3+
        combinations = []
        for higher_v in v_values:  # -1, 0, +1 for qdot3, qdot4, ...
            for v1 in v_values:
                for v2 in v_values:
                    vel_combo = [0.0] * n_vel_dims
                    # Set first two velocities
                    vel_combo[0] = v_mins[0] + (v1 + 1) * 0.5 * (v_maxs[0] - v_mins[0])
                    vel_combo[1] = v_mins[1] + (v2 + 1) * 0.5 * (v_maxs[1] - v_mins[1])
                    # Set higher dimensions to the same fixed value
                    for i in range(2, n_vel_dims):
                        vel_combo[i] = v_mins[i] + (higher_v + 1) * 0.5 * (v_maxs[i] - v_mins[i])
                    combinations.append(vel_combo)
        return combinations

def _create_position_grid(particle, pos_indices=[0, 1], resolution=100, position_bounds_override=None):
    """Create position grid for the specified position indices."""
    # Use override bounds if provided, otherwise use training bounds
    if position_bounds_override is not None:
        position_bounds = position_bounds_override
    else:
        position_bounds = particle.train_hyperparams['position_bounds']
    
    scale_factor = particle.scale
    
    # Handle case where we only have one position dimension
    if len(position_bounds) == 1:
        pos_indices = [0]  # Override to single dimension
        
    # Ensure indices are within bounds
    max_pos_dim = len(position_bounds)
    pos_indices = [idx for idx in pos_indices if idx < max_pos_dim]
    
    if len(pos_indices) == 1:
        # 1D case
        idx = pos_indices[0]
        a, b = position_bounds[idx]
        # Create unscaled grid for display
        q_range_display = np.linspace(a, b, resolution)
        # Create scaled grid for network evaluation (this is the key: we DON'T scale here)
        q_range_scaled = q_range_display  # Keep original values
        return q_range_display, q_range_scaled, None, pos_indices
    else:
        # 2D case  
        idx1, idx2 = pos_indices[:2]  # Take first two indices
        a, b = position_bounds[idx1]
        c, d = position_bounds[idx2]
        
        # Create unscaled grids for display
        q1_range_display = np.linspace(a, b, resolution)
        q2_range_display = np.linspace(c, d, resolution)
        q1_grid_display, q2_grid_display = np.meshgrid(q1_range_display, q2_range_display, indexing='xy')
        
        # Create grids for network evaluation (DON'T scale here)
        q1_grid_scaled, q2_grid_scaled = q1_grid_display, q2_grid_display
        
        return (q1_grid_display, q2_grid_display), (q1_grid_scaled, q2_grid_scaled), None, pos_indices

def _create_animation_frames(particle, pos_grid_display, pos_grid_scaled, pos_indices, model, function_type, n_frames=40, velocity_bounds_override=None):
    """Create animation frames with synchronized velocity movement."""
    v_mins, v_maxs = _extract_velocity_bounds(particle, velocity_bounds_override)
    n_vel_dims = len(v_mins)
    
    # Create synchronized velocity sweep
    frames = []
    for i in range(n_frames):
        t = i / (n_frames - 1)  # 0 to 1
        
        # All velocities move together, scaled to their respective bounds
        vel_values = []
        for j in range(n_vel_dims):
            # Map t from [0,1] to [v_min, v_max] for each dimension
            v_j = v_mins[j] + t * (v_maxs[j] - v_mins[j])
            vel_values.append(v_j)
        
        # Compute function values for this velocity combination
        frame_data = _compute_function_values(particle, pos_grid_display, pos_grid_scaled, pos_indices, vel_values, model, function_type)
        frames.append(frame_data)
    
    return frames

def _compute_function_values(particle, pos_grid_display, pos_grid_scaled, pos_indices, vel_values, model, function_type, compute_analytical=True, sympy_func=None):
    """Compute Lagrangian or mass matrix determinant values for both analytical and learned."""
    from dataset_creation import normalize_testing_data
    
    scale_factor = particle.scale
    if model is not None:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
    else:
        device = torch.device('cpu')
        dtype = torch.float32
    
    if len(pos_indices) == 1:
        # 1D system
        q_flat_scaled = pos_grid_scaled.reshape(-1, 1)  # Shape: (N, 1)
        n_points = len(q_flat_scaled)
    else:
        # 2D+ system
        q1_grid_scaled, q2_grid_scaled = pos_grid_scaled
        q_flat_scaled = np.column_stack([q1_grid_scaled.ravel(), q2_grid_scaled.ravel()])  # Shape: (N, 2)
        n_points = q1_grid_scaled.size
    
    # Create full state vectors using SCALED positions
    n_pos_dims = particle.dof // 2
    n_vel_dims = len(vel_values)
    
    # Create position vectors (fill missing dimensions with zeros)
    full_positions = np.zeros((n_points, n_pos_dims))
    if len(pos_indices) == 1:
        full_positions[:, pos_indices[0]] = q_flat_scaled[:, 0]
    else:
        # For 2D case: pos_indices[0] gets q_flat_scaled[:, 0], pos_indices[1] gets q_flat_scaled[:, 1]
        for i, idx in enumerate(pos_indices):
            full_positions[:, idx] = q_flat_scaled[:, i]
    
    # Create velocity vectors (already in correct scale)
    full_velocities = np.zeros((n_points, n_vel_dims))
    for i, vel_val in enumerate(vel_values):
        full_velocities[:, i] = vel_val
    
    # Combine into state vectors
    state_vectors = np.hstack([full_positions, full_velocities])
    state_tensor = torch.tensor(state_vectors, dtype=dtype, device=device)
    
    # Normalize the data
    normalized_input = normalize_testing_data([[state_tensor], [torch.zeros_like(state_tensor)]], 
                                             n_pos_dims, 
                                             torch.tensor(scale_factor, device=device, dtype=dtype))[0][0]
    
    results = {}
    
    # Compute function values
    with torch.no_grad():
        if function_type == 'lagrangian':
            # Learned values
            learned_tensor = model.lagrangian(normalized_input)
            learned_values = learned_tensor.detach().cpu().numpy()
            results['learned'] = learned_values.reshape(pos_grid_display[0].shape if len(pos_indices) > 1 else -1)
            
            # Analytical values
            if compute_analytical:
                analytical_tensor = particle.lagrangian(normalized_input)
                analytical_values = analytical_tensor.detach().cpu().numpy()
                results['analytical'] = analytical_values.reshape(pos_grid_display[0].shape if len(pos_indices) > 1 else -1)
                
        elif function_type == 'determinant':
            from torch.func import vmap, hessian
            
            # Learned determinant
            _, hess = model.batch_jac_and_hes(model.network, normalized_input)
            mass_matrix = hess[:, n_vel_dims:, n_vel_dims:]
            learned_determinants_tensor = torch.linalg.det(mass_matrix)
            learned_determinants = learned_determinants_tensor.detach().cpu().numpy()
            results['learned'] = learned_determinants.reshape(pos_grid_display[0].shape if len(pos_indices) > 1 else -1)
            
            # Analytical determinant
            if compute_analytical:
                def true_kinetic_energy_from_state(x):
                    q, qt = torch.split(x, n_pos_dims, dim=-1)
                    q = torch.unsqueeze(q, 0)
                    qt = torch.unsqueeze(qt, 0)
                    return particle.kinetic(q, qt)
                
                true_hessian_func = vmap(hessian(true_kinetic_energy_from_state))
                true_hess_batch = torch.squeeze(true_hessian_func(normalized_input), 1)
                true_mass_matrix = true_hess_batch[:, n_pos_dims:, n_pos_dims:]  # Extract velocity part
                true_determinants_tensor = torch.linalg.det(true_mass_matrix)
                true_determinants = true_determinants_tensor.detach().cpu().numpy()
                results['analytical'] = true_determinants.reshape(pos_grid_display[0].shape if len(pos_indices) > 1 else -1)
    
    return results

def plot_lagrangian_systematic(particle, model, save_path=None, dtype=torch.float32, device=None, 
                               pos_indices=[0, 1], resolution=100, create_animation=True, verbose=True,
                               position_bounds_override=None, velocity_bounds_override=None):
    """
    Create systematic Lagrangian visualizations for academic publications.
    
    Args:
        pos_indices: Which position coordinates to use as axes [default: [0,1]]
        resolution: Grid resolution for high-quality plots
        create_animation: Whether to create animation videos
        position_bounds_override: Manual position bounds [(min1,max1), (min2,max2), ...]
        velocity_bounds_override: Manual velocity bounds [(min1,max1), (min2,max2), ...]
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Create output directories
    debug_dir = None
    if save_path:
        debug_dir = os.path.join(save_path, "debug_results", "lagrangian_systematic")
        try:
            os.makedirs(debug_dir, exist_ok=True)
            if verbose:
                print(f"Lagrangian plots will be saved to: {debug_dir}")
        except Exception as e:
            print(f"Warning: Could not create directory {debug_dir}: {e}")
            debug_dir = None
    
    model.eval()
    
    # Create position grid
    q_grid_display, q_grid_scaled, _, actual_pos_indices = _create_position_grid(particle, pos_indices, resolution, position_bounds_override)
    
    if len(actual_pos_indices) == 1:
        # 1D system: single plot - FIX: pass all 8 arguments including velocity_bounds_override
        _plot_1d_systematic(particle, model, q_grid_display, q_grid_scaled, actual_pos_indices[0], debug_dir, verbose, velocity_bounds_override)
    else:
        # Multi-D system: Option B approach
        pos_grid_display = q_grid_display
        pos_grid_scaled = q_grid_scaled
        _plot_multid_systematic(particle, model, pos_grid_display, pos_grid_scaled, actual_pos_indices, debug_dir, verbose, 'lagrangian', velocity_bounds_override)
    
    # Create animation if requested
    if create_animation and debug_dir and len(actual_pos_indices) >= 1:
        pos_grid = pos_grid_display if len(actual_pos_indices) > 1 else q_grid_display
        pos_grid_for_eval = pos_grid_scaled if len(actual_pos_indices) > 1 else q_grid_scaled
        _create_lagrangian_animation(particle, model, pos_grid, pos_grid_for_eval, actual_pos_indices, debug_dir, verbose, velocity_bounds_override)

def plot_mass_matrix_determinant_systematic(particle, model, save_path=None, dtype=torch.float32, device=None,
                                           pos_indices=[0, 1], resolution=100, create_animation=True, verbose=True,
                                           position_bounds_override=None, velocity_bounds_override=None):
    """
    Create systematic mass matrix determinant visualizations for academic publications.
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Create output directories  
    debug_dir = None
    if save_path:
        debug_dir = os.path.join(save_path, "debug_results", "mass_matrix_systematic")
        try:
            os.makedirs(debug_dir, exist_ok=True)
            if verbose:
                print(f"Mass matrix plots will be saved to: {debug_dir}")
        except Exception as e:
            print(f"Warning: Could not create directory {debug_dir}: {e}")
            debug_dir = None
    elif verbose:
        print("No save_path provided - mass matrix plots will not be saved to file")
    
    model.eval()
    
    # Create position grid
    q_grid_display, q_grid_scaled, _, actual_pos_indices = _create_position_grid(particle, pos_indices, resolution, position_bounds_override)
    
    if len(actual_pos_indices) == 1:
        # 1D system: single plot - FIX: pass all 8 arguments including velocity_bounds_override
        _plot_1d_determinant_systematic(particle, model, q_grid_display, q_grid_scaled, actual_pos_indices[0], debug_dir, verbose, velocity_bounds_override)
    else:
        # Multi-D system: Option B approach  
        pos_grid_display = q_grid_display
        pos_grid_scaled = q_grid_scaled
        _plot_multid_systematic(particle, model, pos_grid_display, pos_grid_scaled, actual_pos_indices, debug_dir, verbose, 'determinant', velocity_bounds_override)
    
    # Create animation if requested
    if create_animation and debug_dir and len(actual_pos_indices) >= 1:
        pos_grid = pos_grid_display if len(actual_pos_indices) > 1 else q_grid_display
        pos_grid_for_eval = pos_grid_scaled if len(actual_pos_indices) > 1 else q_grid_scaled
        _create_determinant_animation(particle, model, pos_grid, pos_grid_for_eval, actual_pos_indices, debug_dir, verbose, velocity_bounds_override)

def inspect_network_vitals(model, save_path=None, verbose=True):
    """
    Prints statistics about the weights and biases of each layer in the model.
    """
    debug_dir = None
    if save_path:
        debug_dir = os.path.join(save_path, "debug_results")
        try:
            os.makedirs(debug_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create debug directory {debug_dir}: {e}")
            debug_dir = None
    elif verbose:
        print("No save_path provided - output will not be saved to file")
    
    output = []
    output.append("--- Network Weight & Bias Vitals ---")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            output.append(f"Layer: {name}")
            output.append(f"  - Shape: {param.data.shape}")
            output.append(f"  - Mean:  {param.data.mean():.4f}")
            output.append(f"  - Std:   {param.data.std():.4f}")
            output.append(f"  - Max:   {param.data.max():.4f}")
            output.append(f"  - Min:   {param.data.min():.4f}")
            output.append("-" * 20)

    if verbose:
        for line in output:
            print(line)
    
    # Save output to file
    if debug_dir:
        try:
            with open(os.path.join(debug_dir, "network_vitals.txt"), 'w') as f:
                f.write('\n'.join(output))
            if verbose:
                print(f"Network vitals saved to {debug_dir}/network_vitals.txt")
        except Exception as e:
            print(f"Warning: Could not save network vitals to file: {e}")

def _plot_1d_systematic(particle, model, q_range_display, q_range_scaled, pos_idx, debug_dir, verbose, velocity_bounds_override=None):
    """Plot 1D system: q vs qdot with function as color."""
    from dataset_creation import normalize_testing_data
    
    # Get velocity bounds for the single velocity dimension
    v_mins, v_maxs = _extract_velocity_bounds(particle, velocity_bounds_override)
    v_min, v_max = v_mins[0], v_maxs[0]
    
    # Create q-qdot grid using DISPLAY coordinates for plotting, SCALED coordinates for evaluation
    qdot_range = np.linspace(v_min, v_max, 100)
    Q_display, QDOT = np.meshgrid(q_range_display, qdot_range, indexing='xy')
    Q_scaled, _ = np.meshgrid(q_range_scaled, qdot_range, indexing='xy')
    
    # Prepare data
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    scale_factor = particle.scale
    
    n_points = Q_scaled.size
    positions = np.zeros((n_points, 1))
    positions[:, 0] = Q_scaled.ravel()
    velocities = QDOT.ravel().reshape(-1, 1)
    
    state_vectors = np.hstack([positions, velocities])
    state_tensor = torch.tensor(state_vectors, dtype=dtype, device=device)
    
    # Normalize
    normalized_input = normalize_testing_data([[state_tensor], [torch.zeros_like(state_tensor)]], 
                                             1, torch.tensor(scale_factor, device=device, dtype=dtype))[0][0]
    
    # Plot both analytical and learned
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'1D System: Position vs Velocity Analysis', fontsize=16, fontweight='bold')
    
    # Get coordinate name
    coord_name = f'θ{pos_idx+1}' if pos_idx in particle.angle_indices else f'q{pos_idx+1}'
    
    # Analytical Lagrangian
    with torch.no_grad():
        L_true_tensor = particle.lagrangian(normalized_input)
        L_true = L_true_tensor.detach().cpu().numpy().reshape(Q_display.shape)
    
    im1 = axes[0,0].imshow(L_true, extent=[q_range_display.min(), q_range_display.max(), v_min, v_max], 
                        origin='lower', aspect='auto', cmap='viridis')
    axes[0,0].set_xlabel(f'{coord_name}')
    axes[0,0].set_ylabel(f'{coord_name}̇')
    axes[0,0].set_title('Analytical Lagrangian', fontweight='bold')
    fig.colorbar(im1, ax=axes[0,0], label='L')
    
    # Learned Lagrangian
    with torch.no_grad():
        L_learned_tensor = model.lagrangian(normalized_input)
        L_learned = L_learned_tensor.detach().cpu().numpy().reshape(Q_display.shape)
    
    im2 = axes[0,1].imshow(L_learned, extent=[q_range_display.min(), q_range_display.max(), v_min, v_max], 
                        origin='lower', aspect='auto', cmap='viridis')
    axes[0,1].set_xlabel(f'{coord_name}')
    axes[0,1].set_ylabel(f'{coord_name}̇')
    axes[0,1].set_title('Learned Lagrangian', fontweight='bold')
    fig.colorbar(im2, ax=axes[0,1], label='L')
    
    # Difference plot
    L_diff = L_learned - L_true
    im3 = axes[1,0].imshow(L_diff, extent=[q_range_display.min(), q_range_display.max(), v_min, v_max], 
                        origin='lower', aspect='auto', cmap='coolwarm')
    axes[1,0].set_xlabel(f'{coord_name}')
    axes[1,0].set_ylabel(f'{coord_name}̇')
    axes[1,0].set_title('Difference (Learned - Analytical)', fontweight='bold')
    fig.colorbar(im3, ax=axes[1,0], label='ΔL')
    
    # Error statistics
    axes[1,1].text(0.1, 0.8, f'Max Error: {np.max(np.abs(L_diff)):.6f}', transform=axes[1,1].transAxes, fontsize=12)
    axes[1,1].text(0.1, 0.7, f'Mean Error: {np.mean(np.abs(L_diff)):.6f}', transform=axes[1,1].transAxes, fontsize=12)
    axes[1,1].text(0.1, 0.6, f'RMS Error: {np.sqrt(np.mean(L_diff**2)):.6f}', transform=axes[1,1].transAxes, fontsize=12)
    axes[1,1].set_title('Error Statistics', fontweight='bold')
    axes[1,1].axis('off')
    
    if debug_dir:
        plt.savefig(os.path.join(debug_dir, "lagrangian_1d_systematic.png"), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        if verbose:
            print(f"1D Lagrangian plot saved")
    
    if verbose:
        plt.show()
    else:
        plt.close()

def _plot_1d_determinant_systematic(particle, model, q_range_display, q_range_scaled, pos_idx, debug_dir, verbose, velocity_bounds_override=None):
    """Plot 1D system: mass matrix determinant.""" 
    from dataset_creation import normalize_testing_data
    from torch.func import vmap, hessian
    
    # Get velocity bounds - FIX: now uses velocity_bounds_override parameter
    v_mins, v_maxs = _extract_velocity_bounds(particle, velocity_bounds_override)
    v_min, v_max = v_mins[0], v_maxs[0]
    
    # Create grid
    qdot_range = np.linspace(v_min, v_max, 100)
    Q, QDOT = np.meshgrid(q_range, qdot_range, indexing='xy')
    
    # Prepare data
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    scale_factor = particle.scale
    
    n_points = Q.size
    positions = np.zeros((n_points, 1))
    positions[:, 0] = Q.ravel()
    velocities = QDOT.ravel().reshape(-1, 1)
    
    state_vectors = np.hstack([positions, velocities])
    state_tensor = torch.tensor(state_vectors, dtype=dtype, device=device)
    normalized_input = normalize_testing_data([[state_tensor], [torch.zeros_like(state_tensor)]], 
                                             1, torch.tensor(scale_factor, device=device, dtype=dtype))[0][0]
    
    # Plot determinants
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'1D System: Mass Matrix Determinant Analysis', fontsize=16, fontweight='bold')
    
    coord_name = f'θ{pos_idx+1}' if pos_idx in particle.angle_indices else f'q{pos_idx+1}'
    
    # True determinant
    def true_kinetic_energy_from_state(x):
        q, qt = torch.split(x, 1, dim=-1)
        q = torch.unsqueeze(q, 0)
        qt = torch.unsqueeze(qt, 0)
        return particle.kinetic(q, qt)
    
    true_hessian_func = vmap(hessian(true_kinetic_energy_from_state))
    true_hess_batch = torch.squeeze(true_hessian_func(normalized_input), 1)
    true_mass_matrix = true_hess_batch[:, 1:, 1:]  # Extract velocity part
    true_determinants = torch.linalg.det(true_mass_matrix).cpu().numpy().reshape(Q.shape)
    
    im1 = axes[0].imshow(true_determinants, extent=[q_range.min(), q_range.max(), v_min, v_max], 
                        origin='lower', aspect='auto', cmap='coolwarm',
                        norm=SymLogNorm(linthresh=1e-6))
    axes[0].set_xlabel(f'{coord_name}')
    axes[0].set_ylabel(f'{coord_name}̇')
    axes[0].set_title('True Mass Matrix Det.', fontweight='bold')
    fig.colorbar(im1, ax=axes[0], label='det(M)')
    
    # Learned determinant
    with torch.no_grad():
        _, hess = model.batch_jac_and_hes(model.network, normalized_input)
        mass_matrix = hess[:, 1:, 1:]
        learned_determinants = torch.linalg.det(mass_matrix).detach().cpu().numpy().reshape(Q.shape)
    
    im2 = axes[1].imshow(learned_determinants, extent=[q_range.min(), q_range.max(), v_min, v_max], 
                        origin='lower', aspect='auto', cmap='coolwarm',
                        norm=SymLogNorm(linthresh=1e-6))
    axes[1].set_xlabel(f'{coord_name}')
    axes[1].set_ylabel(f'{coord_name}̇')
    axes[1].set_title('Learned Mass Matrix Det.', fontweight='bold')
    fig.colorbar(im2, ax=axes[1], label='det(M)')
    
    if debug_dir:
        plt.savefig(os.path.join(debug_dir, "mass_matrix_determinant_1d_systematic.png"), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        if verbose:
            print(f"1D Mass matrix plot saved")
    
    if verbose:
        plt.show()
    else:
        plt.close()

def _plot_multid_systematic(particle, model, pos_grid_display, pos_grid_scaled, pos_indices, debug_dir, verbose, function_type, velocity_bounds_override=None):
    """Plot multi-dimensional systematic grids using Option B."""
    q1_grid_display, q2_grid_display = pos_grid_display
    
    # Get velocity combinations
    vel_combinations = _get_velocity_combinations(particle, velocity_bounds_override=velocity_bounds_override)
    n_vel_dims = len(_extract_velocity_bounds(particle, velocity_bounds_override)[0])
    
    coord_names = []
    for i, idx in enumerate(pos_indices):
        if idx in particle.angle_indices:
            coord_names.append(f'θ{idx+1}')
        else:
            coord_names.append(f'q{idx+1}')
    
    if n_vel_dims <= 2:
        # 2D velocity space - single 3x3 grid
        n_combinations = len(vel_combinations)
        n_cols = int(np.sqrt(n_combinations))
        n_rows = (n_combinations + n_cols - 1) // n_cols
        
        _create_single_grid_plot(particle, model, pos_grid_display, pos_grid_scaled, pos_indices, vel_combinations, 
                               debug_dir, verbose, function_type, coord_names, n_rows, n_cols)
    else:
        # Higher dimensions - create 3 separate 3x3 grids (Option B)
        _create_multiple_grid_plots(particle, model, pos_grid_display, pos_grid_scaled, pos_indices, vel_combinations,
                                   debug_dir, verbose, function_type, coord_names, n_vel_dims)

def _create_single_grid_plot(particle, model, pos_grid_display, pos_grid_scaled, pos_indices, vel_combinations,
                           debug_dir, verbose, function_type, coord_names, n_rows, n_cols):
    """Create single 3x3 grid plot for 2D velocity systems."""
    # Create both analytical and learned plots
    for plot_type in ['analytical', 'learned']:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        if plot_type == 'analytical':
            fig.suptitle(f'Analytical {function_type.title()} - {coord_names[0]} vs {coord_names[1]}', 
                        fontsize=16, fontweight='bold')
        else:  # learned
            fig.suptitle(f'Learned {function_type.title()} - {coord_names[0]} vs {coord_names[1]}', 
                        fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, vel_combo in enumerate(vel_combinations):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Compute function values (returns dict with both analytical and learned)
            values_dict = _compute_function_values(particle, pos_grid_display, pos_grid_scaled, pos_indices, vel_combo, model, function_type)
            
            # Plot the requested type
            extent = [pos_grid_display[0].min(), pos_grid_display[0].max(), 
                     pos_grid_display[1].min(), pos_grid_display[1].max()]
            
            if plot_type == 'analytical':
                values = values_dict['analytical']
                cmap = 'viridis' if function_type == 'lagrangian' else 'coolwarm'
                norm = None if function_type == 'lagrangian' else SymLogNorm(linthresh=1e-6)
            else:  # learned
                values = values_dict['learned']
                cmap = 'viridis' if function_type == 'lagrangian' else 'coolwarm'
                norm = None if function_type == 'lagrangian' else SymLogNorm(linthresh=1e-6)
            
            im = ax.imshow(values, extent=extent, origin='lower', aspect='auto', cmap=cmap, norm=norm)
            
            title = f'$\\dot{{q}}_1 = {vel_combo[0]:.2f}$, $\\dot{{q}}_2 = {vel_combo[1]:.2f}$'
                
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(coord_names[0])
            ax.set_ylabel(coord_names[1])
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for idx in range(len(vel_combinations), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if debug_dir:
            filename = f"{function_type}_{plot_type}_grid_systematic.png"
            plt.savefig(os.path.join(debug_dir, filename), dpi=300, bbox_inches='tight', facecolor='white')
            if verbose:
                print(f"Grid plot saved: {filename}")
                
            # Save individual subplots
            individual_dir = os.path.join(debug_dir, "individual_plots", f"{function_type}_{plot_type}")
            os.makedirs(individual_dir, exist_ok=True)
            
            for idx, vel_combo in enumerate(vel_combinations):
                # Create individual plot
                fig_ind, ax_ind = plt.subplots(1, 1, figsize=(6, 5))
                
                # Compute function values again for this specific combination
                values_dict = _compute_function_values(particle, pos_grid_display, pos_grid_scaled, pos_indices, vel_combo, model, function_type)
                
                extent = [pos_grid_display[0].min(), pos_grid_display[0].max(), 
                         pos_grid_display[1].min(), pos_grid_display[1].max()]
                
                if plot_type == 'analytical':
                    values = values_dict['analytical']
                    cmap = 'viridis' if function_type == 'lagrangian' else 'coolwarm'
                    norm = None if function_type == 'lagrangian' else SymLogNorm(linthresh=1e-6)
                else:  # learned
                    values = values_dict['learned']
                    cmap = 'viridis' if function_type == 'lagrangian' else 'coolwarm'
                    norm = None if function_type == 'lagrangian' else SymLogNorm(linthresh=1e-6)
                
                im_ind = ax_ind.imshow(values, extent=extent, origin='lower', aspect='auto', cmap=cmap, norm=norm)
                title = f'$\\dot{{q}}_1 = {vel_combo[0]:.2f}$, $\\dot{{q}}_2 = {vel_combo[1]:.2f}$'
                ax_ind.set_title(title, fontsize=12, fontweight='bold')
                ax_ind.set_xlabel(coord_names[0], fontsize=12)
                ax_ind.set_ylabel(coord_names[1], fontsize=12)
                
                cbar_ind = plt.colorbar(im_ind, ax=ax_ind)
                if function_type == 'lagrangian':
                    cbar_ind.set_label('L', fontsize=12)
                else:
                    cbar_ind.set_label('det(M)', fontsize=12)
                
                plt.tight_layout()
                
                individual_filename = f"{function_type}_{plot_type}_vel_{idx+1:02d}.png"
                plt.savefig(os.path.join(individual_dir, individual_filename), dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig_ind)
            
            if verbose:
                print(f"Individual plots saved to: {individual_dir}")
        
        if verbose:
            plt.show()
        else:
            plt.close()

def _create_multiple_grid_plots(particle, model, pos_grid_display, pos_grid_scaled, pos_indices, vel_combinations,
                               debug_dir, verbose, function_type, coord_names, n_vel_dims):
    """Create multiple 3x3 grid plots for higher-dimensional systems (Option B)."""
    # Group combinations by higher-dimension values
    grouped_combinations = {}
    v_values = [-1, 0, 1]
    
    for i, higher_v in enumerate(v_values):
        group_key = f"higher_dims_{higher_v}"
        grouped_combinations[group_key] = vel_combinations[i*9:(i+1)*9]
    
    for group_name, group_combinations in grouped_combinations.items():
        higher_v_val = group_name.split('_')[-1]
        
        # Create both analytical and learned plots for each group
        for plot_type in ['analytical', 'learned']:
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            
            if plot_type == 'analytical':
                fig.suptitle(f'Analytical {function_type.title()} - {coord_names[0]} vs {coord_names[1]} '
                            f'(Higher dims: {higher_v_val})', fontsize=16, fontweight='bold')
            else:  # learned
                fig.suptitle(f'Learned {function_type.title()} - {coord_names[0]} vs {coord_names[1]} '
                            f'(Higher dims: {higher_v_val})', fontsize=16, fontweight='bold')
            
            for idx, vel_combo in enumerate(group_combinations):
                row = idx // 3
                col = idx % 3
                ax = axes[row, col]
                
                # Compute function values
                values_dict = _compute_function_values(particle, pos_grid_display, pos_grid_scaled, pos_indices, vel_combo, model, function_type)
                
                # Plot the requested type
                extent = [pos_grid_display[0].min(), pos_grid_display[0].max(), 
                         pos_grid_display[1].min(), pos_grid_display[1].max()]
                
                if plot_type == 'analytical':
                    values = values_dict['analytical']
                    cmap = 'viridis' if function_type == 'lagrangian' else 'coolwarm'
                    norm = None if function_type == 'lagrangian' else SymLogNorm(linthresh=1e-6)
                else:  # learned
                    values = values_dict['learned']
                    cmap = 'viridis' if function_type == 'lagrangian' else 'coolwarm'
                    norm = None if function_type == 'lagrangian' else SymLogNorm(linthresh=1e-6)
                
                im = ax.imshow(values, extent=extent, origin='lower', aspect='auto', cmap=cmap, norm=norm)
                title = f'$\\dot{{q}}_1 = {vel_combo[0]:.2f}$, $\\dot{{q}}_2 = {vel_combo[1]:.2f}$'
                    
                ax.set_title(title, fontsize=10)
                ax.set_xlabel(coord_names[0])
                ax.set_ylabel(coord_names[1])
                plt.colorbar(im, ax=ax)
            
            # Hide unused subplots
            for idx in range(len(group_combinations), 3 * 3):
                row = idx // 3
                col = idx % 3
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            if debug_dir:
                filename = f"{function_type}_{plot_type}_grid_{group_name}_systematic.png"
                plt.savefig(os.path.join(debug_dir, filename), dpi=300, bbox_inches='tight', facecolor='white')
                if verbose:
                    print(f"Grid plot saved: {filename}")
                
                # Save individual subplots
                individual_dir = os.path.join(debug_dir, "individual_plots", f"{function_type}_{plot_type}_{group_name}")
                os.makedirs(individual_dir, exist_ok=True)
                
                for idx, vel_combo in enumerate(group_combinations):
                    # Create individual plot
                    fig_ind, ax_ind = plt.subplots(1, 1, figsize=(6, 5))
                    
                    # Compute function values again for this specific combination
                    values_dict = _compute_function_values(particle, pos_grid_display, pos_grid_scaled, pos_indices, vel_combo, model, function_type)
                    
                    extent = [pos_grid_display[0].min(), pos_grid_display[0].max(), 
                             pos_grid_display[1].min(), pos_grid_display[1].max()]
                    
                    if plot_type == 'analytical':
                        values = values_dict['analytical']
                        cmap = 'viridis' if function_type == 'lagrangian' else 'coolwarm'
                        norm = None if function_type == 'lagrangian' else SymLogNorm(linthresh=1e-6)
                    else:  # learned
                        values = values_dict['learned']
                        cmap = 'viridis' if function_type == 'lagrangian' else 'coolwarm'
                        norm = None if function_type == 'lagrangian' else SymLogNorm(linthresh=1e-6)
                    
                    im_ind = ax_ind.imshow(values, extent=extent, origin='lower', aspect='auto', cmap=cmap, norm=norm)
                    title = f'$\\dot{{q}}_1 = {vel_combo[0]:.2f}$, $\\dot{{q}}_2 = {vel_combo[1]:.2f}$'
                    ax_ind.set_title(title, fontsize=12, fontweight='bold')
                    ax_ind.set_xlabel(coord_names[0], fontsize=12)
                    ax_ind.set_ylabel(coord_names[1], fontsize=12)
                    
                    cbar_ind = plt.colorbar(im_ind, ax=ax_ind)
                    if function_type == 'lagrangian':
                        cbar_ind.set_label('L', fontsize=12)
                    else:
                        cbar_ind.set_label('det(M)', fontsize=12)
                    
                    plt.tight_layout()
                    
                    individual_filename = f"{function_type}_{plot_type}_{group_name}_vel_{idx+1:02d}.png"
                    plt.savefig(os.path.join(individual_dir, individual_filename), dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig_ind)
                
                if verbose:
                    print(f"Individual plots saved to: {individual_dir}")
        
        if verbose:
            plt.show()
        else:
            plt.close()

def _create_lagrangian_animation(particle, model, pos_grid_display, pos_grid_scaled, pos_indices, debug_dir, verbose, velocity_bounds_override=None):
    """Create Lagrangian animation with synchronized velocity movement."""
    try:
        import matplotlib.animation as animation
        
        # Create frames
        frames = _create_animation_frames(particle, pos_grid_display, pos_grid_scaled, pos_indices, model, 'lagrangian', n_frames=40, velocity_bounds_override=velocity_bounds_override)
        
        if len(pos_indices) == 1:
            _create_1d_animation(frames, pos_grid_display, pos_indices, debug_dir, verbose, 'lagrangian')
        else:
            _create_2d_animation(frames, pos_grid_display, pos_indices, debug_dir, verbose, 'lagrangian')
    except ImportError:
        if verbose:
            print("Animation skipped: matplotlib.animation not available")
    except Exception as e:
        if verbose:
            print(f"Animation creation failed: {e}")

def _create_determinant_animation(particle, model, pos_grid_display, pos_grid_scaled, pos_indices, debug_dir, verbose, velocity_bounds_override=None):
    """Create mass matrix determinant animation."""
    try:
        import matplotlib.animation as animation
        
        frames = _create_animation_frames(particle, pos_grid_display, pos_grid_scaled, pos_indices, model, 'determinant', n_frames=40, velocity_bounds_override=velocity_bounds_override)
        
        if len(pos_indices) == 1:
            _create_1d_animation(frames, pos_grid_display, pos_indices, debug_dir, verbose, 'determinant')
        else:
            _create_2d_animation(frames, pos_grid_display, pos_indices, debug_dir, verbose, 'determinant')
    except ImportError:
        if verbose:
            print("Animation skipped: matplotlib.animation not available")
    except Exception as e:
        if verbose:
            print(f"Animation creation failed: {e}")

def _create_2d_animation(frames, pos_grid, pos_indices, debug_dir, verbose, function_type):
    """Create 2D animation for multi-dimensional systems."""
    import matplotlib.animation as animation
    
    q1_grid, q2_grid = pos_grid
    extent = [q1_grid.min(), q1_grid.max(), q2_grid.min(), q2_grid.max()]
    
    coord_names = []
    for idx in pos_indices:
        coord_names.append(f'θ{idx+1}' if idx in getattr(frames[0], 'angle_indices', []) else f'q{idx+1}')
    
    # Create animations for analytical and learned
    for anim_type in ['analytical', 'learned']:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Initialize plot based on type
        if anim_type == 'analytical':
            if function_type == 'lagrangian':
                im = ax.imshow(frames[0]['analytical'], extent=extent, origin='lower', aspect='auto', cmap='viridis')
                ax.set_title('Analytical Lagrangian Animation', fontweight='bold')
                cbar = plt.colorbar(im, ax=ax, label='L (Analytical)')
            else:
                im = ax.imshow(frames[0]['analytical'], extent=extent, origin='lower', aspect='auto', cmap='coolwarm',
                              norm=SymLogNorm(linthresh=1e-6))
                ax.set_title('Analytical Mass Matrix Determinant Animation', fontweight='bold')
                cbar = plt.colorbar(im, ax=ax, label='det(M) (Analytical)')
        else:  # learned
            if function_type == 'lagrangian':
                im = ax.imshow(frames[0]['learned'], extent=extent, origin='lower', aspect='auto', cmap='viridis')
                ax.set_title('Learned Lagrangian Animation', fontweight='bold')
                cbar = plt.colorbar(im, ax=ax, label='L (Learned)')
            else:
                im = ax.imshow(frames[0]['learned'], extent=extent, origin='lower', aspect='auto', cmap='coolwarm',
                              norm=SymLogNorm(linthresh=1e-6))
                ax.set_title('Learned Mass Matrix Determinant Animation', fontweight='bold')
                cbar = plt.colorbar(im, ax=ax, label='det(M) (Learned)')
        
        ax.set_xlabel(coord_names[0])
        ax.set_ylabel(coord_names[1])
        
        def animate(frame_idx):
            if anim_type == 'analytical':
                data = frames[frame_idx]['analytical']
            else:  # learned
                data = frames[frame_idx]['learned']
            
            im.set_array(data)
            im.set_clim(vmin=data.min(), vmax=data.max())
            return [im]
        
        ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=150, blit=True, repeat=True)
        
        if debug_dir:
            # Try MP4 first, fallback to GIF
            try:
                filename = f"{function_type}_{anim_type}_animation.mp4"
                ani.save(os.path.join(debug_dir, filename), writer='ffmpeg', fps=8, dpi=150)
                if verbose:
                    print(f"Animation saved: {filename}")
            except Exception as e:
                try:
                    filename = f"{function_type}_{anim_type}_animation.gif"
                    ani.save(os.path.join(debug_dir, filename), writer='pillow', fps=5, dpi=100)
                    if verbose:
                        print(f"Animation saved as GIF: {filename} (ffmpeg not available)")
                except Exception as e2:
                    if verbose:
                        print(f"Animation save failed: {e2}")
        
        plt.close()

def _create_1d_animation(frames, pos_grid, pos_indices, debug_dir, verbose, function_type):
    """Create 1D animation.""" 
    import matplotlib.animation as animation
    
    # Create animations for analytical and learned
    for anim_type in ['analytical', 'learned']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Initialize plot based on type
        if anim_type == 'analytical':
            line, = ax.plot(pos_grid, frames[0]['analytical'], linewidth=2, color='blue')
            ax.set_title(f'Analytical {function_type.title()} Animation - 1D System', fontweight='bold')
        else:  # learned
            line, = ax.plot(pos_grid, frames[0]['learned'], linewidth=2, color='red')
            ax.set_title(f'Learned {function_type.title()} Animation - 1D System', fontweight='bold')
        
        ax.set_xlabel(f'Position')
        ax.set_ylabel(f'{function_type.title()}')
        ax.grid(True, alpha=0.3)
        
        # Set y-limits based on all frames for this animation type
        if anim_type == 'analytical':
            all_values = np.concatenate([frame['analytical'] for frame in frames])
        else:  # learned
            all_values = np.concatenate([frame['learned'] for frame in frames])
        
        ax.set_ylim(all_values.min() * 1.1, all_values.max() * 1.1)
        
        def animate(frame_idx):
            if anim_type == 'analytical':
                data = frames[frame_idx]['analytical']
            else:  # learned
                data = frames[frame_idx]['learned']
            
            line.set_ydata(data)
            return [line]
        
        ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=150, blit=True, repeat=True)
        
        if debug_dir:
            # Try MP4 first, fallback to GIF
            try:
                filename = f"{function_type}_{anim_type}_animation_1d.mp4"
                ani.save(os.path.join(debug_dir, filename), writer='ffmpeg', fps=8, dpi=150)
                if verbose:
                    print(f"1D Animation saved: {filename}")
            except Exception as e:
                try:
                    filename = f"{function_type}_{anim_type}_animation_1d.gif"
                    ani.save(os.path.join(debug_dir, filename), writer='pillow', fps=5, dpi=100)
                    if verbose:
                        print(f"1D Animation saved as GIF: {filename} (ffmpeg not available)")
                except Exception as e2:
                    if verbose:
                        print(f"1D Animation save failed: {e2}")
        
        plt.close()