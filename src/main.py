import math
import numpy as np
import matplotlib.pyplot as plt 
import argparse
import torch
import os
import time
import sys

# Add architectures directory to Python path for compatibility
# This allows PyTorch to find architecture modules when loading some models
architectures_path = os.path.join(os.path.dirname(__file__), 'architectures')
if architectures_path not in sys.path:
    sys.path.insert(0, architectures_path)

from physical_systems import initialize_particle_system
from file_exports import load_model, load_particle, save_history, save_model_and_particle, load_history, load_trajectory_test_results
from evaluate import debug_model, create_trajectory_tests, trajectory_test, plot_trajectory_test_comparisons, plot_training_validation_curves
from lagrangian_demo import run_lagrangian_demo
from train import train_from_scratch, train_from_imported

def setup():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LNN training and testing')
    
    # System selection
    parser.add_argument('--physics', type=str, default="double_pendulum", help='Which physical system do you consider? (For the names, check physical_systems.py)')
    parser.add_argument('--mode', type=str, default="test", help='Training mode (test, train_from_scratch, train_from_imported)')
    parser.add_argument('--load_path', type=str, default="", help='Path to the model to be loaded')
    parser.add_argument('--save_path', type=str, default="", help='Path to the model to be saved')
    parser.add_argument('--dtype', type=str, default="float32", help='Data type (float32 or float64)')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use (cuda or cpu)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--load_history', action='store_true', help='Enable loading of training history')
    parser.add_argument('--load_trajectory', action='store_true', help='Enable loading of trajectory data')
    parser.add_argument('--no_demo', action='store_true', help='Disable demo mode')
    parser.add_argument('--no_debug', action='store_true', help='Disable debug mode')
    parser.add_argument('--no_trajectory', action='store_true', help='Disable trajectory mode')
    parser.add_argument('--model_name', type=str, default="LNN", help='Name of the model to use')
    parser.add_argument('--lambda_val', type=float, default=5,
                        help='Lambda penalty value for negative eigenvalue regularization (default: 5)')

    # Physical system parameters - generic list
    parser.add_argument('--physical_params', type=float, nargs='*', help='Physical parameters in order: for double_pendulum [m1, m2, l1, l2, g], for harmonic_oscillator [m, k], etc.')
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--num_samples', type=int, help='Number of training samples')
    parser.add_argument('--total_epochs', type=int, help='Total training epochs')
    parser.add_argument('--minibatch_size', type=int, help='Minibatch size')
    parser.add_argument('--eta_min', type=float, help='Minimum learning rate for scheduler')
    
    # Model parameters
    parser.add_argument('--num_layers', type=int, help='Number of layers')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension')
    
    # Validation parameters
    parser.add_argument('--validation_position_bounds', type=float, nargs='*', help='Validation position bounds as flat list: [min1, max1, min2, max2, ...]')
    parser.add_argument('--validation_velocity_bounds', type=float, nargs='*', help='Validation velocity bounds as flat list: [min1, max1, min2, max2, ...]')
    
    # Test parameters
    parser.add_argument('--test_position_bounds', type=float, nargs='*', help='Test position bounds as flat list: [min1, max1, min2, max2, ...]')
    parser.add_argument('--test_velocity_bounds', type=float, nargs='*', help='Test velocity bounds as flat list: [min1, max1, min2, max2, ...]')
    
    # Training bounds parameters
    parser.add_argument('--train_position_bounds', type=float, nargs='*', help='Training position bounds as flat list: [min1, max1, min2, max2, ...]')
    parser.add_argument('--train_velocity_bounds', type=float, nargs='*', help='Training velocity bounds as flat list: [min1, max1, min2, max2, ...]')
    
    parser.add_argument('--num_tests', type=int, help='Number of test trajectories to generate')
    parser.add_argument('--time_step', type=float, help='Time step for integration')
    parser.add_argument('--time_bounds', type=float, nargs=2, help='Time bounds as [start, end]')
    parser.add_argument('--animate', action='store_true', help='If you want to animate the trajectories (may take a long time depending on the system)')

    parser.add_argument('--training_data_path', type=str, default='', help='If you have a dataset for training already, provide its path. Format: [state, differentiated_state] where state has the shape (number_of_batches, dimension).')
    args = parser.parse_args()

    physics = args.physics
    mode = args.mode
    load_path = args.load_path
    save_path = args.save_path

    # Set dtype
    if args.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = torch.float32
    
    # Set device
    device = torch.device(args.device)

    verbose = args.verbose

    # Create particle based on physics
    particle = initialize_particle_system(physics)

    
    # Load particle parameters
    if load_path:
        try:
            particle = load_particle(particle, load_path)
        except Exception as e:
            print(f"Warning: Could not load particle parameters: {e}, using defaults")
    
    # Apply physical parameters from command line args (takes precedence)
    if args.physical_params is not None:
        if len(args.physical_params) != len(particle.param_names):
            print(f"Warning: Expected {len(particle.param_names)} physical parameters {particle.param_names}, got {len(args.physical_params)}")
        else:
            for i, (param_name, param_value) in enumerate(zip(particle.param_names, args.physical_params)):
                if hasattr(particle, param_name):
                    setattr(particle, param_name, param_value)
                    print(f"Set {param_name} = {param_value}")
    
    # Training hyperparameters
    lr = args.lr if args.lr is not None else particle.train_hyperparams['lr']
    num_samples = args.num_samples if args.num_samples is not None else particle.train_hyperparams['num_samples']
    total_epochs = args.total_epochs if args.total_epochs is not None else particle.train_hyperparams['total_epochs']
    minibatch_size = args.minibatch_size if args.minibatch_size is not None else particle.train_hyperparams['minibatch_size']
    eta_min = args.eta_min if args.eta_min is not None else particle.train_hyperparams['eta_min']
    
    # Update particle's hyperparameters
    particle.train_hyperparams['lr'] = lr
    particle.train_hyperparams['num_samples'] = num_samples
    particle.train_hyperparams['total_epochs'] = total_epochs
    particle.train_hyperparams['minibatch_size'] = minibatch_size
    particle.train_hyperparams['eta_min'] = eta_min
    
    # Model parameters
    num_layers = args.num_layers if args.num_layers is not None else particle.model_params['num_layers']
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else particle.model_params['hidden_dim']
    
    # Update particle's model parameters
    particle.model_params['num_layers'] = num_layers
    particle.model_params['hidden_dim'] = hidden_dim

    if args.time_bounds is not None:
        particle.test_params['time_bounds'] = tuple(args.time_bounds)
 
    particle.test_params['num_tests'] = args.num_tests if args.num_tests is not None else particle.test_params['num_tests']
    particle.test_params['time_step'] = args.time_step if args.time_step is not None else particle.test_params['time_step']

    # Parse validation bounds from command line and update particle directly
    if args.validation_position_bounds is not None:
        # Convert flat list to tuples: [min1, max1, min2, max2] -> [(min1, max1), (min2, max2)]
        bounds_list = args.validation_position_bounds
        if len(bounds_list) % 2 != 0:
            print("Warning: validation_position_bounds must have even number of elements")
        else:
            particle.test_params['validation_position_bounds'] = [(bounds_list[i], bounds_list[i+1]) for i in range(0, len(bounds_list), 2)]
    
    if args.validation_velocity_bounds is not None:
        # Convert flat list to tuples: [min1, max1, min2, max2] -> [(min1, max1), (min2, max2)]
        bounds_list = args.validation_velocity_bounds
        if len(bounds_list) % 2 != 0:
            print("Warning: validation_velocity_bounds must have even number of elements")
        else:
            particle.test_params['validation_velocity_bounds'] = [(bounds_list[i], bounds_list[i+1]) for i in range(0, len(bounds_list), 2)]

    # Parse test bounds from command line and update particle directly
    if args.test_position_bounds is not None:
        bounds_list = args.test_position_bounds
        if len(bounds_list) % 2 != 0:
            print("Warning: test_position_bounds must have even number of elements")
        else:
            particle.test_params['position_bounds'] = [(bounds_list[i], bounds_list[i+1]) for i in range(0, len(bounds_list), 2)]
    
    if args.test_velocity_bounds is not None:
        bounds_list = args.test_velocity_bounds
        if len(bounds_list) % 2 != 0:
            print("Warning: test_velocity_bounds must have even number of elements")
        else:
            particle.test_params['velocity_bounds'] = [(bounds_list[i], bounds_list[i+1]) for i in range(0, len(bounds_list), 2)]

    # Parse training bounds from command line and update particle directly
    if args.train_position_bounds is not None:
        bounds_list = args.train_position_bounds
        if len(bounds_list) % 2 != 0:
            print("Warning: train_position_bounds must have even number of elements")
        else:
            particle.train_hyperparams['position_bounds'] = [(bounds_list[i], bounds_list[i+1]) for i in range(0, len(bounds_list), 2)]
    
    if args.train_velocity_bounds is not None:
        bounds_list = args.train_velocity_bounds
        if len(bounds_list) % 2 != 0:
            print("Warning: train_velocity_bounds must have even number of elements")
        else:
            particle.train_hyperparams['velocity_bounds'] = [(bounds_list[i], bounds_list[i+1]) for i in range(0, len(bounds_list), 2)]

    # Set default save path if not provided
    if save_path == "":
        default_save_path = "./outputs"
        class_name = particle.__class__.__name__
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(default_save_path, f"{physics}_{timestamp}")

    # Validate load_path if loading is requested
    if args.load_history and not load_path:
        raise ValueError("--load_path must be provided when --load_history is specified")
    
    if args.load_trajectory and not load_path:
        raise ValueError("--load_path must be provided when --load_trajectory is specified")
    
    # Check if load_path exists when needed
    if (args.load_history or args.load_trajectory) and not os.path.exists(load_path):
        raise ValueError(f"Load path does not exist: {load_path}")

    return particle, args.model_name, mode, save_path, load_path, device, dtype, verbose, args.load_history, args.load_trajectory, args.no_demo, args.no_debug, args.no_trajectory, args.lambda_val, args.animate, args.training_data_path




def run(particle, save_path, mode, training_data_path='', load_path="", model_name="LNN", lambda_val=5, dtype=torch.float32, device=None):

    if mode == "train_from_scratch":
        model, history = train_from_scratch(particle, save_path, lambda_val, dtype, device,
                                            training_data_path,
                                          particle.test_params.get('validation_position_bounds'),
                                          particle.test_params.get('validation_velocity_bounds'), normalize=True, experiment_name=model_name)
        save_model_and_particle(particle, model, save_path)
        save_history(history, save_path)
        return model, history
    if load_path == "":
        raise ValueError("Load path must be provided in test mode")
    if mode == "test":
        model = load_model(load_path, device, dtype)
        #particle.scale_constants([84.64669036865234, 133.2147216796875])
        #particle.scale_constants([10.396069526672363, 9.808507919311523])
        return model, None
    elif mode == "train_from_imported":
        model = load_model(load_path, device, dtype)
        model, history = train_from_imported(particle, model, lambda_val, save_path, dtype, device,
                                            training_data_path,
                                            has_scheduler=True,
                                            validation_position_bounds=particle.test_params.get('validation_position_bounds'),
                                            validation_velocity_bounds=particle.test_params.get('validation_velocity_bounds'))
        save_model_and_particle(particle, model, save_path)
        save_history(history, save_path)
        return model, history
    else:
        print("Invalid training scenario")





def main():
    particle, model_name, mode, save_path, load_path, device, dtype, verbose, load_history_flag, load_trajectory_flag, no_demo_flag, no_debug_flag, no_trajectory_flag, lambda_val, animate_flag, training_data_path = setup()

    if not no_demo_flag:
        run_lagrangian_demo(particle)

    model, history = run(particle, save_path, mode, training_data_path, load_path, model_name, lambda_val, dtype, device)

    # Load history if requested
    if load_history_flag:
        try:
            history = load_history(load_path)
            if verbose:
                print(f"Successfully loaded training history from {load_path}")
        except Exception as e:
            print(f"Warning: Could not load training history from {load_path}: {e}")
            history = None

    # Load trajectory data if requested
    trajectory_data = None
    if load_trajectory_flag:
        try:
            trajectory_data = load_trajectory_test_results(load_path)
            if verbose:
                print(f"Successfully loaded trajectory test results from {load_path}")
                print(f"Loaded {len(trajectory_data)} trajectory test cases")
        except Exception as e:
            print(f"Warning: Could not load trajectory data from {load_path}: {e}")
            trajectory_data = None

    # Plot training/validation curves if history is available
    if history is not None:
        plot_training_validation_curves(history, save_path, verbose=verbose)
    # Use validation bounds from particle for both validation and plots

    if not no_debug_flag:
        if verbose:
            print(f"DEBUG: particle.dof = {particle.dof}")
            print(f"DEBUG: particle.dof//2 = {particle.dof//2}")
            print(f"DEBUG: train position_bounds = {particle.train_hyperparams['position_bounds']}")
            print(f"DEBUG: train velocity_bounds = {particle.train_hyperparams['velocity_bounds']}")
            print(f"DEBUG: validation position_bounds = {particle.test_params.get('validation_position_bounds')}")
            print(f"DEBUG: validation velocity_bounds = {particle.test_params.get('validation_velocity_bounds')}")

        debug_model(particle, model, save_path, dtype, device, verbose=verbose,
           position_bounds_override=particle.test_params.get('validation_position_bounds'),
           velocity_bounds_override=particle.test_params.get('validation_velocity_bounds'))
    
    if not no_trajectory_flag:
        # Use loaded trajectory data if available, otherwise create new tests
        if trajectory_data is not None:
            # Use loaded trajectory data
            q_qdot_test_list, nn_test_list, qdot_qdotdot_test_list, t_test = trajectory_data
            if verbose:
                print("Using loaded trajectory data for evaluation")
        else:
            # Create new trajectory tests and run them
            if verbose:
                print("Creating new trajectory tests")
            q_qdot_test_list, qdot_qdotdot_test_list, t_test = create_trajectory_tests(particle, dtype, device)
            nn_test_list = trajectory_test(particle, model, q_qdot_test_list, qdot_qdotdot_test_list, t_test, dtype, device, save_path, ode_dtype=torch.float64, num_time_splits=15, verbose=verbose)
    
        # Plot trajectory comparisons with the new animation features
        plot_trajectory_test_comparisons(particle, q_qdot_test_list, nn_test_list, t_test, save_path, animate=animate_flag, verbose=verbose)

# Run main

if __name__ == "__main__":
    main()