import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import json
import time
from LNN import LNN, custom_initialize_weights
from dataset_creation import create_training_data, normalize_training_data
def run_hyper_parameter_experiment(exp_config,
                   # Data related
                   train_ds_global, # Pass the global TensorDataset
                   # Model construction related (assuming these are global or from particle)
                   particle_dof_param,
                   num_layers_param,
                   activation_fn_module_param,
                   hidden_dim_param,
                   device_param,
                   dtype_param,
                   # LNN class and init function
                   initialize_model_weights_fn_param # Pass your weight init function
                   ):
    """
    Runs a single training experiment based on the given configuration
    and saves the results to a JSON file.
    """
    print(f"\n{'='*25}\n--- Starting Experiment: {exp_config['name']} ---")
    print(f"Config: LR={exp_config['lr']}, Epochs={exp_config['epochs']}, Scheduler Details below.")

    # For reproducibility of this specific run (optional)
    # torch.manual_seed(exp_config.get('seed', 42)) # Can add seed to config
    # if device_param.type == 'cuda':
    #     torch.cuda.manual_seed_all(exp_config.get('seed', 42))

    # 1. Re-initialize Model
    model = LNN(particle_dof_param,
                      num_layers_param,
                      activation_fn_module_param,
                      hidden_dim_param,
                      device_param) # LNN.__init__ should handle .to(device, dtype)
    
    initialize_model_weights_fn_param(model)
    
    # model.to(device_param, dtype=dtype_param) # Ensure this is done correctly, LNN init or here

    # DataParallel (optional, ensure it's re-applied if model is re-instantiated)
    # if torch.cuda.device_count() > 1 and not isinstance(model, torch.nn.DataParallel):
    #     print(f"Wrapping model with nn.DataParallel for {torch.cuda.device_count()} GPUs.")
    #     model = torch.nn.DataParallel(model)
    # model.to(device_param, dtype=dtype_param) # Final .to() after any wrapping

    # 2. Setup DataLoader, Optimizer, and Scheduler
    # Create DataLoader for this run (shuffling is good)
    current_train_loader = DataLoader(train_ds_global, 
                                      batch_size=exp_config['batch_size'], 
                                      shuffle=True, 
                                      num_workers=2, # If your OS supports it
                                      pin_memory=True if device_param.type == 'cuda' else False)
    
    exp_initial_lr = exp_config['lr']
    exp_num_epochs_to_run = exp_config['epochs']
    
    optimizer = torch.optim.RAdam(model.parameters(), lr=exp_initial_lr, weight_decay=1e-6, foreach=True)
    scheduler = exp_config['scheduler_fn'](optimizer, exp_num_epochs_to_run)
    print(f"Scheduler: {scheduler}")

    # 3. Run Training
    start_time = time.time()
    training_history = {"train_loss": [], "lr": [], "error": None} # Initialize
    success_status = False
    try:
        # Assuming LNN.fit takes num_epochs, train_loader, optimizer, scheduler, grad_clip
        training_history = model.fit(num_epochs_to_run=exp_num_epochs_to_run,
                                     train_loader=current_train_loader,
                                     optimizer=optimizer,
                                     scheduler=scheduler,
                                     gradient_clip_norm=exp_config.get("gradient_clip_norm", 1.0))
        success_status = True
    except Exception as e:
        print(f"!!! Experiment {exp_config['name']} FAILED with error: {e} !!!")
        import traceback
        traceback.print_exc()
        training_history["error"] = str(e)
        success_status = False
    end_time = time.time()
    duration_seconds = end_time - start_time

    # 4. Prepare Results
    results_to_save = {
        "experiment_name": exp_config['name'],
        "config_params": {
            "initial_lr": exp_config['lr'],
            "scheduler_name": exp_config['scheduler_fn'].__name__ if hasattr(exp_config['scheduler_fn'], '__name__') else str(exp_config['scheduler_fn']),
            "scheduler_config_details": str(scheduler), # Captures the instantiated scheduler's state
            "epochs_intended": exp_num_epochs_to_run,
            "batch_size": exp_config['batch_size'],
            "gradient_clip_norm": exp_config.get("gradient_clip_norm", 1.0),
            "num_layers": num_layers_param,
            "hidden_dim": hidden_dim_param,
            "activation_fn": str(activation_fn_module_param)
        },
        "training_duration_seconds": round(duration_seconds, 2),
        "training_history": training_history,
        "final_loss": training_history['train_loss'][-1] if success_status and training_history['train_loss'] else None,
        "status": "success" if success_status else "failed"
    }

    # 5. Save Results
    results_dir = "experiment_results_double_pendulum"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_filename = os.path.join(results_dir, f"{exp_config['name']}_{timestamp}.json")
    pt_filename = os.path.join(results_dir, f"{exp_config['name']}_{timestamp}.pt")
    try:
        with open(json_filename, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        print(f"Results for {exp_config['name']} saved to {json_filename}")
        with open(pt_filename, 'wb') as f:
            torch.save(model, f)
        print(f"The model weights for {exp_config['name']} saved to {pt_filename}")
    except Exception as e:
        print(f"!!! Failed to save results for {exp_config['name']} to {json_filename}. Error: {e} !!!")

    print(f"--- Experiment {exp_config['name']} Finished. Duration: {duration_seconds:.2f}s ---")
    return results_to_save, model

# %%
def hyperparameter_search(particle, dtype, device):
    # --- Global Setup (from your script) ---
    # Example:
    # Set seeds at the very beginning for reproducibility of the entire sequence
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42) # For all GPUs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Initialize Particle (specific to your setup) ---
    # Ensure 'particle' is your configured double_pendulum instance
    # particle = double_pendulum(m1=1, m2=1, l1=1, l2=1, g=9.8) # Example

    # --- Data Preparation (once) ---
    print("Preparing training data...")
    total_data_points = particle.train_hyperparams['num_samples']
    position_start_end = particle.train_hyperparams['position_bounds']
    velocity_start_end = particle.train_hyperparams['velocity_bounds']
    train_seed = particle.train_hyperparams['train_seed']
    
    # Create data on CPU to be safe with DataLoader num_workers
    training_data_cpu = create_training_data(total_data_points, particle.dof // 2,
                                             position_start_end, velocity_start_end,
                                             seed=train_seed, device=torch.device('cpu'))
    scale_factor_cpu, training_data_norm_cpu = normalize_training_data(training_data_cpu, particle.dof // 2)
    q_qdot_train_cpu, qdot_qdotdot_train_cpu = training_data_norm_cpu
    train_ds_global_main = TensorDataset(q_qdot_train_cpu, qdot_qdotdot_train_cpu) # Global dataset
    print(f"Training data prepared. Number of samples: {len(train_ds_global_main)}")
    print(f"Scale factor from normalization (used for test set later): {scale_factor_cpu}")

    # --- Model Parameters (from your particle setup) ---
    lnn_num_layers = particle.model_params['num_layers']
    lnn_activation_fn_module = particle.model_params['activation_fn'] # This should be an instance, e.g., nn.Softplus()
    lnn_hidden_dim = particle.model_params['hidden_dim']
    lnn_particle_dof = particle.dof

    # --- Experiment Configurations ---
    # (Copied from previous response, ensure imports like optim are available)
    experiment_configs = [
        # ... (your list of 7-8 experiment dicts) ...
        {
            "name": "Exp3_4e-4_CosineAnneal_TmaxEps",
            "lr": 6e-4,
            "scheduler_fn": lambda opt, exp_eps: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=exp_eps, eta_min=1e-6),
            "epochs": 300, "gradient_clip_norm": 1.0,
            "batch_size": 256
        },
    ]
    
    # --- Loop Through and Run Experiments ---
    all_run_summaries = []
    overall_start_time = time.time()
    MAX_TOTAL_TIME_MINUTES = 12 * 60 - 10 # Total time budget minus a small buffer

    for idx, current_exp_config in enumerate(experiment_configs):
        # Check remaining time
        current_total_elapsed_minutes = (time.time() - overall_start_time) / 60
        if current_total_elapsed_minutes > MAX_TOTAL_TIME_MINUTES:
            print(f"Total time limit approaching ({MAX_TOTAL_TIME_MINUTES} mins). Stopping before experiment {current_exp_config['name']}.")
            break

        print(f"\nStarting experiment {idx + 1}/{len(experiment_configs)}: {current_exp_config['name']}")
        
        results, model = run_hyper_parameter_experiment(
            exp_config=current_exp_config,
            train_ds_global=train_ds_global_main,
            particle_dof_param=lnn_particle_dof,
            num_layers_param=lnn_num_layers,
            activation_fn_module_param=lnn_activation_fn_module,
            hidden_dim_param=lnn_hidden_dim,
            device_param=device,
            dtype_param=dtype,
            initialize_model_weights_fn_param=custom_initialize_weights # Your weight init function
        )
        all_run_summaries.append(results) # Store full results if needed, or just summary

    # --- Print Final Summary ---
    print("\n\n{'='*25}\n--- All Attempted Experiments Summary ---")
    for summary_res in all_run_summaries:
        # Ensure keys exist before trying to access them for the print summary
        final_loss_str = f"{summary_res.get('final_loss', 'N/A'):.6f}" if summary_res.get('final_loss') is not None else "N/A"
        print(f"  Experiment: {summary_res.get('experiment_name', 'Unknown')}")
        print(f"    Status: {summary_res.get('status', 'Unknown')}")
        print(f"    Final Loss: {final_loss_str}")
        print(f"    Duration: {summary_res.get('training_duration_seconds', 0):.2f}s")
        if summary_res.get('status') == 'failed':
            print(f"    Error: {summary_res.get('training_history', {}).get('error', 'No error message captured')}")
        print("-" * 10)
    
    total_script_duration = (time.time() - overall_start_time) / 60
    print(f"Total script duration: {total_script_duration:.2f} minutes.")
