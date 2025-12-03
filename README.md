# Hessian Regularized Lagrangian Neural Networks

A modular, robust implementation of Lagrangian Neural Networks (LNNs) with Hessian regularization in PyTorch. This repository accompanies the paper *"Improving Lagrangian Neural Networks with Hessian Regularization"*.

## Overview

Lagrangian Neural Networks can learn arbitrary Lagrangians from trajectory data, but their unusual optimization objective leads to significant training instabilities. This implementation addresses these challenges through:

- **Hessian Regularization**: Penalizes unphysical signatures in the Lagrangian's second derivatives with respect to velocities, preventing the network from learning unstable dynamics
- **Improved Activation Functions**: GeLU activation functions better suited to learning Lagrangians
- **Physics-Aware Coordinate Scaling**: Improves stability across different physical systems
- **Extended Training Bounds**: Helps networks learn periodic boundary conditions

## Features

- Support for multiple physical systems:
  - **Classical Mechanics**:
    - Constant force motion
    - Harmonic oscillator (1D spring)
    - Spring pendulum (2D)
    - Double pendulum (chaotic system)
    - Triple pendulum (highly chaotic system)
  - **Differential Geometry**:
    - Sphere geodesic (free particle on a sphere)
  - **Relativistic Systems**:
    - AdS₂ geodesic (2D Anti-de Sitter spacetime)
    - AdS₃ geodesic (3D Anti-de Sitter spacetime)
    - AdS₄ geodesic (4D Anti-de Sitter spacetime)
- Energy conservation through Lagrangian mechanics
- Trajectory visualizations with animations
- Comprehensive evaluation and analysis tools

## Project Structure

```
Hessian Regularized LNN/
├── src/
│   ├── main.py                 # Main entry point
│   ├── analytical_tools.py     # Analytical tools for evaluation
│   ├── hyperparameter_search.py # Hyperparameter search utilities
│   ├── architectures/          # Neural network architectures
│   │   ├── gelu_LNN.py         # GeLU LNN architecture
│   │   ├── relativistic_LNN.py # Relativistic LNN with Lorentzian signature
│   │   └── ...                 # Other architectures
│   ├── physical_systems.py     # Physical system implementations
│   ├── LNN.py                  # Base neural network architectures
│   ├── train.py                # Training functions
│   ├── evaluate.py             # Evaluation and plotting
│   ├── file_exports.py         # Save/load functionality
│   ├── ode_solve.py            # Numerical integration
│   ├── dataset_creation.py     # Data generation
│   ├── lagrangian_demo.py      # Solving by Lagrangian demonstration
│   └── variance_preserving_initializations.py  # Custom initializations
├── outputs/                    # Generated outputs and saved models
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (optional but recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/abdullahumuth/hessian_regularized_lnn.git
   cd hessian_regularized_lnn
   ```

2. Create a virtual environment:
   ```bash
   python -m venv lnn-env
   source lnn-env/bin/activate  # On Windows: lnn-env\Scripts\activate
   ```

3. Install PyTorch with CUDA support (adjust for your CUDA version):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```

4. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. For video animations, install FFmpeg separately:
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from https://ffmpeg.org/

## Supported Physical Systems

### Classical Mechanics

| System | `--physics` flag | DOF | Description |
|--------|------------------|-----|-------------|
| Constant Force | `constant_force` | 2 | Simple motion under constant force (e.g., gravity) |
| Harmonic Oscillator | `harmonic_oscillator` | 2 | 1D spring oscillator |
| Spring Pendulum | `spring_pendulum` | 4 | 2D pendulum with elastic spring |
| Double Pendulum | `double_pendulum` | 4 | Chaotic coupled pendulum system |
| Triple Pendulum | `triple_pendulum` | 6 | Highly chaotic three-mass pendulum |

### Differential Geometry

| System | `--physics` flag | DOF | Description |
|--------|------------------|-----|-------------|
| Sphere Geodesic | `sphere_geodesic` | 4 | Free particle constrained to sphere surface |

### Relativistic Systems

| System | `--physics` flag | DOF | Description |
|--------|------------------|-----|-------------|
| AdS₂ Geodesic | `ads2_geodesic` | 4 | Geodesic in 2D Anti-de Sitter spacetime |
| AdS₃ Geodesic | `ads3_geodesic` | 6 | Geodesic in 3D Anti-de Sitter spacetime |
| AdS₄ Geodesic | `ads4_geodesic` | 8 | Geodesic in 4D Anti-de Sitter spacetime |

> **Important**: For relativistic systems (those with spacetime coordinates as inputs), you must use `--model_name relativistic` to enable the Lorentzian signature regularization. See [Relativistic Systems](#relativistic-systems) for details.

## Usage

### Command Line Interface

#### Basic Usage
```bash
cd src
python main.py [options]
```

#### System Selection
- `--physics SYSTEM` - Physical system to use (see tables above)

#### Model Selection
- `--model_name NAME` - Name of the model architecture to use:
  - `LNN` (default) - Standard LNN with eigenvalue regularization for classical systems
  - `relativistic` - LNN with Lorentzian signature regularization for relativistic systems

#### Mode Selection
- `--mode MODE` - Operation mode:
  - `train_from_scratch` - Train a new model from scratch
  - `train_from_imported` - Continue training from existing model
  - `test` - Test existing model and generate visualizations

#### File Paths
- `--load_path PATH` - Path to load existing model/data from
- `--save_path PATH` - Path to save outputs (auto-generated if not specified)
- `--load_history` - Load existing training history
- `--load_trajectory` - Load existing trajectory data for evaluation
- `--training_data_path PATH` - Path to custom training dataset

#### Physical Parameters
- `--physical_params P1 P2 ...` - Physical parameters in order:
  - Double pendulum: `[m1, m2, l1, l2, g]`
  - Triple pendulum: `[m1, m2, m3, l1, l2, l3, g]`
  - Harmonic oscillator: `[m, k]`
  - Spring pendulum: `[m, k, g]`
  - Constant force: `[m, g]`
  - Sphere geodesic: `[m, R]`
  - AdS geodesics: `[L]` (AdS length scale)

#### Training Hyperparameters
- `--lr RATE` - Learning rate (default: 1e-3)
- `--num_samples N` - Number of training samples
- `--total_epochs N` - Total training epochs
- `--minibatch_size N` - Minibatch size
- `--eta_min RATE` - Minimum learning rate for cosine annealing scheduler
- `--lambda_val VALUE` - Regularization strength for Hessian penalty (default: 5)

#### Model Parameters
- `--num_layers N` - Number of neural network layers
- `--hidden_dim N` - Hidden layer dimension

#### Bounds Configuration
- `--train_position_bounds` / `--train_velocity_bounds` - Training data bounds
- `--validation_position_bounds` / `--validation_velocity_bounds` - Validation bounds
- `--test_position_bounds` / `--test_velocity_bounds` - Test trajectory bounds

#### Test Configuration
- `--num_tests N` - Number of test trajectories to generate
- `--time_step DT` - Time step for integration
- `--time_bounds START END` - Time bounds for trajectory simulation

#### System Configuration
- `--dtype TYPE` - Data type: `float32` or `float64` (default: float32)
- `--device DEVICE` - Device: `cuda` or `cpu` (default: cuda)
- `--verbose` - Enable verbose output
- `--animate` - Generate trajectory animations

#### Feature Toggles
- `--no_demo` - Disable Lagrangian demo mode
- `--no_debug` - Disable debug evaluation mode
- `--no_trajectory` - Disable trajectory testing

### Training Examples

#### Classical Systems

```bash
# Train double pendulum with Hessian regularization
python main.py --physics double_pendulum --mode train_from_scratch --total_epochs 300 --lambda_val 5 --verbose

# Train triple pendulum (highly chaotic)
python main.py --physics triple_pendulum --mode train_from_scratch --total_epochs 500 --verbose

# Train spring pendulum
python main.py --physics spring_pendulum --mode train_from_scratch --total_epochs 500 --verbose

# Train sphere geodesic
python main.py --physics sphere_geodesic --mode train_from_scratch --total_epochs 50 --verbose

# Continue training from existing model
python main.py --physics double_pendulum --mode train_from_imported --load_path ./outputs/my_model --total_epochs 100
```

#### Relativistic Systems

For relativistic systems (those with spacetime coordinates as inputs), you **must** specify `--model_name relativistic` to use the appropriate Lorentzian signature regularization:

```bash
# Train AdS2 geodesic
python main.py --physics ads2_geodesic --model_name relativistic --mode train_from_scratch --total_epochs 120 --verbose

# Train AdS3 geodesic
python main.py --physics ads3_geodesic --model_name relativistic --mode train_from_scratch --total_epochs 150 --verbose

# Train AdS4 geodesic
python main.py --physics ads4_geodesic --model_name relativistic --mode train_from_scratch --total_epochs 200 --verbose
```

### Testing Examples

```bash
# Test classical model
python main.py --physics double_pendulum --mode test --load_path ./outputs/double_pendulum_model --verbose

# Test relativistic model
python main.py --physics ads4_geodesic --model_name relativistic --mode test --load_path ./outputs/ads4_model --verbose

# Test with animations
python main.py --physics double_pendulum --mode test --load_path ./outputs/model --animate --verbose

# Quick test without debug or trajectory evaluation
python main.py --physics double_pendulum --mode test --load_path ./outputs/model --no_debug --no_trajectory
```

## Output Structure

The system generates outputs in directories with experiment-specific naming:

```
outputs/
└── {experiment_name}/
    ├── model.pt                    # Trained PyTorch model
    ├── particle.json               # Physical system parameters and configuration
    ├── history.csv                 # Training loss history (epoch, train_loss, val_loss)
    ├── history_metadata.json       # Training metadata and hyperparameters
    ├── debug_results/              # Debug and diagnostic outputs
    ├── test_results/               # Test evaluation results
    ├── trajectory_test_results/    # Numerical trajectory comparison data
    └── validation_results/         # Validation analysis
```

### Example Experiment Types

Based on typical experimental configurations, the system supports various experiment naming patterns:

- **Architecture Experiments**: `GeLU/`, `Softplus/`, `xtanhkx/`
- **Initialization Experiments**: `GeLUInitialized/`, `SoftplusInitialized/`
- **Hybrid Experiments**: `QuadraticHybrid/`, `xtanhkxGeLUHybrid/`
- **Normalization Studies**: `DyTNoNormalization/`, `LNNNoNormalization/`
- **Specialized Variants**: `GeLUInitializedNoExtraTerm/`, `SoftplusNaiveInitialized/`

### File Descriptions

- **model.pt**: Complete trained PyTorch model with all learned parameters
- **particle.json**: Serialized physical system containing all configuration and parameters
- **history.csv**: Training progress with loss values over epochs
- **history_metadata.json**: Complete training configuration, hyperparameters, and experiment settings
- **debug_results/**: Debug outputs, intermediate results, and diagnostic information
- **test_results/**: Comprehensive test evaluation including visualizations and error analysis
- **trajectory_test_results/**: Numerical trajectory data for comparison between analytical and predicted solutions
- **validation_results/**: Validation analysis and performance metrics

## Hessian Regularization

The primary source of training instability in LNNs stems from computing the inverse of the Lagrangian's Hessian with respect to generalized velocities (the mass matrix). When this matrix approaches singularity, training becomes unstable.

### Classical Systems (Eigenvalue Regularization)

For classical systems, we penalize negative eigenvalues of the mass matrix:

$$\text{Loss} = \text{MAE}(\ddot{q}_{\text{real}}, \ddot{q}_{\text{predicted}}) + \lambda \sum_{i} |\lambda_i| \text{ for } \lambda_i < 0$$

This ensures the mass matrix remains positive semi-definite, which is physically required for stable classical dynamics.

### Relativistic Systems (Lorentzian Regularization)

For relativistic systems where spacetime coordinates are inputs, the mass matrix must have Lorentzian signature (one negative eigenvalue for the timelike direction). We enforce:
- The $M_{00}$ component (timelike) must be non-positive
- The spatial submatrix must be positive semi-definite

This allows LNNs to learn geodesic motion in curved spacetimes and extract metric tensor elements.

The `--lambda_val` parameter controls regularization strength (default: 5). Higher values enforce stricter constraints but may slow convergence.

## Theory

### Lagrangian Neural Networks

LNNs learn the Lagrangian function $L = T - V$ directly. The equations of motion are derived through the Euler-Lagrange equation:

$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) - \frac{\partial L}{\partial q} = 0$$

This ensures energy conservation and physical consistency. The acceleration is computed as:

$$\ddot{q} = \left(\nabla_{\dot{q}}\nabla_{\dot{q}}^T L\right)^{-1}\left[\nabla_q L - \left(\nabla_q \nabla_{\dot{q}}^T L\right)\dot{q}\right]$$

### Anti-de Sitter Spacetime

The AdS geodesic systems model particle motion in Anti-de Sitter spacetime using the Poincaré patch metric:

$$ds^2 = \frac{L^2}{z^2} \left(-dt^2 + dx^2 + \cdots + dz^2\right)$$

These systems are relevant to the AdS/CFT correspondence in theoretical physics. Our Lorentzian regularization enables LNNs to learn such metrics directly from trajectory data.

## Troubleshooting

Common issues and solutions:

1. **CUDA out of memory during training**: Reduce `--minibatch_size` or `--num_samples`
2. **CUDA out of memory during ODE solving**: This happens during trajectory testing. Increase time splitting with the `num_time_splits` parameter in the code, or reduce the number of test cases
3. **Animation failures**: Install FFmpeg or check file permissions
4. **Training instability**: Increase `--lambda_val` for stronger regularization or lower learning rate with `--lr`
5. **Import errors**: Ensure all requirements are installed: `pip install -r requirements.txt`
6. **Long trajectory computation**: The system automatically uses time-splitting for memory management. Check console output for progress
7. **Model loading errors**: Ensure the model path exists and contains both `model.pt` and `particle.json`
8. **Relativistic system not training properly**: Ensure you're using `--model_name relativistic` for systems with spacetime inputs
9. **Coordinate singularity issues**: Some systems (e.g., sphere geodesic at poles, AdS at z=0) have coordinate singularities; ensure training bounds avoid these regions

## Citation

If you use this code in your research, please cite the following paper:

We are preparing a formal citation for the paper. Please check back later for the updated citation information.

## License

This project is licensed under the Apache 2.0 license.

## Contact

For questions or issues, please open a GitHub issue or contact me at abdullah.hamzaogullari@std.bogazici.edu.tr
