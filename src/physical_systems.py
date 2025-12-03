import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from mpl_toolkits.mplot3d import Axes3D

class PhysicalSystemBase(ABC):
    """
    Base template class for physical systems in Lagrangian Neural Networks.
    All physical systems should inherit from this class to ensure completeness.
    """
    
    def __init__(self):
        # These must be defined by subclasses
        self.dof = None  # Degrees of freedom (total dimensions, including velocities)
        self.angle_indices = []  # List of indices that represent angles (for periodic boundary conditions)
        self.scale = []  # Scaling factors for each coordinate
        self.param_names = []  # Names of physical parameters
        
        # These dictionaries must be defined by subclasses
        self.test_params = {}  # Parameters for testing/validation
        self.train_hyperparams = {}  # Training hyperparameters
        self.model_params = {}  # Neural network model parameters
    
    @abstractmethod
    def scale_constants(self, scale):
        """Set scaling constants for coordinates."""
        pass
    
    @abstractmethod
    def kinetic(self, q, q_dot, *args):
        """Calculate kinetic energy T(q, q_dot)."""
        pass
    
    @abstractmethod
    def potential(self, q, q_dot, *args):
        """Calculate potential energy V(q, q_dot)."""
        pass
    
    @abstractmethod
    def lagrangian(self, x):
        """Calculate Lagrangian L = T - V. Must handle both single state and batch inputs, and both numpy arrays and torch tensors."""
        pass
    
    @abstractmethod
    def energy(self, x):
        """Calculate total energy E = T + V. Must handle both torch.Tensor and np.ndarray."""
        pass
    
    @abstractmethod
    def solve_acceleration(self, q, qdot):
        """Solve for accelerations given positions and velocities."""
        pass
    
    @abstractmethod
    def plot_solved_dynamics(self, t, path, labelstr="", **kwargs):
        """Plot the dynamics of the system. Should return a matplotlib figure."""
        pass
    
    @abstractmethod
    def plot_lagrangian(self, t, path, lagrangian, labelstr=""):
        """Plot the Lagrangian over time."""
        pass
    
    def validate_implementation(self):
        """Validate that all required attributes and methods are properly implemented."""
        errors = []
        
        # Check required attributes
        if self.dof is None:
            errors.append("dof (degrees of freedom) must be set")
        if not isinstance(self.angle_indices, list):
            errors.append("angle_indices must be a list")
        if not isinstance(self.scale, list) or len(self.scale) == 0:
            errors.append("scale must be a non-empty list")
        if not isinstance(self.param_names, list) or len(self.param_names) == 0:
            errors.append("param_names must be a non-empty list")
        
        # Check required parameter dictionaries
        required_test_params = ['toy_position', 'toy_velocity', 'toy_time_dataset', 'time_step', 
                               'time_bounds', 'position_bounds', 'velocity_bounds', 'test_seed', 
                               'num_tests', 'validation_position_bounds', 'validation_velocity_bounds']
        for param in required_test_params:
            if param not in self.test_params:
                errors.append(f"test_params missing required key: {param}")
        
        required_train_params = ['lr', 'num_samples', 'test_ratio', 'total_epochs', 'minibatch_size', 
                                'train_seed', 'position_bounds', 'velocity_bounds', 'eta_min']
        for param in required_train_params:
            if param not in self.train_hyperparams:
                errors.append(f"train_hyperparams missing required key: {param}")
        
        required_model_params = ['num_layers', 'activation_fn', 'hidden_dim']
        for param in required_model_params:
            if param not in self.model_params:
                errors.append(f"model_params missing required key: {param}")
        
        if errors:
            raise ValueError("Implementation validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True

# %%
class constant_force(PhysicalSystemBase):

    def __init__(self, m=1.0, g=9.81):
        super().__init__()
        self.m = m
        self.g = g
        self.dof = 2
        self.angle_indices = []
        self.scale = [1.0]
        self.param_names = ['m', 'g']
        self.test_params = {'toy_position' : torch.tensor([0.48]),
                            'toy_velocity' : torch.tensor([0.31]),
                            'toy_time_dataset' : np.arange(0, 4, 0.001),
                            'time_step' : 0.001,
                            'time_bounds' : (0,4),
                            'position_bounds' : ((0,1),),
                            'velocity_bounds' : ((-2,2),),
                            'test_seed': 96,
                            'num_tests' : 4,
                            'validation_position_bounds' : ((0,1),),
                            'validation_velocity_bounds' : ((-2,2),),
                            }
        self.train_hyperparams = {'lr': 1e-3,
                                  'num_samples' : 4000,
                                  'test_ratio': 0.1,
                                  'total_epochs': 120,
                                  'minibatch_size': 64,
                                  'train_seed': 86,
                                  'position_bounds' : ((0,1),),
                                  'velocity_bounds' : ((-2,2),),
                                  'eta_min': 1e-6,
                                  }
        self.model_params = {'num_layers' : 3,
                             'activation_fn' : nn.GELU(),
                             'hidden_dim' : 128,
                            }
        
        # Validate completeness
        self.validate_implementation()

    def scale_constants(self, scale):
        self.scale = scale
    
    def kinetic(self, q, q_dot):
        if len(q.shape) == 1:
            w1 = q_dot[0] * self.scale[0]
            return 0.5 * self.m * w1**2
        w1 = q_dot[:,0] * self.scale[0]
        return 0.5 * self.m * w1**2
    
    def potential(self, q, q_dot):
        if len(q.shape) == 1:
            x1 = q[0] * self.scale[0]
            return self.m * self.g * x1
        x1 = q[:,0] * self.scale[0]
        return self.m * self.g * x1
    
    # NEED LAGRANGIAN TO PRODUCE SCALARS, NOT TENSORS!
    def lagrangian(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        q, qt = torch.split(x, self.dof//2, dim=-1)
        T = self.kinetic(q, qt)
        V = self.potential(q, qt)
        return T - V

    def energy(self, x):
        if isinstance(x, torch.Tensor):
            if len(x.shape)==1:
                x = x.unsqueeze(0)
            q, qt = torch.split(x, self.dof//2, dim=-1)
        elif isinstance(x, np.ndarray): 
            if len(x.shape)==1:
                x = np.expand_dims(x, axis=0)
            q, qt = np.split(x, self.dof//2, axis=-1)
        T = self.kinetic(q, qt)
        V = self.potential(q, qt)
        return T + V
    
    def solve_acceleration(self, q, qdot):
        if isinstance(q, torch.Tensor):
            qdtt = torch.zeros_like(q)
        elif isinstance(q, np.ndarray):
            qdtt = np.zeros_like(q)
        
        qdtt[:, 0] = -self.g/self.scale[0]
        return qdtt
    
    def plot_solved_dynamics(self, t, path, labelstr = "", **kwargs):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Constant Force Dynamics - {labelstr}', fontsize=14)
        
        # Position vs time
        axes[0,0].plot(t, path[:, 0], label=f'Position - {labelstr}', **kwargs)
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Position (m)')
        axes[0,0].set_title('Position vs Time')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # Velocity vs time
        axes[0,1].plot(t, path[:, 1], label=f'Velocity - {labelstr}', **kwargs)
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Velocity (m/s)')
        axes[0,1].set_title('Velocity vs Time')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # Phase space
        axes[1,0].plot(path[:, 0], path[:, 1], label=f'Trajectory - {labelstr}', **kwargs)
        axes[1,0].set_xlabel('Position (m)')
        axes[1,0].set_ylabel('Velocity (m/s)')
        axes[1,0].set_title('Phase Space')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        # Energy vs time
        energy = [self.energy(x) for x in path]
        axes[1,1].plot(t, energy, label=f'Total Energy - {labelstr}', **kwargs)
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Energy (J)')
        axes[1,1].set_title('Energy Conservation')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        plt.tight_layout()
        return fig
        
    def plot_lagrangian(self, t, path, lagrangian, labelstr = ""):
        plt.plot(t, [lagrangian(l) for l in path], label = labelstr)



# %%
class harmonic_oscillator_spring_1d(PhysicalSystemBase):

    def __init__(self, m=1.0, k=1.0):
        super().__init__()
        self.m = m
        self.k = k
        self.dof = 2
        self.angle_indices = []
        self.scale = [1.0]
        self.param_names = ['m', 'k']
        self.test_params = {'toy_position' : torch.tensor([0.3]),
                            'toy_velocity' : torch.tensor([1.0]),
                            'toy_time_dataset' : np.arange(0, 4, 0.001),
                            'time_step' : 0.001,
                            'time_bounds' : (0,4),
                            'position_bounds' : ((0,1),),
                            'velocity_bounds' : ((-2,2),),
                            'test_seed': 96,
                            'num_tests' : 4,
                            'validation_position_bounds' : ((-1,1),),
                            'validation_velocity_bounds' : ((-2,2),),
                            }
        self.train_hyperparams = {'lr': 1e-3,
                                  'num_samples' : 4000,
                                  'test_ratio': 0.1,
                                  'total_epochs': 120,
                                  'minibatch_size': 64,
                                  'train_seed': 86,
                                  'position_bounds': ((-1,1),),
                                  'velocity_bounds' : ((-2,2),),
                                  'eta_min': 1e-6,
                                  }
        self.model_params = {'num_layers' : 3,
                             'activation_fn' : nn.GELU(),
                             'hidden_dim' : 128,
                            }
        
        # Validate completeness
        self.validate_implementation()

    def scale_constants(self, scale):
        self.scale = scale
        
    def kinetic(self, q, q_dot):
        if len(q.shape) == 1:
            w1 = q_dot[0] * self.scale[0]
            return 0.5 * self.m * w1**2
        w1 = q_dot[:,0] * self.scale[0]
        return 0.5 * self.m * w1**2
    
    def potential(self, q, q_dot):
        if len(q.shape) == 1:
            x1 = q[0] * self.scale[0]
            return 0.5 * self.k * x1**2
        x1 = q[:,0] * self.scale[0]
        return 0.5 * self.k * x1**2
    # NEED LAGRANGIAN TO PRODUCE SCALARS, NOT TENSORS!
    def lagrangian(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        q, qt = torch.split(x, self.dof//2, dim=-1)
        T = self.kinetic(q, qt)
        V = self.potential(q, qt)
        return T - V
    
    def energy(self, x):
        if isinstance(x, torch.Tensor):
            if len(x.shape)==1:
                x = x.unsqueeze(0)
            q, qt = torch.split(x, self.dof//2, dim=-1)
        elif isinstance(x, np.ndarray): 
            if len(x.shape)==1:
                x = np.expand_dims(x, axis=0)
            q, qt = np.split(x, self.dof//2, axis=-1)
        T = self.kinetic(q, qt)
        V = self.potential(q, qt)
        return T + V
    
    def solve_acceleration(self, q, qdot):
        if isinstance(q, torch.Tensor):
            qdtt = torch.zeros_like(q)
        elif isinstance(q, np.ndarray):
            qdtt = np.zeros_like(q)
        
        # Scale the position for calculation
        x1 = q[:, 0] * self.scale[0]
        qdtt[:, 0] = (-self.k * x1 / self.m) / self.scale[0]
        return qdtt
    
    def plot_solved_dynamics(self, t, path, labelstr = "", **kwargs):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Harmonic Oscillator Dynamics - {labelstr}', fontsize=14)
        
        # Position vs time
        axes[0,0].plot(t, path[:, 0], label=f'Position - {labelstr}', **kwargs)
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Position (m)')
        axes[0,0].set_title('Position vs Time')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # Velocity vs time
        axes[0,1].plot(t, path[:, 1], label=f'Velocity - {labelstr}', **kwargs)
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Velocity (m/s)')
        axes[0,1].set_title('Velocity vs Time')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # Phase space (should be elliptical for harmonic oscillator)
        axes[1,0].plot(path[:, 0], path[:, 1], label=f'Trajectory - {labelstr}', **kwargs)
        axes[1,0].set_xlabel('Position (m)')
        axes[1,0].set_ylabel('Velocity (m/s)')
        axes[1,0].set_title('Phase Space (Should be Elliptical)')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        axes[1,0].axis('equal')
        
        # Energy vs time (should be constant)
        energy = [self.energy(x) for x in path]
        axes[1,1].plot(t, energy, label=f'Total Energy - {labelstr}', **kwargs)
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Energy (J)')
        axes[1,1].set_title('Energy Conservation')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        plt.tight_layout()
        return fig

    def plot_lagrangian(self, t, path, lagrangian, labelstr = ""):
        plt.plot(t, [lagrangian(l) for l in path], label = labelstr)


class spring_pendulum(PhysicalSystemBase):

    def __init__(self, m=1.0, k=40.0, g=9.81):
        super().__init__()
        # CORRECTED: Added mass 'm'
        self.m = m
        self.k = k
        self.g = g
        self.dof = 4
        self.angle_indices = [1]
        self.scale = [1.0, 1.0]
        # CORRECTED: Added 'm' to param_names
        self.param_names = ['m', 'k', 'g']
        self.test_params = {'toy_position' : torch.tensor([1.1, 0.5]),
                              'toy_velocity' : torch.tensor([0.0, 0.0]),
                              'toy_time_dataset' : np.arange(0, 5, 0.1),
                              'time_step' : 0.1,
                              'time_bounds' : (0,50),
                              'position_bounds': ((0.5,1.5),(0,2*np.pi)),
                              'velocity_bounds': ((-1,1), (-1,1)),
                              'test_seed': 55,
                              'num_tests' : 4,
                              'validation_position_bounds': ((0.1, 5),(0,2*np.pi)),
                              'validation_velocity_bounds': ((-5,5), (-5,5)),
                              }
        self.train_hyperparams = {'lr': 5e-4,
                                    'num_samples' : 60000,
                                    'test_ratio': 0.1,
                                    'total_epochs': 300,
                                    'minibatch_size': 128,
                                    'train_seed': 86,
                                    'position_bounds': ((0.05, 5),(-np.pi/2,2*np.pi+np.pi/2)),
                                    'velocity_bounds': ((-5,5), (-5,5)),
                                    'eta_min': 1e-7,
                                    }
        self.model_params = {'num_layers' : 4,
                               'activation_fn' : nn.GELU(),
                               'hidden_dim' : 500,
                              }

        # Validate completeness
        self.validate_implementation()

    def scale_constants(self, scale):
        self.scale = scale

    def kinetic(self, q, qt, cos=None):
        t1, t2, w1, w2 = q[:,0], q[:,1], qt[:,0], qt[:,1]
        t1, t2 = t1*self.scale[0], t2*self.scale[1]%(2*np.pi)
        w1, w2 = w1*self.scale[0], w2*self.scale[1]
        # CORRECTED: Kinetic energy is 0.5 * m * v^2
        return 0.5 * self.m * (w1**2 + (t1*w2)**2)

    def potential(self, q, qt, cos=torch.cos):
        t1, t2, w1, w2 = q[:,0], q[:,1], qt[:,0], qt[:,1]
        t1, t2 = t1*self.scale[0], t2*self.scale[1]%(2*np.pi)
        w1, w2 = w1*self.scale[0], w2*self.scale[1]
        # CORRECTED: Standard gravitational potential is -m*g*h where h = r*cos(theta)
        # The spring potential is 0.5*k*(r-l0)^2, assuming l0=1.
        return -self.m * self.g * t1 * cos(t2) + 0.5 * self.k * (t1 - 1)**2

    def lagrangian(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        q, qt = torch.split(x, self.dof//2, dim=-1)
        T = self.kinetic(q, qt, torch.cos)
        V = self.potential(q, qt, torch.cos)
        return T - V

    def energy(self, x):
        if isinstance(x, torch.Tensor):
            if len(x.shape)==1:
                x = x.unsqueeze(0)
            q, qt = torch.split(x, self.dof//2, dim=-1)
            cos = torch.cos
        elif isinstance(x, np.ndarray):
            if len(x.shape)==1:
                x = np.expand_dims(x, axis=0)
            q, qt = np.split(x, self.dof//2, axis=-1)
            cos = np.cos
        T = self.kinetic(q, qt, cos)
        V = self.potential(q, qt, cos)
        return T + V

    def solve_acceleration(self, q, qdot):
        if isinstance(q, torch.Tensor):
            qdtt = torch.zeros_like(q)
            sin = torch.sin
            cos = torch.cos
        elif isinstance(q, np.ndarray):
            qdtt = np.zeros_like(q)
            sin = np.sin
            cos = np.cos

        # Apply standard scaling pattern
        t1, t2, w1, w2 = q[:,0], q[:,1], qdot[:,0], qdot[:,1]
        t1, t2 = t1*self.scale[0], t2*self.scale[1]%(2*np.pi)
        w1, w2 = w1*self.scale[0], w2*self.scale[1]
        
        # CORRECTED: Physics equations derived from the proper Lagrangian
        # r̈ = rθ̇² + g*cos(θ) - (k/m)*(r-1)
        # θ̈ = (-g*sin(θ) - 2*ṙ*θ̇)/r

        g1 = t1*w2**2 + self.g*cos(t2) - (self.k/self.m)*(t1-1)
        g2 = (-self.g*sin(t2) - 2*w1*w2)/t1

        # Scale back the accelerations
        qdtt[:, 0] = g1/self.scale[0]
        qdtt[:, 1] = g2/self.scale[1]

        return qdtt

    def to_cartesian(self, q_qdot):
        '''
        Polar coords to xy
        '''
        if isinstance(q_qdot, np.ndarray):
            xy = np.zeros_like(q_qdot)
            sin = np.sin
            cos = np.cos
        elif isinstance(q_qdot, torch.Tensor):
            xy = torch.zeros_like(q_qdot)
            sin = torch.sin
            cos = torch.cos

        # Apply standard scaling pattern
        t1, t2 = q_qdot[:, 0], q_qdot[:, 1]
        t1, t2 = t1*self.scale[0], t2*self.scale[1]%(2*np.pi)

        xy[:, 0] = t1*sin(t2)
        xy[:, 1] = -t1*cos(t2)
        return xy
    
    def plot_solved_dynamics(self, t, path, labelstr = "", **kwargs):
        xy = self.to_cartesian(path)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Spring Pendulum Dynamics - {labelstr}', fontsize=14)
        
        # Apply scaling for display
        t1_scaled = path[:, 0] * self.scale[0]
        t2_scaled = (path[:, 1] * self.scale[1]) % (2*np.pi)
        w1_scaled = path[:, 2] * self.scale[0]  
        w2_scaled = path[:, 3] * self.scale[1]
        
        # Cartesian trajectory with spring visualization
        axes[0,0].plot(xy[:, 0], xy[:, 1], label=f'Trajectory - {labelstr}', **kwargs)
        axes[0,0].scatter(xy[0, 0], xy[0, 1], color='green', s=50, label='Start', zorder=5)
        axes[0,0].scatter(xy[-1, 0], xy[-1, 1], color='red', s=50, label='End', zorder=5)
        
        # Draw spring at start and end positions
        n_coils = 8
        for i, (pos_idx, color, alpha, label_prefix) in enumerate([(0, 'green', 0.5, 'Start'), (-1, 'red', 0.5, 'End')]):
            x_mass, y_mass = xy[pos_idx, 0], xy[pos_idx, 1]
            spring_length = np.sqrt(x_mass**2 + y_mass**2)
            
            # Create zigzag spring pattern
            t_spring = np.linspace(0, 1, n_coils * 4 + 1)
            spring_x = np.zeros_like(t_spring)
            spring_y = np.zeros_like(t_spring)
            
            for j in range(len(t_spring)):
                # Linear interpolation along the spring direction
                spring_x[j] = t_spring[j] * x_mass
                spring_y[j] = t_spring[j] * y_mass
                
                # Add zigzag perpendicular displacement
                if j > 0 and j < len(t_spring) - 1:
                    # Perpendicular direction
                    if spring_length > 0:  # Avoid division by zero
                        perp_x = -y_mass / spring_length * 0.05 * spring_length
                        perp_y = x_mass / spring_length * 0.05 * spring_length
                        
                        # Zigzag pattern
                        zigzag = (-1) ** (j // 2) * np.sin(j * np.pi / 2)
                        spring_x[j] += perp_x * zigzag
                        spring_y[j] += perp_y * zigzag
            
            axes[0,0].plot(spring_x, spring_y, color=color, alpha=alpha, linewidth=2, 
                          label=f'{label_prefix} Spring')
            
        # Add mass representation
        axes[0,0].scatter(xy[0, 0], xy[0, 1], color='green', s=100, marker='o', 
                         edgecolors='black', linewidth=2, zorder=10)
        axes[0,0].scatter(xy[-1, 0], xy[-1, 1], color='red', s=100, marker='o', 
                         edgecolors='black', linewidth=2, zorder=10)
        
        axes[0,0].set_xlabel('X Position (m)')
        axes[0,0].set_ylabel('Y Position (m)')
        axes[0,0].set_title('Cartesian Trajectory with Spring')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        axes[0,0].axis('equal')
        
        # Radial distance vs time (use scaled values for display)
        axes[0,1].plot(t, t1_scaled, label=f'Radial Distance - {labelstr}', **kwargs)
        axes[0,1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Natural Length')
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Radial Distance (m)')
        axes[0,1].set_title('Spring Length vs Time')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # Angular position vs time (use scaled values for display)
        axes[0,2].plot(t, t2_scaled, label=f'Angle - {labelstr}', **kwargs)
        axes[0,2].set_xlabel('Time (s)')
        axes[0,2].set_ylabel('Angle (rad)')
        axes[0,2].set_title('Angular Position vs Time')
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].legend()
        
        # Radial velocity vs time (use scaled values for display)
        axes[1,0].plot(t, w1_scaled, label=f'Radial Velocity - {labelstr}', **kwargs)
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Radial Velocity (m/s)')
        axes[1,0].set_title('Radial Velocity vs Time')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        # Angular velocity vs time (use scaled values for display)
        axes[1,1].plot(t, w2_scaled, label=f'Angular Velocity - {labelstr}', **kwargs)
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Angular Velocity (rad/s)')
        axes[1,1].set_title('Angular Velocity vs Time')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        # Energy vs time (should be constant)
        energy = [self.energy(x) for x in path]
        axes[1,2].plot(t, energy, label=f'Total Energy - {labelstr}', **kwargs)
        axes[1,2].set_xlabel('Time (s)')
        axes[1,2].set_ylabel('Energy (J)')
        axes[1,2].set_title('Energy Conservation')
        axes[1,2].grid(True, alpha=0.3)
        axes[1,2].legend()
        
        plt.tight_layout()
        return fig
    

    def plot_lagrangian(self, t, path, lagrangian, labelstr = ""):
        plt.plot(t, [lagrangian(l) for l in path], label = labelstr)



class double_pendulum(PhysicalSystemBase):
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.8):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.dof = 4
        self.angle_indices = [0,1]
        self.scale = [1.0, 1.0]
        self.param_names = ['m1', 'm2', 'l1', 'l2', 'g']
        self.test_params = {'toy_position' : torch.tensor([1.1, 0.5]),
                            'toy_velocity' : torch.tensor([0.0, 0.0]),
                            'toy_time_dataset' : np.arange(0, 5, 0.1),
                            'time_step' : 0.1,
                            'time_bounds' : (0,100),
                            'position_bounds': ((0,2*np.pi),(0,2*np.pi)),
                            'velocity_bounds': ((-0.1,0.1), (-0.1,0.1)),
                            'test_seed': 96,
                            'num_tests' : 4,
                            'validation_position_bounds': ((0,2*np.pi),(0,2*np.pi)),
                            'validation_velocity_bounds': ((-10,10), (-10,10)),
                            }
        self.train_hyperparams = {'lr': 1e-3,
                                  'num_samples' : 60000,
                                  'test_ratio': 0.1,
                                  'total_epochs': 300,
                                  'minibatch_size': 128,
                                  'train_seed': 86,
                                  'position_bounds': ((-np.pi/2,2*np.pi + np.pi/2), (-np.pi/2, 2*np.pi + np.pi/2)),
                                  'velocity_bounds': ((-10,10), (-10,10)),
                                  'eta_min': 1e-6,
                                  }
        
        self.model_params = {'num_layers' : 4,
                             'activation_fn' : nn.GELU(),
                             'hidden_dim' : 500,
                            }
        
        # Validate completeness
        self.validate_implementation()

    def scale_constants(self, scale):
        self.scale = scale
    
    def kinetic(self, q, q_dot, cos=torch.cos):
        t1, t2, w1, w2 = q[:,0],q[:,1], q_dot[:,0], q_dot[:,1]
        t1, t2 = t1*self.scale[0]%(2*np.pi), t2*self.scale[1]%(2*np.pi)
        w1, w2 = w1*self.scale[0], w2*self.scale[1]
        T1 = 0.5 * self.m1 * (self.l1 * w1)**2
        T2 = 0.5 * self.m2 * ((self.l1 * w1)**2 + (self.l2 * w2)**2 + 2 * self.l1 * self.l2 * w1 * w2 * cos((t1 - t2)))
        T = T1 + T2
        return T
    
    def potential(self, q, q_dot, cos=torch.cos):
        t1, t2, w1, w2 = q[:,0],q[:,1], q_dot[:,0], q_dot[:,1]
        t1, t2 = t1*self.scale[0]%(2*np.pi), t2*self.scale[1]%(2*np.pi)
        w1, w2 = w1*self.scale[0], w2*self.scale[1]
        y1 = -self.l1 * cos(t1)
        y2 = y1 - self.l2 * cos(t2)
        V = self.m1 * self.g * y1 + self.m2 * self.g * y2
        return V
    
    
    # NEED LAGRANGIAN TO PRODUCE SCALARS, NOT TENSORS!
    def lagrangian(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        q, qt = torch.split(x, self.dof//2, dim=-1)
        T = self.kinetic(q, qt, torch.cos)
        V = self.potential(q, qt, torch.cos)
        return T - V

    def energy(self, x):
        if isinstance(x, torch.Tensor):
            if len(x.shape)==1:
                x = x.unsqueeze(0)
            q, qt = torch.split(x, self.dof//2, dim=-1)
            cos = torch.cos
        elif isinstance(x, np.ndarray): 
            if len(x.shape)==1:
                x = np.expand_dims(x, axis=0)
            q, qt = np.split(x, self.dof//2, axis=-1)
            cos = np.cos
        T = self.kinetic(q, qt, cos)
        V = self.potential(q, qt, cos)
        return T + V
    
    
    def solve_acceleration(self, q, qdot):
        if isinstance(q, torch.Tensor):
            qdtt = torch.zeros_like(q)
            sin = torch.sin
            cos = torch.cos
        elif isinstance(q, np.ndarray):
            qdtt = np.zeros_like(q)
            sin = np.sin
            cos = np.cos

        # Apply scaling
        t1 = (q[:, 0] * self.scale[0]) % (2 * np.pi)
        t2 = (q[:, 1] * self.scale[1]) % (2 * np.pi)
        w1 = qdot[:, 0] * self.scale[0]
        w2 = qdot[:, 1] * self.scale[1]
        
        a1 = (self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * cos(t1 - t2)
        a2 = (self.l1 / self.l2) * cos(t1 - t2)
        f1 = -(self.l2 / self.l1) * (self.m2 / (self.m1 + self.m2)) * (w2**2) * sin(t1 - t2) - (self.g / self.l1) * sin(t1)
        f2 = (self.l1 / self.l2) * (w1**2) * sin(t1 - t2) - (self.g / self.l2) * sin(t2)
        g1 = (f1 - a1 * f2) / (1 - a1 * a2)
        g2 = (f2 - a2 * f1) / (1 - a1 * a2)
        qdtt[:, 0] = g1/self.scale[0]
        qdtt[:, 1] = g2/self.scale[1]
        return qdtt
    
    def to_cartesian(self, state):
        
        if isinstance(state, torch.Tensor):
            xy = torch.zeros_like(state)
            sin = torch.sin
            cos = torch.cos
        elif isinstance(state, np.ndarray):
            xy = np.zeros_like(state)
            sin = np.sin
            cos = np.cos
        t1, t2 = state[:,0], state[:,1]
        t1, t2 = t1*self.scale[0], t2*self.scale[1]

        x1 = self.l1 * sin(t1)
        y1 = -self.l1 * cos(t1)
        x2 = x1 + self.l2 * sin(t2)
        y2 = y1 - self.l2 * cos(t2)
        xy[:, 0] = x1
        xy[:, 1] = y1
        xy[:, 2] = x2
        xy[:, 3] = y2
        return xy
    
    
    def plot_solved_dynamics(self, t, path, labelstr = "", **kwargs):
        xy = self.to_cartesian(path)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Double Pendulum Dynamics - {labelstr}', fontsize=14)
        
        # Trajectory of both masses
        axes[0,0].plot(xy[:, 0], xy[:, 1], label=f'Mass 1 - {labelstr}', **kwargs)
        axes[0,0].plot(xy[:, 2], xy[:, 3], label=f'Mass 2 - {labelstr}', linestyle='--', **kwargs)
        # Show pendulum structure at start and end
        axes[0,0].plot([0, xy[0,0], xy[0,2]], [0, xy[0,1], xy[0,3]], 'go-', alpha=0.5, label='Start Config')
        axes[0,0].plot([0, xy[-1,0], xy[-1,2]], [0, xy[-1,1], xy[-1,3]], 'ro-', alpha=0.5, label='End Config')
        axes[0,0].set_xlabel('X Position (m)')
        axes[0,0].set_ylabel('Y Position (m)')
        axes[0,0].set_title('Cartesian Trajectories')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        axes[0,0].axis('equal')
        
        # Angular positions vs time
        axes[0,1].plot(t, path[:, 0], label=f'θ₁ - {labelstr}', **kwargs)
        axes[0,1].plot(t, path[:, 1], label=f'θ₂ - {labelstr}', linestyle='--', **kwargs)
        axes[0,1].set_xlabel('Time (s)')
        axes[0,1].set_ylabel('Angle (rad)')
        axes[0,1].set_title('Angular Positions vs Time')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        # Angular velocities vs time
        axes[0,2].plot(t, path[:, 2], label=f'ω₁ - {labelstr}', **kwargs)
        axes[0,2].plot(t, path[:, 3], label=f'ω₂ - {labelstr}', linestyle='--', **kwargs)
        axes[0,2].set_xlabel('Time (s)')
        axes[0,2].set_ylabel('Angular Velocity (rad/s)')
        axes[0,2].set_title('Angular Velocities vs Time')
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].legend()
        
        # Phase space for first pendulum
        axes[1,0].plot(path[:, 0], path[:, 2], label=f'θ₁ Phase Space - {labelstr}', **kwargs)
        axes[1,0].set_xlabel('θ₁ (rad)')
        axes[1,0].set_ylabel('ω₁ (rad/s)')
        axes[1,0].set_title('Mass 1 Phase Space')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        # Phase space for second pendulum
        axes[1,1].plot(path[:, 1], path[:, 3], label=f'θ₂ Phase Space - {labelstr}', **kwargs)
        axes[1,1].set_xlabel('θ₂ (rad)')
        axes[1,1].set_ylabel('ω₂ (rad/s)')
        axes[1,1].set_title('Mass 2 Phase Space')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        
        # Energy vs time (should be constant)
        energy = [self.energy(x) for x in path]
        axes[1,2].plot(t, energy, label=f'Total Energy - {labelstr}', **kwargs)
        axes[1,2].set_xlabel('Time (s)')
        axes[1,2].set_ylabel('Energy (J)')
        axes[1,2].set_title('Energy Conservation')
        axes[1,2].grid(True, alpha=0.3)
        axes[1,2].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_lagrangian(self, t, path, lagrangian, labelstr = ""):
        plt.plot(t, [lagrangian(l) for l in path], label = labelstr)



class triple_pendulum(PhysicalSystemBase):
    def __init__(self, m1=1.0, m2=1.0, m3=1.0, l1=1.0, l2=1.0, l3=1.0, g=9.8):
        super().__init__()
        self.m1, self.m2, self.m3 = m1, m2, m3
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.g = g

        self.dof = 6  # 3 angles, 3 angular velocities
        self.angle_indices = [0, 1, 2]
        self.scale = [1.0, 1.0, 1.0]
        self.param_names = ['m1', 'm2', 'm3', 'l1', 'l2', 'l3', 'g']

        self.test_params = {
            'toy_position': torch.tensor([1.1, 0.5, 0.2]),
            'toy_velocity': torch.tensor([0.0, 0.0, 0.0]),
            'toy_time_dataset': np.arange(0, 5, 0.1),
            'time_step': 0.1,
            'time_bounds': (0, 100),
            'position_bounds': ((0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)),
            'velocity_bounds': ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)),
            'test_seed': 96,
            'num_tests': 4,
            'validation_position_bounds': ((0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)),
            'validation_velocity_bounds': ((-10, 10), (-10, 10), (-10, 10)),
        }
        
        self.train_hyperparams = {
            'lr': 1e-3,
            'num_samples': 60000,
            'test_ratio': 0.1,
            'total_epochs': 300,
            'minibatch_size': 128,
            'train_seed': 82,
            'position_bounds': ((-np.pi/2, 2*np.pi + np.pi/2), 
                              (-np.pi/2, 2*np.pi + np.pi/2), 
                              (-np.pi/2, 2*np.pi + np.pi/2)),
            'velocity_bounds': ((-10, 10), (-10, 10), (-10, 10)),
            'eta_min': 1e-6,
        }
        
        self.model_params = {
            'num_layers': 4,
            'activation_fn': nn.GELU(),
            'hidden_dim': 500,
        }
        
        self.validate_implementation()

    def scale_constants(self, scale):
        self.scale = scale

    def kinetic(self, q, q_dot, cos=torch.cos):
        t1, t2, t3 = q[:, 0], q[:, 1], q[:, 2]
        w1, w2, w3 = q_dot[:, 0], q_dot[:, 1], q_dot[:, 2]
        
        # Apply scaling and periodic boundary
        t1 = (t1 * self.scale[0]) % (2 * np.pi)
        t2 = (t2 * self.scale[1]) % (2 * np.pi)
        t3 = (t3 * self.scale[2]) % (2 * np.pi)
        w1 = w1 * self.scale[0]
        w2 = w2 * self.scale[1]
        w3 = w3 * self.scale[2]
        
        T = (0.5 * (self.m1 + self.m2 + self.m3) * self.l1**2 * w1**2 +
             0.5 * (self.m2 + self.m3) * self.l2**2 * w2**2 +
             0.5 * self.m3 * self.l3**2 * w3**2 +
             (self.m2 + self.m3) * self.l1 * self.l2 * w1 * w2 * cos(t1 - t2) +
             self.m3 * self.l1 * self.l3 * w1 * w3 * cos(t1 - t3) +
             self.m3 * self.l2 * self.l3 * w2 * w3 * cos(t2 - t3))
        return T

    def potential(self, q, q_dot, cos=torch.cos):
        t1, t2, t3 = q[:, 0], q[:, 1], q[:, 2]
        
        # Apply scaling and periodic boundary
        t1 = (t1 * self.scale[0]) % (2 * np.pi)
        t2 = (t2 * self.scale[1]) % (2 * np.pi)
        t3 = (t3 * self.scale[2]) % (2 * np.pi)
        
        y1 = -self.l1 * cos(t1)
        y2 = y1 - self.l2 * cos(t2)
        y3 = y2 - self.l3 * cos(t3)
        
        V = self.g * (self.m1 * y1 + self.m2 * y2 + self.m3 * y3)
        return V

    def lagrangian(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        q, qt = torch.split(x, self.dof // 2, dim=-1)
        T = self.kinetic(q, qt, torch.cos)
        V = self.potential(q, qt, torch.cos)
        return T - V

    def energy(self, x):
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            q, qt = torch.split(x, self.dof // 2, dim=-1)
            cos = torch.cos
        elif isinstance(x, np.ndarray):
            if len(x.shape) == 1:
                x = np.expand_dims(x, axis=0)
            q, qt = np.split(x, 2, axis=-1)
            cos = np.cos
        
        T = self.kinetic(q, qt, cos)
        V = self.potential(q, qt, cos)
        return T + V

    def solve_acceleration(self, q, qdot):
        if isinstance(q, torch.Tensor):
            qdtt = torch.zeros_like(q)
            sin = torch.sin
            cos = torch.cos
            solve = torch.linalg.solve
        elif isinstance(q, np.ndarray):
            qdtt = np.zeros_like(q)
            sin = np.sin
            cos = np.cos
            solve = np.linalg.solve
        
        # Apply scaling
        t1 = (q[:, 0] * self.scale[0]) % (2 * np.pi)
        t2 = (q[:, 1] * self.scale[1]) % (2 * np.pi)
        t3 = (q[:, 2] * self.scale[2]) % (2 * np.pi)
        w1 = qdot[:, 0] * self.scale[0]
        w2 = qdot[:, 1] * self.scale[1]
        w3 = qdot[:, 2] * self.scale[2]
        
        batch_size = q.shape[0]
        
        # Initialize mass matrix M and force vector F
        if isinstance(q, torch.Tensor):
            M = torch.zeros(batch_size, 3, 3, device=q.device)
            F = torch.zeros(batch_size, 3, device=q.device)
        else:
            M = np.zeros((batch_size, 3, 3))
            F = np.zeros((batch_size, 3))
        
        # Mass matrix M(q) - symmetric
        M[:, 0, 0] = (self.m1 + self.m2 + self.m3) * self.l1**2
        M[:, 0, 1] = (self.m2 + self.m3) * self.l1 * self.l2 * cos(t1 - t2)
        M[:, 0, 2] = self.m3 * self.l1 * self.l3 * cos(t1 - t3)
        M[:, 1, 0] = M[:, 0, 1]  # Symmetry
        M[:, 1, 1] = (self.m2 + self.m3) * self.l2**2
        M[:, 1, 2] = self.m3 * self.l2 * self.l3 * cos(t2 - t3)
        M[:, 2, 0] = M[:, 0, 2]  # Symmetry
        M[:, 2, 1] = M[:, 1, 2]  # Symmetry
        M[:, 2, 2] = self.m3 * self.l3**2
        
        # Force vector F(q, q_dot)
        F[:, 0] = (-(self.m2 + self.m3) * self.l1 * self.l2 * w2**2 * sin(t1 - t2)
                   - self.m3 * self.l1 * self.l3 * w3**2 * sin(t1 - t3)
                   - (self.m1 + self.m2 + self.m3) * self.g * self.l1 * sin(t1))
        
        F[:, 1] = ((self.m2 + self.m3) * self.l1 * self.l2 * w1**2 * sin(t1 - t2)
                   - self.m3 * self.l2 * self.l3 * w3**2 * sin(t2 - t3)
                   - (self.m2 + self.m3) * self.g * self.l2 * sin(t2))
        
        F[:, 2] = (self.m3 * self.l1 * self.l3 * w1**2 * sin(t1 - t3)
                   + self.m3 * self.l2 * self.l3 * w2**2 * sin(t2 - t3)
                   - self.m3 * self.g * self.l3 * sin(t3))
        
        # Solve M * q_ddot = F for q_ddot
        if isinstance(q, torch.Tensor):
            q_ddot = solve(M, F.unsqueeze(-1)).squeeze(-1)
        else:
            q_ddot = solve(M, F[:, :, np.newaxis]).squeeze(-1)
        
        # Scale back the accelerations
        qdtt[:, 0] = q_ddot[:, 0] / self.scale[0]
        qdtt[:, 1] = q_ddot[:, 1] / self.scale[1]
        qdtt[:, 2] = q_ddot[:, 2] / self.scale[2]
        
        return qdtt

    def to_cartesian(self, state):
        if isinstance(state, torch.Tensor):
            xy = torch.zeros(state.shape[0], 6, device=state.device)
            sin = torch.sin
            cos = torch.cos
        else:
            xy = np.zeros((state.shape[0], 6))
            sin = np.sin
            cos = np.cos
        
        t1, t2, t3 = state[:, 0], state[:, 1], state[:, 2]
        
        # Apply scaling
        t1 = (t1 * self.scale[0]) % (2 * np.pi)
        t2 = (t2 * self.scale[1]) % (2 * np.pi)
        t3 = (t3 * self.scale[2]) % (2 * np.pi)
        
        x1 = self.l1 * sin(t1)
        y1 = -self.l1 * cos(t1)
        x2 = x1 + self.l2 * sin(t2)
        y2 = y1 - self.l2 * cos(t2)
        x3 = x2 + self.l3 * sin(t3)
        y3 = y2 - self.l3 * cos(t3)
        
        xy[:, 0] = x1
        xy[:, 1] = y1
        xy[:, 2] = x2
        xy[:, 3] = y2
        xy[:, 4] = x3
        xy[:, 5] = y3
        
        return xy

    def plot_solved_dynamics(self, t, path, labelstr = "", **kwargs):
        xy = self.to_cartesian(path)
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Triple Pendulum Dynamics - {labelstr}', fontsize=16)
        
        # Trajectory of all three masses
        axes[0, 0].plot(xy[:, 0], xy[:, 1], label=f'Mass 1 - {labelstr}', **kwargs)
        axes[0, 0].plot(xy[:, 2], xy[:, 3], label=f'Mass 2 - {labelstr}', linestyle='--', **kwargs)
        axes[0, 0].plot(xy[:, 4], xy[:, 5], label=f'Mass 3 - {labelstr}', linestyle=':', **kwargs)
        
        # Show pendulum structure at start and end
        axes[0, 0].plot([0, xy[0, 0], xy[0, 2], xy[0, 4]], 
                        [0, xy[0, 1], xy[0, 3], xy[0, 5]], 
                        'go-', alpha=0.5, linewidth=2, label='Start')
        axes[0, 0].plot([0, xy[-1, 0], xy[-1, 2], xy[-1, 4]], 
                        [0, xy[-1, 1], xy[-1, 3], xy[-1, 5]], 
                        'ro-', alpha=0.5, linewidth=2, label='End')
        axes[0, 0].set_xlabel('X Position (m)')
        axes[0, 0].set_ylabel('Y Position (m)')
        axes[0, 0].set_title('Cartesian Trajectories')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].axis('equal')
        
        # Angular positions vs time
        axes[0, 1].plot(t, path[:, 0], label=f'θ₁ - {labelstr}', **kwargs)
        axes[0, 1].plot(t, path[:, 1], label=f'θ₂ - {labelstr}', linestyle='--', **kwargs)
        axes[0, 1].plot(t, path[:, 2], label=f'θ₃ - {labelstr}', linestyle=':', **kwargs)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Angle (rad)')
        axes[0, 1].set_title('Angular Positions vs Time')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Angular velocities vs time
        axes[0, 2].plot(t, path[:, 3], label=f'ω₁ - {labelstr}', **kwargs)
        axes[0, 2].plot(t, path[:, 4], label=f'ω₂ - {labelstr}', linestyle='--', **kwargs)
        axes[0, 2].plot(t, path[:, 5], label=f'ω₃ - {labelstr}', linestyle=':', **kwargs)
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Angular Velocity (rad/s)')
        axes[0, 2].set_title('Angular Velocities vs Time')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        
        # Phase space for each pendulum
        axes[1, 0].plot(path[:, 0] % (2*np.pi), path[:, 3], 
                        label=f'θ₁ Phase - {labelstr}', **kwargs)
        axes[1, 0].set_xlabel('θ₁ (rad)')
        axes[1, 0].set_ylabel('ω₁ (rad/s)')
        axes[1, 0].set_title('Mass 1 Phase Space')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].plot(path[:, 1] % (2*np.pi), path[:, 4], 
                        label=f'θ₂ Phase - {labelstr}', **kwargs)
        axes[1, 1].set_xlabel('θ₂ (rad)')
        axes[1, 1].set_ylabel('ω₂ (rad/s)')
        axes[1, 1].set_title('Mass 2 Phase Space')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        axes[1, 2].plot(path[:, 2] % (2*np.pi), path[:, 5], 
                        label=f'θ₃ Phase - {labelstr}', **kwargs)
        axes[1, 2].set_xlabel('θ₃ (rad)')
        axes[1, 2].set_ylabel('ω₃ (rad/s)')
        axes[1, 2].set_title('Mass 3 Phase Space')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        
        # Energy vs time (should be conserved)
        energy = [self.energy(x) for x in path]
        axes[2, 0].plot(t, energy, label=f'Total Energy - {labelstr}', **kwargs)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Energy (J)')
        axes[2, 0].set_title('Energy Conservation')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].legend()
        
        # Lagrangian vs time
        lagrangian_vals = [self.lagrangian(x).item() for x in path]
        axes[2, 1].plot(t, lagrangian_vals, label=f'Lagrangian - {labelstr}', **kwargs)
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Lagrangian (J)')
        axes[2, 1].set_title('Lagrangian vs Time')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()
        
        # Angular momentum (optional but interesting)
        axes[2, 2].axis('off')  # Or implement angular momentum plot
        
        plt.tight_layout()
        return fig

    def plot_lagrangian(self, t, path, lagrangian, labelstr=""):
        plt.plot(t, [lagrangian(l) for l in path], label=labelstr)
        plt.xlabel('Time (s)')
        plt.ylabel('Lagrangian (J)')
        plt.title('Lagrangian vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)


class sphere_geodesic(PhysicalSystemBase):
    """
    Describes the motion of a free particle constrained to the surface of a
    sphere of radius R. The path it follows is a geodesic, which corresponds
    to a great circle on the sphere. The Lagrangian contains only a kinetic term.
    """
    def __init__(self, m=1.0, R=1.0):
        super().__init__()
        self.m = m
        self.R = R
        self.dof = 4  # (theta, phi, theta_dot, phi_dot)
        self.angle_indices = [0, 1]
        self.scale = [1.0, 1.0]
        self.param_names = ['m', 'R']

        self.test_params = {
            'toy_position': torch.tensor([np.pi / 3, np.pi / 6]),
            'toy_velocity': torch.tensor([0.5, 1.0]),
            'toy_time_dataset': np.arange(0, 10, 0.05),
            'time_step': 0.05,
            'time_bounds': (0, 20),
            'position_bounds': ((np.pi/4, np.pi - np.pi/4), (0, 2 * np.pi)), # Avoid poles
            'velocity_bounds': ((-5, 5), (-5, 5)),
            'test_seed': 927,
            'num_tests': 4,
            'validation_position_bounds': ((np.pi/10, np.pi - np.pi/10), (0, 2 * np.pi)), # Avoid poles
            'validation_velocity_bounds': ((-5, 5), (-5, 5)),
        }
        
        self.train_hyperparams = {
            'lr': 1e-3,
            'num_samples': 50000,
            'test_ratio': 0.1,
            'total_epochs': 50,
            'minibatch_size': 128,
            'train_seed': 847,
            'position_bounds': ((np.pi/10, np.pi - np.pi/10), (-np.pi/2, 2*np.pi + np.pi/2)), # Avoid poles
            'velocity_bounds': ((-5, 5), (-5, 5)),
            'eta_min': 1e-6,
        }
        
        self.model_params = {
            'num_layers': 4,
            'activation_fn': nn.GELU(),
            'hidden_dim': 500,
        }
        
        self.validate_implementation()

    def scale_constants(self, scale):
        self.scale = scale

    def kinetic(self, q, q_dot, sin=torch.sin):
        # T = 1/2 m R^2 (theta_dot^2 + sin^2(theta) phi_dot^2)
        theta, _ = q[:, 0], q[:, 1]
        theta_dot, phi_dot = q_dot[:, 0], q_dot[:, 1]

        # Apply scaling
        theta = theta * self.scale[0]
        # phi is not needed for the kinetic energy formula itself
        theta_dot = theta_dot * self.scale[0]
        phi_dot = phi_dot * self.scale[1]
        
        T = 0.5 * self.m * self.R**2 * (theta_dot**2 + (sin(theta)**2) * phi_dot**2)
        return T

    def potential(self, q, q_dot, sin=None):
        # For a geodesic, potential energy is zero (or constant)
        if isinstance(q, torch.Tensor):
            return torch.zeros(q.shape[0], device=q.device)
        return 0.0

    def lagrangian(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        q, qt = torch.split(x, self.dof // 2, dim=-1)
        # Potential is zero, so L = T
        return self.kinetic(q, qt, torch.sin)

    def energy(self, x):
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            q, qt = torch.split(x, self.dof // 2, dim=-1)
            sin = torch.sin
        elif isinstance(x, np.ndarray):
            if len(x.shape) == 1:
                x = np.expand_dims(x, axis=0)
            q, qt = np.split(x, 2, axis=-1)
            sin = np.sin
        # Total energy E = T + V = T + 0 = T
        return self.kinetic(q, qt, sin)

    def solve_acceleration(self, q, qdot):
        if isinstance(q, torch.Tensor):
            qdtt = torch.zeros_like(q)
            sin, cos = torch.sin, torch.cos
        elif isinstance(q, np.ndarray):
            qdtt = np.zeros_like(q)
            sin, cos = np.sin, np.cos

        # Geodesic equations for a sphere
        # theta_ddot = sin(theta) * cos(theta) * phi_dot^2
        # phi_ddot = -2 * cot(theta) * theta_dot * phi_dot
        
        theta, _ = q[:, 0], q[:, 1]
        theta_dot, phi_dot = qdot[:, 0], qdot[:, 1]

        # Apply scaling
        theta = theta * self.scale[0]
        theta_dot = theta_dot * self.scale[0]
        phi_dot = phi_dot * self.scale[1]

        # Add a small epsilon to avoid division by zero at the poles (coordinate singularity)
        eps = 1e-9
        
        # Calculate accelerations in scaled coordinates
        acc_theta = sin(theta) * cos(theta) * phi_dot**2
        acc_phi = -2 * (cos(theta) / (sin(theta) + eps)) * theta_dot * phi_dot
        
        # Scale back the accelerations
        qdtt[:, 0] = acc_theta / self.scale[0]
        qdtt[:, 1] = acc_phi / self.scale[1]
        
        return qdtt

    def to_cartesian(self, state):
        # Convert from (theta, phi) to (x, y, z)
        if isinstance(state, torch.Tensor):
            xyz = torch.zeros(state.shape[0], 3, device=state.device)
            sin, cos = torch.sin, torch.cos
        else:
            xyz = np.zeros((state.shape[0], 3))
            sin, cos = np.sin, np.cos
            
        theta, phi = state[:, 0], state[:, 1]
        
        # Apply scaling
        theta = theta * self.scale[0]
        phi = phi * self.scale[1]

        xyz[:, 0] = self.R * sin(theta) * cos(phi)  # x
        xyz[:, 1] = self.R * sin(theta) * sin(phi)  # y
        xyz[:, 2] = self.R * cos(theta)             # z
        
        return xyz

    def plot_solved_dynamics(self, t, path, labelstr="", **kwargs):
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'Geodesic on a Sphere - {labelstr}', fontsize=16)
        
        # 1. 3D Trajectory Plot
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        
        # Draw the sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = self.R * np.outer(np.cos(u), np.sin(v))
        y_sphere = self.R * np.outer(np.sin(u), np.sin(v))
        z_sphere = self.R * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x_sphere, y_sphere, z_sphere, color='b', alpha=0.1, rstride=5, cstride=5)
        
        # Plot the trajectory
        xyz_path = self.to_cartesian(path)
        ax1.plot(xyz_path[:, 0], xyz_path[:, 1], xyz_path[:, 2], label=f'Trajectory - {labelstr}', **kwargs)
        ax1.scatter(xyz_path[0, 0], xyz_path[0, 1], xyz_path[0, 2], s=50, c='green', label='Start')
        ax1.scatter(xyz_path[-1, 0], xyz_path[-1, 1], xyz_path[-1, 2], s=50, c='red', label='End')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Trajectory on Sphere')
        ax1.legend()
        
        # 2. Energy Conservation Plot
        ax2 = fig.add_subplot(2, 2, 2)
        energy = np.array([self.energy(x) for x in path])
        ax2.plot(t, energy, label=f'Total Energy - {labelstr}', **kwargs)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Energy (J)')
        ax2.set_title('Energy Conservation (Should be Constant)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Angular Positions Plot
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(t, path[:, 0], label=f'θ (polar) - {labelstr}', **kwargs)
        ax3.plot(t, path[:, 1], label=f'φ (azimuthal) - {labelstr}', linestyle='--', **kwargs)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angle (rad)')
        ax3.set_title('Angular Positions vs. Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Angular Velocities Plot
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(t, path[:, 2], label=f'θ_dot - {labelstr}', **kwargs)
        ax4.plot(t, path[:, 3], label=f'φ_dot - {labelstr}', linestyle='--', **kwargs)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Angular Velocity (rad/s)')
        ax4.set_title('Angular Velocities vs. Time')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def plot_lagrangian(self, t, path, lagrangian, labelstr=""):
        # Since L = E for this system, this plot is redundant with energy
        # but implemented for completeness.
        plt.plot(t, [lagrangian(l) for l in path], label=labelstr)
        plt.xlabel('Time (s)')
        plt.ylabel('Lagrangian (J)')
        plt.title('Lagrangian vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)


class ads2_geodesic(PhysicalSystemBase):
    """
    Describes the geodesic motion in 2D Anti-de Sitter spacetime (Poincaré patch).
    State vector is (t, z, t_dot, z_dot).
    The metric is ds^2 = (L^2/z^2) * (-dt^2 + dz^2).
    """
    def __init__(self, L=1.0):
        super().__init__()
        if L <= 0:
            raise ValueError("AdS length L must be positive.")
        self.L = L

        self.dof = 4
        self.angle_indices = []
        self.position_log_scale_indices = [1]  # z is at index 1
        self.scale = [1.0, 1.0]  # For (t, z)
        self.param_names = ['L']

        # Using the agreed-upon conservative parameters
        self.test_params = {
            'toy_position': torch.tensor([0.0, 1.0 * self.L]),
            'toy_velocity': torch.tensor([0.0, 0.0]), # t_dot will be set by enforce_constraints
            'toy_time_dataset': np.arange(0.0, 1.0, 0.01),
            'time_step': 0.01,
            'time_bounds': (0.0, 1.0),
            'position_bounds': ((0.0, 0.0), (0.5 * self.L, 3.0 * self.L)),
            'velocity_bounds': ((0.0, 0.0), (-0.25, 0.25)),
            'test_seed': 101,
            'num_tests': 4,
            'validation_position_bounds': ((-5.0, 5.0), (0.5 * self.L, 6.0 * self.L)),
            'validation_velocity_bounds': ((0.0, 0.0), (-0.6, 0.6)),
        }
        
        self.train_hyperparams = {
            'lr': 1e-3,
            'num_samples': 60000,
            'test_ratio': 0.1,
            'total_epochs': 120,
            'minibatch_size': 128,
            'train_seed': 1010,
            'position_bounds': ((-5.0, 5.0), (0.5 * self.L, 6.0 * self.L)),
            'velocity_bounds': ((0.0, 0.0), (-0.6, 0.6)),
            'eta_min': 1e-7,
        }
        
        self.model_params = {
            'num_layers': 4,
            'activation_fn': nn.GELU(),
            'hidden_dim': 500,
        }
        
        self.validate_implementation()

    def _apply_scale_pos_vel(self, x, xdot=None):
        if xdot is None:
            q, q_dot = torch.split(x, self.dof // 2, dim=-1)
        else:
            q, q_dot = x, xdot
        
        # Vectorized scaling - more efficient and avoids in-place operations
        scale_tensor = torch.tensor(self.scale, device=q.device, dtype=q.dtype) if isinstance(q, torch.Tensor) else np.array(self.scale)
        q_scaled = q * scale_tensor
        q_dot_scaled = q_dot * scale_tensor
        return q_scaled, q_dot_scaled

    def scale_constants(self, scale):
        self.scale = scale

    def lagrangian(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        q_s, qt_s = self._apply_scale_pos_vel(x)
        z = q_s[:, 1]
        t_dot, z_dot = qt_s[:, 0], qt_s[:, 1]
        
        metric_factor = (self.L / (z + 1e-9))**2
        L_val = 0.5 * metric_factor * (-t_dot**2 + z_dot**2)
        return L_val

    def kinetic(self, q, q_dot, **kwargs):
        x = torch.cat([q, q_dot], dim=-1) if isinstance(q, torch.Tensor) else np.concatenate([q, q_dot], axis=-1)
        return self.lagrangian(x)

    def potential(self, q, q_dot, **kwargs):
        return torch.zeros(q.shape[0], device=q.device) if isinstance(q, torch.Tensor) else 0.0

    def energy(self, x):
        return 2 * self.lagrangian(x)

    def enforce_constraints(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
            
        corrected = state.clone()
        q_s, qt_s = self._apply_scale_pos_vel(corrected)

        z = q_s[:, 1]
        z_dot = qt_s[:, 1]

        t_dot_sq = z_dot**2 + (z**2 / self.L**2)
        t_dot = torch.sqrt(torch.clamp(t_dot_sq, min=1e-12))
        
        corrected[:, 2] = t_dot / self.scale[0] # t_dot is at index 2 of full state
        return corrected

    def solve_acceleration(self, q, qdot):
        if isinstance(q, torch.Tensor):
            qdtt = torch.zeros_like(q)
        else:
            qdtt = np.zeros_like(q)

        q_s, qdot_s = self._apply_scale_pos_vel(q, qdot)
        z = q_s[:, 1]
        t_dot, z_dot = qdot_s[:, 0], qdot_s[:, 1]
        
        eps = 1e-9
        z_safe = z + eps
        
        t_ddot = (2.0 / z_safe) * t_dot * z_dot
        z_ddot = (1.0 / z_safe) * (t_dot**2 + z_dot**2)

        qdtt[:, 0] = t_ddot / self.scale[0]
        qdtt[:, 1] = z_ddot / self.scale[1]
        return qdtt
        
    def plot_solved_dynamics(self, t, path, labelstr="", **kwargs):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Geodesic in AdS2 Spacetime (L={self.L}) - {labelstr}', fontsize=16)

        ax1 = axes[0, 0]
        ax1.plot(path[:, 0], path[:, 1], label=f'z(t) - {labelstr}', **kwargs)
        ax1.axhline(0, color='gray', linestyle='--', label='Boundary (z=0)')
        ax1.set_title('Trajectory in (t, z) Plane')
        ax1.set_xlabel('t (Coordinate Time)')
        ax1.set_ylabel('z (Radial)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        energy_vals = np.array([self.energy(p) for p in path])
        ax2.plot(t, energy_vals, label=f'$g_{{\mu\nu}} v^\mu v^\nu$', **kwargs)
        ax2.axhline(-1, color='r', linestyle='--', label='Expected Value (-1)')
        ax2.set_title('Constraint Check')
        ax2.set_xlabel('τ (Proper Time)')
        ax2.set_ylim(-1.1, -0.9)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        ax3.plot(t, path[:, 1], label=f'z(τ)', **kwargs)
        ax3.set_title('Radial Position vs. Proper Time')
        ax3.set_xlabel('τ (Proper Time)')
        ax3.set_ylabel('z')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.plot(t, path[:, 3], label=f'z_dot(τ)', **kwargs)
        ax4.plot(t, path[:, 2], label=f't_dot(τ)', linestyle='--', **kwargs)
        ax4.set_title('Velocities vs. Proper Time')
        ax4.set_xlabel('τ (Proper Time)')
        ax4.set_ylabel('Velocity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def plot_lagrangian(self, t, path, lagrangian, labelstr=""):
        energy_vals = [self.energy(l) for l in path]
        plt.plot(t, energy_vals, label=f'2*L = g_μν*ẋ^μ*ẋ^ν (= -1) - {labelstr}')
        plt.xlabel('Proper Time τ'); plt.ylabel('Value'); plt.title('Lagrangian Check')
        plt.legend(); plt.grid(True, alpha=0.3)

class ads3_geodesic(PhysicalSystemBase):
    """
    Describes the geodesic motion of a test particle in 3D Anti-de Sitter
    spacetime, using the Poincaré patch coordinates. The state vector is
    (t, x, z, t_dot, x_dot, z_dot), where the dot represents differentiation
    with respect to the particle's proper time (tau).
    The metric is ds^2 = (L^2/z^2) * (-dt^2 + dx^2 + dz^2).
    """
    def __init__(self, L=1.0):
        super().__init__()
        if L <= 0:
            raise ValueError("AdS length L must be positive.")
        self.L = L  # AdS length scale

        self.dof = 6
        self.angle_indices = []
        self.position_log_scale_indices = [2]
        self.scale = [1.0, 1.0, 1.0]
        self.param_names = ['L']

        self.test_params = {
            'toy_position': torch.tensor([0.0, 0.0, 1.0 * self.L]),
            'toy_velocity': torch.tensor([0.0, 0.5, 0.1]),
            'toy_time_dataset': np.arange(0, 1.0, 0.01),
            'time_step': 0.001,
            'time_bounds': (0, 0.1),
            'position_bounds': ((0., 0.), (-5. * self.L, 5. * self.L), (0.5 * self.L, 5. * self.L)),
            'velocity_bounds': ((0., 0.), (-2., 2.), (-2., 2.)),
            'test_seed': 301,
            'num_tests': 4,
            'validation_position_bounds': ((-5.0, 5.0), (-5. * self.L, 5. * self.L), (0.2 * self.L, 10. * self.L)),
            'validation_velocity_bounds': ((0., 0.), (-3., 3.), (-3., 3.)),
        }
        
        self.train_hyperparams = {
            'lr': 1e-3,
            'num_samples': 75000,
            'test_ratio': 0.1,
            'total_epochs': 150,
            'minibatch_size': 128,
            'train_seed': 302,
            'position_bounds': ((-5.0, 5.0), (-5. * self.L, 5. * self.L), (0.2 * self.L, 10. * self.L)),
            'velocity_bounds': ((0., 0.), (-3., 3.), (-3., 3.)),
            'eta_min': 1e-6,
        }
        
        self.model_params = {
            'num_layers': 4,
            'activation_fn': nn.GELU(),
            'hidden_dim': 500,
        }
        
        self.validate_implementation()

    def _apply_scale_pos_vel(self, q, q_dot):
        # Vectorized scaling - more efficient and avoids in-place operations
        scale_tensor = torch.tensor(self.scale, device=q.device, dtype=q.dtype) if isinstance(q, torch.Tensor) else np.array(self.scale)
        q_scaled = q * scale_tensor
        q_dot_scaled = q_dot * scale_tensor
        return q_scaled, q_dot_scaled

    def scale_constants(self, scale):
        self.scale = scale

    def lagrangian(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        q, qt = torch.split(x, self.dof // 2, dim=-1)

        q_s, qt_s = self._apply_scale_pos_vel(q, qt)
        z = q_s[:, 2]
        t_dot, x_dot, z_dot = qt_s[:, 0], qt_s[:, 1], qt_s[:, 2]
        
        # CORRECTED: Added self.L**2
        metric_factor = (self.L / (z + 1e-9))**2
        L_val = 0.5 * metric_factor * (-t_dot**2 + x_dot**2 + z_dot**2)
        return L_val

    def kinetic(self, q, q_dot, **kwargs):
        x = torch.cat([q, q_dot], dim=-1) if isinstance(q, torch.Tensor) else np.concatenate([q, q_dot], axis=-1)
        return self.lagrangian(x)

    def potential(self, q, q_dot, **kwargs):
        return torch.zeros(q.shape[0], device=q.device) if isinstance(q, torch.Tensor) else 0.0

    def energy(self, x):
        return 2 * self.lagrangian(x)

    def enforce_constraints(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
            
        corrected = state.clone()
        q, qt = torch.split(corrected, self.dof // 2, dim=-1)
        q_s, qt_s = self._apply_scale_pos_vel(q, qt)

        z = q_s[:, 2]
        x_dot = qt_s[:, 1]
        z_dot = qt_s[:, 2]

        # CORRECTED: Added division by self.L**2
        t_dot_sq = x_dot**2 + z_dot**2 + (z**2 / self.L**2)
        
        t_dot = torch.sqrt(torch.clamp(t_dot_sq, min=1e-12))
        
        # Unscale t_dot before placing it back in the state vector
        corrected[:, 3] = t_dot / self.scale[0]
        return corrected
        
    def solve_acceleration(self, q, qdot):
        if isinstance(q, torch.Tensor):
            qdtt = torch.zeros_like(q)
        else:
            qdtt = np.zeros_like(q)

        q_s, qdot_s = self._apply_scale_pos_vel(q, qdot)
        z = q_s[:, 2]
        t_dot, x_dot, z_dot = qdot_s[:, 0], qdot_s[:, 1], qdot_s[:, 2]
        
        eps = 1e-9
        z_safe = z + eps
        
        # This part remains correct as L cancels out
        t_ddot = (2.0 / z_safe) * t_dot * z_dot
        x_ddot = (2.0 / z_safe) * x_dot * z_dot
        z_ddot = (1.0 / z_safe) * (t_dot**2 - x_dot**2 + z_dot**2)

        qdtt[:, 0] = t_ddot / self.scale[0]
        qdtt[:, 1] = x_ddot / self.scale[1]
        qdtt[:, 2] = z_ddot / self.scale[2]
        
        return qdtt
        
    # ... (plotting methods remain the same) ...
    def to_cartesian(self, path):
        # For AdS3, we plot the (x, z) projection
        return path[:, [1, 2]]

    def plot_solved_dynamics(self, t, path, labelstr="", **kwargs):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Geodesic in AdS3 Spacetime (L={self.L}) - {labelstr}', fontsize=16)

        # 1. Trajectory in (x, z) plane
        ax1 = axes[0, 0]
        xz_path = self.to_cartesian(path)
        ax1.plot(xz_path[:, 0], xz_path[:, 1], label=f'Trajectory - {labelstr}', **kwargs)
        ax1.axhline(0, color='gray', linestyle='--', label='Boundary (z=0)')
        ax1.set_title('Trajectory in (x, z) Plane')
        ax1.set_xlabel('x')
        ax1.set_ylabel('z (Radial)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Conserved Quantity Check (2*L)
        ax2 = axes[0, 1]
        energy_vals = np.array([self.energy(p) for p in path])
        ax2.plot(t, energy_vals, label=f'$g_{{\mu\nu}} v^\mu v^\nu$ - {labelstr}', **kwargs)
        ax2.axhline(-1, color='r', linestyle='--', label='Expected Value (-1)')
        ax2.set_title('Constraint Check (Should be -1)')
        ax2.set_xlabel('Proper Time τ')
        ax2.set_ylabel('Value')
        ax2.set_ylim(-1.1, -0.9)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Coordinates vs. Time
        ax3 = axes[1, 0]
        ax3.plot(t, path[:, 0], label=f't(τ) - {labelstr}', **kwargs)
        ax3.plot(t, path[:, 1], label=f'x(τ) - {labelstr}', linestyle='--', **kwargs)
        ax3.plot(t, path[:, 2], label=f'z(τ) - {labelstr}', linestyle=':', **kwargs)
        ax3.set_title('Coordinates vs. Proper Time')
        ax3.set_xlabel('Proper Time τ')
        ax3.set_ylabel('Coordinate Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Velocities vs. Time
        ax4 = axes[1, 1]
        ax4.plot(t, path[:, 3], label=f't_dot(τ) - {labelstr}', **kwargs)
        ax4.plot(t, path[:, 4], label=f'x_dot(τ) - {labelstr}', linestyle='--', **kwargs)
        ax4.plot(t, path[:, 5], label=f'z_dot(τ) - {labelstr}', linestyle=':', **kwargs)
        ax4.set_title('Velocities vs. Proper Time')
        ax4.set_xlabel('Proper Time τ')
        ax4.set_ylabel('Velocity')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def plot_lagrangian(self, t, path, lagrangian, labelstr=""):
        energy_vals = [self.energy(l) for l in path]
        plt.plot(t, energy_vals, label=f'2*L = g_μν*ẋ^μ*ẋ^ν (= -1) - {labelstr}')
        plt.xlabel('Proper Time τ')
        plt.ylabel('Value')
        plt.title('Lagrangian Check')
        plt.legend()
        plt.grid(True, alpha=0.3)

class ads4_geodesic(PhysicalSystemBase):
    """
    Describes the geodesic motion in 4D Anti-de Sitter spacetime (Poincaré patch).
    State vector is (t, x, y, z, t_dot, x_dot, y_dot, z_dot).
    The metric is ds^2 = (L^2/z^2) * (-dt^2 + dx^2 + dy^2 + dz^2).
    """
    def __init__(self, L=1.0):
        super().__init__()
        if L <= 0:
            raise ValueError("AdS length L must be positive.")
        self.L = L

        self.dof = 8
        self.angle_indices = []
        self.position_log_scale_indices = [3]  # z is at index 3
        self.scale = [1.0, 1.0, 1.0, 1.0]
        self.param_names = ['L']

        # Using the agreed-upon conservative parameters
        self.test_params = {
            'toy_position': torch.tensor([0.0, 0.0, 0.0, 1.0 * self.L]),
            'toy_velocity': torch.tensor([0.0, 0.05, 0.0, 0.0]), # t_dot set by enforce_constraints
            'toy_time_dataset': np.arange(0.0, 1.0, 0.01),
            'time_step': 0.01,
            'time_bounds': (0, 0.5),
            'position_bounds': ((0.,0.), (-1.,1.), (-1.,1.), (0.5*self.L, 3.0*self.L)),
            'velocity_bounds': ((0.,0.), (-1., 1.), (-1., 1.), (-0.25, 0.25)),
            'test_seed': 103,
            'num_tests': 4,
            'validation_position_bounds': ((-5.0, 5.0), (-5.,5.), (-5.,5.), (0.5*self.L, 6.0*self.L)),
            'validation_velocity_bounds': ((0.,0.), (-5.,5.), (-5.,5.), (-0.6, 0.6)),
        }
        
        self.train_hyperparams = {
            'lr': 1e-3,
            'num_samples': 120000,
            'test_ratio': 0.1,
            'total_epochs': 200,
            'minibatch_size': 128,
            'train_seed': 303,
            'position_bounds': ((-5.0, 5.0), (-5.,5.), (-5.,5.), (0.5*self.L, 6.0*self.L)),
            'velocity_bounds': ((0.,0.), (-5.,5.), (-5.,5.), (-0.6, 0.6)),
            'eta_min': 1e-7,
        }
        
        self.model_params = {
            'num_layers': 4,
            'activation_fn': nn.GELU(),
            'hidden_dim': 500,
        }
        
        self.validate_implementation()
    
    def _apply_scale_pos_vel(self, q, q_dot):
        # Vectorized scaling - more efficient and avoids in-place operations
        scale_tensor = torch.tensor(self.scale, device=q.device, dtype=q.dtype) if isinstance(q, torch.Tensor) else np.array(self.scale)
        q_scaled = q * scale_tensor
        q_dot_scaled = q_dot * scale_tensor
        return q_scaled, q_dot_scaled

    def scale_constants(self, scale):
        self.scale = scale

    def lagrangian(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        q, qt = torch.split(x, self.dof // 2, dim=-1)

        q_s, qt_s = self._apply_scale_pos_vel(q, qt)
        z = q_s[:, 3]
        t_dot, x_dot, y_dot, z_dot = qt_s[:, 0], qt_s[:, 1], qt_s[:, 2], qt_s[:, 3]
        
        metric_factor = (self.L / (z + 1e-9))**2
        L_val = 0.5 * metric_factor * (-t_dot**2 + x_dot**2 + y_dot**2 + z_dot**2)
        return L_val

    def kinetic(self, q, q_dot, **kwargs):
        x = torch.cat([q, q_dot], dim=-1) if isinstance(q, torch.Tensor) else np.concatenate([q, q_dot], axis=-1)
        return self.lagrangian(x)

    def potential(self, q, q_dot, **kwargs):
        return torch.zeros(q.shape[0], device=q.device) if isinstance(q, torch.Tensor) else 0.0

    def energy(self, x):
        return 2 * self.lagrangian(x)

    def enforce_constraints(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
            
        corrected = state.clone()
        q, qt = torch.split(corrected, self.dof // 2, dim=-1)
        q_s, qt_s = self._apply_scale_pos_vel(q, qt)

        z = q_s[:, 3]
        x_dot, y_dot, z_dot = qt_s[:, 1], qt_s[:, 2], qt_s[:, 3]

        t_dot_sq = x_dot**2 + y_dot**2 + z_dot**2 + (z**2 / self.L**2)
        t_dot = torch.sqrt(torch.clamp(t_dot_sq, min=1e-12))
        
        corrected[:, 4] = t_dot / self.scale[0] # t_dot is at index 4 of full state
        return corrected

    def solve_acceleration(self, q, qdot):
        if isinstance(q, torch.Tensor):
            qdtt = torch.zeros_like(q)
        else:
            qdtt = np.zeros_like(q)

        q_s, qdot_s = self._apply_scale_pos_vel(q, qdot)
        z = q_s[:, 3]
        t_dot, x_dot, y_dot, z_dot = qdot_s[:, 0], qdot_s[:, 1], qdot_s[:, 2], qdot_s[:, 3]
        
        eps = 1e-9
        z_safe = z + eps
        
        t_ddot = (2.0 / z_safe) * t_dot * z_dot
        x_ddot = (2.0 / z_safe) * x_dot * z_dot
        y_ddot = (2.0 / z_safe) * y_dot * z_dot
        z_ddot = (1.0 / z_safe) * (t_dot**2 - x_dot**2 - y_dot**2 + z_dot**2)

        qdtt[:, 0] = t_ddot / self.scale[0]
        qdtt[:, 1] = x_ddot / self.scale[1]
        qdtt[:, 2] = y_ddot / self.scale[2]
        qdtt[:, 3] = z_ddot / self.scale[3]
        return qdtt

    def to_cartesian(self, path):
        return path[:, [1, 2, 3]] # x, y, z

    def plot_solved_dynamics(self, t, path, labelstr="", **kwargs):
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f'Geodesic in AdS4 Spacetime (L={self.L}) - {labelstr}', fontsize=16)

        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        xyz_path = self.to_cartesian(path)
        ax1.plot(xyz_path[:, 0], xyz_path[:, 1], xyz_path[:, 2], label=f'Trajectory', **kwargs)
        ax1.set_title('Trajectory in (x, y, z) Space')
        ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
        ax1.legend()

        ax2 = fig.add_subplot(2, 2, 2)
        energy_vals = np.array([self.energy(p) for p in path])
        ax2.plot(t, energy_vals, label=f'$g_{{\mu\nu}} v^\mu v^\nu$', **kwargs)
        ax2.axhline(-1, color='r', linestyle='--', label='Expected Value (-1)')
        ax2.set_title('Constraint Check')
        ax2.set_xlabel('τ (Proper Time)')
        ax2.set_ylim(-1.1, -0.9); ax2.legend(); ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(t, path[:, 3], label=f'z(τ)', **kwargs)
        ax3.set_title('Radial Position vs. Proper Time')
        ax3.set_xlabel('τ (Proper Time)'); ax3.set_ylabel('z'); ax3.legend(); ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(t, path[:, 4], label=f't_dot', **kwargs)
        ax4.plot(t, path[:, 7], label=f'z_dot', linestyle='--', **kwargs)
        ax4.set_title('Key Velocities vs. Proper Time')
        ax4.set_xlabel('τ (Proper Time)'); ax4.set_ylabel('Velocity'); ax4.legend(); ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
        
    def plot_lagrangian(self, t, path, lagrangian, labelstr=""):
        energy_vals = [self.energy(l) for l in path]
        plt.plot(t, energy_vals, label=f'2*L = g_μν*ẋ^μ*ẋ^ν (= -1) - {labelstr}')
        plt.xlabel('Proper Time τ'); plt.ylabel('Value'); plt.title('Lagrangian Check')
        plt.legend(); plt.grid(True, alpha=0.3)

def initialize_particle_system(physics):
    if (physics == "constant_force"):
        particle = constant_force()
    elif (physics == "harmonic_oscillator"):
        particle = harmonic_oscillator_spring_1d()
    elif (physics == "spring_pendulum"):
        particle = spring_pendulum()
    elif (physics == "double_pendulum"):
        particle = double_pendulum()
    elif (physics == "triple_pendulum"):
        particle = triple_pendulum()
    elif (physics == "sphere_geodesic"):
        particle = sphere_geodesic()
    elif (physics == "ads2_geodesic"):
        particle = ads2_geodesic()
    elif (physics == "ads3_geodesic"):
        particle = ads3_geodesic()
    elif (physics == "ads4_geodesic"):
        particle = ads4_geodesic()
    else:
        raise ValueError(f"Unknown physics type: '{physics}'.")
    return particle