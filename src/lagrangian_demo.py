import torch
import matplotlib.pyplot as plt
import numpy as np
from analytical_tools import ode_solve_analytic, torch_solve_ode


def run_lagrangian_demo(particle):

    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def to_torch(x, dtype=torch.float32):
        if isinstance(x, torch.Tensor):
            return x.clone().detach().type(dtype)
        return torch.tensor(np.asarray(x), dtype=dtype)

    def compute_energy_array(xs):
        """Call particle.energy for each state in xs; try numpy first, fallback to torch tensor."""
        out = []
        for x in xs:
            try:
                val = particle.energy(x)
            except Exception:
                # fallback: convert to torch
                try:
                    xt = to_torch(x)
                    val = particle.energy(xt)
                except Exception as e:
                    raise RuntimeError(f"particle.energy failed for an input state: {e}")
            out.append(float(val))
        return np.array(out)

    # -------------------------
    # main
    # -------------------------
    # store and set scale constants
    scale_factor = getattr(particle, "scale", None)
    particle.scale_constants([1.0 for _ in range(particle.dof // 2)])

    # get time and initial states
    t = np.asarray(particle.test_params['toy_time_dataset'])
    q0 = particle.test_params['toy_position']
    qt0 = particle.test_params['toy_velocity']

    # Check if particle has enforce_constraints method
    has_constraints = hasattr(particle, 'enforce_constraints') and callable(getattr(particle, 'enforce_constraints'))
    
    # If constraints exist, enforce them on initial conditions
    if has_constraints:
        initial_state = torch.cat([to_torch(q0), to_torch(qt0)]).unsqueeze(0)
        initial_state = particle.enforce_constraints(initial_state)
        q0 = initial_state[0, :particle.dof//2]
        qt0 = initial_state[0, particle.dof//2:]

    # analytic solve (may return numpy array or list or torch)
    path = ode_solve_analytic(q0, qt0, t, particle.solve_acceleration)
    path_arr = to_numpy(path)  # shape should be (len(t), dof)

    # prepare torch initial state and time for your torch ODE solver
    tx0 = torch.cat([to_torch(q0), to_torch(qt0)])
    tt = to_torch(t)

    # Lagrangian-based trajectory (torch)
    tpath = torch_solve_ode(tx0, tt, lambda x: particle.lagrangian(x).squeeze())
    tpath_arr = to_numpy(tpath)

    # ensure 2D (time, dof)
    if tpath_arr.ndim == 1:
        tpath_arr = tpath_arr.reshape(len(t), -1)
    if path_arr.ndim == 1:
        path_arr = path_arr.reshape(len(t), -1)

    # -------------------------
    # individual trajectory plots (they create their own figures)
    # -------------------------
    particle.plot_solved_dynamics(t, path_arr, "Analytical solution", color='orange').show()

    particle.plot_solved_dynamics(t, tpath_arr, "Lagrangian-based solution", color='purple').show()

    # -------------------------
    # comparison figure: 1x2 subplots
    # left: coordinate-by-coordinate overlays (each coordinate's analytic vs Lagrangian)
    # right: energy comparison
    # -------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Position comparison (coordinate-by-coordinate)
    dof_half = particle.dof // 2
    pos_analytical = path_arr[:, :dof_half]
    pos_lagrangian = tpath_arr[:, :dof_half]

    for i in range(dof_half):
        ax1.plot(t, pos_analytical[:, i], label=f'Analytical q{i+1}', linewidth=2.5, alpha=0.85)
        ax1.plot(t, pos_lagrangian[:, i], linestyle='--', label=f'Lagrangian q{i+1}', linewidth=2.5, alpha=0.85)

    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Position', fontsize=14)
    ax1.set_title('Position Overlay (Direct Comparison)', fontsize=16, pad=12)
    ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax1.grid(True, alpha=0.25)

    # Energy conservation plot (robust evaluation)
    En = compute_energy_array(path_arr)
    Ent = compute_energy_array(tpath_arr)

    ax2.plot(t[:len(En)], En, linewidth=2.5, label='Analytical Solution')
    ax2.plot(t[:len(Ent)], Ent, linewidth=2.5, linestyle='--', label='Lagrangian-based Solution')
    ax2.set_xlabel('Time', fontsize=14)
    ax2.set_ylabel('Energy', fontsize=14)
    ax2.set_title('Energy Conservation', fontsize=16, pad=12)
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax2.grid(True, alpha=0.25)

    # global title and layout (reserve space for suptitle)
    fig.suptitle('Lagrangian Mechanics Demonstration\nComparison of Solutions', fontsize=18, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    # -------------------------
    # error analysis plot (MSE over time)
    # -------------------------
    # robust MSE: handle potential shape mismatches by trimming to the shortest time axis
    min_len = min(path_arr.shape[0], tpath_arr.shape[0])
    mse = np.sum((tpath_arr[:min_len] - path_arr[:min_len])**2, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(t[:min_len], mse, linewidth=2)
    plt.fill_between(t[:min_len], mse, alpha=0.3)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Mean Squared Error', fontsize=10)
    plt.title('Solution Difference Between Methods\n(Demonstration)', fontsize=12, pad=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # restore scale constants
    if scale_factor is not None:
        particle.scale_constants(scale_factor)
