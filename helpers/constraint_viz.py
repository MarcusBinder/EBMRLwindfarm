"""
Visualization utilities for constraint composition.

All functions return matplotlib figures, compatible with both
direct saving and wandb/tensorboard logging via writer.add_figure().
"""

from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def plot_energy_landscape(
    actor: torch.nn.Module,
    obs: torch.Tensor,
    positions: torch.Tensor,
    mask: Optional[torch.Tensor],
    surrogate: torch.nn.Module,
    lam: float,
    recep_profile: Optional[torch.Tensor] = None,
    influence_profile: Optional[torch.Tensor] = None,
    grid_res: int = 80,
    yaw_max_deg: float = 30.0,
) -> Optional[plt.Figure]:
    """
    2-panel heatmap: actor energy (left) and composed energy (right).

    Sweeps T0 × T1 yaw over a grid with T2 fixed at 0.
    Only works for EBT actors (requires compute_energy method).

    Returns None for non-EBT actors.
    """
    if not hasattr(actor, "compute_energy"):
        return None

    device = next(actor.parameters()).device
    actor.eval()

    with torch.no_grad():
        turbine_emb = actor.encode(obs, positions, mask, recep_profile, influence_profile)

    n_turb = turbine_emb.shape[1]

    # Grid over T0 × T1 in normalized [-1, 1] space
    t_vals = torch.linspace(-1, 1, grid_res, device=device)
    T0, T1 = torch.meshgrid(t_vals, t_vals, indexing="ij")
    n_pts = grid_res * grid_res

    actions = torch.zeros(n_pts, n_turb, 1, device=device)
    actions[:, 0, 0] = T0.flatten()
    actions[:, 1, 0] = T1.flatten()
    # T2 (and any others) stay at 0

    emb_exp = turbine_emb.expand(n_pts, -1, -1)

    # Compute energies in chunks to avoid OOM
    chunk = 2048
    E_actor_list, E_cons_list = [], []
    with torch.no_grad():
        for i in range(0, n_pts, chunk):
            a_chunk = actions[i : i + chunk]
            e_chunk = emb_exp[i : i + chunk]
            E_actor_list.append(actor.compute_energy(e_chunk, a_chunk).cpu())
            E_cons_list.append(surrogate(a_chunk, None).cpu())

    E_actor = torch.cat(E_actor_list).squeeze(-1).numpy().reshape(grid_res, grid_res)
    E_cons = torch.cat(E_cons_list).squeeze(-1).numpy().reshape(grid_res, grid_res)
    E_composed = E_actor + lam * E_cons

    # Convert axis to degrees
    deg_vals = t_vals.cpu().numpy() * yaw_max_deg
    extent = [deg_vals[0], deg_vals[-1], deg_vals[0], deg_vals[-1]]

    # Known optima for multi_modal (degrees)
    unconstrained = (-16.0, -17.3)
    constrained = (22.7, -9.3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Clip extremes for better color range
    vmin_a, vmax_a = np.percentile(E_actor, [2, 98])
    im1 = ax1.imshow(
        E_actor.T, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=vmin_a, vmax=vmax_a,
    )
    ax1.plot(*unconstrained, "w*", markersize=15, markeredgecolor="k", label="Unconstrained opt")
    ax1.set_title("Actor Energy $E_{\\mathrm{actor}}(T_0, T_1)$")
    ax1.set_xlabel("T0 yaw (deg)")
    ax1.set_ylabel("T1 yaw (deg)")
    ax1.legend(loc="upper right")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    vmin_c, vmax_c = np.percentile(E_composed, [2, 98])
    im2 = ax2.imshow(
        E_composed.T, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=vmin_c, vmax=vmax_c,
    )
    ax2.plot(*unconstrained, "w*", markersize=15, markeredgecolor="k", label="Unconstrained opt")
    ax2.plot(*constrained, "r*", markersize=15, markeredgecolor="k", label="Constrained opt")
    ax2.set_title(f"Composed Energy $E_{{\\mathrm{{actor}}}} + {lam} \\cdot E_{{\\mathrm{{constraint}}}}$")
    ax2.set_xlabel("T0 yaw (deg)")
    ax2.set_ylabel("T1 yaw (deg)")
    ax2.legend(loc="upper right")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Energy Landscape (T2 = 0)", fontsize=14)
    fig.tight_layout()
    return fig


def plot_optimization_trajectories(
    actor: torch.nn.Module,
    obs: torch.Tensor,
    positions: torch.Tensor,
    mask: Optional[torch.Tensor],
    surrogate: torch.nn.Module,
    lam: float,
    num_candidates: int = 8,
    num_steps: int = 20,
    lr: float = 0.1,
    recep_profile: Optional[torch.Tensor] = None,
    influence_profile: Optional[torch.Tensor] = None,
    grid_res: int = 80,
    yaw_max_deg: float = 30.0,
) -> Optional[plt.Figure]:
    """
    Composed energy landscape with gradient descent trajectories overlaid.
    EBT only.
    """
    if not hasattr(actor, "compute_energy"):
        return None

    device = next(actor.parameters()).device
    actor.eval()

    with torch.no_grad():
        turbine_emb = actor.encode(obs, positions, mask, recep_profile, influence_profile)

    n_turb = turbine_emb.shape[1]

    # --- Compute background heatmap ---
    t_vals = torch.linspace(-1, 1, grid_res, device=device)
    T0, T1 = torch.meshgrid(t_vals, t_vals, indexing="ij")
    n_pts = grid_res * grid_res

    actions_grid = torch.zeros(n_pts, n_turb, 1, device=device)
    actions_grid[:, 0, 0] = T0.flatten()
    actions_grid[:, 1, 0] = T1.flatten()
    emb_exp = turbine_emb.expand(n_pts, -1, -1)

    chunk = 2048
    E_list = []
    with torch.no_grad():
        for i in range(0, n_pts, chunk):
            a_c = actions_grid[i : i + chunk]
            e_c = emb_exp[i : i + chunk]
            ea = actor.compute_energy(e_c, a_c)
            ec = surrogate(a_c, None)
            E_list.append((ea + lam * ec).cpu())
    E_composed = torch.cat(E_list).squeeze(-1).numpy().reshape(grid_res, grid_res)

    # --- Run optimization with trajectory recording ---
    extra_energy_fns = [(surrogate.per_turbine_energy, lam)] if lam > 0 else []
    emb_cand = turbine_emb.expand(num_candidates, -1, -1)

    actions = torch.randn(num_candidates, n_turb, 1, device=device).clamp(-1, 1)
    trajectory = [actions.detach().cpu().clone()]

    with torch.enable_grad():
        for _ in range(num_steps):
            actions = actions.detach().requires_grad_(True)
            energy = actor._compose_per_turbine_energy(emb_cand, actions, None, extra_energy_fns)
            grad = torch.autograd.grad(energy.sum(), actions)[0]
            actions = (actions - lr * grad).clamp(-1, 1)
            trajectory.append(actions.detach().cpu().clone())

    # --- Plot ---
    deg_vals = t_vals.cpu().numpy() * yaw_max_deg
    extent = [deg_vals[0], deg_vals[-1], deg_vals[0], deg_vals[-1]]

    fig, ax = plt.subplots(figsize=(8, 7))
    vmin, vmax = np.percentile(E_composed, [2, 98])
    ax.imshow(
        E_composed.T, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.8,
    )

    # Plot trajectories
    colors = plt.cm.Set1(np.linspace(0, 1, num_candidates))
    for c in range(num_candidates):
        t0_path = [trajectory[s][c, 0, 0].item() * yaw_max_deg for s in range(len(trajectory))]
        t1_path = [trajectory[s][c, 1, 0].item() * yaw_max_deg for s in range(len(trajectory))]
        ax.plot(t0_path, t1_path, "-o", color=colors[c], markersize=3, linewidth=1.5, alpha=0.8)
        ax.plot(t0_path[0], t1_path[0], "s", color=colors[c], markersize=6)  # start
        ax.plot(t0_path[-1], t1_path[-1], "D", color=colors[c], markersize=6)  # end

    ax.plot(22.7, -9.3, "r*", markersize=15, markeredgecolor="k", label="Constrained opt")
    ax.plot(-16.0, -17.3, "w*", markersize=15, markeredgecolor="k", label="Unconstrained opt")
    ax.set_xlabel("T0 yaw (deg)")
    ax.set_ylabel("T1 yaw (deg)")
    ax.set_title(f"Optimization Trajectories ($\\lambda={lam}$, {num_candidates} candidates, {num_steps} steps)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_yaw_vs_lambda(
    agent: Any,
    envs: Any,
    surrogate: torch.nn.Module,
    lambdas: List[float],
    eval_steps: int,
    device: torch.device,
) -> plt.Figure:
    """
    Per-turbine mean yaw angle as a function of guidance scale lambda.
    Works for both EBT and diffusion actors.
    """
    results = _sweep_lambdas(agent, envs, surrogate, lambdas, eval_steps, device)

    fig, ax = plt.subplots(figsize=(8, 5))
    n_turb = results[0]["yaw_per_turb"].shape[0] if results[0]["yaw_per_turb"] is not None else 0

    lam_arr = np.array(lambdas)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_turb, 1)))
    for t in range(n_turb):
        yaws = [r["yaw_per_turb"][t] for r in results]
        ax.plot(lam_arr, yaws, "-o", color=colors[t], label=f"T{t}", linewidth=2)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Guidance scale $\\lambda$")
    ax.set_ylabel("Mean yaw angle (deg)")
    ax.set_title("Per-Turbine Yaw vs Constraint Strength")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_power_vs_lambda(
    agent: Any,
    envs: Any,
    surrogate: torch.nn.Module,
    lambdas: List[float],
    eval_steps: int,
    device: torch.device,
) -> plt.Figure:
    """
    Mean power output as a function of guidance scale lambda.
    Works for both EBT and diffusion actors.
    """
    results = _sweep_lambdas(agent, envs, surrogate, lambdas, eval_steps, device)

    fig, ax = plt.subplots(figsize=(8, 5))
    powers = [r["power"] for r in results]
    ax.plot(lambdas, powers, "-o", color="C0", linewidth=2)
    ax.set_xlabel("Guidance scale $\\lambda$")
    ax.set_ylabel("Mean power (W)")
    ax.set_title("Power vs Constraint Strength")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sweep_lambdas(
    agent: Any,
    envs: Any,
    surrogate: torch.nn.Module,
    lambdas: List[float],
    eval_steps: int,
    device: torch.device,
) -> List[dict]:
    """Run agent at each lambda value, return per-turbine yaw + power."""
    results = []
    for lam in lambdas:
        gfn = surrogate if lam > 0 else None
        obs, _ = envs.reset()
        yaw_per_turb_all: List[np.ndarray] = []
        powers: List[float] = []

        for _ in range(eval_steps):
            with torch.no_grad():
                act = agent.act(envs, obs, guidance_fn=gfn, guidance_scale=lam)
            obs, rew, _, _, info = envs.step(act)

            if "yaw angles agent" in info:
                yaw = np.array(info["yaw angles agent"])
                yaw_flat = yaw[0] if yaw.ndim > 1 else yaw
                yaw_per_turb_all.append(yaw_flat)
            if "Power agent" in info:
                powers.append(float(np.mean(info["Power agent"])))

        yaw_arr = np.array(yaw_per_turb_all) if yaw_per_turb_all else None
        results.append({
            "yaw_per_turb": yaw_arr.mean(axis=0) if yaw_arr is not None else None,
            "power": np.mean(powers) if powers else 0.0,
        })
    return results
