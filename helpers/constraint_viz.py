"""
Visualization utilities for constraint composition.

All functions return matplotlib figures, compatible with both
direct saving and wandb/tensorboard logging via writer.add_figure().
"""

from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def plot_yaw_trajectory(
    agent: Any,
    envs: Any,
    surrogate: torch.nn.Module,
    lambdas: List[float],
    num_steps: int,
    device: torch.device,
) -> plt.Figure:
    """
    Per-turbine yaw angles over timesteps for each lambda.

    Shows the agent walking from one optimum to another through
    small delta steps when the constraint is applied.
    Works for both EBT and diffusion actors.
    """
    n_lambdas = len(lambdas)
    fig, axes = plt.subplots(n_lambdas, 1, figsize=(10, 2.5 * n_lambdas), sharex=True)
    if n_lambdas == 1:
        axes = [axes]

    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    for i, lam in enumerate(lambdas):
        gfn = surrogate if lam > 0 else None
        obs, _ = envs.reset()
        yaw_history = []

        with torch.no_grad():
            for _ in range(num_steps):
                act = agent.act(envs, obs, guidance_fn=gfn, guidance_scale=lam)
                obs, _, _, _, info = envs.step(act)
                if "yaw angles agent" in info:
                    yaw = np.array(info["yaw angles agent"])
                    yaw_history.append(yaw[0] if yaw.ndim > 1 else yaw)

        if not yaw_history:
            continue

        yaw_arr = np.array(yaw_history)  # (num_steps, n_turb)
        ax = axes[i]
        for t in range(yaw_arr.shape[1]):
            ax.plot(yaw_arr[:, t], color=colors[t % len(colors)],
                    label=f"T{t}", linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.set_ylabel("Yaw (deg)")
        ax.set_title(f"$\\lambda = {lam}$", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Yaw Trajectory Under Constraint", fontsize=13)
    fig.tight_layout()
    return fig


def plot_local_energy_landscape(
    actor: torch.nn.Module,
    obs: torch.Tensor,
    positions: torch.Tensor,
    mask: Optional[torch.Tensor],
    surrogate: torch.nn.Module,
    lam: float,
    current_action: torch.Tensor,
    recep_profile: Optional[torch.Tensor] = None,
    influence_profile: Optional[torch.Tensor] = None,
    delta_range_deg: float = 5.0,
    grid_res: int = 60,
    yaw_max_deg: float = 30.0,
) -> Optional[plt.Figure]:
    """
    Local energy heatmap in delta-action space around the current state.

    Sweeps delta-T0 × delta-T1 over a small range around current_action,
    showing how the constraint deflects the local gradient.
    EBT only (requires compute_energy method).

    Args:
        current_action: (1, n_turb, 1) current action in normalized [-1, 1] space
        delta_range_deg: range to sweep around current state in degrees
    """
    if not hasattr(actor, "compute_energy"):
        return None

    device = next(actor.parameters()).device
    actor.eval()

    with torch.no_grad():
        turbine_emb = actor.encode(obs, positions, mask, recep_profile, influence_profile)

    n_turb = turbine_emb.shape[1]

    # Get action scaling from actor buffers
    action_scale = actor.action_scale.item()
    action_bias = actor.action_bias_val.item()

    # Current action center (normalized)
    center = current_action.squeeze(0)  # (n_turb, 1)
    delta_range_norm = delta_range_deg / yaw_max_deg

    # Grid over delta-T0 × delta-T1
    d_vals = torch.linspace(-delta_range_norm, delta_range_norm, grid_res, device=device)
    D0, D1 = torch.meshgrid(d_vals, d_vals, indexing="ij")
    n_pts = grid_res * grid_res

    # Actions = center + delta (clamped to [-1, 1])
    actions = center.unsqueeze(0).expand(n_pts, -1, -1).clone()
    actions[:, 0, 0] = (center[0, 0] + D0.flatten()).clamp(-1, 1)
    actions[:, 1, 0] = (center[1, 0] + D1.flatten()).clamp(-1, 1)

    emb_exp = turbine_emb.expand(n_pts, -1, -1)

    # Compute energies in chunks
    chunk = 2048
    E_actor_list, E_cons_list = [], []
    with torch.no_grad():
        for i in range(0, n_pts, chunk):
            a_chunk = actions[i : i + chunk]
            e_chunk = emb_exp[i : i + chunk]
            E_actor_list.append(actor.compute_energy(e_chunk, a_chunk).cpu())
            # Scale actions before surrogate (matches _compose_per_turbine_energy)
            a_scaled = a_chunk * action_scale + action_bias
            E_cons_list.append(surrogate(a_scaled, None).cpu())

    E_actor = torch.cat(E_actor_list).squeeze(-1).numpy().reshape(grid_res, grid_res)
    E_cons = torch.cat(E_cons_list).squeeze(-1).numpy().reshape(grid_res, grid_res)
    E_composed = E_actor + lam * E_cons

    # Axes in degrees (delta from center)
    deg_delta = d_vals.cpu().numpy() * yaw_max_deg
    center_deg = center.cpu().numpy().flatten() * yaw_max_deg
    # Absolute yaw on axes
    t0_deg = center_deg[0] + deg_delta
    t1_deg = center_deg[1] + deg_delta
    extent = [t0_deg[0], t0_deg[-1], t1_deg[0], t1_deg[-1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Actor energy (local)
    vmin_a, vmax_a = np.percentile(E_actor, [2, 98])
    im1 = ax1.imshow(
        E_actor.T, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=vmin_a, vmax=vmax_a,
    )
    ax1.plot(center_deg[0], center_deg[1], "w+", markersize=12, markeredgewidth=2,
             label=f"Current ({center_deg[0]:.1f}, {center_deg[1]:.1f})")
    # Find and mark actor minimum
    min_idx = np.unravel_index(E_actor.argmin(), E_actor.shape)
    ax1.plot(t0_deg[min_idx[0]], t1_deg[min_idx[1]], "r*", markersize=12,
             markeredgecolor="k", label="Local min")
    ax1.set_title("Actor Energy (local)")
    ax1.set_xlabel("T0 yaw (deg)")
    ax1.set_ylabel("T1 yaw (deg)")
    ax1.legend(fontsize=8)
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    # Panel 2: Composed energy (local)
    vmin_c, vmax_c = np.percentile(E_composed, [2, 98])
    im2 = ax2.imshow(
        E_composed.T, origin="lower", extent=extent, aspect="auto",
        cmap="viridis", vmin=vmin_c, vmax=vmax_c,
    )
    ax2.plot(center_deg[0], center_deg[1], "w+", markersize=12, markeredgewidth=2,
             label=f"Current ({center_deg[0]:.1f}, {center_deg[1]:.1f})")
    # Find and mark composed minimum
    min_idx_c = np.unravel_index(E_composed.argmin(), E_composed.shape)
    ax2.plot(t0_deg[min_idx_c[0]], t1_deg[min_idx_c[1]], "r*", markersize=12,
             markeredgecolor="k", label="Local min")
    ax2.set_title(f"Composed Energy ($\\lambda={lam}$)")
    ax2.set_xlabel("T0 yaw (deg)")
    ax2.set_ylabel("T1 yaw (deg)")
    ax2.legend(fontsize=8)
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle(f"Local Energy Landscape ($\\pm{delta_range_deg}°$ from current state)", fontsize=13)
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
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for t in range(n_turb):
        yaws = [r["yaw_per_turb"][t] for r in results]
        ax.plot(lam_arr, yaws, "-o", color=colors[t % len(colors)], label=f"T{t}", linewidth=2)

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
