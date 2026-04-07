#!/usr/bin/env python3
"""
Find wind farm layouts where constraining one turbine's yaw range
causes qualitatively different optimal strategies for the WHOLE farm.

Uses PyWake with the EXACT same configuration as WindGym:
  - Blondel_Cathelain_2020 (Gaussian wake deficit)
  - JimenezWakeDeflection
  - CrespoHernandez turbulence model
  - DTU10MW turbine (D=178.3m)
  - WS=10 m/s, TI=0.07

Usage:
    python scripts/find_multimodal_layout.py
    python scripts/find_multimodal_layout.py --constraint-deg 10 --top-k 20
    python scripts/find_multimodal_layout.py --save-plots results/
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import differential_evolution

from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.site import UniformSite
from py_wake.turbulence_models import CrespoHernandez

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Constants — matching WindGym pywake backend exactly ─────────────
D = 178.3   # DTU10MW rotor diameter (m)
WS = 10.0   # freestream wind speed (m/s)  — WindGym default
TI = 0.07   # turbulence intensity          — WindGym default
YAW_MAX = 30.0  # maximum yaw angle (degrees)


# ── Layout definitions (3–5 turbines, asymmetric) ──────────────────
def get_layouts() -> Dict[str, np.ndarray]:
    """Layouts with lateral offsets designed to create sign-asymmetric
    wake deflection effects.

    Key insight: downstream turbines placed ~0.3-0.5D off the wake
    centre line experience huge power differences from +yaw vs -yaw
    on upstream turbines (Jimenez deflection is directional).
    """
    return {
        # ── 3-turbine layouts ──
        # Row with middle turbine offset — creates partial wake
        "offset_mid_3": np.array([
            [0, 0], [4 * D, 0.5 * D], [8 * D, 0],
        ]),
        # Wider offset variant
        "offset_mid_3w": np.array([
            [0, 0], [4 * D, 0.8 * D], [8 * D, 0],
        ]),
        # Tight triangle — close spacing amplifies deflection effects
        "tight_triangle": np.array([
            [0, 0], [3 * D, 0.5 * D], [1.5 * D, 2 * D],
        ]),
        # Asymmetric V — one arm closer than the other
        "asym_V": np.array([
            [0, 0], [3 * D, 2 * D], [4 * D, -1.5 * D],
        ]),
        # Stagger with close spacing
        "stagger_close": np.array([
            [0, 0], [3 * D, 0.3 * D], [6 * D, -0.3 * D],
        ]),

        # ── 4-turbine layouts ──
        # Diamond with offset — key: two upstream turbines both
        # partially waking a downstream turbine from different sides
        "diamond_4": np.array([
            [0, 0], [3 * D, 1.5 * D], [3 * D, -1.5 * D], [6 * D, 0],
        ]),
        # Staggered 2x2
        "stag_2x2": np.array([
            [0, 0], [4 * D, 0],
            [2 * D, 0.5 * D], [6 * D, 0.5 * D],
        ]),
        # Irregular 4 — random-ish positions
        "irreg_4": np.array([
            [0, 0.3 * D], [3.5 * D, -0.2 * D],
            [2 * D, 2.5 * D], [6 * D, 1 * D],
        ]),
        # Kite shape
        "kite_4": np.array([
            [0, 0], [2 * D, 1.5 * D], [2 * D, -1.5 * D], [5 * D, 0],
        ]),

        # ── 5-turbine layouts ──
        # Two front rows + 1 back, staggered
        "stag_5": np.array([
            [0, 0], [0, 3 * D],
            [3 * D, 0.5 * D], [3 * D, 2.5 * D],
            [6 * D, 1.5 * D],
        ]),
        # Irregular pentagon
        "irreg_5": np.array([
            [0, 0], [2 * D, 2 * D],
            [4 * D, 0.5 * D], [5 * D, 3 * D],
            [8 * D, 1.5 * D],
        ]),
        # Arrow formation (2 front + 2 mid + 1 back)
        "arrow_5": np.array([
            [0, 1 * D], [0, -1 * D],
            [3 * D, 0.5 * D], [3 * D, -0.5 * D],
            [6 * D, 0],
        ]),
    }


# ── Result container ────────────────────────────────────────────────
@dataclass
class SearchResult:
    layout_name: str
    wd: float
    constrained_turbine: int
    unconstrained_yaws_deg: np.ndarray
    constrained_yaws_deg: np.ndarray
    unconstrained_power: float
    constrained_power: float
    strategy_shift_deg: float
    power_cost_pct: float
    has_sign_flip: bool
    n_turbines: int = 0


# ── PyWake setup (matching WindGym exactly) ─────────────────────────
def make_farm(positions: np.ndarray):
    """Create a PyWake farm model with WindGym-identical settings."""
    wt = DTU10MW()
    site = UniformSite()
    site.initial_position = positions
    return Blondel_Cathelain_2020(
        site, windTurbines=wt,
        turbulenceModel=CrespoHernandez(),
        deflectionModel=JimenezWakeDeflection(),
    )


def eval_power(farm, x, y, wd, yaw_deg):
    """Evaluate total farm power for a single yaw config and wind direction."""
    res = farm(x=x, y=y, wd=[wd], ws=[WS], TI=TI, yaw=yaw_deg, tilt=0)
    return float(res.Power.values.sum())


def eval_power_per_turbine(farm, x, y, wd, yaw_deg):
    """Evaluate per-turbine power."""
    res = farm(x=x, y=y, wd=[wd], ws=[WS], TI=TI, yaw=yaw_deg, tilt=0)
    return res.Power.values.squeeze()


# ── Grid search (for 3–4 turbines) ─────────────────────────────────
def build_yaw_grid(n_turbines: int, yaw_max: float, yaw_step: float) -> np.ndarray:
    """All yaw angle combinations. Returns (C, N) array in degrees."""
    yaw_values = np.arange(-yaw_max, yaw_max + yaw_step / 2, yaw_step)
    grids = np.meshgrid(*[yaw_values] * n_turbines, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=-1)


def search_layout_grid(
    layout_name: str,
    positions: np.ndarray,
    wd_array: np.ndarray,
    yaw_configs_deg: np.ndarray,
    constraint_deg: float,
) -> List[SearchResult]:
    """Exhaustive grid search for one layout across all wind directions."""
    n_turbines = positions.shape[0]
    n_configs = yaw_configs_deg.shape[0]
    n_wd = len(wd_array)
    farm = make_farm(positions)
    x, y = positions[:, 0], positions[:, 1]

    # Each call evaluates all wind dirs at once
    power_matrix = np.zeros((n_configs, n_wd))
    for ci in range(n_configs):
        yaw = yaw_configs_deg[ci].tolist()
        res = farm(x=x, y=y, wd=wd_array, ws=[WS], TI=TI, yaw=yaw, tilt=0)
        power_matrix[ci, :] = res.Power.values.sum(axis=0).squeeze()

    # Precompute constraint masks
    constraint_masks = {
        t: np.abs(yaw_configs_deg[:, t]) <= constraint_deg + 0.01
        for t in range(n_turbines)
    }

    return _analyse_power_matrix(
        layout_name, n_turbines, power_matrix, yaw_configs_deg,
        constraint_masks, wd_array, constraint_deg,
    )


# ── Optimization-based search (for 5+ turbines) ────────────────────
def search_layout_optim(
    layout_name: str,
    positions: np.ndarray,
    wd_array: np.ndarray,
    constraint_deg: float,
    n_restarts: int = 8,
) -> List[SearchResult]:
    """Use scipy optimization for layouts too large for grid search."""
    n_turbines = positions.shape[0]
    farm = make_farm(positions)
    x, y = positions[:, 0], positions[:, 1]

    results: List[SearchResult] = []

    for wd in wd_array:
        # Objective: MINIMIZE negative power
        def neg_power(yaw_deg_flat):
            return -eval_power(farm, x, y, wd, yaw_deg_flat.tolist())

        # Unconstrained optimization
        bounds_free = [(-YAW_MAX, YAW_MAX)] * n_turbines
        best_power = -np.inf
        best_yaws = np.zeros(n_turbines)
        for _ in range(n_restarts):
            x0 = np.random.uniform(-YAW_MAX, YAW_MAX, n_turbines)
            try:
                res = differential_evolution(
                    neg_power, bounds_free, seed=None,
                    maxiter=50, tol=0.01, polish=True,
                    init="latinhypercube", popsize=10,
                )
                if -res.fun > best_power:
                    best_power = -res.fun
                    best_yaws = res.x.copy()
            except Exception:
                pass

        # For each turbine, find constrained optimum
        for t in range(n_turbines):
            bounds_constrained = list(bounds_free)
            bounds_constrained[t] = (-constraint_deg, constraint_deg)

            c_best_power = -np.inf
            c_best_yaws = np.zeros(n_turbines)
            for _ in range(n_restarts):
                x0 = np.random.uniform(-YAW_MAX, YAW_MAX, n_turbines)
                x0[t] = np.clip(x0[t], -constraint_deg, constraint_deg)
                try:
                    res = differential_evolution(
                        neg_power, bounds_constrained, seed=None,
                        maxiter=50, tol=0.01, polish=True,
                        init="latinhypercube", popsize=10,
                    )
                    if -res.fun > c_best_power:
                        c_best_power = -res.fun
                        c_best_yaws = res.x.copy()
                except Exception:
                    pass

            other = [i for i in range(n_turbines) if i != t]
            shift = sum(abs(best_yaws[i] - c_best_yaws[i]) for i in other)
            cost = ((best_power - c_best_power) / best_power * 100
                    if best_power > 0 else 0)
            sign_flip = any(
                best_yaws[i] * c_best_yaws[i] < 0 and abs(best_yaws[i]) > 5
                for i in other
            )

            results.append(SearchResult(
                layout_name=layout_name,
                wd=wd,
                constrained_turbine=t,
                unconstrained_yaws_deg=np.round(best_yaws, 1).copy(),
                constrained_yaws_deg=np.round(c_best_yaws, 1).copy(),
                unconstrained_power=best_power,
                constrained_power=c_best_power,
                strategy_shift_deg=shift,
                power_cost_pct=cost,
                has_sign_flip=sign_flip,
                n_turbines=n_turbines,
            ))

    return results


def _analyse_power_matrix(
    layout_name, n_turbines, power_matrix, yaw_configs_deg,
    constraint_masks, wd_array, constraint_deg,
) -> List[SearchResult]:
    """Extract results from a precomputed (n_configs, n_wd) power matrix."""
    results: List[SearchResult] = []

    for wi, wd in enumerate(wd_array):
        powers = power_matrix[:, wi]
        best_idx = int(np.argmax(powers))
        best_yaws = yaw_configs_deg[best_idx]
        best_power = powers[best_idx]

        for t in range(n_turbines):
            mask = constraint_masks[t]
            constrained_powers = powers[mask]
            if constrained_powers.size == 0:
                continue
            c_best_local = int(np.argmax(constrained_powers))
            c_best_idx = np.where(mask)[0][c_best_local]
            c_yaws = yaw_configs_deg[c_best_idx]
            c_power = constrained_powers[c_best_local]

            other = [i for i in range(n_turbines) if i != t]
            shift = sum(abs(best_yaws[i] - c_yaws[i]) for i in other)
            cost = (best_power - c_power) / best_power * 100 if best_power > 0 else 0
            sign_flip = any(
                best_yaws[i] * c_yaws[i] < 0 and abs(best_yaws[i]) > 5
                for i in other
            )

            results.append(SearchResult(
                layout_name=layout_name,
                wd=wd,
                constrained_turbine=t,
                unconstrained_yaws_deg=best_yaws.copy(),
                constrained_yaws_deg=c_yaws.copy(),
                unconstrained_power=best_power,
                constrained_power=c_power,
                strategy_shift_deg=shift,
                power_cost_pct=cost,
                has_sign_flip=sign_flip,
                n_turbines=n_turbines,
            ))

    return results


# ── Visualization ───────────────────────────────────────────────────
def plot_result(result: SearchResult, positions: np.ndarray,
                constraint_deg: float, save_path: Optional[str] = None):
    """Plot a single result: layout + yaw comparison + power comparison."""
    n = result.n_turbines
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Left: farm layout with yaw arrows ──
    ax = axes[0]
    wd_rad = np.deg2rad(270 - result.wd)

    for label, yaws, color in [
        ("Unconstrained", result.unconstrained_yaws_deg, "#2196F3"),
        ("Constrained", result.constrained_yaws_deg, "#F44336"),
    ]:
        for i in range(n):
            px, py = positions[i]
            face_angle = wd_rad + np.pi  # faces into wind
            yaw_rad = np.deg2rad(yaws[i])
            rotor_angle = face_angle + yaw_rad
            # Rotor line
            rx = D / 2 * np.cos(rotor_angle + np.pi / 2)
            ry = D / 2 * np.sin(rotor_angle + np.pi / 2)
            ax.plot([px - rx, px + rx], [py - ry, py + ry],
                    color=color, lw=3, alpha=0.8)
            # Nacelle arrow
            arrow_len = D * 0.5
            dx = arrow_len * np.cos(rotor_angle)
            dy = arrow_len * np.sin(rotor_angle)
            ax.annotate("", xy=(px + dx, py + dy), xytext=(px, py),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    # Turbine labels
    for i in range(n):
        px, py = positions[i]
        ct_marker = " *" if i == result.constrained_turbine else ""
        ax.text(px, py - D * 1.0,
                f"T{i}{ct_marker}\n"
                f"unc: {result.unconstrained_yaws_deg[i]:+.0f}°\n"
                f"con: {result.constrained_yaws_deg[i]:+.0f}°",
                ha="center", fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Wind arrow
    xmin = positions[:, 0].min() - 3 * D
    ymid = positions[:, 1].mean()
    ax.annotate(f"Wind {result.wd:.0f}°", xy=(xmin + 2 * D, ymid),
                xytext=(xmin, ymid),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2),
                fontsize=10, ha="center", va="center", color="gray")

    ax.set_aspect("equal")
    ct = result.constrained_turbine
    ax.set_title(f"{result.layout_name} | WD={result.wd:.0f}°\n"
                 f"Constrain T{ct} to ±{constraint_deg:.0f}°", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    margin = 2 * D
    ax.set_xlim(positions[:, 0].min() - 4 * D, positions[:, 0].max() + margin)
    ax.set_ylim(positions[:, 1].min() - margin * 1.5, positions[:, 1].max() + margin)
    ax.legend(handles=[
        Line2D([0], [0], color="#2196F3", lw=3, label="Unconstrained"),
        Line2D([0], [0], color="#F44336", lw=3, label="Constrained"),
    ], loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Centre: yaw angle comparison bars ──
    ax = axes[1]
    x_pos = np.arange(n)
    width = 0.35
    ax.bar(x_pos - width / 2, result.unconstrained_yaws_deg, width,
           label="Unconstrained", color="#2196F3", alpha=0.8)
    ax.bar(x_pos + width / 2, result.constrained_yaws_deg, width,
           label="Constrained", color="#F44336", alpha=0.8)
    ax.axhspan(-constraint_deg, constraint_deg, alpha=0.1, color="red",
               label=f"T{ct} limit ±{constraint_deg:.0f}°")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"T{i}{'*' if i == ct else ''}" for i in range(n)])
    ax.set_ylabel("Yaw angle (°)")
    ax.set_title("Per-turbine yaw angles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", lw=0.5)

    # ── Right: power comparison ──
    ax = axes[2]
    farm = make_farm(positions)
    x_arr, y_arr = positions[:, 0], positions[:, 1]
    p_unc = eval_power_per_turbine(
        farm, x_arr, y_arr, result.wd,
        result.unconstrained_yaws_deg.tolist()) / 1e6
    p_con = eval_power_per_turbine(
        farm, x_arr, y_arr, result.wd,
        result.constrained_yaws_deg.tolist()) / 1e6

    ax.bar(x_pos - width / 2, p_unc, width,
           label=f"Unconstrained ({p_unc.sum():.2f} MW)", color="#2196F3", alpha=0.8)
    ax.bar(x_pos + width / 2, p_con, width,
           label=f"Constrained ({p_con.sum():.2f} MW)", color="#F44336", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"T{i}{'*' if i == ct else ''}" for i in range(n)])
    ax.set_ylabel("Power (MW)")
    ax.set_title(f"Per-turbine power | Cost: {result.power_cost_pct:.1f}%")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Strategy shift: {result.strategy_shift_deg:.0f}° on free turbines | "
        f"Sign flip: {'YES' if result.has_sign_flip else 'no'}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fname = (f"{save_path}/{result.layout_name}_wd{result.wd:.0f}"
                 f"_T{result.constrained_turbine}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Find layouts where per-turbine yaw constraints "
                    "cause global strategy shifts."
    )
    parser.add_argument("--yaw-max", type=float, default=30.0,
                        help="Maximum yaw angle (degrees)")
    parser.add_argument("--yaw-step", type=float, default=5.0,
                        help="Yaw grid step for 3-4 turbine layouts (degrees)")
    parser.add_argument("--wd-step", type=float, default=5.0,
                        help="Wind direction step (degrees)")
    parser.add_argument("--constraint-deg", type=float, default=10.0,
                        help="Per-turbine yaw constraint limit")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top results to display")
    parser.add_argument("--save-plots", type=str, default=None,
                        help="Directory to save plots")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--layouts", nargs="+", default=None,
                        help="Only search these layouts")
    args = parser.parse_args()

    layouts = get_layouts()
    if args.layouts:
        layouts = {k: v for k, v in layouts.items() if k in args.layouts}

    wd_array = np.arange(0, 360, args.wd_step)

    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)

    print("=" * 80)
    print("  MULTIMODAL LAYOUT SEARCH")
    print(f"  PyWake: Blondel-Cathelain 2020 + Jimenez + CrespoHernandez")
    print(f"  DTU10MW (D={D}m) | WS={WS}m/s | TI={TI}")
    print(f"  Layouts: {len(layouts)} | WD: 0-355° / {args.wd_step}° | "
          f"Constraint: ±{args.constraint_deg}°")
    print("=" * 80)

    all_results: List[SearchResult] = []

    for layout_name, positions in layouts.items():
        n_turb = positions.shape[0]
        t0 = time.time()

        if n_turb <= 4:
            # Grid search — feasible
            yaw_configs = build_yaw_grid(n_turb, args.yaw_max, args.yaw_step)
            print(f"\n  [{layout_name}] {n_turb} turb, "
                  f"{len(yaw_configs)} yaw configs (grid) ...",
                  end=" ", flush=True)
            results = search_layout_grid(
                layout_name, positions, wd_array, yaw_configs, args.constraint_deg
            )
        else:
            # Optimization — for 5+ turbines
            # Use fewer wind directions with optim (each one is expensive)
            wd_coarse = np.arange(0, 360, max(args.wd_step, 10))
            print(f"\n  [{layout_name}] {n_turb} turb, "
                  f"optim ({len(wd_coarse)} wd) ...",
                  end=" ", flush=True)
            results = search_layout_optim(
                layout_name, positions, wd_coarse, args.constraint_deg,
            )

        elapsed = time.time() - t0
        # Filter: meaningful shift and reasonable power cost
        good = [r for r in results
                if r.strategy_shift_deg > 5 and 0.3 < r.power_cost_pct < 30]
        print(f"done in {elapsed:.1f}s ({len(good)} interesting cases)")
        all_results.extend(good)

    # Sort by strategy shift
    all_results.sort(key=lambda r: r.strategy_shift_deg, reverse=True)

    # Deduplicate: best per (layout, wd rounded to 10°, turbine)
    seen = set()
    deduped = []
    for r in all_results:
        key = (r.layout_name, round(r.wd / 10) * 10, r.constrained_turbine)
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    all_results = deduped
    top_k = all_results[:args.top_k]

    # Print results table
    print(f"\n{'=' * 80}")
    print(f"  TOP {len(top_k)} RESULTS")
    print(f"{'=' * 80}")

    if not top_k:
        print("  No cases found with strategy shift > 5° and power cost 0.3-30%.")
        print("  Try: --constraint-deg 5, different layouts, or --wd-step 2")
        print(f"\nDone.")
        return

    hdr_yaw_w = max(5 * r.n_turbines + 6 for r in top_k)
    print(f"  {'#':<3} {'Layout':<16} {'WD':>4} {'T':>2} "
          f"{'Shift':>6} {'Cost':>6} {'Flip':>4}  "
          f"{'Unconstrained':>{hdr_yaw_w}}  {'Constrained':>{hdr_yaw_w}}")
    print(f"  {'─'*3} {'─'*16} {'─'*4} {'─'*2} "
          f"{'─'*6} {'─'*6} {'─'*4}  {'─'*hdr_yaw_w}  {'─'*hdr_yaw_w}")

    for rank, r in enumerate(top_k, 1):
        unc_str = "[" + ",".join(f"{y:+.0f}" for y in r.unconstrained_yaws_deg) + "]"
        con_str = "[" + ",".join(f"{y:+.0f}" for y in r.constrained_yaws_deg) + "]"
        print(f"  {rank:<3} {r.layout_name:<16} {r.wd:>3.0f}° T{r.constrained_turbine} "
              f"{r.strategy_shift_deg:>5.0f}° {r.power_cost_pct:>5.1f}% "
              f"{'YES' if r.has_sign_flip else ' no':>4}  "
              f"{unc_str:>{hdr_yaw_w}}  {con_str:>{hdr_yaw_w}}")

    # Detailed top 3
    print(f"\n{'=' * 80}")
    print(f"  DETAILED TOP RESULTS")
    print(f"{'=' * 80}")
    for rank, r in enumerate(top_k[:3], 1):
        ct = r.constrained_turbine
        print(f"\n  ─── Case {rank}: {r.layout_name}, wd={r.wd:.0f}°, "
              f"constrain T{ct} to ±{args.constraint_deg:.0f}° ───")
        print(f"  Unconstrained: "
              f"[{', '.join(f'{y:+.0f}' for y in r.unconstrained_yaws_deg)}]° "
              f"→ {r.unconstrained_power / 1e6:.3f} MW")
        print(f"  Constrained:   "
              f"[{', '.join(f'{y:+.0f}' for y in r.constrained_yaws_deg)}]° "
              f"→ {r.constrained_power / 1e6:.3f} MW")
        print(f"  Shift: {r.strategy_shift_deg:.0f}° | "
              f"Cost: {r.power_cost_pct:.1f}% | "
              f"Sign flip: {'YES' if r.has_sign_flip else 'no'}")
        for i in range(r.n_turbines):
            marker = " ← constrained" if i == ct else ""
            delta = r.constrained_yaws_deg[i] - r.unconstrained_yaws_deg[i]
            print(f"    T{i}: {r.unconstrained_yaws_deg[i]:+6.1f}° → "
                  f"{r.constrained_yaws_deg[i]:+6.1f}° (Δ={delta:+.0f}°){marker}")

    # Plot
    if not args.no_plot and top_k:
        for r in top_k[:3]:
            pos = layouts[r.layout_name]
            plot_result(r, pos, args.constraint_deg, save_path=args.save_plots)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
