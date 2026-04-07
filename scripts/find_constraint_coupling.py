#!/usr/bin/env python3
"""
Find wind farm layouts where constraining one turbine's yaw range (±10°)
causes qualitatively different optimal strategies for OTHER turbines.

Three search strategies (prongs):
  1. Parametric 3-4 turbine chains at 5-10D spacing
  2. Subsets of the r1 training grid (3×2 at 7D×5D, guaranteed ≥5D)
  3. Bilevel optimization: directly optimize layout to maximize strategy shift

Critical improvement over find_multimodal_layout.py: per-turbine power
validation — rejects "phantom shifts" where shifted turbines produce 0 MW.

PyWake config matches WindGym exactly:
  Blondel_Cathelain_2020 + JimenezWakeDeflection + CrespoHernandez + DTU10MW

Usage:
    python scripts/find_constraint_coupling.py --prongs 3              # bilevel only (~15 min)
    python scripts/find_constraint_coupling.py --prongs 1 2 --coarse   # broad sweep (~40 min)
    python scripts/find_constraint_coupling.py --prongs 1 2 3          # full search (~3+ hours)
"""

import argparse
import itertools
import sys
import time

# Force unbuffered output for progress reporting
sys.stdout.reconfigure(line_buffering=True)
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.site import UniformSite
from py_wake.turbulence_models import CrespoHernandez

# ── Constants ─────────────────────────────────────────────────────────
D = 178.3              # DTU10MW rotor diameter (m)
MIN_SPACING_D = 5.0    # hard physical minimum (5D = 891.5m)
MIN_POWER_MW = 0.5     # minimum power for a "real" turbine contribution
YAW_MAX = 30.0         # maximum yaw angle (degrees)
CONSTRAINT_DEG = 10.0  # default per-turbine constraint

# (WS m/s, TI) sweep points
WS_TI_COMBOS = [
    (8, 0.04), (9, 0.06), (10, 0.07), (11, 0.05), (12, 0.03),
]

# r1 training layout: 3×2 grid at 7D × 5D (all pairs ≥5D)
R1_POSITIONS = np.array([
    [0.0, 0.0],
    [1248.1, 0.0],
    [2496.2, 0.0],
    [0.0, 891.5],
    [1248.1, 891.5],
    [2496.2, 891.5],
])


# ── Data structures ──────────────────────────────────────────────────
@dataclass
class CouplingResult:
    layout_name: str
    positions: np.ndarray
    wd: float
    ws: float
    ti: float
    constrained_turbine: int
    unconstrained_yaws: np.ndarray
    constrained_yaws: np.ndarray
    unconstrained_power_total: float  # watts
    constrained_power_total: float
    unconstrained_power_per_turb: np.ndarray = field(default_factory=lambda: np.array([]))
    constrained_power_per_turb: np.ndarray = field(default_factory=lambda: np.array([]))
    total_shift_deg: float = 0.0
    max_shift_deg: float = 0.0
    power_cost_pct: float = 0.0
    has_sign_flip: bool = False
    n_sign_flips: int = 0
    min_shifted_power_mw: float = 0.0
    n_turbines: int = 0
    prong: int = 0


# ── PyWake evaluation engine ─────────────────────────────────────────
def make_farm(positions: np.ndarray):
    """Create PyWake farm model with WindGym-identical settings."""
    wt = DTU10MW()
    site = UniformSite()
    site.initial_position = positions
    return Blondel_Cathelain_2020(
        site, windTurbines=wt,
        turbulenceModel=CrespoHernandez(),
        deflectionModel=JimenezWakeDeflection(),
    )


def eval_per_turbine_power(
    farm, x, y, wd: float, ws: float, ti: float, yaw_deg
) -> np.ndarray:
    """Per-turbine power in watts for a single (WD, WS, TI, yaw) config."""
    res = farm(x=x, y=y, wd=[wd], ws=[ws], TI=ti, yaw=yaw_deg, tilt=0)
    return res.Power.values.squeeze()  # (n_turb,)


def check_min_spacing(positions: np.ndarray, min_d: float = MIN_SPACING_D) -> bool:
    """Check all pairwise turbine distances ≥ min_d rotor diameters."""
    min_dist = min_d * D
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_dist - 0.1:  # small tolerance
                return False
    return True


def pairwise_distances_D(positions: np.ndarray) -> np.ndarray:
    """Return upper-triangle of pairwise distances in rotor diameters."""
    n = len(positions)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(positions[i] - positions[j]) / D)
    return np.array(dists)


def build_yaw_grid(n_turbines: int, yaw_max: float, yaw_step: float) -> np.ndarray:
    """All yaw angle combinations. Returns (C, N) array in degrees."""
    yaw_values = np.arange(-yaw_max, yaw_max + yaw_step / 2, yaw_step)
    grids = np.meshgrid(*[yaw_values] * n_turbines, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=-1)


# ── Per-turbine power validation (THE critical fix) ──────────────────
def validate_candidate(
    farm, x, y, result: CouplingResult,
) -> Optional[CouplingResult]:
    """Re-evaluate with per-turbine power. Return None if invalid."""
    unc_power = eval_per_turbine_power(
        farm, x, y, result.wd, result.ws, result.ti,
        result.unconstrained_yaws.tolist(),
    )
    con_power = eval_per_turbine_power(
        farm, x, y, result.wd, result.ws, result.ti,
        result.constrained_yaws.tolist(),
    )

    ct = result.constrained_turbine
    n = len(x)
    other = [i for i in range(n) if i != ct]
    shifts = np.abs(result.unconstrained_yaws - result.constrained_yaws)
    shifted = [i for i in other if shifts[i] > 5.0]

    if not shifted:
        return None

    # Check ALL shifted turbines have > MIN_POWER_MW in BOTH optima
    min_shifted = min(
        min(unc_power[i] for i in shifted),
        min(con_power[i] for i in shifted),
    )
    if min_shifted < MIN_POWER_MW * 1e6:
        return None

    result.unconstrained_power_per_turb = unc_power.copy()
    result.constrained_power_per_turb = con_power.copy()
    result.min_shifted_power_mw = min_shifted / 1e6
    return result


# ── Grid search engine ───────────────────────────────────────────────
def search_grid(
    layout_name: str,
    positions: np.ndarray,
    wd_array: np.ndarray,
    ws: float,
    ti: float,
    yaw_configs: np.ndarray,
    constraint_deg: float,
    shift_threshold: float = 10.0,
    prong: int = 1,
) -> Tuple[List[CouplingResult], List[CouplingResult]]:
    """Grid search for one layout. Returns (validated, near_misses)."""
    n_turb = positions.shape[0]
    n_configs = yaw_configs.shape[0]
    n_wd = len(wd_array)
    farm = make_farm(positions)
    x, y = positions[:, 0], positions[:, 1]

    # Phase 1: total-power matrix (vectorized over WD)
    power_matrix = np.zeros((n_configs, n_wd))
    for ci in range(n_configs):
        yaw = yaw_configs[ci].tolist()
        res = farm(x=x, y=y, wd=wd_array, ws=[ws], TI=ti, yaw=yaw, tilt=0)
        power_matrix[ci, :] = res.Power.values.sum(axis=0).squeeze()

    # Constraint masks
    constraint_masks = {
        t: np.abs(yaw_configs[:, t]) <= constraint_deg + 0.01
        for t in range(n_turb)
    }

    # Phase 1: extract candidates
    candidates = []
    for wi, wd in enumerate(wd_array):
        powers = power_matrix[:, wi]
        best_idx = int(np.argmax(powers))
        best_yaws = yaw_configs[best_idx]
        best_power = powers[best_idx]

        for t in range(n_turb):
            mask = constraint_masks[t]
            cp = powers[mask]
            if cp.size == 0:
                continue
            c_local = int(np.argmax(cp))
            c_idx = np.where(mask)[0][c_local]
            c_yaws = yaw_configs[c_idx]
            c_power = cp[c_local]

            other = [i for i in range(n_turb) if i != t]
            shifts_arr = np.abs(best_yaws - c_yaws)
            total_shift = sum(shifts_arr[i] for i in other)
            max_shift = max(shifts_arr[i] for i in other) if other else 0
            cost = (best_power - c_power) / best_power * 100 if best_power > 0 else 0
            sign_flips = [
                i for i in other
                if best_yaws[i] * c_yaws[i] < 0 and abs(best_yaws[i]) > 5
            ]

            if total_shift < shift_threshold or not (0.1 < cost < 40):
                continue

            candidates.append(CouplingResult(
                layout_name=layout_name,
                positions=positions.copy(),
                wd=wd, ws=ws, ti=ti,
                constrained_turbine=t,
                unconstrained_yaws=best_yaws.copy(),
                constrained_yaws=c_yaws.copy(),
                unconstrained_power_total=best_power,
                constrained_power_total=c_power,
                total_shift_deg=total_shift,
                max_shift_deg=max_shift,
                power_cost_pct=cost,
                has_sign_flip=len(sign_flips) > 0,
                n_sign_flips=len(sign_flips),
                n_turbines=n_turb,
                prong=prong,
            ))

    # Phase 2: per-turbine power validation
    validated = []
    near_misses = []
    for c in candidates:
        result = validate_candidate(farm, x, y, c)
        if result is not None:
            validated.append(result)
        elif c.total_shift_deg >= 15:
            # Keep as near-miss for diagnostics
            # Still evaluate per-turbine power for diagnostic output
            unc_p = eval_per_turbine_power(farm, x, y, c.wd, c.ws, c.ti,
                                           c.unconstrained_yaws.tolist())
            con_p = eval_per_turbine_power(farm, x, y, c.wd, c.ws, c.ti,
                                           c.constrained_yaws.tolist())
            c.unconstrained_power_per_turb = unc_p.copy()
            c.constrained_power_per_turb = con_p.copy()
            near_misses.append(c)

    return validated, near_misses


# ── Prong 1: Parametric layout generation ────────────────────────────
def generate_chain_layouts_3turb() -> Dict[str, np.ndarray]:
    """3-turbine chains with T0 at origin, parameterized T1 and T2."""
    layouts = {}
    d1_range = [5, 6, 7, 8, 9, 10]
    off1_range = [-1.5, -1.0, -0.5, 0.3, 0.5, 1.0, 1.5]
    d2_offsets = [4, 5, 6, 7, 8]  # d2 = d1 + this
    off2_range = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    for d1 in d1_range:
        for off1 in off1_range:
            for d2_off in d2_offsets:
                d2 = d1 + d2_off
                if d2 > 20:
                    continue
                for off2 in off2_range:
                    pos = np.array([
                        [0, 0],
                        [d1 * D, off1 * D],
                        [d2 * D, off2 * D],
                    ])
                    if not check_min_spacing(pos):
                        continue
                    name = f"c3_{d1}d_{off1:+.1f}_{d2}d_{off2:+.1f}"
                    layouts[name] = pos
    return layouts


def generate_chain_layouts_4turb() -> Dict[str, np.ndarray]:
    """4-turbine layouts: 2×2 grids and diamonds at ≥5D spacing."""
    layouts = {}
    # 2×2 grids at various spacings
    for dx in [5, 6, 7, 8]:
        for dy in [5, 6, 7]:
            for off in [0.0, 0.5, 1.0, -0.5, -1.0]:
                pos = np.array([
                    [0, 0],
                    [dx * D, 0],
                    [off * D, dy * D],
                    [(dx + off) * D, dy * D],
                ])
                if check_min_spacing(pos):
                    name = f"g4_{dx}x{dy}_{off:+.1f}"
                    layouts[name] = pos

    # Diamonds
    for d in [5, 6, 7, 8]:
        for w in [3, 4, 5]:
            pos = np.array([
                [0, 0],
                [d * D, w * D],
                [d * D, -w * D],
                [2 * d * D, 0],
            ])
            if check_min_spacing(pos):
                name = f"dia4_{d}d_{w}w"
                layouts[name] = pos

    # Staggered chains
    for d in [5, 6, 7, 8]:
        for off in [0.5, 1.0, 1.5, -0.5, -1.0, -1.5]:
            pos = np.array([
                [0, 0],
                [d * D, off * D],
                [2 * d * D, 0],
                [3 * d * D, off * D],
            ])
            if check_min_spacing(pos):
                name = f"stag4_{d}d_{off:+.1f}"
                layouts[name] = pos

    return layouts


# ── Prong 2: r1 layout subsets ───────────────────────────────────────
def generate_r1_subsets() -> Dict[str, np.ndarray]:
    """All 3/4/5-turbine subsets of the r1 grid."""
    layouts = {}
    n = len(R1_POSITIONS)
    for size in [3, 4, 5]:
        for combo in itertools.combinations(range(n), size):
            pos = R1_POSITIONS[list(combo)]
            name = f"r1_{''.join(str(i) for i in combo)}"
            layouts[name] = pos
    return layouts


# ── Prong 3: Bilevel layout optimization ─────────────────────────────
def search_bilevel(
    constraint_deg: float,
    ws: float,
    ti: float,
    yaw_configs_3: np.ndarray,
    n_restarts: int = 20,
) -> Tuple[List[CouplingResult], List[CouplingResult]]:
    """Optimize 3-turbine layout positions to maximize strategy shift."""
    validated = []
    near_misses = []

    # Outer bounds: T1 and T2 positions in rotor diameters
    # T0 fixed at origin. Also optimize WD.
    bounds = [
        (4 * D, 14 * D),    # T1_x
        (-6 * D, 6 * D),    # T1_y
        (8 * D, 22 * D),    # T2_x
        (-6 * D, 6 * D),    # T2_y
        (0, 359),           # WD
    ]

    def objective(params):
        t1x, t1y, t2x, t2y, wd = params
        positions = np.array([[0, 0], [t1x, t1y], [t2x, t2y]])

        if not check_min_spacing(positions):
            return 0.0  # penalty: no shift

        farm = make_farm(positions)
        x, y = positions[:, 0], positions[:, 1]

        # Total power for all yaw configs at this single WD
        powers = np.zeros(len(yaw_configs_3))
        for ci in range(len(yaw_configs_3)):
            yaw = yaw_configs_3[ci].tolist()
            res = farm(x=x, y=y, wd=[wd], ws=[ws], TI=ti, yaw=yaw, tilt=0)
            powers[ci] = float(res.Power.values.sum())

        # Find best unconstrained
        best_idx = int(np.argmax(powers))
        best_yaws = yaw_configs_3[best_idx]
        best_power = powers[best_idx]

        if best_power <= 0:
            return 0.0

        # Try constraining each turbine
        best_shift = 0.0
        for t in range(3):
            mask = np.abs(yaw_configs_3[:, t]) <= constraint_deg + 0.01
            cp = powers[mask]
            if cp.size == 0:
                continue
            c_local = int(np.argmax(cp))
            c_idx = np.where(mask)[0][c_local]
            c_yaws = yaw_configs_3[c_idx]
            c_power = cp[c_local]

            other = [i for i in range(3) if i != t]
            total_shift = sum(abs(best_yaws[i] - c_yaws[i]) for i in other)
            cost = (best_power - c_power) / best_power * 100

            if not (0.1 < cost < 40):
                continue

            # Quick power validation for the objective
            if total_shift > 10:
                unc_p = eval_per_turbine_power(farm, x, y, wd, ws, ti,
                                               best_yaws.tolist())
                con_p = eval_per_turbine_power(farm, x, y, wd, ws, ti,
                                               c_yaws.tolist())
                shifts_arr = np.abs(best_yaws - c_yaws)
                shifted = [i for i in other if shifts_arr[i] > 5.0]
                if shifted:
                    min_p = min(
                        min(unc_p[i] for i in shifted),
                        min(con_p[i] for i in shifted),
                    )
                    if min_p < MIN_POWER_MW * 1e6:
                        continue  # phantom shift
                    best_shift = max(best_shift, total_shift)

        return -best_shift  # minimize negative shift

    for restart in range(n_restarts):
        try:
            res = differential_evolution(
                objective, bounds, seed=restart * 42 + 7,
                maxiter=80, tol=0.5, polish=False,
                init="latinhypercube", popsize=12,
            )
            if -res.fun > 5:  # found some shift
                t1x, t1y, t2x, t2y, wd = res.x
                positions = np.array([[0, 0], [t1x, t1y], [t2x, t2y]])
                if not check_min_spacing(positions):
                    continue

                farm = make_farm(positions)
                x, y = positions[:, 0], positions[:, 1]

                # Re-evaluate to extract full result
                powers = np.zeros(len(yaw_configs_3))
                for ci in range(len(yaw_configs_3)):
                    yaw = yaw_configs_3[ci].tolist()
                    r = farm(x=x, y=y, wd=[wd], ws=[ws], TI=ti, yaw=yaw, tilt=0)
                    powers[ci] = float(r.Power.values.sum())

                best_idx = int(np.argmax(powers))
                best_yaws = yaw_configs_3[best_idx]
                best_power = powers[best_idx]

                for t in range(3):
                    mask = np.abs(yaw_configs_3[:, t]) <= constraint_deg + 0.01
                    cp = powers[mask]
                    if cp.size == 0:
                        continue
                    c_local = int(np.argmax(cp))
                    c_idx = np.where(mask)[0][c_local]
                    c_yaws = yaw_configs_3[c_idx]
                    c_power = cp[c_local]

                    other = [i for i in range(3) if i != t]
                    shifts_arr = np.abs(best_yaws - c_yaws)
                    total_shift = sum(shifts_arr[i] for i in other)
                    max_shift = max(shifts_arr[i] for i in other)
                    cost = (best_power - c_power) / best_power * 100 if best_power > 0 else 0
                    sign_flips = [
                        i for i in other
                        if best_yaws[i] * c_yaws[i] < 0 and abs(best_yaws[i]) > 5
                    ]

                    if total_shift < 10 or not (0.1 < cost < 40):
                        continue

                    cand = CouplingResult(
                        layout_name=f"bilevel_r{restart}",
                        positions=positions.copy(),
                        wd=wd, ws=ws, ti=ti,
                        constrained_turbine=t,
                        unconstrained_yaws=best_yaws.copy(),
                        constrained_yaws=c_yaws.copy(),
                        unconstrained_power_total=best_power,
                        constrained_power_total=c_power,
                        total_shift_deg=total_shift,
                        max_shift_deg=max_shift,
                        power_cost_pct=cost,
                        has_sign_flip=len(sign_flips) > 0,
                        n_sign_flips=len(sign_flips),
                        n_turbines=3,
                        prong=3,
                    )
                    result = validate_candidate(farm, x, y, cand)
                    if result is not None:
                        validated.append(result)
                    elif total_shift >= 15:
                        unc_p = eval_per_turbine_power(
                            farm, x, y, wd, ws, ti, best_yaws.tolist())
                        con_p = eval_per_turbine_power(
                            farm, x, y, wd, ws, ti, c_yaws.tolist())
                        cand.unconstrained_power_per_turb = unc_p.copy()
                        cand.constrained_power_per_turb = con_p.copy()
                        near_misses.append(cand)
        except Exception as e:
            print(f"    bilevel restart {restart} failed: {e}")

    return validated, near_misses


# ── 5-turbine optimization search ────────────────────────────────────
def search_optim(
    layout_name: str,
    positions: np.ndarray,
    wd_array: np.ndarray,
    ws: float,
    ti: float,
    constraint_deg: float,
    n_restarts: int = 10,
    prong: int = 2,
) -> Tuple[List[CouplingResult], List[CouplingResult]]:
    """Scipy optimization for 5-turbine layouts."""
    n_turb = positions.shape[0]
    farm = make_farm(positions)
    x, y = positions[:, 0], positions[:, 1]
    validated = []
    near_misses = []

    for wd in wd_array:
        def neg_power(yaw_flat):
            res = farm(x=x, y=y, wd=[wd], ws=[ws], TI=ti,
                       yaw=yaw_flat.tolist(), tilt=0)
            return -float(res.Power.values.sum())

        # Unconstrained
        bounds_free = [(-YAW_MAX, YAW_MAX)] * n_turb
        best_power = -np.inf
        best_yaws = np.zeros(n_turb)
        for _ in range(n_restarts):
            try:
                res = differential_evolution(
                    neg_power, bounds_free,
                    maxiter=30, tol=0.1, polish=True,
                    init="latinhypercube", popsize=8,
                )
                if -res.fun > best_power:
                    best_power = -res.fun
                    best_yaws = res.x.copy()
            except Exception:
                pass

        if best_power <= 0:
            continue

        # Constrained per turbine
        for t in range(n_turb):
            bounds_c = list(bounds_free)
            bounds_c[t] = (-constraint_deg, constraint_deg)

            c_best_power = -np.inf
            c_best_yaws = np.zeros(n_turb)
            for _ in range(n_restarts):
                try:
                    res = differential_evolution(
                        neg_power, bounds_c,
                        maxiter=30, tol=0.1, polish=True,
                        init="latinhypercube", popsize=8,
                    )
                    if -res.fun > c_best_power:
                        c_best_power = -res.fun
                        c_best_yaws = res.x.copy()
                except Exception:
                    pass

            other = [i for i in range(n_turb) if i != t]
            shifts_arr = np.abs(best_yaws - c_best_yaws)
            total_shift = sum(shifts_arr[i] for i in other)
            max_shift = max(shifts_arr[i] for i in other) if other else 0
            cost = (best_power - c_best_power) / best_power * 100
            sign_flips = [
                i for i in other
                if best_yaws[i] * c_best_yaws[i] < 0 and abs(best_yaws[i]) > 5
            ]

            if total_shift < 10 or not (0.1 < cost < 40):
                continue

            cand = CouplingResult(
                layout_name=layout_name,
                positions=positions.copy(),
                wd=wd, ws=ws, ti=ti,
                constrained_turbine=t,
                unconstrained_yaws=np.round(best_yaws, 1).copy(),
                constrained_yaws=np.round(c_best_yaws, 1).copy(),
                unconstrained_power_total=best_power,
                constrained_power_total=c_best_power,
                total_shift_deg=total_shift,
                max_shift_deg=max_shift,
                power_cost_pct=cost,
                has_sign_flip=len(sign_flips) > 0,
                n_sign_flips=len(sign_flips),
                n_turbines=n_turb,
                prong=prong,
            )
            result = validate_candidate(farm, x, y, cand)
            if result is not None:
                validated.append(result)
            elif total_shift >= 15:
                unc_p = eval_per_turbine_power(farm, x, y, wd, ws, ti,
                                               best_yaws.tolist())
                con_p = eval_per_turbine_power(farm, x, y, wd, ws, ti,
                                               c_best_yaws.tolist())
                cand.unconstrained_power_per_turb = unc_p.copy()
                cand.constrained_power_per_turb = con_p.copy()
                near_misses.append(cand)

    return validated, near_misses


# ── Reporting ─────────────────────────────────────────────────────────
def print_result_detail(rank: int, r: CouplingResult, constraint_deg: float):
    """Print detailed result with per-turbine power."""
    ct = r.constrained_turbine
    n = r.n_turbines
    print(f"\n  ═══ #{rank}: {r.layout_name} | WD={r.wd:.0f}° WS={r.ws} TI={r.ti}"
          f" | Constrain T{ct} to ±{constraint_deg:.0f}° ═══")

    # Spacing
    dists = pairwise_distances_D(r.positions)
    pairs = list(itertools.combinations(range(n), 2))
    spacing_str = "  ".join(f"T{i}-T{j}={d:.1f}D" for (i, j), d in zip(pairs, dists))
    all_ok = all(d >= MIN_SPACING_D - 0.01 for d in dists)
    print(f"  Spacing: {spacing_str}  {'✓' if all_ok else '✗'} all ≥{MIN_SPACING_D:.0f}D")

    # Positions
    pos_str = "  ".join(f"T{i}=({r.positions[i, 0]/D:.1f}D,{r.positions[i, 1]/D:.1f}D)"
                        for i in range(n))
    print(f"  Layout:  {pos_str}")

    # Yaw angles
    unc_str = "[" + ", ".join(f"{y:+.0f}" for y in r.unconstrained_yaws) + "]°"
    con_str = "[" + ", ".join(f"{y:+.0f}" for y in r.constrained_yaws) + "]°"
    print(f"  Unc yaw: {unc_str}  Total: {r.unconstrained_power_total/1e6:.3f} MW")
    print(f"  Con yaw: {con_str}  Total: {r.constrained_power_total/1e6:.3f} MW")

    # Per-turbine detail
    if len(r.unconstrained_power_per_turb) > 0:
        for i in range(n):
            unc_y = r.unconstrained_yaws[i]
            con_y = r.constrained_yaws[i]
            unc_p = r.unconstrained_power_per_turb[i] / 1e6
            con_p = r.constrained_power_per_turb[i] / 1e6
            delta = con_y - unc_y
            marker = ""
            if i == ct:
                marker = " ← CONSTRAINED"
            elif abs(delta) > 5:
                flip = " FLIP" if unc_y * con_y < 0 and abs(unc_y) > 5 else ""
                marker = f" Δ={delta:+.0f}°{flip}"
            print(f"    T{i}: yaw {unc_y:+6.0f}° → {con_y:+6.0f}°  "
                  f"power {unc_p:.2f} → {con_p:.2f} MW{marker}")

    print(f"  Summary: shift={r.total_shift_deg:.0f}° (max {r.max_shift_deg:.0f}°) | "
          f"cost={r.power_cost_pct:.1f}% | "
          f"sign flips={r.n_sign_flips} | "
          f"min shifted power={r.min_shifted_power_mw:.2f} MW | "
          f"prong={r.prong}")


def print_diagnostics(
    near_misses: List[CouplingResult],
    all_shifts: List[float],
    n_total_combos: int,
):
    """Print diagnostic info when no validated results found."""
    print(f"\n{'=' * 80}")
    print(f"  DIAGNOSTICS — No validated results found")
    print(f"{'=' * 80}")
    print(f"  Total (layout × WD × WS × TI × turbine) combos evaluated: {n_total_combos}")

    if all_shifts:
        shifts = np.array(all_shifts)
        print(f"\n  Shift distribution (before per-turbine validation):")
        for thresh in [5, 10, 15, 20, 25, 30]:
            count = np.sum(shifts >= thresh)
            print(f"    ≥{thresh}°: {count} cases")
        print(f"    Max shift: {shifts.max():.1f}°")

    if near_misses:
        print(f"\n  Near-misses (shift ≥15° but failed power validation):")
        near_misses.sort(key=lambda r: r.total_shift_deg, reverse=True)
        for i, r in enumerate(near_misses[:10]):
            ct = r.constrained_turbine
            unc_str = "[" + ",".join(f"{y:+.0f}" for y in r.unconstrained_yaws) + "]"
            con_str = "[" + ",".join(f"{y:+.0f}" for y in r.constrained_yaws) + "]"
            print(f"    {i+1}. {r.layout_name} WD={r.wd:.0f}° T{ct} "
                  f"shift={r.total_shift_deg:.0f}°")
            print(f"       Unc: {unc_str}  Con: {con_str}")
            if len(r.unconstrained_power_per_turb) > 0:
                for j in range(r.n_turbines):
                    if j == ct:
                        continue
                    delta = abs(r.constrained_yaws[j] - r.unconstrained_yaws[j])
                    if delta > 5:
                        unc_mw = r.unconstrained_power_per_turb[j] / 1e6
                        con_mw = r.constrained_power_per_turb[j] / 1e6
                        flag = " ← LOW" if min(unc_mw, con_mw) < MIN_POWER_MW else ""
                        print(f"       T{j}: Δ={delta:.0f}° power={unc_mw:.2f}/{con_mw:.2f} MW{flag}")
    else:
        print(f"\n  No near-misses (no candidates with shift ≥15° at all)")

    print(f"\n  Suggestions:")
    print(f"    - Try --constraint-deg 5 (tighter constraint)")
    print(f"    - Try wider spacing range (d1 up to 12D)")
    print(f"    - Run with --prongs 3 --bilevel-restarts 40 for more bilevel coverage")
    print(f"    - Consider relaxing MIN_SPACING_D to 4.0 if 5D is too restrictive")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Find layouts where per-turbine constraints cause "
                    "global strategy shifts. Three search prongs."
    )
    parser.add_argument("--constraint-deg", type=float, default=CONSTRAINT_DEG)
    parser.add_argument("--yaw-max", type=float, default=YAW_MAX)
    parser.add_argument("--yaw-step", type=float, default=5.0,
                        help="Yaw grid step (5=fine, 10=coarse)")
    parser.add_argument("--wd-step", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--prongs", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--coarse", action="store_true",
                        help="Use 10° yaw step for prong 1")
    parser.add_argument("--bilevel-restarts", type=int, default=20)
    parser.add_argument("--ws-ti-idx", type=int, nargs="+", default=None,
                        help="Only run specific WS/TI combo indices (0-4)")
    parser.add_argument("--save-plots", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    ws_ti_list = WS_TI_COMBOS
    if args.ws_ti_idx is not None:
        ws_ti_list = [WS_TI_COMBOS[i] for i in args.ws_ti_idx]

    prongs = set(args.prongs)
    wd_array = np.arange(0, 360, args.wd_step)

    yaw_step_p1 = 10.0 if args.coarse else args.yaw_step
    yaw_grid_3 = build_yaw_grid(3, args.yaw_max, args.yaw_step)
    yaw_grid_3_p1 = build_yaw_grid(3, args.yaw_max, yaw_step_p1)
    yaw_grid_4_coarse = build_yaw_grid(4, args.yaw_max, 10.0)

    print("=" * 80)
    print("  CONSTRAINT-COUPLING SEARCH")
    print(f"  PyWake: Blondel-Cathelain 2020 + Jimenez + CrespoHernandez")
    print(f"  DTU10MW (D={D}m) | Min spacing: {MIN_SPACING_D}D = {MIN_SPACING_D*D:.0f}m")
    print(f"  Constraint: ±{args.constraint_deg}° | Yaw max: ±{args.yaw_max}°")
    print(f"  WS/TI combos: {ws_ti_list}")
    print(f"  Prongs: {sorted(prongs)} | WD step: {args.wd_step}°")
    if 1 in prongs:
        print(f"  Prong 1 yaw step: {yaw_step_p1}° ({'coarse' if args.coarse else 'fine'})")
    print("=" * 80)

    all_validated: List[CouplingResult] = []
    all_near_misses: List[CouplingResult] = []
    all_shifts: List[float] = []
    n_total_combos = 0
    t_start = time.time()

    # ── Prong 1: Parametric chains ────────────────────────────────────
    if 1 in prongs:
        print(f"\n{'─' * 40}")
        print(f"  PRONG 1: Parametric layouts")
        print(f"{'─' * 40}")

        layouts_3 = generate_chain_layouts_3turb()
        layouts_4 = generate_chain_layouts_4turb()
        print(f"  Generated {len(layouts_3)} 3-turb + {len(layouts_4)} 4-turb layouts")

        for ws, ti in ws_ti_list:
            print(f"\n  --- WS={ws} TI={ti} ---")

            # 3-turbine grid search
            for li, (name, pos) in enumerate(layouts_3.items()):
                if li % 50 == 0:
                    print(f"    3-turb: {li}/{len(layouts_3)} ...", flush=True)
                v, nm = search_grid(
                    name, pos, wd_array, ws, ti,
                    yaw_grid_3_p1, args.constraint_deg, prong=1,
                )
                all_validated.extend(v)
                all_near_misses.extend(nm)
                if v:
                    print(f"    ★ {name}: {len(v)} validated results!")

            # 4-turbine grid search (always coarse 10° step)
            for li, (name, pos) in enumerate(layouts_4.items()):
                if li % 20 == 0:
                    print(f"    4-turb: {li}/{len(layouts_4)} ...", flush=True)
                v, nm = search_grid(
                    name, pos, wd_array, ws, ti,
                    yaw_grid_4_coarse, args.constraint_deg, prong=1,
                )
                all_validated.extend(v)
                all_near_misses.extend(nm)
                if v:
                    print(f"    ★ {name}: {len(v)} validated results!")

    # ── Prong 2: r1 subsets ───────────────────────────────────────────
    if 2 in prongs:
        print(f"\n{'─' * 40}")
        print(f"  PRONG 2: r1 layout subsets")
        print(f"{'─' * 40}")

        r1_layouts = generate_r1_subsets()
        print(f"  Generated {len(r1_layouts)} subsets")

        for ws, ti in ws_ti_list:
            print(f"\n  --- WS={ws} TI={ti} ---")
            for name, pos in r1_layouts.items():
                n_turb = len(pos)
                if n_turb <= 4:
                    yg = yaw_grid_3 if n_turb == 3 else yaw_grid_4_coarse
                    v, nm = search_grid(
                        name, pos, wd_array, ws, ti,
                        yg, args.constraint_deg, prong=2,
                    )
                    all_validated.extend(v)
                    all_near_misses.extend(nm)
                    if v:
                        print(f"    ★ {name}: {len(v)} validated results!")
                else:
                    # 5-turbine: use optimization (reduced budget)
                    wd_coarse = np.arange(0, 360, 20)  # coarse WD for 5-turb
                    v, nm = search_optim(
                        name, pos, wd_coarse, ws, ti,
                        args.constraint_deg, n_restarts=3, prong=2,
                    )
                    all_validated.extend(v)
                    all_near_misses.extend(nm)
                    if v:
                        print(f"    ★ {name}: {len(v)} validated results!")

    # ── Prong 3: Bilevel optimization ─────────────────────────────────
    if 3 in prongs:
        print(f"\n{'─' * 40}")
        print(f"  PRONG 3: Bilevel layout optimization")
        print(f"{'─' * 40}")
        print(f"  Restarts: {args.bilevel_restarts}")

        for ws, ti in ws_ti_list:
            print(f"\n  --- WS={ws} TI={ti} ---", flush=True)
            t0 = time.time()
            v, nm = search_bilevel(
                args.constraint_deg, ws, ti, yaw_grid_3,
                n_restarts=args.bilevel_restarts,
            )
            elapsed = time.time() - t0
            all_validated.extend(v)
            all_near_misses.extend(nm)
            print(f"    Done in {elapsed:.1f}s — {len(v)} validated, {len(nm)} near-misses")

    # ── Results ───────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"  SEARCH COMPLETE — {elapsed_total:.0f}s elapsed")
    print(f"  Validated results: {len(all_validated)}")
    print(f"  Near-misses: {len(all_near_misses)}")
    print(f"{'=' * 80}")

    if not all_validated:
        print_diagnostics(all_near_misses, all_shifts, n_total_combos)
        return

    # Sort by total shift, deduplicate
    all_validated.sort(key=lambda r: r.total_shift_deg, reverse=True)
    seen = set()
    deduped = []
    for r in all_validated:
        # Deduplicate: round WD to 10°, same layout prefix + constrained turbine
        prefix = r.layout_name.split("_")[0]  # e.g., "c3" or "r1"
        key = (prefix, round(r.wd / 10) * 10, r.constrained_turbine,
               tuple(np.round(r.positions.ravel(), -1)))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    top_k = deduped[:args.top_k]

    # Summary table
    print(f"\n  TOP {len(top_k)} RESULTS")
    print(f"  {'#':<3} {'Layout':<24} {'WD':>4} {'WS':>3} {'TI':>5} "
          f"{'CT':>2} {'MaxΔ':>5} {'TotΔ':>5} {'Cost':>6} {'Flip':>4} "
          f"{'MinP':>5} {'Pr':>2}")
    print(f"  {'─'*3} {'─'*24} {'─'*4} {'─'*3} {'─'*5} "
          f"{'─'*2} {'─'*5} {'─'*5} {'─'*6} {'─'*4} {'─'*5} {'─'*2}")

    for rank, r in enumerate(top_k, 1):
        print(f"  {rank:<3} {r.layout_name:<24} {r.wd:>3.0f}° {r.ws:>3.0f} {r.ti:>5.3f} "
              f"T{r.constrained_turbine} {r.max_shift_deg:>4.0f}° {r.total_shift_deg:>4.0f}° "
              f"{r.power_cost_pct:>5.1f}% "
              f"{'YES' if r.has_sign_flip else ' no':>4} "
              f"{r.min_shifted_power_mw:>4.1f}M {r.prong:>2}")

    # Detailed top results
    for rank, r in enumerate(top_k[:5], 1):
        print_result_detail(rank, r, args.constraint_deg)

    print(f"\nDone. Total time: {elapsed_total:.0f}s")


if __name__ == "__main__":
    main()
