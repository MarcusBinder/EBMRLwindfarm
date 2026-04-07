# Constraint Coupling Search Results

Results from `scripts/find_constraint_coupling.py` — finding layouts where constraining
one turbine's yaw to +-10 deg causes other turbines to adopt different optimal yaw angles.

**PyWake config**: Blondel_Cathelain_2020 + JimenezWakeDeflection + CrespoHernandez + DTU10MW (D=178.3m)

**Search scope**: 1548 three-turbine + 96 four-turbine parametric layouts, all with >=5D pairwise spacing, full 360 deg WD sweep at 5 deg resolution, WS/TI sweep.

## Best Result: stag4_5d_+0.5

**Layout**: 4-turbine staggered chain, 5D spacing, +0.5D lateral offset

```
Positions (meters):
  T0 = (0.0,     0.0)
  T1 = (891.5,  89.15)    = (5D, +0.5D)
  T2 = (1783.0,  0.0)     = (10D, 0)
  T3 = (2674.5, 89.15)    = (15D, +0.5D)

Pairwise spacing:
  T0-T1 = 5.0D    T0-T2 = 10.0D   T0-T3 = 15.0D
  T1-T2 = 5.0D    T1-T3 = 10.0D   T2-T3 = 5.0D
  All >= 5D (891.5m)
```

**Conditions**: WD = 85 deg, WS = 10 m/s, TI = 0.07

**Constrained turbine**: T1 limited to +-10 deg (was +-30 deg)

### Optimal yaw angles

| Turbine | Unconstrained | Constrained | Shift | Unc Power | Con Power |
|---------|--------------|-------------|-------|-----------|-----------|
| T0      | 0 deg        | 0 deg       | 0 deg | 2.76 MW   | 1.82 MW   |
| **T1**  | **+20 deg**  | **+10 deg** | **CT** | 5.68 MW  | 6.09 MW   |
| T2      | -10 deg      | 0 deg       | **+10 deg** | 1.76 MW | 1.07 MW |
| T3      | -20 deg      | -10 deg     | **+10 deg** | 5.73 MW | 6.57 MW |

- Total shift on free turbines: **20 deg** (T2: 10 deg + T3: 10 deg)
- Power cost: **2.4%** (15.92 MW -> 15.54 MW)
- Min power on shifted turbines: **1.07 MW** (T2 in constrained case) - well above 0.5 MW threshold
- All shifted turbines have meaningful power (>0.5 MW) in BOTH optima

### Physical interpretation

At WD=85 deg (wind from slightly north of east), the staggered chain creates a
sequential wake pattern where each turbine partially wakes the next. T1 yaws +20 deg
to steer its wake away from T2. When T1 is constrained to +-10 deg, it can only
partially steer, so the *entire downstream chain* (T2 and T3) adopts less aggressive
yaw angles. T0's power also drops (2.76 -> 1.82 MW) as a secondary chain effect,
even though its yaw doesn't change.

### Robustness across conditions

The same qualitative pattern (Unc: [0, +20, -10, -20], Con: [0, +10, 0, -10]) holds
across a wide range of wind speeds and turbulence intensities:

| WS (m/s) | TI   | Total Shift | Cost  | Min Power on shifted turbines |
|----------|------|-------------|-------|-------------------------------|
| 8        | 0.07 | 20 deg      | 3.0%  | 0.44 MW (T2, marginal)        |
| 9        | 0.07 | 20 deg      | 2.9%  | 0.71 MW                       |
| **10**   | **0.07** | **20 deg** | **2.4%** | **1.07 MW**              |
| 11       | 0.07 | 20 deg      | 2.2%  | 1.44 MW                       |
| 9        | 0.05 | 20 deg      | 2.1%  | 0.41 MW (marginal)            |

**Recommended showcase**: WS=10 or WS=11 at TI=0.07 (highest power on shifted turbines).

## Other Notable Results

### 3-turbine echelon: c3_5d_-1.0_10d_-1.0

```
T0 = (0, 0),  T1 = (5D, -1.0D),  T2 = (10D, -1.0D)
WD = 275 deg, WS = 10 m/s, TI = 0.07
Constrain T0 to +-10 deg
```

| Turbine | Unconstrained | Constrained | Power change |
|---------|--------------|-------------|--------------|
| **T0**  | **-20 deg**  | **-10 deg** | 5.73 -> 6.57 MW (CT) |
| T1      | +15 deg      | +15 deg     | 6.05 -> 5.43 MW |
| T2      | 0 deg        | 0 deg       | 4.33 -> 2.94 MW |

Free turbines don't shift yaw, but T2's power drops by 32% (4.33 -> 2.94 MW) purely
from the chain effect of T0's reduced wake steering. This shows constraint propagation
through *power redistribution* even without yaw angle changes.

### 4-turbine staggered 7D: stag4_7d_+0.5

```
T0 = (0, 0),  T1 = (7D, +0.5D),  T2 = (14D, 0),  T3 = (21D, +0.5D)
WD = 85 deg, WS = 10 m/s, TI = 0.07
Constrain T1 to +-10 deg
```

| Turbine | Unconstrained | Constrained | Shift |
|---------|--------------|-------------|-------|
| T0      | 0 deg        | 0 deg       | 0 deg |
| **T1**  | **-20 deg**  | **-10 deg** | **CT** |
| T2      | -10 deg      | 0 deg       | **+10 deg** |
| T3      | -20 deg      | -20 deg     | 0 deg |

Total shift: 10 deg (only T2 shifts). Power cost: 4.2%.

## Search Methodology

### What was searched
- **Prong 1**: 1548 parametric 3-turbine chains (d1 in 5-10D, offsets +-0.3 to +-1.5D)
  plus 96 four-turbine layouts (grids, diamonds, staggered chains)
- **Prong 2**: 41 subsets (3/4/5-turb) of the r1 training layout (3x2 grid at 7Dx5D)
- **Prong 3**: Bilevel optimization (scipy differential_evolution over layout positions)
- Grid search: 10 deg step (coarse), refined to 5 deg on promising cases

### Key finding: inline symmetry artifacts

Many apparent "sign flips" (e.g., -20 deg -> +20 deg on T2) turned out to be artifacts
of symmetric inline waking where +- yaw produces identical wake deflection. The telltale:
the "flipped" turbine has identical power in both cases (e.g., 5.73 MW -> 5.73 MW =
cos^2(20 deg) x rated). These are NOT genuine strategy changes.

### Fundamental physics limitation

The Jimenez deflection model produces wake displacement of ~0.27D at 5D downstream
for 20 deg yaw. With Gaussian wake width ~1D at 5D, the deflection-to-width ratio
(delta/sigma ~ 0.3) creates smooth, unimodal power landscapes. This limits genuine
strategy shifts to ~10 deg per turbine at >=5D spacing. Larger shifts require either
tighter spacing (<5D), a stronger deflection model, or lower TI (narrower wakes).

## Files

- Search script: `scripts/find_constraint_coupling.py`
- Surface plots: `results/coupling_stag4_5d_+0.5_wd85.png` (and 3 others)
- Previous search (legacy): `scripts/find_multimodal_layout.py`
