I'm working on an Energy-Based Model (EBM) reinforcement learning project for wind
farm yaw control. I want to demonstrate that applying a load constraint on a SINGLE
turbine (limiting its yaw to ±10°) causes the ENTIRE farm to adopt a qualitatively
different optimal yaw configuration — not just clip that one turbine.

## The task

Find a specific (layout, wind_direction, wind_speed, TI) configuration where:
1. The unconstrained optimal yaw angles involve multiple turbines yawing cooperatively
2. Constraining ONE turbine to ±10° causes OTHER turbines to change their yaw by
   15°+ (ideally with sign flips)
3. ALL turbines that shift must have meaningful power (>0.5 MW) in BOTH the
   unconstrained and constrained optima — this is critical
4. Use at most 5 turbines
5. ALL turbine pairs must be spaced at least 5 rotor diameters apart (5D = 891.5m
   for DTU10MW). This is a hard physical constraint — real wind farms don't place
   turbines closer than this.

## PyWake setup (must match exactly — this is what WindGym uses)

```python
from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
from py_wake.deflection_models.jimenez import JimenezWakeDeflection
from py_wake.turbulence_models import CrespoHernandez
from py_wake.site import UniformSite
from py_wake.examples.data.dtu10mw import DTU10MW

wt = DTU10MW()  # D = 178.3m
site = UniformSite()
site.initial_position = positions
farm = Blondel_Cathelain_2020(
    site, windTurbines=wt,
    turbulenceModel=CrespoHernandez(),
    deflectionModel=JimenezWakeDeflection(),
)
res = farm(x=x, y=y, wd=[wd], ws=[ws], TI=ti, yaw=yaw_list, tilt=0)
total_power = res.Power.values.sum()
per_turbine = res.Power.values.squeeze()  # shape (n_turb,)
```

Acceptable ranges: WS = 8–12 m/s, TI = 0.02–0.09.
PyWake runs at ~3ms per call — grid search over 2197 configs (13^3) is ~6 seconds.

## What has already been tried (DO NOT REPEAT)

### Dead ends:
- **Symmetric inline rows** (e.g., 3 turbines at 5D spacing, WD=270°): The Jimenez
  model uses sin(γ)·cos²(γ) for deflection, which IS sign-aware. But for inline
  layouts the wake is centered on the downstream turbine, so ±γ deflects equally
  away — no sign asymmetry.

- **Layouts with 3–5D spacing and small lateral offsets**: Downstream turbines end up
  fully waked (0 MW power). Grid search finds "shifts" that are artifacts — the
  shifted turbine produces 0 MW regardless of yaw, so argmax picks arbitrarily.
  ALWAYS verify that shifted turbines have real power (>0.5 MW) in both optima.

- **~30 layouts tested** including triangles, V-shapes, staggered rows, diamonds,
  kites, arrows, irregular formations, echelons, offset rows — across full 360° wind
  directions, TI 0.02–0.09, WS 8–12. No dramatic strategy shifts found when
  requiring all shifted turbines to have meaningful power.

- **5-turbine layouts with scipy optimization**: differential_evolution with
  n_restarts=8, 36 wind directions. Found 0 interesting cases.

- **V-up layout** T0=(0,+0.5D), T1=(0,-0.5D), T2=(5D,0): Found a 15° real shift but
  T0-T1 spacing is only 1D — violates the 5D minimum spacing constraint.

### The fundamental limitation discovered:
The Jimenez deflection (beta=0.1) produces ~0.27D lateral displacement at 5D
downstream for 20° yaw. The Gaussian wake width at 5D is ~1D+. This means:
- At 5D spacing, turbines near the wake center are binary (fully waked ~0 MW, or free)
- No meaningful "partial wake" regime where deflection direction matters enough
- Power landscape is smooth and single-modal
- Wider spacing (7–10D) gives partial waking but even weaker deflection

### Key physics insight:
With a 2-turbine test (T0 upstream, T1 at 0.5D lateral offset, 5D downstream):
- T0 yaw=+20°: T1=6.532 MW (wake deflected away)
- T0 yaw=-20°: T1=1.197 MW (wake deflected toward)
This proves the sign asymmetry EXISTS — the problem is creating layouts where this
asymmetry produces coupled multi-turbine strategy changes at ≥5D spacing.

## What to try next (suggested directions)

1. **Wider spacing (7–10D) with oblique wind**: At larger distances the Gaussian wake
   expands more, creating a wider partial-wake zone. Combined with oblique wind
   angles, multiple turbines could simultaneously be in partial-wake conditions where
   the deflection direction matters.

2. **4-turbine diamond at ≥5D spacing**: T0=(0,0), T1=(5D,3D), T2=(5D,-3D),
   T3=(10D,0). At oblique wind angles, T1 and T2 may both partially wake T3, and
   their cooperative yaw strategies interact. All pairs are ≥5D apart.

3. **Exploit oblique wind on grid layouts**: The project's training layouts (r1, r2,
   r3) are 6-turbine grids at 5D+ spacing. Pick any 3–5 turbine subset and test at
   oblique wind angles (e.g., 240–260°) where multiple turbines are in partial wakes.

## Output format

For each promising case, show:
- Layout positions, wind direction, WS, TI
- Verification that all turbine pairs are ≥5D apart
- Unconstrained optimum: per-turbine yaw angles and per-turbine power (MW)
- Constrained optimum: per-turbine yaw angles and per-turbine power (MW)
- Which turbine is constrained, total shift on free turbines, power cost %
- Confirmation that all shifted turbines have >0.5 MW in both cases
