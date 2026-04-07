# CLAUDE.md — Project Guide for Claude Code

## General notes
- Always work in a new branch when implementing new features

## Project Overview

**EBM + RL + Transformers for Wind Farm Control.** This repo combines Energy-Based Models with reinforcement learning, using a transformer backbone that treats each wind turbine as a token. Originally a Transformer-SAC agent for yaw control (generalizes zero-shot across farm layouts), now being extended with EBM research.

**Environment dependency:** Training requires `WindGym` (open-source wind farm gym). The environment is NOT included in this repo. Evaluation and network development can be done standalone.

## Architecture

The core idea: **turbines are tokens**. A transformer processes variable-length sequences of turbine observations, enabling a single policy to control any farm size.

- **Actor/Critic** (`networks.py`): Transformer encoder → per-turbine embeddings. Actor outputs per-turbine yaw actions, critic mean-pools embeddings then outputs a single farm-level Q-value.
- **Positional encodings** (`positional_encodings/`): 14+ variants — absolute (MLP, sinusoidal, polar), relative bias (MLP, ALiBi, directional), RoPE, spatial, GAT-based. Selected via `--pos_encoding_type` string.
- **Profile encodings** (`profile_encodings/`): CNN/Fourier encoders for wake receptivity/influence profiles. Selected via `--profile_encoding_type` string.
- **Config** (`config.py`): Single `@dataclass Args` parsed by `tyro`. All hyperparameters here.

## Code Conventions

- **Config pattern:** All hyperparameters in `config.py` as a `tyro` dataclass. Never use argparse.
- **Factory pattern for encodings:** `create_positional_encoding()` and `create_profile_encoder()` in `networks.py` map string names to classes.
- **Type hints:** Used throughout (`typing.Optional`, `Tuple`, `List`, etc.)
- **Module organization:** `positional_encodings/` and `profile_encodings/` use `_prefixed.py` private modules re-exported via `__init__.py`.
- **Lazy imports:** Heavy dependencies (PyWake, scipy) are imported lazily in `helpers/__init__.py`.
- **Wind-relative coordinates:** Turbine positions are always transformed to a wind-relative frame before encoding (wind from 270°).

## Key Files

| File | Purpose |
|------|---------|
| `transformer_sac_windfarm.py` | Main SAC training loop (~68KB) |
| `diffusion_sac_windfarm.py` | Diffusion-SAC training loop (diffusion actor + SAC critics) |
| `ebt_sac_windfarm.py` | EBT-SAC training loop (explicit EBM actor + SAC critics) |
| `diffusion.py` | Diffusion actor, load surrogates (per-turbine, travel budget) |
| `ebt.py` | EBT actor (explicit energy head, gradient-descent actions) |
| `networks.py` | Actor, Critic, TQC architectures + encoding factories (~45KB) |
| `config.py` | All CLI args as tyro dataclass |
| `evaluate.py` | Evaluation pipeline |
| `replay_buffer.py` | Experience replay with variable-size turbine sequences |
| `helpers/agent.py` | `WindFarmAgent` — wraps actor for inference |
| `helpers/multi_layout_env.py` | Multi-layout env wrapper (trains on diverse farms) |
| `helpers/helper_funcs.py` | Checkpoint save/load, coordinate transforms |
| `helpers/layouts.py` | Farm layout definitions (turbine x,y positions) |
| `scripts/fetch_wandb_results.py` | Fetch and plot wandb experiment results |
| `scripts/run_sweep.py` | Run hyperparameter sweep experiments |
| `scripts/demo_per_turbine_constraints.py` | Demo per-turbine constraints + travel budget |

## Common Commands

```bash
# SAC training (requires WindGym)
python transformer_sac_windfarm.py --layouts square_1 --total_timesteps 100000 --seed 1

# Diffusion-SAC training
python diffusion_sac_windfarm.py --layouts 3turb --action_type yaw --dt_sim 1 --dt_env 1 --noise_schedule cosine --bc_weight_start 1.0

# EBT-SAC training
python ebt_sac_windfarm.py --layouts 3turb --total_timesteps 100000

# Evaluation
python evaluate.py --checkpoint runs/<run>/checkpoints/step_100000.pt --eval_layouts square_1

# Experiment sweep
python scripts/run_sweep.py --total-timesteps 10000

# Fetch wandb results
python scripts/fetch_wandb_results.py --filter "sweep_"

# Per-turbine constraint demo
python scripts/demo_per_turbine_constraints.py --checkpoint runs/<run>/checkpoints/step_10000.pt

# All config options
python transformer_sac_windfarm.py --help
```

## Dependencies

Core: `torch>=2.0`, `numpy`, `gymnasium`, `tyro`, `wandb`, `matplotlib`, `scipy`
Environment: `py_wake`, `WindGym`

