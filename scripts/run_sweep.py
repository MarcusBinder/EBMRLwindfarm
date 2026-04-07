#!/usr/bin/env python3
"""
Sweep: test training improvements for yaw oscillation fix.

All runs use action_type=yaw, dt_env=1 (best from prior experiments).
Sweeps over noise schedule, action regularization, and BC annealing.

After all runs complete, calls fetch_wandb_results.py to analyze.

Usage:
    python scripts/run_sweep.py                   # Run full sweep
    python scripts/run_sweep.py --dry-run          # Print commands without running
    python scripts/run_sweep.py --total-timesteps 50000  # Override timesteps
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Fixed settings
BASE_ARGS = {
    "layouts": "3turb",
    "dt_sim": 1,
    "dt_env": 1,
    "learning_starts": 1000,
    "batch_size": 256,
    "track": True,
    "save_model": False,
}

# Diffusion improvement configs to test (applied on top of BASE_ARGS)
CONFIGS = [
    ("baseline", {
        "noise_schedule": "linear",
    }),
    ("cosine", {
        "noise_schedule": "cosine",
    }),
    ("cosine_areg01", {
        "noise_schedule": "cosine",
        "action_reg_weight": 0.1,
    }),
    ("cosine_bc_anneal", {
        "noise_schedule": "cosine",
        "bc_weight_start": 1.0,
        "bc_weight_end": 0.0,
        "bc_anneal_steps": 7000,
    }),
    ("cosine_areg_bc", {
        "noise_schedule": "cosine",
        "action_reg_weight": 0.1,
        "bc_weight_start": 1.0,
        "bc_weight_end": 0.0,
        "bc_anneal_steps": 7000,
    }),
]

# Action types to test
ACTION_TYPES = ["yaw", "wind"]

# Build full experiment list: every config x every action type
EXPERIMENTS = [
    (f"{name}_{atype}", {**cfg, "action_type": atype})
    for name, cfg in CONFIGS
    for atype in ACTION_TYPES
]


def build_command(name: str, extra_args: dict, total_timesteps: int,
                  seed: int) -> list[str]:
    """Build the training command for a single experiment."""
    cmd = [sys.executable, "diffusion_sac_windfarm.py"]

    all_args = {**BASE_ARGS, **extra_args}
    all_args["total_timesteps"] = total_timesteps
    all_args["seed"] = seed
    all_args["wandb_project_name"] = "diffusion_windfarm"
    all_args["exp_name"] = f"sweep_{name}_s{seed}"

    for key, val in all_args.items():
        cli_key = key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{cli_key}")
        else:
            cmd.extend([f"--{cli_key}", str(val)])

    return cmd


def run_experiment(name: str, cmd: list[str], dry_run: bool = False) -> bool:
    """Run a single experiment. Returns True on success."""
    cmd_str = " ".join(cmd)
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*70}")
    print(f"CMD: {cmd_str}\n")

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\nFAILED: {name} (exit code {result.returncode}) [{elapsed/60:.1f}min]")
        return False
    else:
        print(f"\nDONE: {name} [{elapsed/60:.1f}min]")
        return True


def analyze_results():
    """Run the wandb results fetcher."""
    print(f"\n{'='*70}")
    print("ANALYZING RESULTS")
    print(f"{'='*70}\n")

    script = Path(__file__).resolve().parent / "fetch_wandb_results.py"
    subprocess.run([
        sys.executable, str(script),
        "--filter", "sweep_",
        "--output-dir", "scripts/wandb_plots",
    ])


def main():
    parser = argparse.ArgumentParser(description="Run training improvement sweep")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--total-timesteps", type=int, default=10000,
                        help="Training steps per experiment (default: 10000)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only experiments matching this substring")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip the wandb analysis step at the end")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.only:
        experiments = [(n, e) for n, e in experiments if args.only in n]

    print(f"Running {len(experiments)} experiments, {args.total_timesteps} steps each")
    print(f"Estimated time: ~{len(experiments) * 7} minutes\n")

    results = {}
    t_total = time.time()

    for name, extra_args in experiments:
        cmd = build_command(name, extra_args, args.total_timesteps, args.seed)
        success = run_experiment(name, cmd, dry_run=args.dry_run)
        results[name] = success

    elapsed_total = time.time() - t_total

    # Summary
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE [{elapsed_total/60:.1f}min total]")
    print(f"{'='*70}")
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")

    # Analyze
    if not args.dry_run and not args.skip_analysis:
        analyze_results()


if __name__ == "__main__":
    main()
