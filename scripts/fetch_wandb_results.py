#!/usr/bin/env python3
"""
Fetch and visualize results from Weights & Biases.

Produces:
  1. A markdown summary table of all runs (printed to stdout)
  2. Time-series plots of key metrics (saved as PNGs)
  3. CSV exports of raw metric data

Usage:
    # Summary table of all runs
    python scripts/fetch_wandb_results.py

    # Filter runs by name regex
    python scripts/fetch_wandb_results.py --filter "cosine|baseline"

    # Specific metrics only
    python scripts/fetch_wandb_results.py --metrics charts/episodic_return yaw/abs_mean_deg

    # Custom output directory
    python scripts/fetch_wandb_results.py --output-dir results/wandb_plots
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb


# Metrics to fetch for time-series plots
DEFAULT_METRICS = [
    "charts/episodic_return",
    "charts/episodic_power",
    "charts/step_reward_mean",
    "losses/actor_loss",
    "losses/qf1_loss",
    "losses/bc_loss",
    "losses/action_reg",
    "yaw/abs_mean_deg",
    "yaw/over_20_frac",
    "debug/bc_weight",
    "debug/action_mean",
    "debug/action_std",
]

# Config keys to show in the summary table
SUMMARY_CONFIG_KEYS = [
    "action_type",
    "dt_env",
    "noise_schedule",
    "num_diffusion_steps",
    "num_inference_steps",
    "diffusion_bc_weight",
    "bc_weight_start",
    "bc_weight_end",
    "action_reg_weight",
    "lr_warmup_steps",
    "beta_end",
    "total_timesteps",
    "seed",
]

# Summary metrics (last value from run)
SUMMARY_METRIC_KEYS = [
    "charts/step_reward_mean",
    "charts/episodic_return",
    "charts/episodic_power",
    "yaw/abs_mean_deg",
    "yaw/over_20_frac",
]


def fetch_runs(project: str, entity: str | None = None,
               name_filter: str | None = None) -> list:
    """Fetch all runs from a wandb project."""
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path)

    if name_filter:
        pattern = re.compile(name_filter)
        runs = [r for r in runs if pattern.search(r.name)]

    return list(runs)


def build_summary_table(runs: list) -> pd.DataFrame:
    """Build a comparison DataFrame from run configs and final metrics."""
    rows = []
    for run in runs:
        row = {
            "name": run.name,
            "state": run.state,
            "runtime_min": round(run.summary.get("_runtime", 0) / 60, 1),
        }
        # Config values
        for key in SUMMARY_CONFIG_KEYS:
            row[key] = run.config.get(key, "")
        # Final metric values
        for key in SUMMARY_METRIC_KEYS:
            val = run.summary.get(key)
            if val is not None:
                row[key.split("/")[-1]] = round(float(val), 4)
            else:
                row[key.split("/")[-1]] = ""
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by final reward descending
    reward_col = "step_reward_mean"
    if reward_col in df.columns:
        df = df.sort_values(reward_col, ascending=False, na_position="last")
    return df


def download_timeseries(runs: list, metrics: list[str],
                        output_dir: Path) -> dict[str, pd.DataFrame]:
    """Download time-series data for specified metrics from all runs."""
    all_data = {}
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    for run in runs:
        print(f"  Fetching history for {run.name}...", file=sys.stderr)
        try:
            history = run.history(keys=metrics, pandas=True)
        except Exception as e:
            print(f"  WARNING: Failed to fetch {run.name}: {e}", file=sys.stderr)
            continue

        if history.empty:
            continue

        all_data[run.name] = history
        history.to_csv(csv_dir / f"{run.name}.csv", index=False)

    return all_data


def plot_metric(all_data: dict[str, pd.DataFrame], metric: str,
                output_dir: Path, step_col: str = "_step"):
    """Plot a single metric across all runs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False

    for run_name, df in all_data.items():
        if metric not in df.columns:
            continue
        series = df[[step_col, metric]].dropna()
        if series.empty:
            continue
        has_data = True
        ax.plot(series[step_col], series[metric], label=run_name, alpha=0.8)

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    safe_name = metric.replace("/", "_")
    fig.savefig(output_dir / f"{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bars(table: pd.DataFrame, output_dir: Path):
    """Bar chart comparing final metrics across runs."""
    metrics_to_plot = ["step_reward_mean", "episodic_return", "abs_mean_deg", "over_20_frac"]
    available = [m for m in metrics_to_plot if m in table.columns]

    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        df_plot = table[["name", metric]].dropna()
        if df_plot.empty:
            continue
        names = [n[:20] for n in df_plot["name"]]
        values = df_plot[metric].astype(float)
        ax.barh(names, values)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_markdown_table(df: pd.DataFrame):
    """Print a DataFrame as a markdown table to stdout."""
    # Select most useful columns, skip empty ones
    cols = [c for c in df.columns if bool(df[c].astype(str).ne("").any())]
    print("\n## Wandb Run Summary\n")
    print(df[cols].to_markdown(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description="Fetch and plot wandb results")
    parser.add_argument("--project", default="diffusion_windfarm",
                        help="Wandb project name")
    parser.add_argument("--entity", default=None,
                        help="Wandb entity (default: auto-detect)")
    parser.add_argument("--output-dir", default="scripts/wandb_plots",
                        help="Directory for plots and CSV exports")
    parser.add_argument("--filter", default=None,
                        help="Regex filter on run names")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Specific metrics to plot (default: all standard metrics)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Only print summary table, skip plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = args.metrics or DEFAULT_METRICS

    # Fetch runs
    print(f"Fetching runs from project '{args.project}'...", file=sys.stderr)
    runs = fetch_runs(args.project, args.entity, args.filter)
    print(f"Found {len(runs)} runs.", file=sys.stderr)

    if not runs:
        print("No runs found.", file=sys.stderr)
        return

    # Summary table
    table = build_summary_table(runs)
    print_markdown_table(table)
    table.to_csv(output_dir / "summary.csv", index=False)

    if args.no_plots:
        return

    # Time-series data
    print("Downloading time-series data...", file=sys.stderr)
    all_data = download_timeseries(runs, metrics, output_dir)

    # Generate plots
    print("Generating plots...", file=sys.stderr)
    for metric in metrics:
        plot_metric(all_data, metric, output_dir)

    # Comparison bar charts
    plot_comparison_bars(table, output_dir)

    print(f"Plots saved to {output_dir}/", file=sys.stderr)
    print(f"CSV data saved to {output_dir}/csv/", file=sys.stderr)


if __name__ == "__main__":
    main()
