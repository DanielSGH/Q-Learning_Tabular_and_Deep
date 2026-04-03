"""
ablation.py — Task 2.2: Ablation study over 4 hyperparameters.

Varies one hyperparameter at a time (others fixed to best defaults).
Runs 5 seeds × ≥3 values × 4 hyperparameters = 60 runs.

Usage:
  python ablation.py                    # full study
  python ablation.py --param lr         # only lr sweep
  python ablation.py --steps 200000     # quick test with fewer steps
  python ablation.py --seeds 5          # number of repetitions
"""

import argparse
import os
import itertools
import numpy as np
from train import train

# ---------------------------------------------------------------------------
# Fixed "best" hyperparameters (baseline for each ablation sweep)
# These are the defaults when sweeping OTHER parameters.
# Tune after you see initial results.
# ---------------------------------------------------------------------------

FIXED = dict(
    total_steps       = 1_000_000,
    hidden_sizes      = (64, 64),
    lr                = 1e-3,
    gamma             = 0.99,
    epsilon_start     = 1.0,
    epsilon_end       = 0.05,
    epsilon_decay_steps = 100_000,
    update_every      = 4,
    use_target_network = False,
    use_replay_buffer  = False,
)

# ---------------------------------------------------------------------------
# Ablation grids — at least 3 values each (assignment requirement)
# ---------------------------------------------------------------------------

ABLATION_GRIDS = {
    "lr": {
        "param_key": "lr",
        "values":    [1e-4, 1e-3, 1e-2],
        "labels":    ["1e-4 (low)", "1e-3 (med)", "1e-2 (high)"],
    },
    "update_every": {
        "param_key": "update_every",
        "values":    [1, 4, 16],
        "labels":    ["1 (high ratio)", "4 (med ratio)", "16 (low ratio)"],
    },
    "hidden_sizes": {
        "param_key": "hidden_sizes",
        "values":    [(32,), (64, 64), (256, 256)],
        "labels":    ["[32] (small)", "[64,64] (med)", "[256,256] (large)"],
    },
    "epsilon_end": {
        "param_key": "epsilon_end",
        "values":    [0.01, 0.05, 0.2],
        "labels":    ["0.01 (low ε)", "0.05 (med ε)", "0.20 (high ε)"],
    },
}


# ---------------------------------------------------------------------------
# Run a sweep for one hyperparameter
# ---------------------------------------------------------------------------

def run_sweep(param_name: str, n_seeds: int, total_steps: int, results_dir: str):
    grid = ABLATION_GRIDS[param_name]
    param_key = grid["param_key"]
    values    = grid["values"]
    labels    = grid["labels"]

    print(f"\n{'='*60}")
    print(f"ABLATION: {param_name} | values={values} | seeds={n_seeds}")
    print(f"{'='*60}")

    for val, label in zip(values, labels):
        run_name = f"ablation_{param_name}_{_val_to_str(val)}"
        for seed in range(n_seeds):
            # Build kwargs: start from fixed, override the swept param
            kwargs = dict(FIXED)
            kwargs["total_steps"] = total_steps
            kwargs[param_key] = val
            kwargs["seed"]       = seed
            kwargs["run_name"]   = run_name
            kwargs["results_dir"] = results_dir

            train(**kwargs)


def _val_to_str(val):
    """Convert a hyperparameter value to a safe filename string."""
    if isinstance(val, tuple):
        return "x".join(str(v) for v in val)
    if isinstance(val, float):
        return str(val).replace(".", "p")
    return str(val)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run DQN ablation study (Task 2.2)")
    p.add_argument("--param",       type=str, default="all",
                   choices=["all", "lr", "update_every", "hidden_sizes", "epsilon_end"],
                   help="Which hyperparameter to sweep (default: all)")
    p.add_argument("--steps",       type=int, default=1_000_000,
                   help="Environment steps per run")
    p.add_argument("--seeds",       type=int, default=5,
                   help="Number of repetitions per configuration")
    p.add_argument("--results_dir", type=str, default="results",
                   help="Where to save results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    params_to_run = (
        list(ABLATION_GRIDS.keys()) if args.param == "all"
        else [args.param]
    )

    for param_name in params_to_run:
        run_sweep(
            param_name=param_name,
            n_seeds=args.seeds,
            total_steps=args.steps,
            results_dir=args.results_dir,
        )

    print("\nAblation study complete.")
    print(f"Results saved to: {args.results_dir}/")
    print("Run plot_ablation.py to generate figures.")
