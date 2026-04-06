import argparse
import os
import subprocess
import sys

FIXED = dict(
    total_steps    = 1_000_000,
    num_envs       = 8,
    lr             = 0.0001,
    gamma          = 0.99,
    batch_size     = 32,
    grad_clip      = 10.0,
    epsilon_start  = 1.0,
    epsilon_end    = 0.01,
    epsilon_decay  = 200_000,
    update_every   = 4,
    warmup_steps   = 10_000,
    hidden_size    = 128,
    log_interval   = 500,
)

ABLATION_GRIDS = {
    "lr": {
        "flag":   "--lr",
        "values": [1e-4, 1e-3, 1e-2],
        "labels": ["1e-4", "1e-3", "1e-2"],
    },

    "update_every": {
        "flag":   "--update_every",
        "values": [1, 4, 16],
        "labels": ["1", "4", "16"],
    },

    "hidden_size": {
        "flag":   "--hidden_size",
        "values": [32, 128, 256],
        "labels": ["32", "128", "256"],
    },

    "epsilon_end": {
        "flag":   "--epsilon_end",
        "values": [0.01, 0.05, 0.20],
        "labels": ["0.01", "0.05", "0.20"],
    },
}

def val_to_str(val):
    if isinstance(val, float):
        return str(val).replace(".", "p")
    return str(val)


def build_cmd(run_name: str, seed: int, overrides: dict, total_steps: int) -> list:
    cmd = [sys.executable, "train.py",
           "--run_name", run_name,
           "--seed",     str(seed),
           "--total_steps", str(total_steps),
           "--outdir",   "results_ablation",
    ]
    merged = dict(FIXED)
    merged.update(overrides)
    merged["total_steps"] = total_steps

    skip = {"total_steps"}
    for key, val in merged.items():
        if key in skip:
            continue
        cmd += [f"--{key}", str(val)]

    return cmd

def run_sweep(param_name: str, n_seeds: int, total_steps: int):
    grid   = ABLATION_GRIDS[param_name]
    flag   = grid["flag"]
    values = grid["values"]
    labels = grid["labels"]

    print(f"\n{'='*60}")
    print(f"ABLATION: {param_name}  values={values}  seeds={n_seeds}")
    print(f"{'='*60}")

    for val, label in zip(values, labels):
        run_name  = f"abl_{param_name}_{val_to_str(val)}"
        override  = {param_name: val}

        for seed in range(n_seeds):
            cmd = build_cmd(run_name, seed, override, total_steps)
            print(f"  [{param_name}={label} | seed={seed}]  ", end="", flush=True)
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f"  ERROR (returncode={result.returncode})")
            else:
                print(f"  done")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--param", type=str, default="all",
                   choices=["all"] + list(ABLATION_GRIDS.keys()),
                   help="Which hyperparameter to sweep (default: all)")
    p.add_argument("--steps", type=int, default=1_000_000,
                   help="Environment steps per run")
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of seeds per configuration")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs("results_ablation", exist_ok=True)

    params_to_run = (
        list(ABLATION_GRIDS.keys()) if args.param == "all"
        else [args.param]
    )

    for param_name in params_to_run:
        run_sweep(param_name, n_seeds=args.seeds, total_steps=args.steps)

    print("\nAblation complete. Results in results_ablation/")
    print("Run: python plot_ablation.py")