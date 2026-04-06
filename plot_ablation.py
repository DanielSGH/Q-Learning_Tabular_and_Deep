"""
plot_ablation.py — Task 2.2: Plot ablation study results.

Reads:  results_ablation/abl_<param>_<value>/seed_<N>.npz
Writes: results_ablation/ablation_<param>.png  (one figure per hyperparameter)

Usage:
    python plot_ablation.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR  = "results_ablation"
LOG_INTERVAL = 500
SMOOTH_WINDOW = 50

ABLATION_GRIDS = {
    "lr": {
        "values": [1e-4, 1e-3, 1e-2],
        "labels": ["lr=1e-4", "lr=1e-3", "lr=1e-2"],
        "colors": ["steelblue", "darkorange", "green"],
        "title":  "Learning Rate Ablation",
    },
    "update_every": {
        "values": [1, 4, 16],
        "labels": ["update_every=1", "update_every=4", "update_every=16"],
        "colors": ["steelblue", "darkorange", "green"],
        "title":  "Update-to-Data Ratio Ablation",
    },
    "hidden_size": {
        "values": [32, 128, 256],
        "labels": ["hidden=32", "hidden=128", "hidden=256"],
        "colors": ["steelblue", "darkorange", "green"],
        "title":  "Network Size Ablation",
    },
    "epsilon_end": {
        "values": [0.01, 0.05, 0.20],
        "labels": ["ε_end=0.01", "ε_end=0.05", "ε_end=0.20"],
        "colors": ["steelblue", "darkorange", "green"],
        "title":  "Exploration Factor Ablation",
    },
}


def val_to_str(val):
    if isinstance(val, float):
        return str(val).replace(".", "p")
    return str(val)


def smooth(arr, window):
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def load_seeds(run_name: str):
    """Load all seed .npz files for a given run_name."""
    run_dir = os.path.join(RESULTS_DIR, run_name)
    if not os.path.exists(run_dir):
        return None
    seed_data = []
    for fname in sorted(os.listdir(run_dir)):
        if fname.endswith(".npz"):
            data = np.load(os.path.join(run_dir, fname))
            seed_data.append((data["steps"], data["returns"]))
    return seed_data if seed_data else None


def plot_param(param_name: str, grid: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    any_plotted = False

    for val, label, color in zip(grid["values"], grid["labels"], grid["colors"]):
        run_name  = f"abl_{param_name}_{val_to_str(val)}"
        seed_data = load_seeds(run_name)

        if seed_data is None:
            print(f"  [warn] no data for {run_name}, skipping")
            continue

        # Interpolate all seeds onto common step grid
        max_steps    = max(s[-1] for s, _ in seed_data)
        common_steps = np.arange(LOG_INTERVAL, max_steps + 1, LOG_INTERVAL)

        curves = []
        for steps, rets in seed_data:
            interp = np.interp(common_steps, steps, rets)
            curves.append(interp)

        arr  = np.array(curves)
        mean = smooth(arr.mean(axis=0), SMOOTH_WINDOW)
        std  = smooth(arr.std(axis=0),  SMOOTH_WINDOW)

        ax.plot(common_steps, mean, label=label, color=color, linewidth=1.5)
        ax.fill_between(common_steps, mean - std, mean + std,
                        alpha=0.2, color=color)
        any_plotted = True

    if not any_plotted:
        print(f"  [skip] no data found for {param_name}")
        plt.close(fig)
        return

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.set_title(grid["title"])
    ax.legend(loc="upper left")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, f"ablation_{param_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out_path}")


def main():
    for param_name, grid in ABLATION_GRIDS.items():
        print(f"Plotting {param_name}...")
        plot_param(param_name, grid)
    print("Done.")


if __name__ == "__main__":
    main()
