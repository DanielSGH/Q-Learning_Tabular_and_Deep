import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS = {
  "Naive":   ("naive",  "steelblue"),
  "Only TN": ("tn",     "darkorange"),
  "Only ER": ("er",     "green"),
  "TN + ER": ("tn_er",  "crimson"),
}

RESULTS_DIR   = "results"
OUTPUT_PATH   = os.path.join(RESULTS_DIR, "comparison_plot.png")
LOG_INTERVAL  = 500
SMOOTH_WINDOW = 20


def smooth_curve(y: np.ndarray, window: int = 20) -> np.ndarray:
  if window <= 1 or len(y) < 2:
    return y
  window = min(window, len(y))
  kernel = np.ones(window, dtype=float) / window
  return np.convolve(y, kernel, mode="same")


def load_seeds(run_name: str, results_dir: str):
  run_dir = os.path.join(results_dir, run_name)
  seed_data = []
  for fname in sorted(os.listdir(run_dir)):
    if fname.endswith(".npz"):
      data = np.load(os.path.join(run_dir, fname))
      seed_data.append((data["steps"], data["returns"]))
  return seed_data


def main():
  fig, ax = plt.subplots(figsize=(8, 5))

  for label, (run_name, color) in CONFIGS.items():
    seed_data = load_seeds(run_name, RESULTS_DIR)
    if not seed_data:
      print(f"[warn] no data found for {run_name}, skipping")
      continue

    max_steps    = max(s[-1] for s, _ in seed_data)
    common_steps = np.arange(LOG_INTERVAL, max_steps + 1, LOG_INTERVAL)

    curves = []
    for steps, rets in seed_data:
      interp = np.interp(common_steps, steps, rets)
      interp = smooth_curve(interp, SMOOTH_WINDOW)
      curves.append(interp)

    arr  = np.array(curves)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    ax.plot(common_steps, mean, label=label, color=color, linewidth=1.5)
    ax.fill_between(common_steps, mean - std, mean + std,
            alpha=0.2, color=color)

  ax.set_xlabel("Environment steps")
  ax.set_ylabel("Episode return")
  ax.set_title("CartPole-v1: Final Comparison Across Seeds")
  ax.legend(loc="upper left")
  ax.set_xlim(left=0)
  ax.set_ylim(bottom=0)
  plt.tight_layout()
  plt.savefig(OUTPUT_PATH, dpi=150)
  plt.close(fig)
  print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
  main()