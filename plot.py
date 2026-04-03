import numpy as np
import matplotlib.pyplot as plt
import os

all_returns = []
all_steps = None

seeds = [0, 1, 2, 3, 4]
for seed in seeds:
    path = f"{os.getcwd()}/results/naive/seed_{seed}.npz_FILES/"
    returns = np.load(path + "returns.npy")
    steps = np.load(path + "steps.npy")
    
    all_returns.append(returns)
    
    if all_steps is None:
      all_steps = steps

all_returns = np.array(all_returns)
mean_returns = np.mean(all_returns, axis=0)
std_returns = np.std(all_returns, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(all_steps, mean_returns, label="Mean Return")
plt.fill_between(all_steps, mean_returns - std_returns, mean_returns + std_returns, alpha=0.3)
optimal_return = 500
plt.axhline(y=optimal_return, color="red", linestyle="--", linewidth=2, label="Optimal Return")
plt.xlabel('Steps')
plt.ylabel('Returns')
plt.title('Average Returns over Steps')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()