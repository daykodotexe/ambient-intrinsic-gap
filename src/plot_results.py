import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import defaultdict

# Load results
with open("results.json") as f:
    results = json.load(f)

# ── 1. ARI vs rho (one line per d_intrinsic) ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

d_intrinsic_vals = sorted(set(r["d_intrinsic"] for r in results))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(d_intrinsic_vals)))

for d_i, color in zip(d_intrinsic_vals, colors):
    rows = [r for r in results if r["d_intrinsic"] == d_i]
    rows.sort(key=lambda r: r["rho_true"])
    rhos = [r["rho_true"] for r in rows]
    aris = [r["ari_mean"] for r in rows]
    stds = [r["ari_std"] for r in rows]

    axes[0].plot(rhos, aris, marker='o', label=f"$d_i={d_i}$", color=color)
    axes[0].fill_between(rhos,
                          np.array(aris) - np.array(stds),
                          np.array(aris) + np.array(stds),
                          alpha=0.15, color=color)

axes[0].axvline(x=10, color='red', linestyle='--', alpha=0.7, label=r'$\rho^* = 10$')
axes[0].set_xlabel(r"Stretching Ratio $\rho = d_a / d_i$", fontsize=12)
axes[0].set_ylabel("k-means ARI", fontsize=12)
axes[0].set_title("Clustering Performance vs Stretching Ratio", fontsize=13)
axes[0].legend(fontsize=9)
axes[0].set_xscale('log')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.05, 1.05)

# ── 2. Failure rate vs rho ────────────────────────────────────────────────────
for d_i, color in zip(d_intrinsic_vals, colors):
    rows = [r for r in results if r["d_intrinsic"] == d_i]
    rows.sort(key=lambda r: r["rho_true"])
    rhos = [r["rho_true"] for r in rows]
    failures = [r["failure_rate"] for r in rows]
    axes[1].plot(rhos, failures, marker='s', label=f"$d_i={d_i}$", color=color)

axes[1].axvline(x=10, color='red', linestyle='--', alpha=0.7, label=r'$\rho^* = 10$')
axes[1].set_xlabel(r"Stretching Ratio $\rho = d_a / d_i$", fontsize=12)
axes[1].set_ylabel("Failure Rate (ARI < 0.5)", fontsize=12)
axes[1].set_title("Failure Rate vs Stretching Ratio", fontsize=13)
axes[1].legend(fontsize=9)
axes[1].set_xscale('log')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig("figure1_ari_vs_rho.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved figure1_ari_vs_rho.png")

# ── 3. Fit scaling law: ARI ~ alpha * rho^(-beta) ────────────────────────────
def power_law(rho, alpha, beta, gamma):
    return alpha * np.array(rho)**(-beta) + gamma

fig2, ax = plt.subplots(figsize=(8, 5))

print("\nScaling law fits (ARI ~ alpha * rho^(-beta) + gamma):")
print(f"{'d_i':>5} {'alpha':>8} {'beta':>8} {'gamma':>8} {'rho* (est)':>12}")

for d_i, color in zip(d_intrinsic_vals, colors):
    rows = [r for r in results if r["d_intrinsic"] == d_i and r["rho_true"] >= 1]
    rows.sort(key=lambda r: r["rho_true"])
    rhos = np.array([r["rho_true"] for r in rows], dtype=float)
    aris = np.array([r["ari_mean"] for r in rows])

    try:
        popt, _ = curve_fit(power_law, rhos, aris,
                            p0=[0.8, 0.5, 0.05],
                            bounds=([0, 0, 0], [2, 5, 1]),
                            maxfev=5000)
        alpha, beta, gamma = popt

        # Estimate rho* as where ARI drops below 0.5
        rho_star = ((alpha / (0.5 - gamma)) ** (1/beta)) if (0.5 - gamma) > 0 else np.nan

        print(f"{d_i:>5} {alpha:>8.3f} {beta:>8.3f} {gamma:>8.3f} {rho_star:>12.2f}")

        rho_smooth = np.logspace(np.log10(rhos[0]), np.log10(rhos[-1]), 200)
        ax.plot(rho_smooth, power_law(rho_smooth, *popt),
                '--', color=color, alpha=0.8)
        ax.scatter(rhos, aris, color=color, label=f"$d_i={d_i}$", zorder=5)

    except Exception as e:
        print(f"{d_i:>5} fit failed: {e}")

ax.axvline(x=10, color='red', linestyle=':', alpha=0.7, label=r'$\rho^* = 10$')
ax.set_xlabel(r"Stretching Ratio $\rho$", fontsize=12)
ax.set_ylabel("k-means ARI", fontsize=12)
ax.set_title(r"Power Law Fit: ARI $\sim \alpha \cdot \rho^{-\beta} + \gamma$", fontsize=13)
ax.set_xscale('log')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figure2_scaling_law.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved figure2_scaling_law.png")

