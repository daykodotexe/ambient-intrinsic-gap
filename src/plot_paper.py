import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

# ── Load data ─────────────────────────────────────────────────────────────────
with open("results.json") as f:
    results = json.load(f)
with open("results_knn.json") as f:
    results_knn = json.load(f)
with open("results_heavytail.json") as f:
    results_ht = json.load(f)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150
})

colors_di = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
d_intrinsic_vals = [2, 3, 5, 8, 10]

def power_law(rho, alpha, beta, gamma):
    return alpha * np.array(rho)**(-beta) + gamma

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Main result: ARI vs rho + scaling law fit
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for d_i, color in zip(d_intrinsic_vals, colors_di):
    rows = sorted([r for r in results if r["d_intrinsic"] == d_i],
                  key=lambda r: r["rho_true"])
    rhos = np.array([r["rho_true"] for r in rows])
    aris = np.array([r["ari_mean"] for r in rows])
    stds = np.array([r["ari_std"] for r in rows])

    axes[0].plot(rhos, aris, marker='o', label=f"$d_i={d_i}$", color=color, linewidth=1.8)
    axes[0].fill_between(rhos, aris - stds, aris + stds, alpha=0.12, color=color)

    # Fit and plot power law
    try:
        rows_fit = [r for r in rows if r["rho_true"] >= 2]
        rhos_f = np.array([r["rho_true"] for r in rows_fit])
        aris_f = np.array([r["ari_mean"] for r in rows_fit])
        popt, _ = curve_fit(power_law, rhos_f, aris_f,
                            p0=[0.8, 0.5, 0.05],
                            bounds=([0,0,0],[2,5,1]), maxfev=5000)
        rho_smooth = np.logspace(np.log10(2), np.log10(100), 300)
        axes[1].plot(rho_smooth, power_law(rho_smooth, *popt),
                     '--', color=color, alpha=0.85, linewidth=1.6)
        axes[1].scatter(rhos_f, aris_f, color=color, s=30,
                        label=f"$d_i={d_i}$", zorder=5)
    except:
        pass

for ax in axes:
    ax.axvline(x=5, color='red', linestyle=':', alpha=0.6, linewidth=1.5, label=r"$\rho^*=5$")
    ax.set_xscale('log')
    ax.set_xlabel(r"Stretching Ratio $\rho = d_a / d_i$", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, framealpha=0.7)

axes[0].set_ylabel("k-means ARI", fontsize=12)
axes[0].set_title("(a) Clustering Failure vs Stretching Ratio", fontsize=12)
axes[1].set_ylabel("k-means ARI", fontsize=12)
axes[1].set_title(r"(b) Power Law Fit: ARI $\sim \alpha\rho^{-\beta}$", fontsize=12)

plt.tight_layout()
plt.savefig("fig1_main_result.pdf", bbox_inches='tight')
plt.savefig("fig1_main_result.png", bbox_inches='tight')
print("Saved fig1_main_result")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Algorithm comparison: k-means vs kNN clf vs kNN reg
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

algo_configs = [
    (results,     "ari_mean",      "ari_std",     "k-means ARI",       "(a) Clustering (k-means)",      "Fails"),
    (results_knn, "clf_acc_mean",  "clf_acc_std",  "kNN Accuracy",      "(b) Classification (kNN)",      "Improves"),
    (results_knn, "reg_r2_mean",   "reg_r2_std",   "kNN Regression R²", "(c) Regression (kNN)",          "Improves"),
]

for ax, (data, mean_key, std_key, ylabel, title, direction) in zip(axes, algo_configs):
    for d_i, color in zip(d_intrinsic_vals, colors_di):
        rows = sorted([r for r in data if r["d_intrinsic"] == d_i],
                      key=lambda r: r["rho_true"])
        rhos = np.array([r["rho_true"] for r in rows])
        means = np.array([r[mean_key] for r in rows])
        stds  = np.array([r[std_key]  for r in rows])

        ax.plot(rhos, means, marker='o', color=color,
                label=f"$d_i={d_i}$", linewidth=1.8, markersize=4)
        ax.fill_between(rhos, means - stds, means + stds, alpha=0.12, color=color)

    ax.axvline(x=5, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlabel(r"$\rho = d_a / d_i$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{title}\n→ {direction} with $\\rho$", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, framealpha=0.7)

plt.tight_layout()
plt.savefig("fig2_algorithm_comparison.pdf", bbox_inches='tight')
plt.savefig("fig2_algorithm_comparison.png", bbox_inches='tight')
print("Saved fig2_algorithm_comparison")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Heavy tail analysis
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

noise_configs = [
    ("gaussian", "Gaussian Noise",   "blue"),
    ("t_df5",    "Mild Heavy Tail\n(t, df=5)", "orange"),
    ("t_df2",    "Heavy Tail\n(t, df=2)",      "red"),
]

d_i_ht = [3, 5, 10]
colors_ht = plt.cm.plasma(np.linspace(0.2, 0.8, 3))

for ax, (noise_key, noise_label, _) in zip(axes, noise_configs):
    for d_i, color in zip(d_i_ht, colors_ht):
        rows = sorted([r for r in results_ht
                       if r["noise"] == noise_key and r["d_intrinsic"] == d_i],
                      key=lambda r: r["rho_true"])
        rhos = np.array([r["rho_true"] for r in rows])
        aris = np.array([r["ari_mean"] for r in rows])
        stds = np.array([r["ari_std"]  for r in rows])

        ax.plot(rhos, aris, marker='o', color=color,
                label=f"$d_i={d_i}$", linewidth=1.8, markersize=4)
        ax.fill_between(rhos, aris - stds, aris + stds, alpha=0.12, color=color)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.2)
    ax.set_xscale('log')
    ax.set_xlabel(r"$\rho = d_a / d_i$", fontsize=11)
    ax.set_ylabel("k-means ARI", fontsize=11)
    ax.set_title(noise_label, fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, framealpha=0.7)

plt.suptitle("Heavy-Tailed Noise Shifts Failure Threshold Earlier",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("fig3_heavytail.pdf", bbox_inches='tight')
plt.savefig("fig3_heavytail.png", bbox_inches='tight')
print("Saved fig3_heavytail")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Beta scaling law + predictor R² comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Beta vs d_i
d_vals   = np.array([2, 3, 5, 8, 10], dtype=float)
beta_vals = np.array([0.607, 0.711, 0.777, 0.813, 0.849])

def log_fit(d, a, b):
    return a * np.log(d) + b

popt, _ = curve_fit(log_fit, d_vals, beta_vals)
d_smooth = np.linspace(2, 10, 200)

axes[0].scatter(d_vals, beta_vals, color='steelblue', s=80, zorder=5, label="Fitted $\\beta$")
axes[0].plot(d_smooth, log_fit(d_smooth, *popt), '--', color='steelblue',
             linewidth=2, label=f"$\\beta = {popt[0]:.3f}\\ln(d_i) + {popt[1]:.3f}$")
axes[0].set_xlabel("Intrinsic Dimension $d_i$", fontsize=12)
axes[0].set_ylabel(r"Decay Exponent $\beta$", fontsize=12)
axes[0].set_title(r"(a) $\beta$ Scales Logarithmically with $d_i$", fontsize=12)
axes[0].legend(fontsize=10)

# Predictor R² bar chart
predictors = ["$d_a$", "$d_i$", r"$\rho$", r"$\log\rho$", "PR", "EVR-50", "logCN",
              r"$\log\rho$+PR", r"$\log\rho$+PR+EVR", "All"]
r2_vals = [0.147, 0.032, 0.203, 0.556, 0.321, 0.313, 0.351, 0.590, 0.605, 0.625]
bar_colors = ['#d62728' if 'rho' in p.lower() or 'log' in p.lower() or 'All' in p
              else '#aec7e8' for p in predictors]
bar_colors = ['#d62728' if i in [3, 7, 8, 9] else '#aec7e8'
              for i in range(len(predictors))]

bars = axes[1].barh(predictors, r2_vals, color=bar_colors, edgecolor='white', height=0.6)
axes[1].set_xlabel("R² with k-means ARI", fontsize=12)
axes[1].set_title("(b) Predictor Comparison", fontsize=12)
axes[1].set_xlim(0, 0.75)
axes[1].axvline(x=0.556, color='red', linestyle=':', alpha=0.6, linewidth=1.5)

for bar, val in zip(bars, r2_vals):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va='center', fontsize=9)

plt.tight_layout()
plt.savefig("fig4_beta_and_predictors.pdf", bbox_inches='tight')
plt.savefig("fig4_beta_and_predictors.png", bbox_inches='tight')
print("Saved fig4_beta_and_predictors")

print("\nAll figures saved. Open with:")
print("  start fig1_main_result.png")
print("  start fig2_algorithm_comparison.png")
print("  start fig3_heavytail.png")
print("  start fig4_beta_and_predictors.png")