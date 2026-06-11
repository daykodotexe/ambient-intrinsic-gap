import json
import numpy as np
from scipy.stats import spearmanr

with open("results.json") as f:
    results = json.load(f)

# For each result, we have: d_ambient, d_intrinsic, rho_true, ari_mean
# We want to compare how well each metric predicts ARI

d_a_vals    = np.array([r["d_ambient"]    for r in results], dtype=float)
d_i_vals    = np.array([r["d_intrinsic"]  for r in results], dtype=float)
rho_vals    = np.array([r["rho_true"]     for r in results], dtype=float)
ari_vals    = np.array([r["ari_mean"]     for r in results])

# Spearman correlation (rank-based, no linearity assumption)
corr_da,  p_da  = spearmanr(d_a_vals,  ari_vals)
corr_di,  p_di  = spearmanr(d_i_vals,  ari_vals)
corr_rho, p_rho = spearmanr(rho_vals,  ari_vals)

# Also test log(rho) since we expect a power law
corr_lrho, p_lrho = spearmanr(np.log(rho_vals), ari_vals)

print("Spearman correlation with ARI (higher |r| = better predictor)")
print("-" * 60)
print(f"Ambient dimension  d_a:       r={corr_da:+.3f}  p={p_da:.2e}")
print(f"Intrinsic dim      d_i:       r={corr_di:+.3f}  p={p_di:.2e}")
print(f"Stretching ratio   rho:       r={corr_rho:+.3f}  p={p_rho:.2e}")
print(f"Log stretching     log(rho):  r={corr_lrho:+.3f}  p={p_lrho:.2e}")

print("\nR-squared (linear fit of each predictor vs ARI)")
print("-" * 60)

from numpy.polynomial import polynomial as P

for name, x in [("d_a", d_a_vals), ("d_i", d_i_vals), ("rho", rho_vals), ("log(rho)", np.log(rho_vals))]:
    coeffs = np.polyfit(x, ari_vals, 1)
    predicted = np.polyval(coeffs, x)
    ss_res = np.sum((ari_vals - predicted)**2)
    ss_tot = np.sum((ari_vals - np.mean(ari_vals))**2)
    r2 = 1 - ss_res / ss_tot
    print(f"{name:>12}:  R²={r2:.3f}")

print("\nConclusion:")
print("rho should have highest |r| and R² — that's your key claim.")