import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from data_generator import generate_dataset, estimate_intrinsic_dim_twonn
import json

def participation_ratio(X):
    """
    Participation Ratio: measures effective dimensionality from eigenspectrum.
    PR = (sum eigenvalues)^2 / sum(eigenvalues^2)
    """
    cov = np.cov(X.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    pr = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
    return pr

def explained_variance_ratio_50(X):
    """
    Number of PCA components needed to explain 50% variance.
    Higher = more complex structure.
    """
    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumvar, 0.50) + 1
    return float(n_components)

def condition_number(X):
    """
    Condition number of covariance matrix.
    High condition number = ill-conditioned = harder for algorithms.
    """
    cov = np.cov(X.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return eigenvalues[-1] / eigenvalues[0]

def run_baseline_experiment(
    d_intrinsic_list=[2, 3, 5, 8, 10],
    rho_list=[1, 2, 5, 10, 20, 30, 50, 75, 100],
    n_samples=600,
    n_clusters=3,
    n_trials=3
):
    records = []

    for d_i in d_intrinsic_list:
        for rho in rho_list:
            d_a = int(d_i * rho)
            if d_a < d_i:
                continue

            aris, prs, evrs, cns, rho_ests = [], [], [], [], []

            for trial in range(n_trials):
                X_parts, labels_true = [], []
                for c in range(n_clusters):
                    X_part, _ = generate_dataset(
                        n_samples=n_samples // n_clusters,
                        d_intrinsic=d_i,
                        d_ambient=d_a,
                        noise=0.5,
                        random_state=trial * 100 + c
                    )
                    offset = np.zeros(d_a)
                    offset[c % d_a] = 3.0
                    X_parts.append(X_part + offset)
                    labels_true.extend([c] * (n_samples // n_clusters))

                X = np.vstack(X_parts)
                X = StandardScaler().fit_transform(X)
                labels_true = np.array(labels_true)

                # Algorithm performance
                km = KMeans(n_clusters=n_clusters, random_state=trial, n_init=10)
                ari = adjusted_rand_score(labels_true, km.fit_predict(X))
                aris.append(ari)

                # Complexity measures
                prs.append(participation_ratio(X))
                evrs.append(explained_variance_ratio_50(X))
                cns.append(np.log(condition_number(X)))  # log scale

                # Rho estimated
                sub = np.random.choice(len(X), min(500, len(X)), replace=False)
                d_est = estimate_intrinsic_dim_twonn(X[sub])
                rho_ests.append(d_a / d_est if d_est > 0 else np.nan)

            records.append({
                "d_intrinsic": d_i,
                "d_ambient": d_a,
                "rho_true": rho,
                "rho_estimated": np.nanmean(rho_ests),
                "ari_mean": np.mean(aris),
                "participation_ratio": np.mean(prs),
                "evr_50": np.mean(evrs),
                "log_condition_number": np.mean(cns),
            })

            print(f"d_i={d_i}, rho={rho:5.1f} | "
                  f"ARI={np.mean(aris):.3f} | "
                  f"PR={np.mean(prs):.1f} | "
                  f"EVR50={np.mean(evrs):.1f} | "
                  f"logCN={np.mean(cns):.2f}")

    with open("results_baselines.json", "w") as f:
        json.dump(records, f, indent=2)
    print("\nSaved results_baselines.json")
    return records


if __name__ == "__main__":
    print("Running baseline comparison experiment...\n")
    records = run_baseline_experiment()

    # ── Correlation analysis ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PREDICTOR COMPARISON (Spearman r with ARI)")
    print("="*60)

    aris    = np.array([r["ari_mean"]              for r in records])
    d_a     = np.array([r["d_ambient"]             for r in records], dtype=float)
    d_i     = np.array([r["d_intrinsic"]           for r in records], dtype=float)
    rho     = np.array([r["rho_true"]              for r in records], dtype=float)
    pr      = np.array([r["participation_ratio"]   for r in records])
    evr     = np.array([r["evr_50"]                for r in records])
    lcn     = np.array([r["log_condition_number"]  for r in records])

    predictors = {
        "d_a (ambient dim)":          d_a,
        "d_i (intrinsic dim)":        d_i,
        "rho (stretching ratio)":     rho,
        "log(rho)":                   np.log(rho),
        "Participation Ratio":        pr,
        "EVR-50 (PCA components)":    evr,
        "log(Condition Number)":      lcn,
    }

    from numpy.polynomial import polynomial as P
    print(f"\n{'Predictor':<30} {'Spearman r':>12} {'p-value':>12} {'R²':>8}")
    print("-" * 66)

    for name, x in predictors.items():
        r, p = spearmanr(x, aris)
        coeffs = np.polyfit(x, aris, 1)
        pred = np.polyval(coeffs, x)
        ss_res = np.sum((aris - pred)**2)
        ss_tot = np.sum((aris - np.mean(aris))**2)
        r2 = 1 - ss_res / ss_tot
        print(f"{name:<30} {r:>+12.3f} {p:>12.2e} {r2:>8.3f}")