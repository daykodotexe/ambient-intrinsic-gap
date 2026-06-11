import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from data_generator import generate_dataset, estimate_intrinsic_dim_twonn

def run_experiment(
    d_intrinsic_list=[2, 3, 5, 8, 10],
    rho_list=[1, 2, 5, 10, 20, 30, 50, 75, 100],
    n_samples=600,
    n_clusters=3,
    n_trials=5
):
    """
    For each (d_intrinsic, rho) pair, generate data and measure k-means ARI.
    Returns results as a list of dicts.
    """
    results = []

    for d_i in d_intrinsic_list:
        for rho in rho_list:
            d_a = int(d_i * rho)
            if d_a < d_i:
                continue

            aris = []
            rho_estimates = []

            for trial in range(n_trials):
                # Generate data with n_clusters distinct blobs
                X_parts = []
                labels_true = []
                for c in range(n_clusters):
                    X_part, _ = generate_dataset(
                        n_samples=n_samples // n_clusters,
                        d_intrinsic=d_i,
                        d_ambient=d_a,
                        noise=0.5,
                        random_state=trial * 100 + c
                    )
                    # Offset each cluster in ambient space
                    offset = np.zeros(d_a)
                    offset[c % d_a] = 3.0
                    X_parts.append(X_part + offset)
                    labels_true.extend([c] * (n_samples // n_clusters))

                X = np.vstack(X_parts)
                labels_true = np.array(labels_true)

                # Run k-means
                km = KMeans(n_clusters=n_clusters, random_state=trial, n_init=10)
                labels_pred = km.fit_predict(X)

                ari = adjusted_rand_score(labels_true, labels_pred)
                aris.append(ari)

                # Estimate rho from data
                d_est = estimate_intrinsic_dim_twonn(X)
                rho_est = d_a / d_est if d_est > 0 else np.nan
                rho_estimates.append(rho_est)

            results.append({
                "d_intrinsic": d_i,
                "d_ambient": d_a,
                "rho_true": rho,
                "rho_estimated": np.mean(rho_estimates),
                "ari_mean": np.mean(aris),
                "ari_std": np.std(aris),
                "failure_rate": np.mean([a < 0.5 for a in aris])
            })

            print(f"d_i={d_i}, d_a={d_a:4d}, rho={rho:5.1f} | "
                  f"ARI={np.mean(aris):.3f} ± {np.std(aris):.3f} | "
                  f"failure={np.mean([a < 0.5 for a in aris]):.2f}")

    return results


if __name__ == "__main__":
    print("Running experiment...\n")
    results = run_experiment()

    # Save results
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results.json")