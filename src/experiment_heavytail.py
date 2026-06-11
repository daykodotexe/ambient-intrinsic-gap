import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from data_generator import estimate_intrinsic_dim_twonn
from scipy.stats import t as t_dist
import json

def generate_heavytail_dataset(n_samples, d_intrinsic, d_ambient,
                                noise_std=0.5, tail_df=2, random_state=42):
    """
    Like generate_dataset but ambient noise is heavy-tailed (t-distribution).
    tail_df: degrees of freedom — lower = heavier tail (df=2 is very heavy)
    """
    rng = np.random.RandomState(random_state)

    X_low = rng.randn(n_samples, d_intrinsic)
    projection = rng.randn(d_intrinsic, d_ambient)
    projection /= np.linalg.norm(projection, axis=0)
    X_high = X_low @ projection

    # Heavy-tailed noise instead of Gaussian
    heavy_noise = t_dist.rvs(df=tail_df, size=(n_samples, d_ambient),
                              random_state=random_state)
    X_high += noise_std * heavy_noise

    return X_high, X_low


def run_heavytail_experiment(
    d_intrinsic_list=[3, 5, 10],
    rho_list=[1, 2, 5, 10, 20, 30, 50],
    n_samples=600,
    n_clusters=3,
    n_trials=5
):
    results = []

    # Test three noise types
    noise_configs = [
        ("gaussian",    "normal",  None),
        ("t_df5",       "mild",    5),
        ("t_df2",       "heavy",   2),
    ]

    for noise_name, noise_label, tail_df in noise_configs:
        print(f"\n--- Noise: {noise_label} ({noise_name}) ---")

        for d_i in d_intrinsic_list:
            for rho in rho_list:
                d_a = int(d_i * rho)
                if d_a < d_i:
                    continue

                aris = []
                for trial in range(n_trials):
                    X_parts, labels_true = [], []
                    for c in range(n_clusters):
                        if tail_df is None:
                            # Gaussian
                            from data_generator import generate_dataset
                            X_part, _ = generate_dataset(
                                n_samples=n_samples // n_clusters,
                                d_intrinsic=d_i,
                                d_ambient=d_a,
                                noise=0.5,
                                random_state=trial * 100 + c
                            )
                        else:
                            X_part, _ = generate_heavytail_dataset(
                                n_samples=n_samples // n_clusters,
                                d_intrinsic=d_i,
                                d_ambient=d_a,
                                noise_std=0.5,
                                tail_df=tail_df,
                                random_state=trial * 100 + c
                            )
                        offset = np.zeros(d_a)
                        offset[c % d_a] = 3.0
                        X_parts.append(X_part + offset)
                        labels_true.extend([c] * (n_samples // n_clusters))

                    X = np.vstack(X_parts)
                    labels_true = np.array(labels_true)
                    X = StandardScaler().fit_transform(X)

                    km = KMeans(n_clusters=n_clusters, random_state=trial, n_init=10)
                    labels_pred = km.fit_predict(X)
                    aris.append(adjusted_rand_score(labels_true, labels_pred))

                print(f"  d_i={d_i}, rho={rho:5.1f} | ARI={np.mean(aris):.3f} ± {np.std(aris):.3f}")

                results.append({
                    "noise": noise_name,
                    "d_intrinsic": d_i,
                    "rho_true": rho,
                    "ari_mean": np.mean(aris),
                    "ari_std": np.std(aris),
                    "failure_rate": np.mean([a < 0.5 for a in aris])
                })

    with open("results_heavytail.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_heavytail.json")
    return results


if __name__ == "__main__":
    run_heavytail_experiment()