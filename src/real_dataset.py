import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data_generator import estimate_intrinsic_dim_twonn

print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_full, y_full = mnist.data, mnist.target.astype(int)

# Use a manageable subset
np.random.seed(42)
idx = np.random.choice(len(X_full), 3000, replace=False)
X, y = X_full[idx], y_full[idx]

# Normalize
X = StandardScaler().fit_transform(X)

print(f"Full MNIST subset: {X.shape}")

results = []

# Test at different ambient dimensions via PCA reduction
ambient_dims = [784, 200, 100, 50, 20, 10]

print(f"\n{'d_ambient':>10} {'d_i_est':>10} {'rho_est':>10} {'ARI':>8}")
print("-" * 45)

for d_a in ambient_dims:
    if d_a < 784:
        X_reduced = PCA(n_components=d_a).fit_transform(X)
    else:
        X_reduced = X.copy()

    # Estimate intrinsic dimension on a small subsample (TwoNN is slow on large N)
    sub_idx = np.random.choice(len(X_reduced), 500, replace=False)
    d_i_est = estimate_intrinsic_dim_twonn(X_reduced[sub_idx])

    rho_est = d_a / d_i_est

    # Run k-means with 10 clusters (MNIST digits 0-9)
    km = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels_pred = km.fit_predict(X_reduced)
    ari = adjusted_rand_score(y, labels_pred)

    print(f"{d_a:>10} {d_i_est:>10.2f} {rho_est:>10.2f} {ari:>8.3f}")

    results.append({
        "d_ambient": d_a,
        "d_i_est": d_i_est,
        "rho_est": rho_est,
        "ari": ari
    })

print("\nDone. Check if ARI drops as rho increases — that validates your synthetic findings on real data.")

import json
with open("mnist_results.json", "w") as f:
    json.dump(results, f, indent=2)