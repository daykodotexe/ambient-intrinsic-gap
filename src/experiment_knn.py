import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from data_generator import generate_dataset, estimate_intrinsic_dim_twonn
import json

def run_knn_experiment(
    d_intrinsic_list=[2, 3, 5, 8, 10],
    rho_list=[1, 2, 5, 10, 20, 30, 50, 75, 100],
    n_samples=600,
    n_trials=5
):
    results = []

    for d_i in d_intrinsic_list:
        for rho in rho_list:
            d_a = int(d_i * rho)
            if d_a < d_i:
                continue

            clf_scores, reg_scores = [], []

            for trial in range(n_trials):
                # Generate data
                X, X_low = generate_dataset(
                    n_samples=n_samples,
                    d_intrinsic=d_i,
                    d_ambient=d_a,
                    noise=0.5,
                    random_state=trial * 99
                )
                X = StandardScaler().fit_transform(X)

                # Classification labels — 4 classes based on quadrant in intrinsic space
                y_clf = (X_low[:, 0] > 0).astype(int) * 2 + (X_low[:, 1] > 0).astype(int)

                # Regression target — smooth function of intrinsic coords
                y_reg = np.sin(X_low[:, 0]) + np.cos(X_low[:, 1])

                # kNN classification accuracy
                clf = KNeighborsClassifier(n_neighbors=5)
                clf_acc = cross_val_score(clf, X, y_clf, cv=3, scoring='accuracy').mean()
                clf_scores.append(clf_acc)

                # kNN regression R²
                reg = KNeighborsRegressor(n_neighbors=5)
                reg_r2 = cross_val_score(reg, X, y_reg, cv=3, scoring='r2').mean()
                reg_scores.append(reg_r2)

            print(f"d_i={d_i}, d_a={d_a:4d}, rho={rho:5.1f} | "
                  f"clf_acc={np.mean(clf_scores):.3f} | "
                  f"reg_r2={np.mean(reg_scores):.3f}")

            results.append({
                "d_intrinsic": d_i,
                "d_ambient": d_a,
                "rho_true": rho,
                "clf_acc_mean": np.mean(clf_scores),
                "clf_acc_std": np.std(clf_scores),
                "clf_failure": np.mean([s < 0.6 for s in clf_scores]),
                "reg_r2_mean": np.mean(reg_scores),
                "reg_r2_std": np.std(reg_scores),
                "reg_failure": np.mean([s < 0.3 for s in reg_scores])
            })

    return results


if __name__ == "__main__":
    print("Running kNN classification + regression experiment...\n")
    results = run_knn_experiment()

    with open("results_knn.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_knn.json")