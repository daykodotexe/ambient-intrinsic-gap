import numpy as np
from sklearn.decomposition import PCA

def generate_dataset(n_samples=500, d_intrinsic=5, d_ambient=50, noise=0.01, random_state=42):
    """
    Generate a dataset with controlled intrinsic and ambient dimension.
    Embeds a d_intrinsic-dimensional structure into d_ambient-dimensional space.
    """
    rng = np.random.RandomState(random_state)
    
    # Generate low-dimensional structure
    X_low = rng.randn(n_samples, d_intrinsic)
    
    # Random projection matrix into ambient space
    projection = rng.randn(d_intrinsic, d_ambient)
    projection /= np.linalg.norm(projection, axis=0)  # normalize columns
    
    # Embed into ambient space
    X_high = X_low @ projection
    
    # Add small noise in ambient dimensions
    X_high += noise * rng.randn(n_samples, d_ambient)
    
    return X_high, X_low

def estimate_intrinsic_dim_twonn(X):
    """
    TwoNN estimator for intrinsic dimensionality.
    Facco et al. (2017)
    """
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # r1 and r2: distances to 1st and 2nd nearest neighbors
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    # Avoid division by zero
    valid = r1 > 0
    mu = r2[valid] / r1[valid]
    
    # MLE estimate
    d_hat = 1.0 / (np.mean(np.log(mu)))
    return d_hat

# Quick test
if __name__ == "__main__":
    X, X_low = generate_dataset(n_samples=500, d_intrinsic=5, d_ambient=50)
    d_est = estimate_intrinsic_dim_twonn(X)
    rho = X.shape[1] / d_est
    
    print(f"Ambient dimension:           {X.shape[1]}")
    print(f"True intrinsic dimension:    5")
    print(f"Estimated intrinsic dim:     {d_est:.2f}")
    print(f"Stretching ratio (rho):      {rho:.2f}")