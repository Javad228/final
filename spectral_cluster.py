import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

def spectral_clustering(X, k, sigma=1.0):
    """
    Perform Spectral Clustering.

    Parameters:
    - X: ndarray of shape (n_samples, n_features), input data points.
    - k: int, number of clusters.
    - sigma: float, scaling parameter for the Gaussian similarity function.

    Returns:
    - labels: ndarray of shape (n_samples,), cluster labels for each point.
    """
    # Step 1: Form the affinity matrix A
    n_samples = X.shape[0]
    A = np.zeros((n_samples, n_samples))
    pairwise_distances = euclidean_distances(X, X)
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                A[i, j] = np.exp(-pairwise_distances[i, j]**2 / (2 * sigma**2))

    # Step 2: Define the diagonal degree matrix D
    D = np.diag(A.sum(axis=1))

    # Step 3: Compute the normalized Laplacian matrix L
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
    L = D_inv_sqrt @ A @ D_inv_sqrt

    # Step 4: Compute the k largest eigenvectors of L
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
    top_k_indices = sorted_indices[:k]
    E = eigenvectors[:, top_k_indices]

    # Step 5: Normalize each row of E to have unit length
    E_normalized = E / np.linalg.norm(E, axis=1, keepdims=True)

    # Step 6: Perform k-means clustering on the rows of E
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(E_normalized)

    return labels

# Example usage:
# Define a set of points (e.g., 2D data points)
X = np.array([
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [8.0, 8.0],
    [1.0, 0.6],
    [9.0, 11.0],
    [8.0, 2.0],
    [10.0, 2.0],
    [9.0, 3.0]
])

# Perform spectral clustering with 2 clusters
k = 2
labels = spectral_clustering(X, k)
print("Cluster labels:")
print(labels)
