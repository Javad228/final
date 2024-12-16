import numpy as np

def compute_centered_gram_matrix(X):
    """
    Compute the centered Gram matrix B = H * X * X.T * H, where H is the centering matrix.

    Parameters:
    - X: ndarray of shape (n_samples, n_features).

    Returns:
    - B: Centered Gram matrix.
    """
    n_samples = X.shape[0]
    H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples  # Centering matrix
    B = H @ X @ X.T @ H  # Compute centered Gram matrix
    return B

def compute_low_dimensional_embedding(B, k):
    """
    Compute the k-dimensional embedding from the centered Gram matrix.

    Parameters:
    - B: Centered Gram matrix.
    - k: Target dimensionality.

    Returns:
    - Y: Low-dimensional embedding.
    """
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select top-k eigenvalues and corresponding eigenvectors
    lambda_k = eigenvalues[:k]
    u_k = eigenvectors[:, :k]

    # Compute embedding
    Y = u_k @ np.diag(np.sqrt(lambda_k))

    return Y

# Input dataset
X = np.array([
    [0.66, 0.68, 0.66],
    [0.04, 0.76, 0.17],
    [0.85, 0.74, 0.71],
    [0.93, 0.39, 0.03]
])

# Compute centered Gram matrix
B = compute_centered_gram_matrix(X)

# Compute 1-dimensional embedding
k = 1
Y = compute_low_dimensional_embedding(B, k)

# Results
print("Centered Gram matrix (B):")
print(B)
print("\n1-dimensional embedding (Y):")
print(Y)
