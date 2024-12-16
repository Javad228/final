import numpy as np
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_clusters, max_iter=100, tol=1e-6, epsilon=1e-6):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon  # Small value added to diagonal of covariance matrices

    def initialize_parameters(self, X):
        """
        Initialize parameters for the GMM (weights, means, covariances).
        """
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_clusters) / self.n_clusters
        self.means = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_clusters)])

    def e_step(self, X):
        """
        Expectation step: Compute responsibilities (gamma).
        """
        n_samples, n_features = X.shape
        self.responsibilities = np.zeros((n_samples, self.n_clusters))

        for k in range(self.n_clusters):
            self.responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covariances[k]
            )
        self.responsibilities /= self.responsibilities.sum(axis=1, keepdims=True)

    def m_step(self, X):
        """
        Maximization step: Update parameters (weights, means, covariances).
        """
        n_samples, n_features = X.shape
        for k in range(self.n_clusters):
            Nk = self.responsibilities[:, k].sum()
            self.weights[k] = Nk / n_samples
            self.means[k] = np.sum(self.responsibilities[:, k][:, np.newaxis] * X, axis=0) / Nk
            diff = X - self.means[k]
            cov_matrix = (self.responsibilities[:, k][:, np.newaxis, np.newaxis] *
                          np.einsum('ij,ik->ijk', diff, diff)).sum(axis=0) / Nk
            # Add small value to the diagonal to ensure positive definiteness
            self.covariances[k] = cov_matrix + self.epsilon * np.eye(n_features)

    def compute_log_likelihood(self, X):
        """
        Compute the log-likelihood of the current model.
        """
        log_likelihood = 0
        for k in range(self.n_clusters):
            log_likelihood += self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covariances[k]
            )
        return np.log(log_likelihood).sum()

    def fit(self, X):
        """
        Fit the GMM to the data using the EM algorithm.
        """
        self.initialize_parameters(X)
        log_likelihood_prev = None

        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            log_likelihood = self.compute_log_likelihood(X)
            if log_likelihood_prev is not None and abs(log_likelihood - log_likelihood_prev) < self.tol:
                print(f"Converged at iteration {iteration}")
                break
            log_likelihood_prev = log_likelihood

    def predict(self, X):
        """
        Assign each sample to the cluster with the highest responsibility.
        """
        self.e_step(X)
        return np.argmax(self.responsibilities, axis=1)


# Actual Sample Data
X = np.array([
    [1.0, 2.0], [1.5, 1.8], [2.0, 2.5], [6.0, 8.0], [6.5, 8.5], [7.0, 9.0],
    [3.0, 1.0], [3.5, 1.5], [3.0, 2.0], [9.0, 9.0], [9.5, 9.5], [10.0, 10.0]
])

# Initialize and fit GMM
gmm = GaussianMixtureModel(n_clusters=3)
gmm.fit(X)

# Predict cluster labels
labels = gmm.predict(X)

print("Cluster labels:")
print(labels)

print("\nCluster means (centroids):")
print(gmm.means)

print("\nCluster covariances:")
print(gmm.covariances)
