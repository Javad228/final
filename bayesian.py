import numpy as np
from numpy.linalg import inv

def discriminant_function(x, mu, sigma):
    """
    Calculates the discriminant function for a given location.

    Args:
        x: Measurement vector (numpy array).
        mu: Mean vector for the location (numpy array).
        sigma: Covariance matrix (numpy array).

    Returns:
        The value of the discriminant function.
    """
    sigma_inv = inv(sigma)
    w = sigma_inv @ mu
    w0 = -0.5 * mu.T @ sigma_inv @ mu
    return w.T @ x + w0

# Problem Data
mu1 = np.array([-2, -1])
mu2 = np.array([1, 3])
sigma = np.array([[9, 7.5], [7.5, 25]])
x_measurement = np.array([0.5, -0.5])

# Calculate Discriminant Function Values
g1_x = discriminant_function(x_measurement, mu1, sigma)
g2_x = discriminant_function(x_measurement, mu2, sigma)

# Determine Location
if g1_x > g2_x:
    decision = 1
else:
    decision = 2

print(f"Discriminant function g1(X): {g1_x}")
print(f"Discriminant function g2(X): {g2_x}")
print(f"Decision: Location {decision}")