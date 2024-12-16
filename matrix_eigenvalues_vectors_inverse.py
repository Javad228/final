import numpy as np

def matrix_operations(matrix):
    """
    Perform matrix operations: determinant, inverse, eigenvalues, and eigenvectors.
    
    Parameters:
        matrix (numpy.ndarray): Input square matrix.

    Returns:
        dict: A dictionary with determinant, inverse, eigenvalues, and eigenvectors.
    """
    results = {}

    # Determinant
    det = np.linalg.det(matrix)
    results["Determinant"] = det

    # Inverse
    if det != 0:
        inverse = np.linalg.inv(matrix)
        results["Inverse"] = inverse
    else:
        results["Inverse"] = "Matrix is singular (no inverse)."

    # Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    results["Eigenvalues"] = eigenvalues
    results["Eigenvectors"] = eigenvectors

    return results

# Example usage
if __name__ == "__main__":
    # Define a matrix
    matrix = np.array([[9, 7.5],
                       [7.5, 25]])
    
    # Perform matrix operations
    results = matrix_operations(matrix)

    # Print results
    print("Matrix:")
    print(matrix)
    print("\nDeterminant:")
    print(results["Determinant"])
    print("\nInverse:")
    print(results["Inverse"])
    print("\nEigenvalues:")
    print(results["Eigenvalues"])
    print("\nEigenvectors:")
    print(results["Eigenvectors"])
