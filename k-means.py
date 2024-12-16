import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# Predefined points
points = np.array([
    [0,1], [0,-1], [1,0], [-1,1], [-2,0],
    [-1,-1]
])

# User-defined initial centroids (not necessarily from the data points)
initial_centroids = np.array([
    [0, 1],  # Arbitrary point
    [-1, 1],  # Arbitrary point
])

print("User-defined Initial Centroids:")
print(initial_centroids)

# K-Means algorithm
k = len(initial_centroids)
centroids = initial_centroids
max_iterations = 10

for iteration in range(max_iterations):
    print(f"\nIteration {iteration + 1}:")
    
    # Step 1: Assign points to the nearest cluster
    distances = pairwise_distances(points, centroids, metric='euclidean')
    labels = np.argmin(distances, axis=1)
    
    print("Distances to Centroids:")
    print(distances)
    print("Cluster Assignments:")
    print(labels)
    
    # Step 2: Update centroids as the mean of points in each cluster
    new_centroids = []
    for i in range(k):
        cluster_points = points[labels == i]
        if len(cluster_points) > 0:
            new_centroid = cluster_points.mean(axis=0)
        else:
            new_centroid = centroids[i]  # Keep old centroid if no points in cluster
        new_centroids.append(new_centroid)
        print(f"Cluster {i + 1} Points:")
        print(cluster_points)
        print(f"Updated Centroid {i + 1}: {new_centroid}")
    
    new_centroids = np.array(new_centroids)
    
    # Check for convergence
    if np.allclose(centroids, new_centroids):
        print("\nConvergence Reached!")
        break
    centroids = new_centroids

# Final Results
print("\nFinal Centroids:")
print(centroids)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-Means Clustering with Arbitrary Initial Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
