import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# Predefined points
points = np.array([
    [1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5],
    [4.5, 5], [3.5, 4.5], [8, 7], [8, 8], [7, 8],
    [5, 8], [6, 7], [7, 6], [2, 2], [2.5, 2.5]
])

# User-defined initial medoids
initial_indices = [0, 7, 10]  # Indices of points to be used as initial medoids
medoids = points[initial_indices]
print("User-defined Initial Medoids:")
print(medoids)

# K-Medoids algorithm
k = len(medoids)
max_iterations = 10

for iteration in range(max_iterations):
    print(f"\nIteration {iteration + 1}:")
    
    # Step 1: Assign points to the nearest medoid
    distances = pairwise_distances(points, medoids, metric='euclidean')
    labels = np.argmin(distances, axis=1)
    
    print("Distances to Medoids:")
    print(distances)
    print("Cluster Assignments:")
    print(labels)
    
    # Step 2: Compute total cost and attempt medoid swaps
    current_cost = sum([distances[i, labels[i]] for i in range(len(points))])
    print(f"Current Cost: {current_cost}")
    
    best_medoids = medoids.copy()
    best_cost = current_cost
    
    for i in range(len(medoids)):
        for j in range(len(points)):
            if np.array_equal(points[j], medoids[i]):
                continue
            
            # Swap medoid i with point j
            temp_medoids = medoids.copy()
            temp_medoids[i] = points[j]
            
            # Compute new cost
            temp_distances = pairwise_distances(points, temp_medoids, metric='euclidean')
            temp_labels = np.argmin(temp_distances, axis=1)
            temp_cost = sum([temp_distances[idx, temp_labels[idx]] for idx in range(len(points))])
            
            if temp_cost < best_cost:
                best_cost = temp_cost
                best_medoids = temp_medoids.copy()
                print(f"Better Medoids Found: {best_medoids} with Cost: {best_cost}")
    
    if np.array_equal(medoids, best_medoids):
        print("\nNo Better Medoids Found. Convergence Reached!")
        break
    medoids = best_medoids

# Final Results
print("\nFinal Medoids:")
print(medoids)

# Plot the results
plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'orange']
for i in range(k):
    cluster_points = points[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i + 1}')
plt.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='x', s=200, label='Medoids')
plt.title('K-Medoids Clustering with User-Defined Initial Medoids')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
