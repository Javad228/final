import numpy as np

# Dataset
# Features (x1, x2) and labels (y)
data = np.array([
    [0, 2, 1],
    [1, 3, 1],
    [3, 1, -1],
    [1, 1, 1],
    [2, -0.5, -1],
    [2, -1, -1],
    [0, 0, 1],
    [1, -1, -1]
])

X = data[:, :2]  # Features (x1, x2)
y = data[:, 2]   # Labels (+1 or -1)

# Initialize weights and bias
w = np.zeros(2)  # Weights (w1, w2)
b = 0            # Bias
learning_rate = 1
max_iterations = 1000  # Stop after a maximum number of iterations

# Perceptron algorithm
for _ in range(max_iterations):
    errors = 0
    for i in range(len(X)):
        # Compute the prediction
        prediction = np.sign(np.dot(w, X[i]) + b)
        
        # Update if misclassified
        if y[i] * (np.dot(w, X[i]) + b) <= 0:
            w += learning_rate * y[i] * X[i]
            b += learning_rate * y[i]
            errors += 1
    # Stop if no misclassifications
    if errors == 0:
        break

# Normalize the weights
norm = np.linalg.norm(w)
w_normalized = w / norm
b_normalized = b / norm

# Output the normalized weights and bias
print(f"Normalized weights: {w_normalized}")
print(f"Normalized bias: {b_normalized}")

# Margin computation
margin = 2 / norm  # Margin for a correctly classified perceptron
print(f"Margin: {margin}")

# Decision boundary function
def decision_boundary(x1):
    return -(w_normalized[0] * x1 + b_normalized) / w_normalized[1]

# Plotting the dataset and the normalized decision boundary
import matplotlib.pyplot as plt

# Positive and negative points
positive = data[data[:, 2] == 1][:, :2]
negative = data[data[:, 2] == -1][:, :2]

plt.scatter(positive[:, 0], positive[:, 1], color='blue', label='+1')
plt.scatter(negative[:, 0], negative[:, 1], color='red', label='-1')

# Plot decision boundary
x1 = np.linspace(-1, 4, 100)
x2 = decision_boundary(x1)
plt.plot(x1, x2, color='green', label='Normalized Decision Boundary')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)
plt.title('Normalized Perceptron Decision Boundary')
plt.show()
