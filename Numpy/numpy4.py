import numpy as np

# Reshaping Arrays
arr = np.arange(1, 10)
reshaped = arr.reshape(3, 3)

# Stacking and Splitting
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Stacking:
vstack = np.vstack([a, b])  # Vertical stack
hstack = np.hstack([a, b])  # Horizontal stack

# Splitting:
split = np.array_split(a, 3)
print(split)

# Linear Algebra
matrix = np.array([[1, 2], [3, 4]])
vector = np.array([1, 0])

dot_product = np.dot(matrix, vector)  # Matrix-vector multiplication
transpose = matrix.T  # Transpose of the matrix
inverse = np.linalg.inv(matrix)  # Inverse of the matrix

print(dot_product)
print(transpose)
print(inverse)