import numpy as np

# Indexing and Slicing

# Indexing:
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[0, 1])  # Element at row 0, column 1: 2

# Slicing:
arr = np.array([0, 1, 2, 3, 4, 5])
print(arr[1:4])  # [1 2 3]
print(arr[:3])   # [0 1 2]
print(arr[::2])  # [0 2 4]

# Boolean masking
arr = np.array([1, 2, 3, 4, 5])
print(arr[arr > 3])  # [4 5]
