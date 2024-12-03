
# Operations on arrays
import numpy as np

#Mathematical operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise operations
print(a + b)  # [5 7 9]
print(a * b)  # [4 10 18]

# Broadcasting
c = np.array([[1], [2], [3]])
print(c + a)  # Adds `a` to each row of `c`

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.sum())        # 21
print(arr.mean())       # 3.5
print(arr.min(axis=0))  # Min along columns: [1 2 3]
