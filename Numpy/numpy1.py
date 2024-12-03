
# 1.Creating arrays from lists
import numpy as np

# From lists
arr = np.array([1, 2, 3])
print (arr)

# With specific ranges
zeros = np.zeros((2, 2))  # 2x2 array of zeros
ones = np.ones((3, 3))    # 3x3 array of ones
arange = np.arange(0, 10, 2)  # 0 to 10 with step 2
linspace = np.linspace(0, 1, 5)  # 5 values between 0 and 1

# Random numbers
rand = np.random.rand(3, 3)  # 3x3 matrix of random numbers between 0 and 1

# Print
print(zeros)
print(ones)
print(arange)
print(linspace)
print(rand)

#Array Attributes
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3)
print(arr.ndim)   # 2
print(arr.size)   # 6
print(arr.dtype)  # int64 (or int32 depending on the platform)
