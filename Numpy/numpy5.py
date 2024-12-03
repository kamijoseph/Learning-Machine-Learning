
#Comparing python lists to numpy arrays
import time
import numpy as np

size = 1000000
list1 = range(size)
list2 = range(size)
start = time.time()
result = [(x + y) for x, y in zip(list1, list2)]
print("Python list time:", time.time() - start)

arr1 = np.arange(size)
arr2 = np.arange(size)
start = time.time()
result = arr1 + arr2
print("NumPy array time:", time.time() - start)

# returns:
#     Python list time: 0.11965489387512207
#     NumPy array time: 0.02398538589477539