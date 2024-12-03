import numpy as np

#Exercises:
#     Create a 4x4 matrix of random integers between 1 and 10.
#     Replace all even numbers in the matrix with -1.
#     Calculate the mean of each column in a given matrix.

matrix = np.random.randint(1, 11, size=(4, 4))
print("4x4 Matrix of random integers:")
print(matrix)

matrix[matrix %2 == 0] = -1
print(matrix)

column_means = matrix.mean(axis=0)
print("Mean of each column:")
print(column_means)