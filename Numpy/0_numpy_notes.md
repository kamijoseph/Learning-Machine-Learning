> > > NumPy (Numerical Python) is a powerful Python library for numerical computing. It provides support for multi-dimensional arrays, matrices, and mathematical operations on these data structures, making it essential for data science, machine learning, and scientific computing.

> > > NumPy Basics.
> > > Key Features:

    Multi-dimensional arrays (ndarrays)
    Broadcasting for element-wise operations
    Vectorized operations for speed

> > > Performance and Memory

    NumPy arrays consume less memory.
    Operations are faster because they use optimized C code.

> > > Real-world Applications:

    Data Preprocessing: Normalize or standardize datasets.
    Signal Processing: Perform Fast Fourier Transforms.
    Machine Learning: Use in algorithms like gradient descent.

> > > METHODS IN NUMPY

> > > Array Creation
> > > np.array(): Create an ndarray from a list, tuple, or other sequence.
> > > np.arange(): Create an array with a range of values (like range() in Python).
> > > np.linspace(): Create an array of evenly spaced numbers over a specified interval.
> > > np.zeros(): Create an array filled with zeros.
> > > np.ones(): Create an array filled with ones.
> > > np.empty(): Create an uninitialized array.
> > > np.eye(): Create a 2D identity matrix.
> > > np.random.rand(): Create an array of random numbers from a uniform distribution between 0 and 1.
> > > np.random.randn(): Create an array of random numbers from a standard normal distribution.
> > > np.random.randint(): Create an array of random integers.

> > > Array Attributes
> > > ndarray.shape: Return the dimensions (shape) of an array.
> > > ndarray.size: Return the total number of elements in an array.
> > > ndarray.ndim: Return the number of dimensions (axes) of an array.
> > > ndarray.dtype: Return the data type of the elements of an array.

> > > Indexing and Slicing
> > > ndarray[index]: Access a specific element by index.
> > > ndarray[start:end]: Slice an array (similar to Python lists).
> > > ndarray[start:end:step]: Slice with a step value.
> > > ndarray[::-1]: Reverse the array.
> > > np.where(): Return elements based on a condition (masking).
> > > np.take(): Return elements at specified indices.
> > > np.nonzero(): Return indices of non-zero elements.

> > > Mathematical Operations
> > > np.add(): Add elements.
> > > np.subtract(): Subtract elements.
> > > np.multiply(): Multiply elements.
> > > np.divide(): Divide elements.
> > > np.sqrt(): Square root of elements.
> > > np.exp(): Exponential function.
> > > np.log(): Natural logarithm.
> > > np.sin(), np.cos(), np.tan(): Trigonometric functions.
> > > np.abs(): Absolute value of elements.
> > > np.mod(): Element-wise modulus.

> > > Aggregation Functions
> > > np.sum(): Sum of elements.
> > > np.mean(): Mean of elements.
> > > np.median(): Median of elements.
> > > np.std(): Standard deviation.
> > > np.var(): Variance of elements.
> > > np.min(): Minimum element.
> > > np.max(): Maximum element.
> > > np.argmin(): Index of the minimum element.
> > > np.argmax(): Index of the maximum element.
> > > np.cumsum(): Cumulative sum.
> > > np.cumprod(): Cumulative product.

> > > Reshaping and Manipulation
> > > ndarray.reshape(): Change the shape of an array.
> > > ndarray.T: Transpose of an array (swap rows and columns).
> > > np.flatten(): Flatten an array into one dimension.
> > > np.ravel(): Flatten an array into one dimension (returns a view).
> > > np.concatenate(): Join arrays along a specified axis.
> > > np.vstack(): Stack arrays vertically (row-wise).
> > > np.hstack(): Stack arrays horizontally (column-wise).
> > > np.split(): Split an array into multiple sub-arrays.
> > > np.tile(): Repeat the array.

> > > Linear Algebra Functions
> > > np.dot(): Dot product of two arrays (matrix multiplication).
> > > np.matmul(): Matrix multiplication (similar to np.dot()).
> > > np.linalg.inv(): Inverse of a matrix.
> > > np.linalg.det(): Determinant of a matrix.
> > > np.linalg.eig(): Eigenvalues and eigenvectors of a matrix.
> > > np.linalg.svd(): Singular Value Decomposition.
> > > np.linalg.norm(): Norm of a vector or matrix.

> > > Random Sampling
> > > np.random.seed(): Set the random seed for reproducibility.
> > > np.random.choice(): Generate a random sample from a given array.
> > > np.random.shuffle(): Shuffle the elements of an array in-place.
> > > np.random.permutation(): Return a permuted array.

> > > Miscellaneous
> > > np.unique(): Return sorted unique elements of an array.
> > > np.diff(): Compute the difference between consecutive elements.
> > > np.tile(): Construct an array by repeating an input array.
> > > np.concatenate(): Join a sequence of arrays along an existing axis.
> > > np.repeat(): Repeat elements of an array.

> > > Working with NaN and Infinite Values
> > > np.isnan(): Test element-wise if values are NaN.
> > > np.isinf(): Test element-wise if values are infinite.
> > > np.nanmean(): Compute the mean while ignoring NaN values.
> > > np.nanstd(): Compute the standard deviation while ignoring NaN values.
> > > np.nanmin(): Find the minimum while ignoring NaN values.

> > > File Input and Output
> > > np.loadtxt(): Load data from a text file.
> > > np.genfromtxt(): Load data from a text file, handling missing values.
> > > np.save(): Save an array to a binary file.
> > > np.load(): Load an array from a binary file.
