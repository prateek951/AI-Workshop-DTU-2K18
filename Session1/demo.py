# @desc Working with Numpy

import numpy as np

# Creating the array using numpy

myarray = np.array([12,2,2,2,2,2])

print(myarray)

# The type of the array is np.ndarray whereas in python it would be list class
print(type(myarray))

# To print the order of the matrix
print(myarray.ndim);

# To print the shape of the array
print(myarray.shape)

# To print a specific element of the array
print(myarray[0])