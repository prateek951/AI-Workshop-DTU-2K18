# @desc Working with Numpy

import numpy as np

# Creating the array using numpy

myarray = np.array([12,2,2,2,2,2])

print(myarray)

# The type of the array is np.ndarray whereas in python it would be list class
print(type(myarray))

# To print the order of the matrix
print(myarray.ndim)

# To print the shape of the array
print(myarray.shape)

# To print a specific element of the array
print(myarray[0])

# To print the last element of the array
# 2 ways of printing the last element of the array 

print(myarray[len(myarray) - 1]);
print(myarray[-1])


# Specific methods on the array that we can use

# type(myarray) returns the type of the array
# myarray.ndim returns the dimensions of the array
# myarray.itemsize returns the size of the datatype that we have used for the array
# myarray.size returns the size of the array
# myarray.dtype returns the datatype of the array

# For defining the 2 dimensional arraty using numpy

myarray_2d = np.array([[13132,32,323,23,23],[32,32,32,323,2]]);
print(myarray_2d)
print(myarray_2d.shape)

# Printing the dimensions of the array
print(myarray_2d.ndim)
print(len(myarray_2d))