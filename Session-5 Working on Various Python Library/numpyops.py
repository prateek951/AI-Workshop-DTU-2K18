my_list = [1,2,3]

# Importing the numpy library
import numpy as np 
arr = np.array(my_list)
arr

# Working with matrices 
my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
np.array(my_matrix)

# Create an array
np.arange(0,10)

# Create an array keeping the step size as 2
np.arange(0,10,2)

# Arrays of zeros
np.zeros(3)

# Array of zeros (3 rows and 3 columns)
np.zeros((3,3))

# Array of ones 
np.ones(3)

# Array of ones(3 rows and 3 columns)
np.ones((3,2))

# Linspace returns an evenly spaced numbers over a specified interval
np.linspace(0,5,10)

# Create an identity matrix
np.eye(4)
# Create an array of random numbers from a uniform distribution from 0 to 1
np.random.rand(5)

# Same thing but from a Normal Distibution centered around 0
np.random.randn(2)
np.random.randn(2,2)

np.random.randint(1,100,10)


arr = np.arange(25)
arr

ranarr = np.random.randint(0,50,10)
ranarr

# Reshape method to get the same data but in new shape
arr.reshape(5,5)


# Finding the maximum element in the array
ranarr.max()

# Finding the index location of the maximum element in the array
ranarr.argmax()

# Finding the index location of the minimum element
ranarr.argmin()

# Print the shape of vector
arr.shape

# Reshape the vector to a 5,5
arr.reshape(5,5)

arr.shape

# Print the datatyoe of the array
arr.dtype


from numpy.random import randint
randint(2,10)


# Indexing and Selection in Numpy array
newarr = np.arange(0,11)
newarr

newarr[8]
# Performing slicing to get the slice of an array
newarr[1:5]
newarr[0:5]
newarr[:6]
newarr[5:]

newarr[0:5] = 100
newarr

newarr = np.arange(0,11)
newarr

# Creating the slice of an array
slice_of_arr = arr[0:6]
slice_of_arr[:] = 99
slice_of_arr


# To create a copy of elements we can use the copy method
arr_copy = arr.copy()
arr
arr_copy[:] = 100
arr


# Indexing of a two dimensional array
arr_2d = np.array([[5,10,15],[20,25,30],[35,40,45]])
arr_2d

# Double bracket format
# Element in the first row and the first column
arr_2d[0][0]
# Using the single bracket format
arr_2d[1,1]
arr_2d[1,2]
arr_2d[2,1]
arr_2d[:2,1:]
# grab the first two rows
arr_2d[:2]
arr_2d[1:2]

# Conditional Selection
arr = np.arange(0,11)
arr

# This will give us a boolean vector
bool_arr = arr > 5
arr[bool_arr]

arr[arr>5]

arr[arr<3]

arr_2dim = np.arange(0,50).reshape(5,10)
arr_2dim

#Create the copy of the two dimensional array
arr_2dim_copy = arr_2dim.copy()
arr_2dim_copy

arr_2dim[2,1:5]
arr_2dim[1,3:5]
arr_2dim[2,3:5]