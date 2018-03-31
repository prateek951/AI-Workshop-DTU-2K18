import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

# Importing the datasets 

dataset = datasets.load_iris()

# Matrix of features
X = dataset.data
Y = dataset.target

# Training set for the features matrix
X_train  = X[0:-10]
print(X_train.shape)

# Training set for the target dependent vector
Y_train = Y[0:-10]
print(Y_train.shape)

# Testing set for the features matrix
X_test = X[-10:]
print(X_test.shape)

# Testing set for the target dependent vector
Y_test = Y[-10:]
print(Y_test)
