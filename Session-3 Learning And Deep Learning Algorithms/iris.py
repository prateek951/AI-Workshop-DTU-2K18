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

# Perform the modelling now
from sklearn.neighbors import KNeighborsClassifier

# Create an instance of the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the KNN classifier to the training set
knn.fit(X_train,Y_train)

# Predict class for the new data point
print(knn.predict([[0.0,2.2,1.1,1.1]]))

# Predicted by the model on the test values
print(knn.predict(X_test))

# Predict for the values that are there in the dataset

print(knn.predict([[6.7,3.1,4.4,1.4]]))

# More predictionss
print(knn.predict([[5.7,2.5,5,2]]))
print(knn.predict([[5,3.6,1.4,.2]]))

