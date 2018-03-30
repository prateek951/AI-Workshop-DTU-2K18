# @desc Code for the knn classifier

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Matrix of features

fruit_feature = np.array([[1,9],[2,8],[3,7],[2,8],[6,4],[9,1],[8,2],[9,1],[1,1],[5,5]])

# Dependent vector

fruit_target = np.array(["Sour","Sour","Sour","Sour","Sweet","Sweet","Sweet","Sour","Sweet","Sweet"])

# Create the object for the knn classifier

myKNN = KNeighborsClassifier()

# Fit the model to the training data

myKNN.fit(fruit_feature,fruit_target)


print(fruit_feature)

# Using the predict method to predict the output

print(myKNN.predict([[3,7]]))
print(myKNN.predict([[1,2]]))


