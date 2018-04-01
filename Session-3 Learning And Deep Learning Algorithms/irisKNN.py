import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
 
from sklearn.datasets import load_iris

dataset = load_iris()

# Segregate the matrix of features and the target variable
features = dataset.data 
target = dataset.target 

from sklearn.cross_validation import train_test_split

X_train,Y_train,X_test,Y_test = train_test_split(features,target,test_size=0.3)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# Create classifier of KNN 
classifier = KNeighborsClassifier(n_neighbors=5)
# Fit the classifier on the training features and the target
classifier.fit(X_train,Y_train)
print(classifier.predict([5.8,2.7,4.1,1.0]))

# Compute the accuracy score
from sklearn.metrics import accuracy_score

pred = classifier.predict(X_test)

print(pred)
print(accuracy_score(Y_test,pred)*100)

from sklearn.linear_model import LogisticRegression

logClassifier = LogisticRegression()
logClassifier.fit(X_train,Y_train)

print(logClassifier.predict([5.8,2.7,4.1,1.0]))

# Predictions of the logistic classifier
logPred = logClassifier.predict(X_test)
print(accuracy_score(Y_test,logPred)*100)

import matplotlib.pyplot as plt
plt.plot(features,target,'ro',color='red')
plt.show()

