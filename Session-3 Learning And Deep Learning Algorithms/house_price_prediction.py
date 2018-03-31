from sklearn import datasets
import pandas as pd 

from sklearn.metrics import confusion_matrix as cm

# Load the dataset from the sklearn
dataset = datasets.load_boston()
print(dataset.data)
print(type(dataset.data))

# Shape of the data
print(dataset.data.shape)

df = pd.DataFrame(dataset.data)
print(df.head())
print(df.columns)

# DESCR the dataset
print(dataset.DESCR)

#Feature names in the dataset
print(dataset.feature_names)

print(dataset.target[0:5])

df.columns = dataset.feature_names 
print(df.head())
print(df.describe())

from sklearn.linear_model import LinearRegression

# Create the object for linear regression
# We will be giving training data to this object
regressor = LinearRegression()

from sklearn import cross_validation 
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(df,dataset.target,test_size=0.02)

# Fit the regressor to the training data
regressor.fit(X_train,Y_train)
# Prediction for the linear regressor object
# print(regressor.predict(dataset.data[0:5]))
print(regressor.predict(X_test))

#Print the actual output
# print(dataset.target[0:5])
print(Y_test)

# In previous case we performed training and testing on the same data
score = regressor.score(X_test,Y_test)
print(score)