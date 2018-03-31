import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Housing_Data.csv')

print(dataset.head())
print(dataset.shape)

X = dataset['lotsize']
Y = dataset['price']

# Reshape X and Y since X and Y are vectors
X  = X.reshape(len(X),1)
Y = Y.reshape(len(Y),1)

X_train = X[:-250]
Y_train = Y[:-250]

X_test = X[-250:]
Y_test = Y[-250:]

plt.scatter(X_train,Y_train,color='blue',marker='*')
plt.title('Linear Regression Model')
plt.xlabel('Lotsize')
plt.ylabel('Price')
plt.show()

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
print(regressor.predict(5000))

# Predictions of price for the lotsize in the test set
plt.scatter(X_test,regressor.predict(X_test))
plt.xlabel('Lotsize for the testing set')
plt.ylabel('Price for the testing set')
plt.plot(X_test,regressor.predict(X_test),color='red')
plt.show()