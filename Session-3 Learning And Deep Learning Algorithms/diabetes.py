from sklearn import datasets
import pandas as pd

from sklearn.metrics import confusion_matrix as cm
from sklearn.linear_model import LinearRegression

dataset = datasets.load_diabetes()

print(dataset.data)

print(type(dataset.data))
print(dataset.data.shape)

df = pd.DataFrame(dataset.data)

print(df.head())
print(df.columns)
print(df.dtypes)

# Split the dataset into the training set and the test set

from sklearn import cross_validation
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(df,dataset.data,test_size=.02)

regressor = LinearRegression()
regressor.fit(X_train,Y_train)
print(regressor.predict(X_test))
print(Y_test)

#Compute the score
score = regressor.score(X_test,Y_test)
print(score)


import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([12,14,18,22,28,34,40,50,60])

def fit(x,y):
    meanxy = (x*y).sum()/len(x*y)
    xmean = x.mean()
    ymean = y.mean()
    x2mean = (x*x).mean()
    m = (meanxy-xmean*ymean)/(x2mean-xmean**2)
    b= ymean-xmean*m
    return m,b    

m,b = fit(x,y)

def predict(m,b,x):
    ypredict = m*x + b
    return ypredict

ypredict = predict(m,b,x)
print(ypredict)

#Now begin with visualisation of the data

plt.scatter(x,y,marker="*")
plt.plot(x,ypredict)
plt.show()

# correlation coefficient with 2 variables/features 
#y=m1x1 + m2x2 + b

# m1 , m2 ->  coefficients  ,  b

x1=np.array([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4])
x2=np.array([5,5,5,5,10,10,10,10,15,15,15,15,20,20,20,20])
y=np.array([41,49,69,65,40,50,58,57,31,36,44,57,19,31,33,43])

def fit2(x1,x2,y):
    x1sq=(x1*x1).mean()
    x2sq=(x2*x2).mean()
    x1x2=(x1*x2).mean()
    x1mean=x1.mean()
    x2mean=x2.mean()
    y=y.mean()
    x1y=(x1*y).mean()
    x2y=(x2*y).mean()
    
    d=x1sq*(x2sq-x2mean**2)-x1x2*(x1x2-x1mean*x2mean)+x1mean*(x1x2*x2mean-x1mean*x2sq)
    dx=x1y*(x2sq-x2mean**2)-x2y*(x1x2-x1mean*x2mean)+y*(x1x2*x2mean-x1mean*x2sq)
    dy=x1sq*(x2y-x2mean*y)-x1x2*(x1y-x1mean*y)+x1mean*(x1y*x2mean-x1mean*x2y)
    dz=x1sq*(x2sq*y-x2mean*x2y)-x1x2*(x1x2*y-x1y*x2mean)+x1mean*(x1x2*x2y-x1y*x2sq)
    m1=dx/d
    m2=dy/d
    b=dz/d
    return m1,m2,b

m1,m2,b = fit2(x1,x2,y)
print(m1)

def predictY(m1,m2,x1,x2,b):
    return m1*x1+m2*x2+b 
y_predict = predictY(m1,m2,x1,x2,b)
print(y_predict)