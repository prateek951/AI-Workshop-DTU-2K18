import numpy as np
import matplotlib.pyplot as plt


mean_01 = np.array([1,0.5])
cov_01 = np.array([[1,0.1],[0.1,1.2]])

mean_02 = np.array([4,5])
cov_02 = np.array([[1,.1],[.1,1.2]])

#Trace the normal distributions
dist_01 = np.random.multivariate_normal(mean_01,cov_01,500)
dist_02 = np.random.multivariate_normal(mean_02,cov_02,500)

print(dist_01.shape,dist_02.shape)

plt.figure(0)
plt.xlim(-5,10)
plt.ylim(-5,10)

plt.scatter(dist_01[:,0],dist_02[:,1])
plt.scatter(dist_02[:,0],dist_02[:,1])
plt.show()

r = dist_01.shape[0] + dist_02.shape[0]
c = dist_01.shape[1] + 1
data = np.zeros((r,c))
print(data.shape)

data[:dist_01.shape[0],:2] = dist_01
data[dist_02.shape[0]:,:2] = dist_02 
data[dist_01.shape[0]:,-1] = 1.0

print(data.mean(axis=0))

np.random.shuffle(data)
print(data[:10])

# Method to compute the euclidean distance between the two points

def distance(x1,x2):
    d = np.sqrt(((x1-x2)**2).sum())
    return d 

def knn(X_train,Y_train,xt,k=7):
    vals = []
    for ix in range(X_train.shape[0]):
        d = distance(X_train[ix],xt)
        vals.append([d,Y_train[ix]])
    sorted_labels = sorted(vals,key=lambda z: z[0])
    neighbours = np.asarray(sorted_labels)[:k,-1]

    freq = np.unique(neighbours,return_counts=True)

    return freq[0][freq[1].argmax()]

test_point = np.array([8,-4])

print(knn(data[:,:2],data[:,-1],test_point))

np.random.shuffle(data)
split = int(.75*data.shape[0])

#Print the split
train_data_X = data[:split,:2]
train_data_y = data[:split,-1]
test_data_X = data[split:,:2]
test_data_y = data[:split,-1]

print(train_data_X.shape,train_data_y.shape)
print(test_data_X.shape,test_data_y.shape)

def get_acc(kx):
    preds = []
    # print kx
    for ix in range(test_data_X.shape[0]):
        preds.append(knn(train_data_X, train_data_y, test_data_X[ix], k=kx))
    preds = np.asarray(preds)
    
    # print preds.shape
    return 100*float((test_data_y == preds).sum())/preds.shape[0]

print(get_acc(7))

for ix in range(0,20):
    print("k:",ix,"| Accuracy:",get_acc(ix))

import pandas as pd 
import datetime 

df = pd.read_csv('../train.csv')
print(df.head())

data = df.values[:2000]
print(data.shape)

# Again split the dataset into the training set and the test set
split = int(0.8* data.shape[0])

X_train = data[:split,1:]
X_test = data[split:,1:]

Y_train = data[:split,0]
Y_test = data[split:,0]

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

plt.figure(0)
plt.imshow(X_train[90].reshape((28,28)),cmap='gray',interpolation='none')
print(Y_train[90])
plt.show()

# Print the accuracy for the model

def get_accuracy(kx):
    preds = []
    for ix in range(X_test.shape[0]):
        start = datetime.datetime.now()
        preds.append(knn(X_train,Y_train,X_test[ix],k=kx))
    preds = np.asarray(preds)

    # prints preds.shape
    return 100* float((Y_test == preds).sum())/preds.shape[0]
print(get_accuracy(kx=15))




