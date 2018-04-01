import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create two classes one for the red points and one for the blue points
xRed = np.array([0.3,0.6,1,1.4,1.7,2])
yRed = np.array([1,4.5,2.3,1.9,8.9,4.1])

xBlue = np.array([3.3,3.5,4,4.5,5.7,6])
yBlue = np.array([7,1.5,6.3,1.9,2.9,7.1])

x = np.array([[0.3,1],[0.6,4.5],[1,2.3],[1.4,1.9],[1.7,8.9],[2,4.1],[3.3,7],[3.5,1.5],[4,6.3],[4.5,1.9],[5.7,2.9],[6,7.1]])
y = np.array([0,0,0,0,0,0,1,1,1,1,1,1])

plt.plot(xRed,yRed,'ro',color='red')
plt.plot(xBlue,yBlue,'ro',color='blue')
plt.plot(3,5,'ro',color='black')
plt.show()


from sklearn.neighbors import KNeighborsClassifier 

# Make the knn classifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='euclidean')
# Fit the knn classifier to the given training data
classifier.fit(x,y)

print(classifier.predict([3,5]))

# Let us define a new class which is the class for the points of the green color 

xGreen = np.array([0.3,.6,1,1.4,1.5,2])
yGreen = np.array([1,4.5,2.3,1.8,8.9,10.2])

x = np.array([[0.3,1],[0.6,4.5],[1,2.3],[1.4,1.9],[1.7,8.9],[2,4.1],
              [3.3,7],[3.5,1.5],[4,6.3],[4.5,1.9],[5.7,2.9],[6,7.1],
             [5.3,7],[5.9,1.5],[6,6.3],[6.5,1.9],[7.7,2.9],[8,7.1]])
y = np.array([0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2])

# Let us do the visualisation stuff
plt.plot(xRed,yRed,'ro',color='red')
plt.plot(xBlue,yBlue,'ro',color='blue')
plt.plot(xGreen,yGreen,'ro',color='green')
plt.plot(7,6,'ro',color='black')
plt.show()

from sklearn.neighbors import KNeighborsClassifier

classifier2 = KNeighborsClassifier(n_neighbors=3)

classifier2.fit(x,y)

print(classifier2.predict([7,6]))


