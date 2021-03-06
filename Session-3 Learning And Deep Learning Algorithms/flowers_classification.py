import pandas as pd

iris_data = pd.read_csv('iris_data.csv')
print(iris_data.head())

# Handle the missing data if any
iris_data = pd.read_csv('iris_data.csv',na_values=['NA'])

print(iris_data.head())

print(iris_data.describe())

import matplotlib.pyplot as plt
import seaborn as sb 

# Temporarily drop the rows with 'NA' values because seaborn package does
# not know what to do with them

print(sb.pairplot(iris_data.dropna(),hue="class"))


# Now as can be seen dataset had some errors where in Iris-versicolor was written as versicolor
# In other case there was a typo mistake in interpreting setosa as setossa
iris_data.loc[iris_data['class']== 'versicolor','class']="Iris-versicolor"
iris_data.loc[iris_data['class']== 'Iris-setossa','class'] = 'Iris-setosa'

print(iris_data['class'].unique())

# Researchers said that for iris setosa it is impossible to have a sepal width below 2.5 cm
# Remove such entries

iris_data = iris_data.loc[(iris_data['class'] != 'Iris-sertosa') | (iris_data['sepal_width_cm'] >= 2.5)]
print(iris_data.loc[iris_data['class']== 'Iris-sertosa','sepal_width_cm'].hist())

# If some entries were in metres convert those entries into centimetres
iris_data.loc[(iris_data['class'] == 'Iris-versicolor') * (iris_data['sepal_length_cm'] < 1.0),'sepal_length_cm']*= 100.0 
print(iris_data[iris_data['class']== 'Iris-versicolor','sepal_length_cm'].hist())


# Take a look at the rows with missing data
print(iris_data.loc[(iris_data['sepal_length_cm'].isnull()) | (iris_data['sepal_width_cm'].isnull()) | (iris_data.loc['petal_length_cm'].isnull()) | iris_data.loc[iris_data['petal_length_cm'].isnull()]])


# To handle missing data we can use the mean imputation process

print(iris_data[iris_data['class']=='Iris-setosa','petal_width_cm'].hist())

# Most of the petal widths for iris-setosa fall within the range of .2 to .3
# let us fill these entries with average measured total petal width
average_petal_width = iris_data[iris_data['class']=="Iris-setosa",'petal_width_cm'].mean()
iris_data.loc[(iris_data['petal_width_cm'].isnull()),'petal_width_cm'] = average_petal_width

print(iris_data.loc[(iris_data['class'=='Iris-setosa']) & (iris_data['petal_width_cm'] == average_petal_width)])

# Check for whether you still have null values in the mined analysed scrapped dataset
print(iris_data.loc[(iris_data['sepal_length_cm'].isnull()) | (iris_data['sepal_width_cm'].isnull()) | (iris_data['petal_length_cm'].isnull()) | (iris_data['petal_width_cm'].isnull())])

# Note: If you don't feel comfortable imputing your data, you can drop all rows with missing data with the dropna() call:

# iris_data.dropna(inplace=True)

print(sb.pairplot(iris_data,hue='class'))

print(sb.pairplot(iris_data))

# Data here is normally distributed for most of the part

plt.figure(figsize=(10,10))

for index,column in enumerate(iris_data.columns):
    if column=='class':
        continue 
    plt.subplot(2,2,index+1)
    print(sb.violinplot(x='class',y=column,data=iris_data))


# Begin with exploratory analysis

all_features = iris_data[['sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm']].values

all_classes = iris_data['class'].values

print(all_classes[:5])

# Now our dataset is ready for the split
from sklearn.cross_validation import train_test_split
(training_inputs,testing_inputs,training_classes,testing_classes) = train_test_split(all_inputs,all_classes,train_size=0.75,random_state=1)

# Using the scale invariant decision tree classifier for the given dataset
from sklearn.tree import DecisionTreeClassifier

# Create the classifier
dt_classifier = DecisionTreeClassifier()

# Train the classifier on the training set
dt_classifier.fit(training_inputs,training_classes)

# Validate the classifier on the testing set based on the accuracy of classification
print(dt_classifier.score(testing_inputs,testing_classes))

model_accuracies = []

for i in range(1000):
    (training_inputs,training_classes,testing_inputs,testing_classes) = train_test_split(all_inputs,all_classes,test_size=0.75)

    # DECISION TREE CLASSIFIER ALGORITHM
       # Create the decision tree classifier
     decision_tree_classifier = DecisionTreeClassifier()
       # Train the classifier on the training set
       decision_tree_classifier.fit(training_inputs,training_classes)
       # Validate the classifier on the test set based on the accuracy of classification
       accuracy = decision_tree_classifier(testing_inputs,testing_classes)
        model_accuracies.append(accuracy)

print(sb.distplot(model_accuracies))


# This problem is the main reason that most data scientists perform k-fold cross-validation on their models: Split the original data set into k subsets, use one of the subsets as the testing set, and the rest of the subsets are used as the training set. This process is then repeated k times such that each subset is used as the testing set exactly once.

# Performing the 10 fold cross validation on the dataset
import numpy as np 
from sklearn.cross_validation import StratifiedKFold

def plot_cv(cv,n_samples):
    masks=[]
    for train,test in cv:
        mask = np.zeros(n_samples,dtype=bool)
        mask[test] = 1
        masks.append(mask)
    plt.figure(figsize=(15,15))
    plt.imshow(masks,interpolation=None)
    plt.ylabel('Fold')
    plt.xlabel('Row #')
plot_cv(StratifiedKFold(all_classes,n_folds=10),len(all_classes))

# Perform 10 fold cross validation code on our model with the following code

from sklearn.cross_validation import cross_val_score

decision_tree_classifier = DecisionTreeClassifier()

# Get the list of scores to get a reasonable estimate of the classifier's performance

cv_scores = cross_val_score(decision_tree_classifier,all_inputs,all_classes,cv=10)
print(sb.distplot(cv_scores))
plt.title('Average Score : {}'.format(np.mean(cv_scores)))













