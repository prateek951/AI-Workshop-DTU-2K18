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
