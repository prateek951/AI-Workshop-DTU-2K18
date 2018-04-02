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


