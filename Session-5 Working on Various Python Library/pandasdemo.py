import pandas as pd 

# Creating a series of elements
c = pd.Series([1,2,3,4,5])
# Creating a dataframe of elements
d = pd.DataFrame([[1,1,2,2],[2,2,2,2]])
e = pd.DataFrame({"a" : [1,2,2],"b" : [7,8,9]})

print(c)
print(d)
print(e["a"][0:3])

# To load the csv file using the pandas
# df = pd.read_excel("Here goes the path for the file that is to be read")

# To print the dataframe shape
# print(df.shape)
# Print the top 5 elements of the dataset
# print(df.head()) 
# print(df["column-name"].mean())
# print(df["column-name"].sum())
# print(df["column-name"].min())
# print(df["column-name"].max())


a = [1,2,21,2,2,32,323,4]

# Slicing in arrays
a = a[4:8]
# Here 4th included and 8th excluded
print(a)

# To begin slicing from 2nd index till last
a = a[2:]
print(a)
a = a[:-1]
print(a)

# Display the whole array
b = [12,2,1,1,42,43,21,423,23]
print(b[:])

import numpy as np
n = np.array(b)
print(n)
#Print the elements from index 1 to index2
print(n[1:3])


# Machine learning is basically is using the past data, to predict the future. The machine learning algorithm decides what importance/ weight to be given to variour factors, while deciding answer,yes or no, eg: spam or not, based on the past data/results. eg: spam data- contact or not, spam keywords

# CSV: A file with a format a,b,c,d .

# Pandas is a package. It is a much faster method for slicing an array, deleting rows/columns.

iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
# iris is the dataset eg: titanic dataset
# dataset can be accessed locally,or using a url
print(iris.head())

# This will print the first five rows of table generally
# iris.head(20) prints the first 20 rows
print(iris.tail())
# iris.tail() prints the last five rows of the table

#no header, therefore 0th row is not known as header, indexing starts from 1st row

# To see only the headings
print(iris.columns)
iris.columns = ['sl','sw','pl','pw','flower_class']
print(iris.columns)

# To print the shape of the dataset
print(iris.shape)

# Print the list of the datatypes
print(iris.dtypes)

# Display the calculations about the columns
print(iris.describe())

#Check on every column of every row if there is a null
# and show true or false if there is a null

print(pd.isnull(iris))

#Columnwise sum of the null values
print(pd.isnull(iris).sum())