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

# iris.columnName will give that column
print(iris.sl)

# Display rows 1 and 2 & columns 1 and 2
print(iris.iloc[1:3,1:3])
# Display the 0th row
print(iris.iloc[0:1,:])

# Find the mean of the column sl
print(iris.sl.mean())
# Find the sim of the columns sl
print(iris.sl.sum())


#NOTE - Pandas make internally the use of numpy

#Putting the null/nan values between the specified indices
iris.iloc[1:3,1:2] = np.nan

# Print the top 5 rows
print(iris.head())

#Check for isNull
print(pd.isnull(iris))

iris.sw.fillna(1,inplace=True)
#Fill null entries with 1 inplace=True means changes are made on the same table
iris.iloc[1:3,1:2] = np.nan
iris.sw.fillna(iris.sw.mean(),inplace=True)
#Fill null entries with sw's column mean value
print(iris.head())

Y = iris['flower_class']
print(Y)


# Deleting a particular column from the dataset
del iris['flower_class']
# after first run, flower class already deleted
print(iris.head())

iris.iloc[1:3,:] = np.nan;
# remove rows with column values having  nan
iris.dropna(inplace=True)

# Print the first five rows of the dataset
print(iris.head())

# Resetting the index
iris.reset_index()
a = iris.reset_index()
print(a)
print(iris.head())

# Drops extra columng now created after reset_index
iris.reset_index(drop=True,inplace=True)
print(iris.head())
