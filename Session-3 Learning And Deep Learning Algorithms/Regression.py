import numpy as np
import pandas as pd

dataset = pd.read_csv('Housing_Data.csv')

print(dataset.head())
print(dataset.shape)

X = dataset['lotsize']
Y = dataset['price']

# Reshape X and Y since X and Y are vectors

