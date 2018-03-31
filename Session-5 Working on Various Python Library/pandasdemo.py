
import pandas as pd 
# Creating a series of elements
c = pd.Series([1,2,3,4,5])
# Creating a dataframe of elements
d = pd.DataFrame([[1,1,2,2],[2,2,2,2]])
e = pd.DataFrame({"a" : [1,2,2],"b" : [7,8,9]})

print(c)
print(d)
print(e["a"][0:3])