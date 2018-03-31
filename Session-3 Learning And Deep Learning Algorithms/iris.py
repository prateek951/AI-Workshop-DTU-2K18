from sklearn import datasets

#Load the dataset

dataset = datasets.load_iris()

# To check the shape for the dataset

print(dataset.data.shape)

# To check the keys for a dictionary
print(dataset.keys())
print(dataset.target)
print(dataset.target_names)
print(dataset.target_names[0])
print(dataset.target_names[1])

# The features of the iris dataset
print(dataset.feature_names)

