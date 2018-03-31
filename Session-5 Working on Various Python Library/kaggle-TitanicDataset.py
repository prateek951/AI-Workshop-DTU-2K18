import pandas as pd 

dataset = pd.read_csv('train.csv')

print(dataset['Sex'])

men_in_titanic = dataset[dataset['Sex'] == 'male']
print(men_in_titanic)

women_in_titanic = dataset[dataset['Sex'] == 'female']
print(women_in_titanic)

persons_who_survived = dataset[dataset.Survived == 1]
print(persons_who_survived)

print(dataset.groupby('Sex').Survived.mean())

men_survived = float(sum(men_in_titanic.Survived)/len(men_in_titanic))

print(men_survived)