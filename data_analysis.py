import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset

data = sns.load_dataset('iris')

#show first few rows of dataset
print(data.head())

#perform exploratory data analysis (eda):


print(data.describe())
#missingvaluesCheck:
print(data.isnull().sum())

#correlation between numerical values
print(data.drop(columns = ["species"]).corr())

#CREATE VISUALISATIONS:

#(i)Pairplot
sns.pairplot(data, hue = "species")
plt.show()

#(ii) Correlation Heatmap
sns.heatmap(data.drop(columns = ["species"]).corr(), annot = True , cmap = 'coolwarm')
plt.show()

#(iii) Boxplot

sns.boxplot(x = 'species', y = 'sepal_length', data = data)
plt.show()