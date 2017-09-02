import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



########## data exploration ###############
'''sns.set(style="ticks", color_codes=True)
ris = sns.load_dataset("iris")
g = sns.pairplot(ris, hue="species")'''


# importing the dataset
iris = datasets.load_iris()
data = iris.data
target = iris.target

print 'the number of values in the target column:'
print target.shape
print 'the number of values in each predictor column and the number of predictors:'
print data.shape

# print the target names
print 'target names:'
print iris.target_names

# transform the data into pandas dataframe
df_data = pd.DataFrame(data, columns = iris.feature_names )
df_target = pd.DataFrame(target )
print 'target values:'
print df_target
# show corolations between each two predictors
print 'corolation matrix:'
print df_data.corr()
# basic statistics about the dataset
print ' data basic statistics: '
print df_data.describe()

#missing data in the preidctors
print 'predictors missing data summary:'
total = df_data.isnull().sum().sort_values(ascending=False)
percent = (df_data.isnull().sum()/df_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print missing_data.head(20)

#missing data in the target
print 'target missing data summary:'
total = df_target.isnull().sum().sort_values(ascending=False)
percent = (df_target.isnull().sum()/df_target.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print missing_data.head(20)



#target variable distribution
ris = sns.load_dataset("iris")
sns.countplot(x="species", data= ris, palette="Greens_d");
plt.show()

#spliting the data into train and test
x_train , x_test, y_train, y_test =  train_test_split(data,target, test_size= 0.3 , random_state= 3)


# y_test distribution
df_train_target = pd.DataFrame(y_train)
#sns.countplot(x="species", data= y_train, palette="Greens_d");
df_train_target.hist()
plt.show()
