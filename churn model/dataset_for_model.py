import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import plotly.tools as tls
import mysql.connector
import pandas as pd
import seaborn as sns
import itertools
from sklearn.preprocessing import  Imputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg


df_additional = pd.read_csv('additional_date_for_subscribers.csv')
#df_additional = pd.read_csv('C:\Users\Home Premium\Desktop\data_for_subscribers.csv')
dataset1 = pd.DataFrame(df_additional,
                        columns=['GK', 'Gender', 'State', 'PaymentType', 'product_id', 'GroupName', 'number_of_orders',
                                 'avg_order_total_price', 'avg_num_units', 'ratio', 'num_of_rows'])



#df_subscribers = pd.read_csv('C:\Users\Home Premium\Desktop\Notactive_subscribers.csv')
df_subscribers = pd.read_csv('subscribers_before_enrichment.csv')
subscribers_data = pd.DataFrame(df_subscribers,
                        columns=[ 'RatePlan', 'MonthlyFee', 'Market', 'tenure'])
print subscribers_data



#creating dataframe of all the subscribers that left the company
total_number_of_orders = dataset1["number_of_orders"].sum()
number_of_orders = dataset1["number_of_orders"].values
ratio = [i / float(total_number_of_orders)  for i in number_of_orders ]
num_of_rows = [round( i * 683264) for i in ratio]

#delete all non relevant columns
additional_data = dataset1.drop(dataset1.columns[[0,6,9,10]],axis=1)
#print additional_data
b = additional_data.values

# take each row in b and repeat it n times depend on the value of num_of_rows on each row
x= np.repeat(b ,num_of_rows,axis=0)
customers_additional_data = pd.DataFrame(x, columns=[ 'Gender', 'State', 'PaymentType', 'product_id', 'GroupName','avg_order_total_price','avg_num_units'])


#join 2 dataframes into one
subscribers_data = pd.concat([customers_additional_data, subscribers_data],axis=1, join='inner')
#print notactive_subscribers_data



print '---------------------------------------------------------------------------------------------------------'

print'################## DATA EXPLORATION ###################'
# the dataset
dataset =  subscribers_data

#basic statistics
print dataset.describe()

#replace all none values with the mode value
dataset['Gender'] = dataset['Gender'].fillna('M')
#print dataset

#get the mode value of state column in order to replace all none values with the mode value
#print dataset.mode()
dataset['State'] = dataset['State'].fillna('NY')


#missing data in the preidctors
print 'predictors missing data summary:'
total = dataset.isnull().sum().sort_values(ascending=False)
percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print missing_data.head(20)

#rename column names
clean_dataset = dataset.rename(columns={'avg_order_total_price': 'total_price', 'avg_num_units': 'num_of_units'})
#print dataset

#create dummy variables in catagorical columns
dataset = pd.get_dummies(clean_dataset, columns=['Gender','State','PaymentType','GroupName','RatePlan','Market'])
print dataset

print 'corolation matrix:'
corr_matrix = dataset.corr()
print corr_matrix



# basic statistics about the dataset
print ' data basic statistics: '
#print dataset.describe()


#splitting the dataset into training and test set
print dataset.shape
x = dataset.ix[:, dataset.columns != 'tenure'].values
y = dataset["tenure"].values
#print x.shape
#print y.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)

  

#feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
#print x_train , x_test




#comparison of the distribution of y_train and y_test 
# y_train
plt.hist(y_train)
plt.title("y_train distribution")
plt.xlabel("y_train Values")
plt.ylabel("Frequency")
plt.show()

# y_test
plt.hist(y_test)
plt.title("y_test distribution")
plt.xlabel("y_test Values")
plt.ylabel("Frequency")
plt.show()







