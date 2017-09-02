import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import plotly.tools as tls
import mysql.connector
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


df = pd.read_csv('C:\Users\Home Premium\Desktop\Social_Network_Ads.csv')
dataset = pd.DataFrame(df, columns = ['User ID','Gender','Age','EstimatedSalary','Purchased'] )
print dataset

x = dataset.iloc[:,[2,3]].values
print x
y = dataset.iloc[:,4].values
print y

# spliting the dataset
x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.3 , random_state = 0 )

# feature scaling

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
print  x_train
x_test = sc_x.transform(x_test)
print  x_test

# fitting logistic regression to the training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print y_pred

# making the confusion matrix
cm = confusion_matrix(y_test,y_pred)
print cm

