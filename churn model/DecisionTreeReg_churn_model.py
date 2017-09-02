import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import plotly.tools as tls
import mysql.connector
import pandas as pd
import seaborn as sns
import itertools
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset_for_model import x_train, y_train,x_test,y_test, dataset, clean_dataset
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import  PCA
#from sklearn.model_selection import  cross_val_score
#from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import  cross_val_score
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn import metrics
import plotly.graph_objs as go
import seaborn as sns
from bokeh.charts import Scatter, output_file, show
from os import system


'----------- creation of the model ------------'
# Decision Tree regression
test_score_results = []
training_score_results = []
predictions = []
depth = [1,2,3,4,5,6,7,8,9,10,20,30,40,50]

for i in depth:
    regr_1 = DecisionTreeRegressor(max_depth= i)
    regr_1.fit(x_train, y_train)
    a = regr_1.score(x_train, y_train)
    training_score_results.append(a)
    g = regr_1.predict(x_test)
    b = regr_1.score(x_test, y_test)
    test_score_results.append(b)


#plot the decision tree graph
regr_2 = DecisionTreeRegressor(max_depth= 10)
regr_2.fit(x)
regr_2.predict(x)
feature_names = list(dataset.columns.values)
export_graphviz(regr_2, out_file = "churntree.dot", feature_names=feature_names)
system("dot -Tpng churntree.dot > churntree.png")
system("gvfs-open churntree.png")

print training_score_results
print test_score_results



#plot of the train and test results
plt.plot(depth, training_score_results)
plt.plot(depth, test_score_results)
plt.legend(['Train Score', 'Test Score'], loc='upper right')
plt.title("Graph for Model Accuracy")
plt.xlabel("tree_depth")
plt.ylabel("Accuracy")
plt.show()



