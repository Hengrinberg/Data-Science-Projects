import matplotlib.pyplot as plt
from sklearn import datasets
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.charts import Scatter, output_file, show

########## data exploration ###############
sns.set(style="ticks", color_codes=True)
ris = sns.load_dataset("diabetes")
g = sns.pairplot(ris)
plt.show()

diabetes = datasets.load_diabetes()

# print df.describe()
data = diabetes.data
df = pd.DataFrame(data)


print df.describe()

'----------- creation of the model ------------'

diabetes = datasets.load_diabetes()

test_score_results = []
training_score_results = []
depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

x = diabetes.data
y = diabetes.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

for i in depth:
    # for k in range(1):
    regr_1 = DecisionTreeRegressor(max_depth=i)
    regr_1.fit(x_train, y_train)
    a = regr_1.score(x_train, y_train)
    training_score_results.append(a)
    g = regr_1.predict(x_test)
    b = regr_1.score(x_test, y_test)
    test_score_results.append(b)

tree.export_graphviz(regr_1, out_file="tree.dot")

print training_score_results
print test_score_results
