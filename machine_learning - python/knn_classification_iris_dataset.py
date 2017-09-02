import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.charts import Scatter, output_file, show

########## data exploration ###############
'''sns.set(style="ticks", color_codes=True)
ris = sns.load_dataset("iris")
g = sns.pairplot(ris, hue="species")'''

iris = datasets.load_iris()
data = iris.data
df = pd.DataFrame(data, columns=iris.feature_names)

print df.describe()
plt.show()

######### creation of the model ###############

# X = iris.data[:, :2]
# Y = iris.target

# print iris.data
# print X
# print iris.feature_names
# print iris.target
# print iris.target_names
test_score_results = []
num_of_neighbors = []
training_score_results = []

k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40]
for n in k:
    x = iris.data
    y = iris.target
    # print x.shape
    # print y.shape
    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=1, n_neighbors=n, p=2,
                               weights='uniform')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=4)  # split the data to training data and test data

    print '##################'

    knn.fit(x_train, y_train)
    b = knn.score(x_train, y_train)
    training_score_results.append(b)
    knn.predict(x_test)
    a = knn.score(x_test, y_test)
    test_score_results.append(a)
    num_of_neighbors.append(n)

print test_score_results
print num_of_neighbors

plt.plot(num_of_neighbors, training_score_results)
plt.plot(num_of_neighbors, test_score_results)
plt.legend(['Train Score', 'Test Score'], loc='upper right')
plt.title("Graph for Model Accuracy")
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.show()

'----------------------------------------------------------------------'

'knn model:'
'''
1) this model doesnt have really training proccess and there is no really decision boundary (its more imaginary) - lazy model

2) choose k neighbors

3) get a test point

4) calculate distance for each one of the training points from the test point

5) normalize the distances in order to make their weight equal

6) sort them 

7) choose k nearest neighbors to the test point

extension (parameters):

- we have two options for weights: uniform , distances
- p  - is the power of minkovski(q) and (1 - Manhattan , 2 - euclidean , more than 2 - minkovski) - punishment on long distances

- TIME COMPLEXITY : O(n^2) - this is the complexity of the worst case of quick sort

advantages:

1) simple to understand


disadvantages:

1) this is a lazy model
2) we don't know which predictor is the most important for the classification
3) bad time complexity because each time we should scan all the training points

* belongs to the decision rule classifiers and is differ from naive base which makes a classification based on highest probability'''
