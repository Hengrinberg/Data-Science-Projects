import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier

########## data exploration ###############


iris = datasets.load_iris()
data = iris.data
df = pd.DataFrame(data)

print df.describe()
plt.show()

######### creation of the model ###############
# implementation of Neural Networks algorithm on Iris dataset


test_score_results = []
training_score_results = []
activation = ['identity', 'logistic','tanh','relu']
max_iter = [10,20,100,200,1000]
num_of_neurons_in_hidden_layer = []
num_of_hidden_layers = []
dict_of_results = {}
list_of_dict_results = []


for i in range(1,11,1):
                x = iris.data
                y = iris.target
                # print x.shape
                # print y.shape
                clf = MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto',
                                    beta_1=0.9, beta_2=0.999, early_stopping=False,
                                    epsilon=1e-08, hidden_layer_sizes=(i, 1), learning_rate='constant',
                                    learning_rate_init=0.001, max_iter=200, momentum=0.9,
                                    nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                                    solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                                    warm_start=False)

                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                                    random_state=4)  # split the data to training data and test data


                num_of_neurons_in_hidden_layer.append(i)
                clf.fit(x_train, y_train)
                b = clf.score(x_train, y_train)
                training_score_results.append(b)
                dict_of_results['train score'] = b
                clf.predict(x_test)
                a = clf.score(x_test, y_test)
                test_score_results.append(a)
                dict_of_results['test score'] = a




plt.plot(num_of_neurons_in_hidden_layer, training_score_results)
plt.plot(num_of_neurons_in_hidden_layer, test_score_results)
plt.legend(['Train Score', 'Test Score'], loc='upper right')
plt.title("Graph for Model Accuracy")
plt.xlabel("number of neurons in hidden layer")
plt.ylabel("Accuracy")
plt.show()