import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB




number_of_predictors = [1,2,3,4]
test_score_results = []

for i in number_of_predictors:
    iris = datasets.load_iris()
    x = iris.data[: , :i]
    y = iris.target


    x_train , x_test, y_train, y_test =  train_test_split(x ,y , test_size= 0.3 , random_state= 2)
    mlt= MultinomialNB()
    print mlt

    mlt.fit(x_train,y_train)
    mlt.predict(x_test)
    b = mlt.score(x_test, y_test)
    test_score_results.append(b)



plt.plot(number_of_predictors, test_score_results)
plt.legend(['Test Score'], loc='upper right')
plt.title("Graph for Model Accuracy")
plt.xlabel("number of predictors")
plt.ylabel("Accuracy")
plt.show()
