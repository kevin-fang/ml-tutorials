# KNN Walkthrough

First run `pip install numpy scipy sklearn` or `conda install numpy scipy scikit-learn` (if you have conda) to install the dependencies.

Line by line analysis:
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
```
This imports and loads the "K Nearest Neighbors Classifier" and the sample datasets. We generally use "X" to represent the input and "y" to represent the output, so we can write `f(X) = y`.

```python
clf = KNeighborsClassifier()
#clf = KNeighborsClassifier(n_neighbors=4)
```
In scikit-learn, all the classifiers are classes. The code above initializes the KNeighborsClassifier; you can specify parameters of the classifier in the parenthesis, (see commented line.)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
```
If we train on all of our data, then we have no new data to test on so we won't be able to tell how accurate our classifier is on information it hasn't seen before (which is kind of the point). Therefore, we use a "train-test-split" to split our data into a training set and a testing set.   
The `test_size` parameter tells the classifier the ratio of training data to testing data. `0.2` means that 20% of the data will be marked as testing data and 80% will be training data.

```python
clf.fit(X_train, y_train)
```
This is where the magic happens! In K-nearest-neighbors, all of the data is added to the graph. In more complicated ones, such as Stochastic Gradient Descent, it tries to minimize an error function using gradient descent.

```python
y_pred = clf.predict(X_test)
print accuracy_score(y_test, y_pred)
```
This tells the classifier to predict an output based on the test input. `accuracy_score(true_data, test_data)` will give you the ratio of correct versus incorrect data.

You should get an accuracy of around 90%. Once you do, either go [here to implement KNearestNeighbors yourself](https://kevin-fang.github.io/ml-tutorials/ScrappyKNN/) or try some other classifiers. Here are some examples of other classifiers - everything else will work the exact same, only the clf variable has to be changed:

```python
from sklearn.tree import DecisionTreeClasifier
clf = DecisionTreeClassifier() # a decision tree classifier

from sklearn.svm import SVC
clf = SVC() # a support vector classifier

from sklearn.linear_model import LinearRegression
clf = LinearRegression() # linear regression

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier() # stochastic gradient descent classifier
```

Note that some of these will require more computing power and time than others. In addition, they may not be perfectly accurate until the **hyperparameters** are updated. For example, in the K-nearest-neighbors classifier, the number of neighbors is 5 by default, but if you run `clf = KNeighborsClassifier(n_neighbors=3)`, you may get better/worse results. If you want to learn more about hyperparameters, we can discuss further.
