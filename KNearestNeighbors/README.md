# KNN Walkthrough

First run `pip install numpy scipy sklearn` to install the dependencies.

Note: this code is taken from the Google Machine Learning tutorial

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
In scikit-learn, all the classifiers are classes. This initializes the KNeighborsClassifier. You can specify parameters ("hyperparameters") of the classifier in the parenthesis, like above in the commented line.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
```
You generally don't want to train a classifier on all your data, so we use a "train-test-split" to split our data into a training set and a testing set.   
The `test_size` parameter tells the classifier the ratio of training data to testing data. `0.2` means that 20% of the data will be marked as testing data and 80% will be training data.

```python
clf.fit(X_train, y_train)
```
This is where the magic happens! In K-nearest-neighbors, it would add all of the data to the graph. In more complicated ones, such as Stochastic Gradient Descent, it tries to minimize an error function using gradient descent.

```python
y_pred = clf.predict(X_test)
print accuracy_score(y_test, y_pred)
```
This tells the classifier to predict an output based on the test input. `accuracy_score(true_data, test_data)` will give you the ratio of correct versus incorrect data.

You should get an accuracy of around 90%. Once you do, either go [here](https://kevin-fang.github.io/ml-tutorials/ScrappyKNN/) to implement KNearestNeighbors yourself or try some other classifiers. Here are some examples of other classifiers - everything else will work the exact same, you just have to change the `clf` variable:
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