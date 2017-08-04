# pip install numpy scipy sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = KNeighborsClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print accuracy_score(y_test, y_pred)
