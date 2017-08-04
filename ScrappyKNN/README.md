# Scrappy K-Nearest-Neighbors

Note: this code is taken from the Google Machine Learning tutorial

```python
from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)
```
This function simply returns the euclidean distance between two points on a graph:   
sqrt((x<sub>1</sub>-x<sub>2</sub>)<sup>2</sup>-(y<sub>1</sub>-y<sub>2</sub>)<sup>2</sup>).

```python
class ScrappyKNN():
```
We want to create a class for our classifier here.
```python
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
```
This is where we usually do the training; since KNN is very simple, we don't have to do any complex math, so this function is very simple in that we just store the inputs here. Some other classifiers such as SGD may use differential equations to optimize a function.
Depending on your data size and the complexity of the classifier, `fit()` can take a fraction of a second to several hours. For example, when you create a deep neural network, `fit` take several hours.

```python
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]
```
`closest()` takes in an input value. It simply loops through `self.X_train` to find the value that is the least distance away from the `row` variable.

```python
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
```
Relatively simple `predict()` function. It simply calls `closest` on the training data, looking for the nearest values and returning them. 

The code after is an exact copy from `knn.py`.