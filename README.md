# Machine Learning Notes

## What is machine learning?  
In the field of machine learning, we want a machine to solve problems without having to explicitly program them to.

This is accomplished by getting a machine to "learn" a pattern. So while we don't explicitly instruct the computer how to solve the problem, we do instruct the computer in how to recognize patterns. These are **machine learning algorithms.**

Most machine learning algorithms have been thoroughly abstracted into black boxes, so to get started, all we need to learn is how to *use* the black boxes, without having to know the specifics of the math.

## Two main types of machine learning
### Supervised Learning

We use supervised learning when we have a training set of inputs and their respective outputs, and want a computer to predict the output from a given input. Regression and Classification are examples of supervised learning and are generally easier to understand.

### Unsupervised Learning

You have only a set of inputs and want a computer to recognize patterns in your data. For example, if you have a bunch of news articles and want to cluster them into groups, PCA, or SVD (singular value decomposition).

We'll only be going over very high level machine learning libraries. This includes scikit-learn and Keras.

---

Some (simple) supervised learning algorithms:
* Decision Tree
* K-nearest neighbors
* Linear/Logistic Regression

Some more complicated ones include recurrent neural nets, stochastic gradient descent, random forest classifiers, etc.

We'll first use a K-nearest neighbors classifier in scikit-learn on sample data, the "black box approach", and then we'll look into the box and write our own K-nearest-neighbors classifier.

First, proceed [here for a simple scikit-learn instance](https://kevin-fang.github.io/ml-tutorials/KNearestNeighbors/) for a simple scikit-learn instance of machine learning. Then, visit [here to implement KNN yourself!](https://kevin-fang.github.io/ml-tutorials/ScrappyKNN/).
