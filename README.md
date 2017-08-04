# Machine Learning Notes

## What is machine learning?  
Get a machine to solve problems without having to explicitly program them to.

## Two main types of machine learning
### Supervised Learning

You have a training set of inputs and their respective outputs, and you want a computer to predict the output from a given input. For example, regression and classification.   
This is what we'll be teaching.

### Unsupervised Learning

You have only a set of inputs and want a computer to recognize patterns in your data. For example, if you have a bunch of news articles and want to cluster them into groups, PCA, or SVD (singular value decomposition).

We'll only be going over very high level machine learning libraries. This includes scikit-learn and Keras.

Some (simple) supervised learning algorithms:
* Decision Tree
* K-nearest neighbors
* Linear Regression
* Logistic Regression

Some more complicated ones include recurrent neural nets, stochastic gradient descent, etc.

We'll first use a K-nearest neighbors classifier in scikit-learn on sample data, and then we'll write our own.

First, proceed [here](https://kevin-fang.github.io/ml-tutorials/KNearestNeighbors/) for a simple scikit-learn instance of machine learning. Then, visit [here](https://kevin-fang.github.io/ml-tutorials/ScrappyKNN/) to implement KNearestNeighbors yourself!