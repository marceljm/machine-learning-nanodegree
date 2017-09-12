# Import, read, and split data
import pandas as pd
data = pd.read_csv('data/data.csv')
import numpy as np
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# np.random.seed() makes the random numbers predictable
# with the seed reset (every time), the same set of numbers will appear every time
np.random.seed(55)

# testing if seed really works...
# print np.random.rand(N)
# print np.random.rand(N)

### Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# TODO: Uncomment one of the three classifiers, and hit "Test Run"
# to see the learning curve. Use these to answer the quiz below.

### Logistic Regression
estimator = LogisticRegression()

### Decision Tree
#estimator = GradientBoostingClassifier()

### Support Vector Machine
#estimator = SVC(kernel='rbf', gamma=1000)

### Neural Network
#estimator = MLPClassifier(max_iter=1000)

from sklearn.model_selection import learning_curve

# It is good to randomize the data before drawing Learning Curves
def randomize(X, y):
    # y.shape[0] returns the number of rows at column 0
    # permutation returns a list of random numbers between 0 and N, without duplicates
    permutation = np.random.permutation(y.shape[0])
    # X[0] returns the row 0, without header
    # X[0][1] or X[0,1] returns the column 1 of the row 0
    # X[0,:] returns all columns of the row 0
    # X[permutation,:] returns X in permutation order
    # y[permutation] returns y in permutation order
    X = X[permutation,:]
    y = y[permutation]
    return X, y

X, y = randomize(X, y)

import matplotlib.pyplot as plt

# reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
# estimator is the classifier
# num_trainings is the number of samples to generate
# linspace(start, stop, num=50) returns evenly spaced numbers (num) over a specified interval (start, stop).
def draw_learning_curves(X, y, estimator, num_trainings):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))

    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
   
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    """

    # axis: along which the means are computed. The default is to compute the mean of the flattened array.
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.plot(train_scores_mean, 'o-', color="g", label="Training score")
    plt.plot(test_scores_mean, 'o-', color="y", label="Cross-validation score")

    plt.legend(loc="best")

    plt.show()

draw_learning_curves(X, y, estimator, 10)
