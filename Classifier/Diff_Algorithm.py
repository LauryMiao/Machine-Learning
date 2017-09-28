#
# classify iris_data from 2 different features with different Algorithms.
# Logistic Regression
# Random Forest
# SVM.SVC
# EnsembleVoteClassifier
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions

# Initializing Classifiers
clf1 = LogisticRegression(random_state=0)
clf2 = RandomForestClassifier(random_state=0)
clf3 = SVC(random_state=0, probability=True)
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
                              weights=[2, 1, 1], voting='soft')

# Loading some example data
X, y = iris_data()
#X = X[:,[0, 2]]  # Select 2 features, since visulization is 2D which just require 2 axes
XX = [X[:,[i,j]] for i in range(4) for j in range(i+1,4)] # combinations of X[:0:4]

# Plotting Decision Regions

gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))

clfs = [clf1, clf2, clf3, eclf]
labels = ['Logistic Regression',
          'Random Forest',
          'RBF kernel SVM',
          'Ensemble']

for row,X in enumerate(XX):
    for clf, lab, grd in zip(clfs,
                             labels,
                             itertools.product([0, 1],
                             repeat=2)):
        clf.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y,
                                    clf=clf, legend=2)
        plt.title(lab)    
    plt.show()