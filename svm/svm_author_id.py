#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




from sklearn.metrics import accuracy_score
#########################################################
### your code goes here ###



# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
#
# print("Running gaussian naive bayes")
#
# t = time()
# clf.fit(features_train, labels_train)
# print("time to train: " + str(round(time() - t, 3)))
#
# t = time()
# pred = clf.predict(features_test)
# print("time to predict: " + str(round(time() - t, 3)))
#
# from sklearn.metrics import accuracy_score
# print("accuracy: " + str(round(accuracy_score(pred, labels_test), 3)))




from sklearn import svm



def svm_train_and_acc(k, c, p):
    clf = svm.SVC(kernel=k, C=c)

    print("Running a linear support vector machine")
    print("kernel=" + k + "   C=" + str(c) + "   " + str(round(100/p)) + "% of training set")

    t = time()

    clf.fit(features_train[:len(features_train)/p], labels_train[:len(features_train)/p])
    print("time to train:   " + str(round(time() - t, 3)))

    t = time()
    pred = clf.predict(features_test)
    print("time to predict: " + str(round(time() - t, 3)))

    print("accuracy:        " + str(round(accuracy_score(pred, labels_test), 3)))

svm_train_and_acc('rbf', 10000, 100)
