#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################

### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

def classify_KNN(n):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    return accuracy_score(pred, labels_test)

def optimal_KNN():
    print("optimal KNN")
    optimal_n = 0
    optimal_accuracy = 0
    for i in range(1, 100):
        acc = classify_KNN(i)
        if acc > optimal_accuracy:
            optimal_accuracy = acc
            optimal_n = i
    print("neighbors: " + str(optimal_n))
    print("accuracy:  " + str(optimal_accuracy))
    print("")
    clf = KNeighborsClassifier(n_neighbors=optimal_n)
    clf.fit(features_train, labels_train)
    return clf

def classify_forest(n):
    clf = RandomForestClassifier(n_estimators=n)
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    return accuracy_score(pred, labels_test)

def optimal_forest():
    print("optimal RandomForest")
    optimal_n = 0
    optimal_accuracy = 0
    for i in range(1, 100):
        acc = classify_forest(i)
        if acc > optimal_accuracy:
            optimal_accuracy = acc
            optimal_n = i
    print("estimators: " + str(optimal_n))
    print("accuracy:   " + str(optimal_accuracy))
    print("")
    clf = RandomForestClassifier(n_estimators=optimal_n)
    clf.fit(features_train, labels_train)
    return clf


def classify_adaboost(n):
    clf = AdaBoostClassifier(n_estimators=n)
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    return accuracy_score(pred, labels_test)

def optimal_adaboost():
    print("optimal AdaBoost")
    optimal_n = 0
    optimal_accuracy = 0
    for i in range(1, 30):
        acc = classify_adaboost(i)
        if acc > optimal_accuracy:
            optimal_accuracy = acc
            optimal_n = i
    print("estimators: " + str(optimal_n))
    print("accuracy:   " + str(optimal_accuracy))
    print("")

    clf = AdaBoostClassifier(n_estimators=optimal_n)
    clf.fit(features_train, labels_train)
    return clf

clf = optimal_KNN()
prettyPicture(clf, features_test, labels_test, 'optimal_knn.png')

clf = optimal_forest()
prettyPicture(clf, features_test, labels_test, 'optimal_forest.png')

clf = optimal_adaboost()
prettyPicture(clf, features_test, labels_test, 'optimal_adaboost.png')
