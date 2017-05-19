#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Prediction of Iris datasets with KNN
=========================================================
This example uses ‘iris’ dataset from internet,
Then, we initialize a KNN model kNeighborsClassifier to predict the class of Iris data


"""
print(__doc__)

# Code Author: HangJie

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

iris = load_iris()
# print(iris.data.shape)
# print(iris.DESCR)

# Randomly sample 25% of the data for testing, and the remaining 75% for training sets
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)

# Data Standardization
# Standardized data to ensure that the variance of the characteristic data for each dimension is 1 and the mean is 0, so that the prediction results are not dominated by some characteristic values that are too large
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# kNN Prediction
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_predict = knn.predict(X_test)

print("the accuracy of KNN:",knn.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=iris.target_names))