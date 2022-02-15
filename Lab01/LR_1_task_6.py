from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn import preprocessing
from utilities import visualize_classifier

input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter = ',')
X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)

classifier_svm = SVC()
classifier_svm.fit(X_train, y_train)

classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)

y_pred_svm = classifier_svm.predict(X_test)
y_pred_nb = classifier_nb.predict(X_test)

num_folds = 3

print('Naive Bayes:')
accuracy_values = cross_val_score(classifier_nb, X, y, scoring = 'accuracy', cv = num_folds)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) 
+ "%")

precision_values = cross_val_score(classifier_nb, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values.mean(), 
2)) + "%")

recall_values = cross_val_score(classifier_nb, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values.mean(), 2)) + 
"%")

f1_values = cross_val_score(classifier_nb, X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")

print('\nSVM:')
accuracy_values = cross_val_score(classifier_svm, X, y, scoring = 'accuracy', cv = num_folds)
print("Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) 
+ "%")

precision_values = cross_val_score(classifier_svm, X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100 * precision_values.mean(), 
2)) + "%")

recall_values = cross_val_score(classifier_svm, X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100 * recall_values.mean(), 2)) + 
"%")

f1_values = cross_val_score(classifier_svm, X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100 * f1_values.mean(), 2)) + "%")

visualize_classifier(classifier_nb, X_test, y_test)
visualize_classifier(classifier_svm, X_test, y_test)