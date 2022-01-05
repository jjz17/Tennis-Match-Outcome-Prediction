import _pickle as cPickle
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import functions as f

data = pd.read_csv(f'..{os.path.sep}data{os.path.sep}wrangled_data.csv', index_col=0)
# url = 'https://raw.githubusercontent.com/jjz17/Tennis-Match-Outcome-Prediction/main/data/wrangled_data.csv'
# data = pd.read_csv(url, index_col=0)

rel_columns = ['s1 fs momentum', 's2 fs momentum',
               's1 fs breaks', 's2 fs breaks', 's1 fs aces',
               's2 fs aces', 's1 fs points', 's2 fs points', 's1 fs win', 's1 win']

ml_data = data[rel_columns]

features, target = f.features_and_target(ml_data)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=3000)

# create the scaler
scaler = MinMaxScaler()

# fit the scaler to the training data(features only)
scaler.fit(X_train)

# Export scaler
scaler_filename = f'..{os.path.sep}models{os.path.sep}tennis_minmax_scaler'
joblib.dump(scaler, scaler_filename)

# transform X_train and X_test based on the (same) scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Parameter grids for Validation/Optimization
logi_param_grid = {"penalty": ['l1', 'l2']}
svc_param_grid = {"penalty": ['l1', 'l2']}
# 17 is optimal for knn
knn_param_grid = {"n_neighbors": [1, 5, 10, 20], "metric": ['euclidean', 'manhattan', 'minkowski']}
nb_param_grid = {}
tree_param_grid = {"criterion": ['gini', 'entropy'], "splitter": ['best', 'random']}

# ML Models
logi_reg = LogisticRegression(solver='liblinear')
knn = KNeighborsClassifier()
lin_svc = LinearSVC(max_iter=1000000)
gauss_nb = GaussianNB()
tree = DecisionTreeClassifier()
tree2 = DecisionTreeClassifier(max_depth=3)

# Dictionary of models
estimators = {
    'Logistic Regression': logi_reg,
    'k-Nearest Neighbor': knn,
    'Support Vector Machine': lin_svc,
    'Gaussian Naive Bayes': gauss_nb,
    'Decision Tree': tree,
    'Second Decision Tree': tree2}

# Dictionary of models with their parameter grids
estimators_with_grids = {
    'Logistic Regression': [logi_reg, logi_param_grid],
    'k-Nearest Neighbor': [knn, knn_param_grid],
    'Support Vector Machine': [lin_svc, svc_param_grid],
    'Gaussian Naive Bayes': [gauss_nb, nb_param_grid],
    'Decision Tree': [tree, tree_param_grid],
    'Second Decision Tree': [tree2, tree_param_grid]}

# Dictionaries to store optimized model objects
best_models = {}

# Train models
f.classifiers_percentage_split(X_train_scaled, X_test_scaled, y_train, y_test, estimators)

# Tune models
f.hyperparameters_tuning(X_train_scaled, X_test_scaled, y_train, y_test, estimators_with_grids, best_models)

# Export Logistic Regression Model
# pickle.dump(best_models['Logistic Regression'], open('../models/tennis_prediction_model2.pk1', 'wb'))
# pickle.dump(best_models['Logistic Regression'], open('tennis_prediction_model.pk1', 'wb'))
# with open(r"models/logreg_model.pickle", "wb") as output_file:
#     cPickle.dump(best_models['Logistic Regression'], output_file)
with open(f'..{os.path.sep}models{os.path.sep}logreg_model.pickle', 'wb') as output_file:
    cPickle.dump(best_models['Logistic Regression'], output_file)

# print(os.path.abspath(os.curdir))
# os.chdir('..')
# print(os.path.abspath(os.curdir))

selected_features = f.recursive_feature_elimination(X_train_scaled, X_test_scaled, y_train, features)

# Extract selected features from the training and testing sets
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Create the scaler
scaler = MinMaxScaler()

# Fit the scaler to the training data(features only)
scaler.fit(X_train_selected)

# Transform X_train and X_test based on the (same) scaler
X_train_selected_scaled = scaler.transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# Replace any potential NaN with 0
X_train_selected_scaled[np.isnan(X_train_selected_scaled)] = 0
X_test_selected_scaled[np.isnan(X_test_selected_scaled)] = 0

# Dictionaries to store optimized model objects
best_selected_models = {}

# Tune models
f.hyperparameters_tuning(X_train_selected_scaled, X_test_selected_scaled, y_train, y_test, estimators_with_grids,
                         best_selected_models)

# print([X_train_selected.columns])
