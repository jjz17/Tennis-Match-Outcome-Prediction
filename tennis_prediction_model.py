import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functions as f
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

url = 'https://raw.githubusercontent.com/jjz17/Tennis-Match-Outcome-Prediction/main/pbp_matches_atp_qual_current.csv'
data = pd.read_csv(url, index_col=0).drop('pbp_id', axis=1)
data

# Drop Wimbledon Final Qualifiers because they are played to best of 5 instead of best of 3
drop_list = data[data['tny_name'] == "Gentlemen'sWimbledonSinglesFinalRoundQualifying"].index
data.drop(drop_list, inplace=True)
data

fs_pbp = []
fs_game_count = []

# iterate through each match
for i in data.index:
    # extract pbp String
    match = data['pbp'][i]
    # extract first set
    first_set = match.split('.')[0]
    # record first set pbp
    fs_pbp.append(first_set)
    # record number of games in first set
    fs_game_count.append(first_set.count(';') + 1)

data['fs_pbp'] = fs_pbp
data['num_fs_games'] = fs_game_count

# Applying functions to transform data
fs_s1_points = []
fs_s2_points = []
fs_s1_momentum = []
fs_s2_momentum = []
fs_s1_breaks = []
fs_s2_breaks = []
fs_s1_aces = []
fs_s2_aces = []
p1_win = []
p1_win_first_set = []

# Iterate through each match(row)
for i in data.index:
    # create list of games in the match
    games = data.loc[i, 'fs_pbp'].split(';')

    # Split games into those served by p1 and p2
    p1_s_games = games[::2]
    p2_s_games = games[1::2]
    tiebreak = []

    # If there is a tiebreak
    if (len(p1_s_games) == 7):
        tiebreak = p1_s_games.pop(6)

    fs_s1_points_count = 0
    fs_s2_points_count = 0
    fs_s1_momentum_count = 0
    fs_s2_momentum_count = 0
    fs_s1_breaks_count = 0
    fs_s2_breaks_count = 0
    fs_s1_aces_count = 0
    fs_s2_aces_count = 0

    # Aggregate data from p1 serving games
    for game in p1_s_games:
        game_data = f.extract_game_data(game)
        fs_s1_points_count += game_data[0]
        fs_s2_points_count += game_data[1]
        fs_s1_momentum_count += game_data[2]
        fs_s2_momentum_count += game_data[3]
        fs_s2_breaks_count += game_data[4]
        fs_s1_aces_count += game_data[5]

    # Aggregate data from p2 serving games
    for game in p2_s_games:
        game_data = f.extract_game_data(game)
        fs_s2_points_count += game_data[0]
        fs_s1_points_count += game_data[1]
        fs_s2_momentum_count += game_data[2]
        fs_s1_momentum_count += game_data[3]
        fs_s1_breaks_count += game_data[4]
        fs_s2_aces_count += game_data[5]

    # Aggregate data from tiebreak
    tb_data = f.extract_tiebreak_data(tiebreak)
    fs_s1_points_count += tb_data[0]
    fs_s2_points_count += tb_data[1]
    fs_s1_momentum_count += tb_data[2]
    fs_s2_momentum_count += tb_data[3]
    fs_s1_aces_count += tb_data[4]
    fs_s2_aces_count += tb_data[5]

    # Add the match's data to the lists
    fs_s1_points.append(fs_s1_points_count)
    fs_s2_points.append(fs_s2_points_count)
    fs_s1_momentum.append(fs_s1_momentum_count)
    fs_s2_momentum.append(fs_s2_momentum_count)
    fs_s1_breaks.append(fs_s1_breaks_count)
    fs_s2_breaks.append(fs_s2_breaks_count)
    fs_s1_aces.append(fs_s1_aces_count)
    fs_s2_aces.append(fs_s2_aces_count)

    p1_win.append(data.loc[i, 'winner'] == 1)

    p1_win_first_set.append(f.is_p1_set_winner(games[-1], len(games)))

data['fs_s1_points'] = fs_s1_points
data['fs_s2_points'] = fs_s2_points
data['fs_s1_momentum'] = fs_s1_momentum
data['fs_s2_momentum'] = fs_s2_momentum
data['fs_s1_breaks'] = fs_s1_breaks
data['fs_s2_breaks'] = fs_s2_breaks
data['fs_s1_aces'] = fs_s1_aces
data['fs_s2_aces'] = fs_s2_aces
data['p1_win'] = p1_win
data['p1_win_fs'] = p1_win_first_set

# label encoding for features

# 1 if p1 win, 0 if p2 win
data['p1_win'] = data['p1_win'].map({True: 1, False: 0})
data['p1_win_fs'] = data['p1_win_fs'].map({True: 1, False: 0})

# Replace na with 0
data = data.fillna(0)

# Data visualization
plot_data = data[['p1_win_fs', 'p1_win']]

plot_data.groupby('p1_win_fs').p1_win.value_counts().unstack(0).plot.barh()

#%%
plot = sns.catplot(x='p1_win', y='fs_s1_points', kind='box', data=data)
plt.show()
#%%
sns.catplot(x='p1_win', y='fs_s2_points', kind='box', data=data)

rel_columns = ['fs_s1_momentum', 'fs_s2_momentum',
               'fs_s1_breaks', 'fs_s2_breaks', 'fs_s1_aces',
               'fs_s2_aces', 'fs_s1_points', 'fs_s2_points', 'p1_win_fs', 'p1_win']

ml_data = data[rel_columns]

features, target = f.features_and_target(ml_data)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=3000)

# create the scaler
scaler = MinMaxScaler()

# fit the scaler to the training data(features only)
scaler.fit(X_train)

# Export scaler
scaler_filename = "tennis_minmax_scaler"
joblib.dump(scaler, scaler_filename)

# transform X_train and X_test based on the (same) scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Parameter grids for Validation/Optimization
logi_param_grid = {"penalty":['l1', 'l2']}
svc_param_grid = {"penalty":['l1', 'l2']}
# 17 is optimal for knn
knn_param_grid = {"n_neighbors":[1, 5, 10, 20], "metric": ['euclidean', 'manhattan', 'minkowski']}
nb_param_grid = {}
tree_param_grid = {"criterion":['gini', 'entropy'], "splitter":['best', 'random']}

# Dictionary of models with their parameter grids
estimators = {
    'Logistic Regression': [LogisticRegression(solver='liblinear'), logi_param_grid],
    'k-Nearest Neighbor': [KNeighborsClassifier(), knn_param_grid],
    'Support Vector Machine': [LinearSVC(max_iter=1000000), svc_param_grid],
    'Gaussian Naive Bayes': [GaussianNB(), nb_param_grid],
    'Decision Tree': [DecisionTreeClassifier(), tree_param_grid],
    'Second Decision Tree': [DecisionTreeClassifier(max_depth=3), tree_param_grid]}

# Dictionaries to store optimized model objects
best_models = {}

# Train models
f.classifiers_percentage_split(X_train_scaled, X_test_scaled, y_train, y_test, estimators)

# Tune models
f.hyperparameters_tuning(X_train_scaled, X_test_scaled, y_train, y_test, estimators, best_models)