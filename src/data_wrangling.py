import os
import pandas as pd
from src import functions as f

# data = pd.read_csv(f'data{os.path.sep}pbp_matches_atp_qual_current.csv', index_col=0).drop(['pbp_id'], axis=1)
url = 'https://raw.githubusercontent.com/jjz17/Tennis-Match-Outcome-Prediction/main/data/pbp_matches_atp_qual_current' \
      '.csv '
data = pd.read_csv(url, index_col=0).drop('pbp_id', axis=1)

# Drop Wimbledon Final Qualifiers because they are played to best of 5 instead of best of 3
drop_list = data[data['tny_name'] == "Gentlemen'sWimbledonSinglesFinalRoundQualifying"].index
data.drop(drop_list, inplace=True)

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

# data.to_csv(f'..data{os.path.sep}wrangled_data.csv, index=False')
data.to_csv(f'wrangled_data.csv', index=False)