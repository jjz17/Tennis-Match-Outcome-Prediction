import os
import pandas as pd
from src import functions as f

data = pd.read_csv(f'..{os.path.sep}data{os.path.sep}pbp_matches_atp_qual_current.csv', index_col=0).drop(['pbp_id'], axis=1)
# url = 'https://raw.githubusercontent.com/jjz17/Tennis-Match-Outcome-Prediction/main/data/pbp_matches_atp_qual_current' \
#       '.csv '
# data = pd.read_csv(url, index_col=0).drop('pbp_id', axis=1)

# Drop Wimbledon Final Qualifiers because they are played to best of 5 instead of best of 3
drop_list = data[data['tny_name'] == "Gentlemen'sWimbledonSinglesFinalRoundQualifying"].index
data.drop(drop_list, inplace=True)

first_set_pbp = []
num_first_set_games = []

# iterate through each match
for i in data.index:
    # extract pbp String
    match = data['pbp'][i]
    # extract first set
    first_set = match.split('.')[0]
    # record first set pbp
    first_set_pbp.append(first_set)
    # record number of games in first set
    num_first_set_games.append(first_set.count(';') + 1)

data['first set pbp'] = first_set_pbp
data['# games in first set'] = num_first_set_games

# Applying functions to transform data
s1_first_set_points = []
s2_first_set_points = []
s1_first_set_momentum = []
s2_first_set_momentum = []
s1_first_set_breaks = []
s2_first_set_breaks = []
s1_first_set_aces = []
s2_first_set_aces = []
s1_win = []
s1_win_first_set = []

# Iterate through each match (row)
for i in data.index:

    # create list of games in the match
    games = data.loc[i, 'first set pbp'].split(';')

    # Split games into those served by p1 and p2
    p1_s_games = games[::2]
    p2_s_games = games[1::2]
    tiebreak = []

    # If there is a tiebreak
    if (len(p1_s_games) == 7):
        tiebreak = p1_s_games.pop(6)

    s1_num_first_set_points = 0
    s2_num_first_set_points = 0
    s1_num_first_set_momentum = 0
    s2_num_first_set_momentum = 0
    s1_num_first_set_breaks = 0
    s2_num_first_set_breaks = 0
    s1_num_first_set_aces = 0
    s2_num_first_set_aces = 0

    # Aggregate data from p1 serving games
    for game in p1_s_games:
        game_data = f.extract_game_data(game)
        s1_num_first_set_points += game_data[0]
        s2_num_first_set_points += game_data[1]
        s1_num_first_set_momentum += game_data[2]
        s2_num_first_set_momentum += game_data[3]
        s2_num_first_set_breaks += game_data[4]
        s1_num_first_set_aces += game_data[5]

    # Aggregate data from p2 serving games
    for game in p2_s_games:
        game_data = f.extract_game_data(game)
        s2_num_first_set_points += game_data[0]
        s1_num_first_set_points += game_data[1]
        s2_num_first_set_momentum += game_data[2]
        s1_num_first_set_momentum += game_data[3]
        s1_num_first_set_breaks += game_data[4]
        s2_num_first_set_aces += game_data[5]

    # Aggregate data from tiebreak
    tb_data = f.extract_tiebreak_data(tiebreak)
    s1_num_first_set_points += tb_data[0]
    s2_num_first_set_points += tb_data[1]
    s1_num_first_set_momentum += tb_data[2]
    s2_num_first_set_momentum += tb_data[3]
    s1_num_first_set_aces += tb_data[4]
    s2_num_first_set_aces += tb_data[5]

    # Add the match's data to the lists
    s1_first_set_points.append(s1_num_first_set_points)
    s2_first_set_points.append(s2_num_first_set_points)
    s1_first_set_momentum.append(s1_num_first_set_momentum)
    s2_first_set_momentum.append(s2_num_first_set_momentum)
    s1_first_set_breaks.append(s1_num_first_set_breaks)
    s2_first_set_breaks.append(s2_num_first_set_breaks)
    s1_first_set_aces.append(s1_num_first_set_aces)
    s2_first_set_aces.append(s2_num_first_set_aces)

    s1_win.append(data.loc[i, 'winner'] == 1)

    s1_win_first_set.append(f.is_s1_set_winner(games[-1], len(games)))

data['s1 fs points'] = s1_first_set_points
data['s2 fs points'] = s2_first_set_points
data['s1 fs momentum'] = s1_first_set_momentum
data['s2 fs momentum'] = s2_first_set_momentum
data['s1 fs breaks'] = s1_first_set_breaks
data['s2 fs breaks'] = s2_first_set_breaks
data['s1 fs aces'] = s1_first_set_aces
data['s2 fs aces'] = s2_first_set_aces
data['s1 win'] = s1_win
data['s1 fs win'] = s1_win_first_set

# label encoding for features

# 1 if p1 win, 0 if p2 win
data['s1 win'] = data['s1 win'].map({True: 1, False: 0})
data['s1 fs win'] = data['s1 fs win'].map({True: 1, False: 0})

# Replace na with 0
data = data.fillna(0)

# data.to_csv(f'..data{os.path.sep}wrangled_data.csv, index=False')
data.to_csv(f'..{os.path.sep}data{os.path.sep}wrangled_data.csv', index=False)