######################
# Import libraries
######################
import numpy as np
import sklearn
import pandas as pd
import streamlit as st
import pickle
import joblib
from PIL import Image

######################
# Custom function
######################
## Calculate set stats

def momentum_check(pbp, tiebreak=False):
    momentum_threshold = 3
    if tiebreak:
        momentum_threshold = 2

    s_momentum = True
    momentum = 0
    if len(pbp) >= momentum_threshold:
        if set(pbp[:momentum_threshold]).issubset({'A', 'S'}):
            momentum += 1
        if set(pbp[:momentum_threshold]).issubset({'R', 'D'}):
            momentum += 1
            s_momentum = False
    return s_momentum, momentum

# Function to extract relevant data from a game pbp string
def extract_game_data(game_pbp):
    s_points = 0
    r_points = 0
    s_momentum = 0
    r_momentum = 0
    breaks = 0
    aces = 0

    # iterate through each point
    for i in range(len(game_pbp)):
        point = game_pbp[i]

        # Handle momentum quantification
        serve_momentum, m_count = momentum_check(game_pbp[i:])
        if m_count > 0:
            if serve_momentum:
                s_momentum += 1
            else:
                r_momentum += 1

        if point == 'S':
            s_points += 1
        elif point == 'A':
            aces += 1
            s_points += 1
        elif point == 'R':
            r_points += 1
        elif point == 'D':
            r_points += 1
        else:
            s_points += 0

    if r_points > s_points:
        breaks = 1

    return [s_points, r_points, s_momentum, r_momentum, breaks, aces]


# Function to extract relevant data from a tiebreak
def extract_tiebreak_data(tiebreak_pbp):
    p1_points = 0
    p2_points = 0
    p1_momentum = 0
    p2_momentum = 0
    p1_aces = 0
    p2_aces = 0

    p1_serving = False

    splits = str(tiebreak_pbp).split('/')

    # iterate through each point
    for points in splits:
        # toggle boolean flag
        p1_serving = not p1_serving

        s_points = 0
        r_points = 0
        s_momentum = 0
        r_momentum = 0
        aces = 0

        # Handle momentum quantification
        serve_momentum, m_count = momentum_check(points, tiebreak=True)
        if m_count > 0:
            if serve_momentum:
                s_momentum += 1
            else:
                r_momentum += 1

        for point in points:

            if point == 'S':
                s_points += 1
            elif point == 'A':
                aces += 1
                s_points += 1
            elif point == 'R':
                r_points += 1
            elif point == 'D':
                r_points += 1
            else:
                s_points += 0

        if p1_serving:
            p1_points += s_points
            p2_points += r_points
            p1_momentum += s_momentum
            p2_momentum += r_momentum
            p1_aces += aces
        else:
            p2_points += s_points
            p1_points += r_points
            p2_momentum += s_momentum
            p1_momentum += r_momentum
            p2_aces += aces

    return [p1_points, p2_points, p1_momentum, p2_momentum, p1_aces, p2_aces]

def is_p1_set_winner(last_game_pbp, num_games):

    # If tiebreak decider
    if num_games > 12:
        p1_points = 0
        p2_points = 0

        switches = last_game_pbp.split('/')
        p1_serves = switches[::2]
        p2_serves = switches[1::2]
        for switch in p1_serves:
            p1_points += switch.count('A')
            p1_points += switch.count('S')
            p2_points += switch.count('R')
            p2_points += switch.count('D')
        for switch in p2_serves:
            p2_points += switch.count('A')
            p2_points += switch.count('S')
            p1_points += switch.count('R')
            p1_points += switch.count('D')
        return p1_points > p2_points
    else:
        s_points = 0
        r_points = 0

        s_points += last_game_pbp.count('A')
        s_points += last_game_pbp.count('S')
        r_points += last_game_pbp.count('R')
        r_points += last_game_pbp.count('D')

        # If p2 serves
        if num_games % 2 == 0:
            return r_points > s_points
        # If p1 serves
        else:
            return s_points > r_points

def compute_set_stats(set_pbp):
    # create list of games in the match
    games = set_pbp.split(';')

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
        game_data = extract_game_data(game)
        fs_s1_points_count += game_data[0]
        fs_s2_points_count += game_data[1]
        fs_s1_momentum_count += game_data[2]
        fs_s2_momentum_count += game_data[3]
        fs_s2_breaks_count += game_data[4]
        fs_s1_aces_count += game_data[5]

    # Aggregate data from p2 serving games
    for game in p2_s_games:
        game_data = extract_game_data(game)
        fs_s2_points_count += game_data[0]
        fs_s1_points_count += game_data[1]
        fs_s2_momentum_count += game_data[2]
        fs_s1_momentum_count += game_data[3]
        fs_s1_breaks_count += game_data[4]
        fs_s2_aces_count += game_data[5]

    # Aggregate data from tiebreak
    tb_data = extract_tiebreak_data(tiebreak)
    fs_s1_points_count += tb_data[0]
    fs_s2_points_count += tb_data[1]
    fs_s1_momentum_count += tb_data[2]
    fs_s2_momentum_count += tb_data[3]
    fs_s1_aces_count += tb_data[4]
    fs_s2_aces_count += tb_data[5]

    p1_fs_win = is_p1_set_winner(games[-1], len(games))

    if p1_fs_win:
        p1_fs_win = 1
    else:
        p1_fs_win = 0

    # SCALE BY CREATING Z-SCORES
    # fs_s1_momentum_count = (fs_s1_momentum_count - 5.56) / 2.760247
    # fs_s2_momentum_count = (fs_s2_momentum_count - 5.467027) / 2.941391
    # fs_s1_breaks_count = (fs_s1_breaks_count - 1.101622) / 0.977256
    # fs_s2_breaks_count = (fs_s2_breaks_count - 1.147027) / 1.013974
    # fs_s1_aces_count = (fs_s1_aces_count - 1.941622) / 1.840163
    # fs_s2_aces_count = (fs_s2_aces_count - 1.690811) / 1.799943
    # fs_s1_points_count = (fs_s1_points_count - 31.331892) / 8.772813
    # fs_s2_points_count = (fs_s2_points_count - 30.710270) / 9.418961

    return [fs_s1_momentum_count, fs_s2_momentum_count, fs_s1_breaks_count, fs_s2_breaks_count,
            fs_s1_aces_count, fs_s2_aces_count, fs_s1_points_count, fs_s2_points_count, p1_fs_win]

    # SCALE INPUTS
    # 	fs_s1_momentum	fs_s2_momentum	fs_s1_breaks	fs_s2_breaks	fs_s1_aces	fs_s2_aces	fs_s1_points	fs_s2_points	p1_win
    # mean	5.560000	5.467027	1.101622	1.147027	1.941622	1.690811	31.331892	30.710270	0.482162
    # std	2.760247	2.941391	0.977256	1.013974	1.840163	1.799943	8.772813	9.418961	0.499952

def scale_features(features):
    features[0] = (features[0] - 5.56) / 2.760247
    features[1] = (features[1] - 5.467027) / 2.941391
    features[2] = (features[2] - 1.101622) / 0.977256
    features[3] = (features[3] - 1.147027) / 1.013974
    features[4] = (features[4] - 1.941622) / 1.840163
    features[5] = (features[5] - 1.690811) / 1.799943
    features[6] = (features[6] - 31.331892) / 8.772813
    features[7] = (features[7] - 30.710270) / 9.418961
    return features

def display_prediction(prediction):
    if prediction == 1:
        return 'P1 wins'
    else:
        return 'P2 wins'

######################
# Page Title
######################

# image = Image.open('solubility-logo.jpg')
#
# st.image(image, use_column_width=True)

st.write("""
# Tennis Match Prediction Web App
This app predicts the outcome of a best of 3-sets tennis match based on the first set

The prediction model is a Logistic Regression Model trained on data obtained from [ Jeff Sackmann](https://github.com/JeffSackmann/tennis_pointbypoint/blob/master/pbp_matches_atp_qual_current.csv).
***
""")

######################
# Input molecules (Side Panel)
######################

st.sidebar.header('User Input First Set PBP')
st.text('Each point is described with one character:\nS (server won)\nR (returner won)\nA (ace)\nD (double fault)\nGames are delimited with the \';\' character\nThe \'/\' character indicates changes of serve during a tiebreak.')

## Read SMILES input
pbp_input = 'SSRSS;ASSS;SSRSS;SSAS;RRRR;SSSS;SRSSS;RRSDR;SSRSS;SSSRS;RRRSSSDSSRSS;RRSRR'

PBP = st.sidebar.text_area("PBP input", pbp_input)
# SMILES = "C\n" + SMILES  # Adds C as a dummy, first item
# SMILES = SMILES.split('\n')

st.header('Input PBP')
PBP

## Calculate molecular descriptors
st.header('Computed set stats')
stats = compute_set_stats(PBP)
labels = ['fs_s1_momentum_count', 'fs_s2_momentum_count', 'fs_s1_breaks_count', 'fs_s2_breaks_count',
            'fs_s1_aces_count', 'fs_s2_aces_count', 'fs_s1_points_count', 'fs_s2_points_count', 'p1_fs_win']
# stats

info = pd.DataFrame()
info['labels'] = labels
info['data'] = stats
# st.table(info.set_index('labels'))
st.table(info)

# scaled_stats2 = scale_features(stats)
st.header('Scaled set stats')
scaler = joblib.load('tennis_minmax_scaler')
scaled_stats = scaler.transform(np.array(stats).reshape(1,-1))
scaled_stats
# scaled_stats2

######################
# Pre-built model
######################

# Reads in saved model
load_model = pickle.load(open('tennis_prediction_model.pk1', 'rb'))



# Apply model to make predictions
prediction = load_model.predict(np.array(scaled_stats).reshape(1,-1))
prob = load_model.predict_proba(np.array(scaled_stats).reshape(1,-1))
# prediction_proba = load_model.predict_proba(X)

st.header('Match Outcome Prediction')
st.text(display_prediction(prediction))
""
'Prob P1 wins'
prob[0][1]
'Prob P2 wins'
prob[0][0]