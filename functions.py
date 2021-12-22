from sklearn.model_selection import GridSearchCV


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


def features_and_target(df):
    features = df.iloc[:, :len(df.columns) - 1]
    target = df.iloc[:, len(df.columns) - 1]
    return features, target


def classifiers_percentage_split(X_train, X_test, y_train, y_test, estimators):
    for estimator_name, estimator_object in estimators.items():
        # Create the model by fitting the training data
        model = estimator_object[0].fit(X=X_train, y=y_train)

        # Make predictions on the test set
        predicted = model.predict(X=X_test)

        # Prediction accuracy
        accuracy = model.score(X_test, y_test)

        print(estimator_name + ':\n\t' + f'Classification accuracy on the test data: {accuracy:.2%}\n')


def hyperparameters_tuning(X_train, X_test, y_train, y_test, estimators, best_models):
    print("Results for Best Models Trained on All Features\n")
    for estimator_name, estimator_objects in estimators.items():
        estimator_model = estimator_objects[0]
        param_grid = estimator_objects[1]

        grid_search = GridSearchCV(estimator_model, param_grid, return_train_score=True, cv=5)

        # Fit the grid search object on the training data (CV will be performed on this)
        grid_search.fit(X=X_train, y=y_train)

        # Grid search results
        print(estimator_name + ":\n")
        print("\tBest estimator: ", grid_search.best_estimator_)
        print("\tBest parameters: ", grid_search.best_params_)
        print("\tBest cross-validation score: ", grid_search.best_score_)
        print("\n")

        model = grid_search.best_estimator_
        #     print("\tR-squared value for training set: ", r2_score(y_train, model.predict(X_train_scaled)))
        #     print("\tMean-squared-error value for training set: ", mean_squared_error(y_train, model.predict(X_train_scaled)))
        accuracy = model.score(X_test, y_test)
        print(accuracy)
        print("\n")

        # Add the best model to dictionary
        best_models[estimator_name] = grid_search.best_estimator_
