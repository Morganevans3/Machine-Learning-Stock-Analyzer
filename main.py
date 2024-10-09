import pandas as pd
import yfinance as yf
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


def is_valid_ticker(ticker):
    # make sure it matches typical stock ticker format
    # used Chatgpt to include ^
    ticker_pattern = r'^[A-Z\^]{1,5}$'

    # check if it matches the pattern
    if re.match(ticker_pattern, ticker):
        return True

    return False


def download_ticker_data(start_date=None, end_date=None, ticker='^SPX', period=None):
    # code to download the stock data from
    # https://towardsdatascience.com/python-how-to-get-live-market-data-less-than-0-1-second-lag-c85ee280ed93
    # we rewrote it into a function and added the possibility to do the max period instead

    if start_date is None:
        # default to 3 years ago
        start_date = datetime.now() - pd.DateOffset(months=36)

    if end_date is None:
        # default to current date
        end_date = datetime.now()

    if period == 'max':
        # defaults to max period
        df = yf.download(ticker)
    else:
        # if specified time period
        df = yf.download(ticker, start=start_date, end=end_date)

    # reset index
    df = df.reset_index()

    print(f'{ticker} is downloaded!')
    return df


def clean_data(df):
    # delete any row with missing values or 0
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
    df = df[df['Volume'] != 0]

    # Do I want to drop outliers in the data?
    # this uses z-score to identify and remove any outliers
    # df = df[(np.abs(stats.zscore(df[['Open', 'High', 'Low', 'Close', 'Adj Close']])) < 3).all(axis=1)]

    # check data types and ensure date column is of datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # check for duplicates and remove if necessary
    df = df.drop_duplicates()
    print('And cleaned!\n')
    return df


def make_model(training, predictors, n_est=100, min_s_s=100):
    # modified code from https://github.com/dataquestio/project-walkthroughs/blob/master/sp_500/market_prediction.ipynb
    # by making it into a function

    # n_estimators is the number of individual decisions trees we want to train
    # higher n_est the more accurate the model is but the longer it takes to run
    # min_sample_split helps with overfitting if they build it too deeply
    # higher min_s_s the less accurate the model but the less it will overfit
    # experiment with this
    # random_states means if we run the same numbers twice it will be in a predictable sequence
    # helps with updating the model because it is based on you
    model = RandomForestClassifier(n_estimators=n_est, min_samples_split=min_s_s, random_state=1)

    # now train the model where we use the predictors
    model.fit(training[predictors], training['Target'])

    return model


def test_accuracy(model, testing, predictors, show=False, name=None):
    # modified code from https://github.com/dataquestio/project-walkthroughs/blob/master/sp_500/market_prediction.ipynb
    # by making it into a function

    # return a precision score for the models accuracy
    # this generates a numpy array of predictions
    prediction = model.predict(testing[predictors])

    # make it into a pandas series
    if name == None:
        prediction = pd.Series(prediction, index=testing.index)
    else:
        prediction = pd.Series(prediction, index=testing.index, name=name)

    # get the precision score
    score = precision_score(testing['Target'], prediction, zero_division=0)

    if show:
        print(f'The precision score is {score} for the predictors - {predictors}.')

    return prediction


def predict(model, training, testing, predictors):
    # modified code from https://github.com/dataquestio/project-walkthroughs/blob/master/sp_500/market_prediction.ipynb

    model.fit(training[predictors], training["Target"])
    # call test_accuracy to get the predictions
    prediction = test_accuracy(model, testing, predictors, name='Predictions')

    # combine the actual target values with the predicted values into a df
    prediction_df = pd.concat([testing['Target'], prediction], axis=1)

    return prediction_df


def custom_predict(model, training, testing, predictors, prob):
    # rewrote the code from predict to include the probability threshold that can be changed

    model.fit(training[predictors], training["Target"])
    # similar to predict but you are getting the prediction probability
    # this gives more control over what is defined as a 0 or 1
    prediction = model.predict_proba(testing[predictors])[:, 1]
    # typical threshold is .5
    # set custom threshold so that the model is more confident the price will go up
    # reduces total number of trading days that will go up but is more confident as to when it will go up
    # we wantto be more confident when it will go up because that is how we make money
    prediction[prediction >= prob] = 1
    prediction[prediction < prob] = 0
    prediction = pd.Series(prediction, index=testing.index, name='Predictions')
    prediction_df = pd.concat([testing['Target'], prediction], axis=1)
    return prediction_df


def backtest(df, model, predictors, prob, window=2500, size=250, custom=False):
    # modified code from https://github.com/dataquestio/project-walkthroughs/blob/master/sp_500/market_prediction.ipynb
    # by including the customization of it

    # window value is the amount of data you are training your model with
    # size value is the size of each window and is based on there being about 250 trading days per year
    # basically take 10 years of data and train your model for a year then go into the next year until 10 years
    results = []

    # iterate through the data window using the size of each window
    # basically go year by year
    for i in range(window, df.shape[0], size):
        # split data into training and testing for the current window
        training = df.iloc[0:i].copy()
        testing = df.iloc[i:(i + size)].copy()

        # make predictions for this window using the predict
        if custom:
            # if custom use the more specific prediction model
            prediction_df = custom_predict(model, training, testing, predictors, prob)
        else:
            # if just using the base model
            prediction_df = predict(model, training, testing, predictors)

        # add predictions for this window
        results.append(prediction_df)

    # combine the results from all windows/years into a df
    results_df = pd.concat(results)

    return results_df


def add_horizon(horizons, df):
    # modified code from https://github.com/dataquestio/project-walkthroughs/blob/master/sp_500/market_prediction.ipynb
    # by refactoring it as a function

    new_predictors = []
    for horizon in horizons:
        # take the rolling average for each horizon
        rolling_avg = df.rolling(horizon).mean()

        ratio_col = f"Close_Ratio_{horizon}"
        # add the ratio between the close and the rolling average
        df[ratio_col] = df['Close'] / rolling_avg['Close']

        # the number of days in the horizon that the price actually went up
        trend_col = f"Trend_{horizon}"
        # shift forward to find the rolling sum of the target
        df[trend_col] = df.shift(1).rolling(horizon).sum()['Target']

        # add these columns to the new predictors list
        new_predictors += [ratio_col, trend_col]

    return new_predictors


def calc_rsi(df, window=14):
    # RSI idea from
    # https://github.com/johnberroa/RandomForest-StockPredictor/blob/master/Technical%20Analysis%20Notebook.ipynb
    # function to calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def print_score(ticker, score, predictors, bt=False, n_est=100, min_s_s=100, prob=0.5):
    # function to print the text explaining the precision score with the aspects of the model included
    if score == 0:
        print(f'\nUsing this back tested model for {ticker}, the precision score is {score} with number of estimates '
              f'being {n_est}, minimum sample size being {min_s_s}, \nthe probability threshold being {prob}, and these '
              f'predictors - {predictors}. This means that the model did not predict any days would increase.')
    elif bt:
        print(f'\nUsing this back tested model for {ticker}, the precision score is {score} with number of estimates '
              f'being {n_est}, minimum sample size being {min_s_s}, \nthe probability threshold being {prob}, and these '
              f'predictors - {predictors}. This means that the model accurately predicted that \nthe price would '
              f'increase about {round((score * 100), 2)}% of the time.')
    else:
        print(f'\nUsing this model for {ticker}, the precision score is {score} with number of estimates '
              f'being {n_est}, minimum sample size being {min_s_s}, \nthe probability threshold being {prob}, and '
              f'these predictors - {predictors}. This means that the model accurately predicted that \nthe price would '
              f'increase about {round((score * 100), 2)}% of the time.')


def check_score(ticker, score, predictors, bt, best_bt, best_precision, best_predictors, n_est=100, min_s_s=100,
                prob=0.5, best_n_est=100, best_min_s_s=100, best_prob=0.5):
    # checks to see if the new precision score from the new model is better than the best precision score saved
    better = False
    if best_bt:
        best_bt_str = 'back tested'
    else:
        best_bt_str = 'not back tested'
    if bt:
        bt_str = 'back tested'
    else:
        bt_str = 'not back tested'
    if score < best_precision:
        print(f'\nThis model had a lower precision score than the model with predictors - {best_predictors}, '
              f'which was about {round((best_precision * 100), 2)}% of the time. \nThis means that you are better off '
              f'not using this model and using the model with number of estimates being {best_n_est}, minimum sample '
              f'size being {best_min_s_s}, \nprobability threshold being {best_prob}, predictors - {best_predictors} '
              f'and {best_bt_str} to invest in {ticker}.')
    else:
        print(f'\nThis model had a higher precision score than the model with predictors - {best_predictors}, '
              f'which was about {round((best_precision * 100), 2)}% of the time. \nThis means that you are better off'
              f' using this algorithm with number of estimates being {n_est}, minimum sample '
              f'size being {min_s_s}, \nprobability threshold being {prob}, predictors - {predictors} and '
              f'{bt_str} to invest in {ticker}.')
        better = True

    return better


if __name__ == '__main__':
    # dictionary of predictors and their descriptions
    pred_dict = {'Close': 'final trading price of the stock for the day',
                 'Volume': 'total number of shares traded during a given trading session indicating the level of '
                           'activity/liquidity in the market and can assess the significance in price movements',
                 'Open': 'price of the first traded stock at the beginning of a trading session and can influence '
                         'investor\'s choices and trading strategies that day',
                 'High': 'highest price the stock reaches during trading session and can indicate potential '
                         'resistance levels',
                 'Low': 'lowest price the stock reaches during trading session and can indicate potential support '
                        'levels',
                 'Horizon': 'time period used for calculating certain technical indicators and can capture loner-term '
                            'trends/patterns',
                 'RSI': 'momentum oscillator that measures speed and change of price movements helping the model '
                        'identify potential trend reversals or momentum shifts in stock price'
                 }

    # save the possible predictors to a list
    poss_pred = ['Close', 'Volume', 'Open', 'High', 'Low', 'Horizon', 'RSI']

    # Welcome message
    print("Welcome to the beginner friendly Machine Learning Stock Market Analysis Tool!")
    print("-----------------------------------------------------------------------------\n")

    # Input: What stock ticker do you want to analyze
    ticker = input("Enter the stock ticker you want to analyze (like S & P 500 is '^SPX', Amazon is 'AMZN', Google is "
                   "'GOOG'; will be defaulted to S & P 500 if incorrect ticker): ")
    print('')

    if is_valid_ticker(ticker):
        # download the ticker data
        df = download_ticker_data(ticker=ticker, period='max')

    else:
        # defaults to s&p 500 if invalid ticker given
        ticker = '^SPX'
        df = download_ticker_data(period='max')

    if len(df) == 0:
        ticker = '^SPX'
        df = download_ticker_data(period='max')

    # cleans the data so there are no missing values or 0 values
    df = clean_data(df)

    # set the index as date
    df.set_index('Date', inplace=True)

    # show the ticker data
    print(f'The cleaned data for {ticker}:')
    print(df)

    print(f'\nNow graphing {ticker}...\n')
    # graph the stock ticker
    plt.figure(figsize=(50, 10))
    plt.plot(df.index, df['Close'])
    plt.title(f'{ticker} Close price.')
    plt.ylabel('Price in dollars.')
    plt.grid(True)
    # from chatgpt to have the x-axis in years
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.show()

    # model can be very accurate but you can still lose a ton of money
    # because you only care about the directionality of the stock price
    # so you want to know if it will go up or down tomorrow

    # use pandas shift to take the previous day's close price
    df['Tomorrow'] = df['Close'].shift(-1)

    # based on tomorrow's price create a target to see if it is greater than today's price
    # this returns bool so make it type int
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)

    print('There is a lot of historical data, so some older data is not useful.')
    print('I would suggest picking a data point at the beginning of a more volatile period (example for ^SPX is '
          '1990-01-02).\n')
    # set up input choice to have where you want to start for machine learning
    start_date = input("Enter the start date for analysis (year-month-day, like '1990-01-02'): ")
    # update df to only include data from that point to current
    df = df.loc[start_date:].copy()

    print(f'\nThis is the new data we are analyzing for {ticker} from {start_date} to present')
    # show the new ticker data we are analyzing
    print(df)

    # save the original df of stock data we are analyzing for the horizons
    original_df = df

    print(f'\nNow graphing the data of {ticker} we are analyzing...')
    # graph the new data
    plt.figure(figsize=(50, 10))
    plt.plot(df.index, df['Close'])
    plt.title(f'{ticker} Close price from {start_date} to Present.')
    plt.ylabel('Price in dollars.')
    plt.grid(True)
    # from chatgpt to have the x-axis in years
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.show()

    print(f'\nWe will now split the {ticker} data into a training and testing set to try to predict if the price will '
          'increase or decrease.')
    print('Because of the dependence of stock prices we cannot use cross validation typically '
          'used in machine learning.')
    print('Cross validation does not take time into consideration and poses the problem '
          'of future data leaking into past data. ')
    print('This does not create an effective stock analysis model for real '
          'world trading scenarios where it cannot as easily generalize to them as in the training.\n')

    # Input: How much data do you want to train and test
    slice_point = int(input("Enter how much data to include for testing, the rest will be in training (integer, e.g., "
                            "100): "))
    # negate it because the beginning of the data will be training and the end is testing
    slice_point *= -1

    training = df.iloc[:slice_point]
    testing = df.iloc[slice_point:]

    # set the predictors we will use
    # do not use tomorrow or target because they will not give accurate analysis in current world
    print('\nNow we are setting the predictors for the base model. They are close, volume, open, high, and low.')
    print('We will first analyze the performance of these predictors and then add more predictors and change the '
          'thresholds to try to get the highest precision for accurately predicting that the price will go up.')
    predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
    predictors_str = ['Close', 'Volume', 'Open', 'High', 'Low']

    # call base model of machine learning to see how accurate the predictors are
    model = make_model(training, predictors)

    # check the precision of the model
    prediction = test_accuracy(model, testing, predictors)

    # get the score to then interpret it
    score = precision_score(testing['Target'], prediction, zero_division=0)

    # print the precision score of the base model
    # print_score(ticker, score, predictors)

    # find the actual proportion of days the market went up
    actual_score = df[df['Target'] == 1].shape[0] / df.shape[0]

    print('\nNow we are backtesting the base model. \nThe backtesting function evaluates the performance of a '
          'machine learning predictive model over different time windows/periods of historical data. \nIt splits '
          'the available data into training and testing sets, trains the model on the training data, and '
          'then evaluates its performance on the testing data. \nUltimately providing insights '
          'into the model\'s effectiveness in predicting stock price movements.')
    # backtest the base model we already created
    predictions_df = backtest(df, model, predictors, .5)
    bt = True

    print('\nThe back tested model made the following predictions: ')
    # check the days the model predicted the stock would go up vs down
    # 0 is down and 1 is up
    print(predictions_df["Predictions"].value_counts())

    # check precision score
    score = precision_score(predictions_df['Target'], predictions_df['Predictions'], zero_division=0)
    print_score(ticker, score, predictors_str, bt=bt)

    if score < actual_score:
        best_bt = False
        best_precision = actual_score
        best_predictors = []
        print(f'\nThe back tested base model had a lower precision score than the actual proportion of days '
              f'{ticker} increased which was about {round((actual_score * 100), 2)}% of the time.\nThis means that'
              f' you are better off not using the algorithm and just investing regularly.')
    else:
        best_bt = bt
        best_precision = score
        best_predictors = predictors_str.copy()
        print(f'\nThe back tested base model had a higher precision score than the actual proportion of days '
              f'{ticker} increased which was about {round((actual_score * 100), 2)}% of the time.\nThis means '
              f'that you are better off using the back tested base algorithm to invest in {ticker}.')

    # since we would be better off not using our algorithm we are going to have to update it
    # give the algorithm more so that it can become more accurate

    curr_n_est = 100
    curr_min_s_s = 100
    curr_prob = 0.5
    best_n_est = 100
    best_min_s_s = 100
    best_prob = 0.5
    # Loop for adding more predictors
    while True:
        # Input: Do you want to continue?
        continue_analysis = input("\nDo you want to continue (yes/no)? ")
        if continue_analysis.lower() != 'yes':
            break
        cont = True

        while cont:
            if best_precision == actual_score:
                print(f'\nThere is currently no best model, randomly trading {ticker} is the best.')
                print(f'Percent of days that increase: {round((best_precision * 100), 2)}%')
            else:
                # show the current best model's attributes
                print(f'\nThe current best model has:')
                print(f'Precision score: {round((best_precision * 100), 2)}%')
                print(f'Predictors: {best_predictors}')
                print(f'Backtested: {best_bt}')
                print(f'Number of estimators: {best_n_est}')
                print(f'Minimum sample split: {best_min_s_s}')
                print(f'Probability threshold: {best_prob}')
                print(f'\nAnd the actual percent of days that the price increases is {round((actual_score * 100), 2)}% '
                      f'(no model).')

            # provide options to change in the model
            print('\nYou have the following options to change in the model:')
            print('1. Predictors - the variables used to make predictions about whether the stock price will go up '
                  'or down, you can add or remove predictors')
            print('2. Number of estimators - number of decisions trees we want to train in the model')
            print('3. Minimum sample split - minimum number of samples required to split an internal node during the '
                  'construction of each decision tree')
            print('4. Probability threshold - determines the level of confidence required for the model to classify an '
                  'instance as positive\n')

            choice = int(input("Enter the number for what you want to change. "))

            changed = True
            no_input = False

            # choice
            if choice == 1:
                # predictor option
                print(f'\nBy analyzing the predictors, the model tries to identify patterns or relationships that '
                      f'can help forecast future stock price movements.')
                # show the current predictors for them to decide if they want to add or remove
                print(f'\nThe predictors of the current model are: ')
                for predictor in predictors_str:
                    # use the dictionary to print the definition
                    print(f'{predictor}: {pred_dict[predictor]}')

                # ask for input
                change = input("\nDo you want to add or remove a predictor (add/remove)? ")

                # find the possible predictors that are not already in predictors
                no_pred = []
                for predictor in poss_pred:
                    if predictor not in predictors_str:
                        no_pred.append(predictor)

                # if they want to add provide the predictors they can add
                if change == 'add' and len(no_pred) != 0:
                    # for the other possible predictors one can add
                    print('\nThe predictors you can add to the current model: ')
                    for predictor in no_pred:
                        print(f'{predictor}: {pred_dict[predictor]}')
                        # print this warning if Horizon is a choice available to add
                        if predictor == 'Horizon':
                            print('* if you choose Horizon, we will clear the rest of the predictors and it will '
                                  'become the only one *')

                    # ask for input of which one like to add
                    print('If you add a predictor and the only predictor is Horizon, then Horizon will be deleted.')
                    add = input('\nWhich predictor would you like to add (type it exactly as it appears)? ')

                    if predictors_str[0] == 'Horizon' and add != 'Horizon':
                        predictors_str.remove('Horizon')
                        predictors = predictors_str
                        df = original_df

                    if add in predictors_str:
                        changed = False

                    # if they want to add horizon call function
                    elif add == 'Horizon':
                        print('\nWe are deleting the other predictors now.')
                        print('We are adding the following horizon approximates: 2 days (2), 1 week (5), 3 months '
                              '(60), 1 year (250), 4 years (1000). These are the closing periods stock traders use '
                              'in real life.')
                        horizons = [2, 5, 60, 250, 1000]
                        predictors_str = ['Horizon']
                        predictors = add_horizon(horizons, df)

                        # drop the dates that have missing data
                        df = df.dropna(subset=df.columns[df.columns != 'Tomorrow'])

                        print(f'\nThis is the new {ticker} data including the horizons:')
                        print(df)

                    elif add == "RSI":
                        # if they want to add RSI to the model
                        print('\nWe are adding the Relative Strength Index as a new predictor, which includes a '
                              'specific lookback period.')
                        print('The lookback period specifies the number of previous day\'s prices that are included in '
                              'the calculation of RSI.')
                        print('Typically the lookback period is 14 days.')
                        per = int(input('\nWhat lookback period do you want to use (integer like 14)? '))
                        df.loc[:, 'RSI'] = calc_rsi(df, window=per)
                        predictors.append(add)
                        predictors_str.append(add)

                    else:
                        predictors.append(add)
                        predictors_str.append(add)

                elif change == 'remove' and len(predictors) != 0:
                    remove_pred = input('\nWhich predictor would you like to remove (type it exactly as it appears)? ')
                    if remove_pred in predictors_str:
                        predictors_str.remove(remove_pred)
                        if remove_pred == "Horizon":
                            predictors = predictors_str
                            df = original_df
                        else:
                            predictors.remove(remove_pred)
                    else:
                        changed = False

                else:
                    changed = False

            elif choice == 2:
                # number of estimators
                print(f'\nThe current number of estimators is {curr_n_est} and the best is {best_n_est}.')
                print(f'The higher n_estimates is, the more accurate the model, but the longer it takes to run.\n')
                n_est = int(input('What number of estimates do you want to use? '))
                if n_est == curr_n_est:
                    changed = False
                else:
                    curr_n_est = n_est

            elif choice == 3:
                # minimum sample split
                print(f'\nThe current minimum sample split is {curr_min_s_s} and the best is {best_min_s_s}.')
                print(
                    f'The min sample split helps with overfitting. The higher the min sample split, the less accurate the'
                    f' model, but the less it will overfit.\n')
                min_s_s = int(input('What minimum sample split do you want to use? '))
                if min_s_s == curr_min_s_s:
                    changed = False
                else:
                    curr_min_s_s = min_s_s

            elif choice == 4:
                # probability threshold
                print(f'\nThe current probability threshold is {curr_prob} and the best is {best_prob}.')
                print('The defaulted probability threshold is 0.5, so you would want to input one between 0.5 and 1.')
                print('The higher the threshold, the more confident the model is that the price will go up, but there '
                      'will be less increase predictions.')
                print('This is important for traders because we want to make money, not lose money.\n')
                prob = float(input('What probability threshold do you want to use? '))
                if prob == curr_prob:
                    changed = False
                else:
                    curr_prob = prob

            else:
                changed = False
                print('Nothing changed!')
                no_input = True

            if not changed and not no_input:
                print('Unable to make the change')
            cont_change = input("\nDo you want to continue changing the model (yes/no)? ")
            if cont_change.lower() != 'yes':
                cont = False

        if changed:
            print('\nRunning new model...\n')

            training = df.iloc[:slice_point]
            testing = df.iloc[slice_point:]

            # make new model with whatever changed
            new_model = make_model(training, predictors, n_est=curr_n_est, min_s_s=curr_min_s_s)

            # run the new model
            predictions_df = backtest(df, new_model, predictors, curr_prob, custom=True)

            # show how many days where either predicted to increase or decrease
            print('This shows how many days the model was predicted to either increase (1) or decrease (0):')
            print(predictions_df['Predictions'].value_counts())

            # check the precision score of the predictions
            score = precision_score(predictions_df['Target'], predictions_df['Predictions'], zero_division=0)

            print_score(ticker, score, predictors_str, bt=bt, n_est=curr_n_est, min_s_s=curr_min_s_s, prob=curr_prob)

            if best_precision == actual_score:
                if score < best_precision:
                    print(f'\nThe back tested model had a lower precision score than the actual proportion of days '
                          f'{ticker} increased which was about {round((actual_score * 100), 2)}% of the time. \nThis '
                          f'means that you are better off not using the model and just investing regularly.')
                else:
                    best_bt = bt
                    best_precision = score
                    best_predictors = predictors_str.copy()
                    best_n_est = curr_n_est
                    best_min_s_s = curr_min_s_s
                    best_prob = curr_prob
                    print(f'\nThe back tested model had a higher precision score than the actual proportion of days '
                          f'{ticker} increased which was about {round((actual_score * 100), 2)}% of the time. \nThis '
                          f'means that you are better off using the back tested base algorithm to invest in {ticker}.')
            else:
                if check_score(ticker, score, predictors_str, bt, best_bt, best_precision, best_predictors, curr_n_est,
                               curr_min_s_s, curr_prob, best_n_est, best_min_s_s, best_prob):
                    best_bt = bt
                    best_precision = score
                    best_predictors = predictors_str
                    best_n_est = curr_n_est
                    best_min_s_s = curr_min_s_s
                    best_prob = curr_prob

            # filter predictions_df where predictions is 1 and target is 0 to show where it predicted incorrectly
            filtered_pred = predictions_df[(predictions_df['Predictions'] == 1) & (predictions_df['Target'] == 0)]

            print('\nThe model incorrectly predicted that the price would go up on these dates when it actually '
                  'did not:')
            # print the filtered df
            print(filtered_pred)

        else:
            print('\nYou did not change anything!')

    print(f"\n\nAnalysis of {ticker} complete!")
    print(f"The best precision score was {best_precision} or {round((best_precision * 100), 2)}% accurate.")
    if best_precision == actual_score:
        print(f"You are better off not using any of the models you created to trade {ticker}.")
    else:
        if best_bt:
            bt_str = 'back tested'
        else:
            bt_str = 'not back tested'
        print(f"The best model you used was {bt_str}, had a number of estimates of {best_n_est}, had a minimum "
              f"sample size of {best_min_s_s}, \nhad a probability threshold of {best_prob}, and included these "
              f"predictors - {best_predictors}.")
