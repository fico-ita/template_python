"""Provide technical details of the proposed low-volatility-momentum strategy.

This module allows the user to make a wallet based on said strategy.

The module contains the following functions:

- `initial_analysis(dict_data, stdout)` - Select top 10 stocks based on stocks' momentum
    scores and volatilities.
- `df_to_windowed_df(dataframe, first_date, last_date, n)` - Transform the stocks
    dataframes in 60-day windowed dataframes.
- `lstm_strategy(dict_data, t, stdout, show_real_returns)` - Executes the proposed
    low-volatility-momentum strategy
- `mount_wallet(sel_stocks, dfs_dict, date, stdout, show_real_returns)` - Returns
    weights for mounting the stocks wallet.
- `optimize_model(scaled_X_train, scaled_y_train, scaled_X_val, scaled_y_val)` - Choose
    the best number of Dense layers and neurons in each for the neural network.
- `prepare_model_data(sel_stocks, t, stdout)` - Creates the data shape for model.
- `train_model(stock, dfs_dict, show_real_returns, stdout)` - Trains model to find best
    Dense layers and neurons numbers combination for each neural network.
- `str_to_datetime(s)` - Transforms date in string format to datetime format.
- `windowed_df_to_date_X_y(windowed_dataframe)` - Transform the stocks dataframes in
    60-day windowed dataframes.

Examples:
    Examples should be written in `doctest` format, and should illustrate how to use the
    function.

    >>> from fico.lstm_strategy import initial_analysis, prepare_model_data
    >>> data_dict = load_data()  # Assuming load_data is imported
    *image of data_dict*
    >>> selected_stocks = initial_analysis(data_dict)
    *image of selected_stocks*
    >>> from fico.lstm_strategy import mount_wallet
    >>> portfolio = mount_wallet(selected_stocks, model_data, show_real_returns = True)
    *image of portfolio*
"""

import copy
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from modules.load_data import load_data
from sklearn.metrics import mean_squared_error
from tensorflow.keras import activations, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
# """Transform the stocks dataframes in 60-day windowed dataframes.

#     Args:
#       Dataframe: stock dataframe
#       String: initial date
#       String: final date

#     Returns:
#       Dataframe: 60-day windowed dataframe
#     """

def initial_analysis(dict_data: object, stdout: bool =True):
    """Select top stocks based on stocks' momentum scores analyzed for 3 periods: 1-month momentum, 3-month and
    6-month momentum, compounded with low volatility filtering. A 'low-volatility-momentum' strategy.

    Args:
        Dict: Dictionary with historical stocks data.

    Returns:
        Array: Selected stocks.
    """
    log_returns = np.log(dict_data["prices"]).diff().fillna(0)

    volatility_window = 63  # Assuming daily data

    # Initialize an empty DataFrame to store momentum scores of each analyzed period, and volatility within the 63-day period:
    momentum_scores_df = pd.DataFrame(index=log_returns.index)
    momentum_scores_21 = pd.DataFrame(index=log_returns.index)
    momentum_scores_63 = pd.DataFrame(index=log_returns.index)
    momentum_scores_126 = pd.DataFrame(index=log_returns.index)
    vol_df = pd.DataFrame(index=log_returns.index)
    volatilities = []
    for column in log_returns:
        # Calculate the rolling standard deviation to measure mean volatility for this stock
        vol_df[f"{column}"] = np.nan_to_num(
            log_returns[column].rolling(volatility_window).std(),
        )
        vol_values = np.nan_to_num(
            log_returns[column].rolling(volatility_window).std().values,
        )
        mean_vol = np.mean(vol_values)
        volatilities.append(mean_vol)

        # Define multiple momentum periods in terms of trading days
        momentum_periods = [21, 63, 126]  # 1-month, 3-month, and 6-month

        # Calculate momentum scores for each period and store in the DataFrame
        for period in momentum_periods:
            momentum_scores = log_returns[column].rolling(period).sum()
            momentum_scores = momentum_scores.shift(-period)
            if period == 21:
                momentum_scores_21[f"{column}"] = np.nan_to_num(momentum_scores)
            elif period == 63:
                momentum_scores_63[f"{column}"] = np.nan_to_num(momentum_scores)
            else:
                momentum_scores_126[f"{column}"] = np.nan_to_num(momentum_scores)

    weight_21 = 0.4
    weight_63 = 0.3
    weight_126 = 0.3

    momentum_scores_df = (
        weight_21 * momentum_scores_21
        + weight_63 * momentum_scores_63
        + weight_126 * momentum_scores_126
    )
    momentum_scores_df.fillna(0)
    vol_df.fillna(0)
    column = "KR"  # Change here to evaluate frequencies of different stocks

    if stdout:
        for period in momentum_periods:
            if period == 21:
                plt.figure()
                sns.histplot(
                    momentum_scores_21[f"{column}"],
                    bins=30,
                    kde=True,
                    color="blue",
                    edgecolor="k",
                )
                plt.title(
                    f"Distribution of '{column}' Stock {period}-Day Momentum Scores (Log Returns)",
                )
                plt.xlabel("Momentum Score")
                plt.ylabel("Frequency")
                plt.grid(True)
            elif period == 63:
                plt.figure()
                sns.histplot(
                    momentum_scores_63[f"{column}"],
                    bins=30,
                    kde=True,
                    color="blue",
                    edgecolor="k",
                )
                plt.title(
                    f"Distribution of '{column}' Stock {period}-Day Momentum Scores (Log Returns)",
                )
                plt.xlabel("Momentum Score")
                plt.ylabel("Frequency")
                plt.grid(True)
            elif period == 126:
                plt.figure()
                sns.histplot(
                    momentum_scores_126[f"{column}"],
                    bins=30,
                    kde=True,
                    color="blue",
                    edgecolor="k",
                )
                plt.title(
                    f"Distribution of '{column}' Stock {period}-Day Momentum Scores (Log Returns)",
                )
                plt.xlabel("Momentum Score")
                plt.ylabel("Frequency")
                plt.grid(True)

        plt.figure()
        sns.histplot(
            momentum_scores_df[f"{column}"],
            bins=30,
            kde=True,
            color="blue",
            edgecolor="k",
        )
        plt.title(
            f"Distribution of '{column}' Stock Weighted (.4, .3, .3) Momentum Scores (Log Returns)",
        )
        plt.xlabel("Momentum Score")
        plt.ylabel("Frequency")
        plt.grid(True)

    # Define thresholds for low volatility and momentum based on volatilities array and momentum_scores_df
    low_volatility_threshold = 0.02
    momentum_threshold = np.log(
        1.1,
    )  # 110% momentum score, an attempt to guarantee nice stock volume with good upward prospect trend.

    # Create boolean DataFrames for low volatility and momentum
    low_volatility_condition = vol_df < low_volatility_threshold
    momentum_condition = momentum_scores_df > momentum_threshold

    # Combine the conditions to select best 10 assets
    selected_assets_df = low_volatility_condition & momentum_condition

    best_assets = []

    for column in selected_assets_df.columns.tolist():
        occurrences = np.count_nonzero(np.array(selected_assets_df[column]))
        best_assets.append({"name": column, "occurrences": occurrences})

    best_assets = sorted(best_assets, key=lambda x: x["occurrences"], reverse=True)
    sel_stocks = [i["name"] for i in best_assets[:10]]

    # Print the maximum volatility
    if stdout:
        print("Maximum volatility: ", np.max(volatilities))

    return sel_stocks


def prepare_model_data(sel_stocks, t, stdout=True):
    """This function creates the data shape for model.

    Args:
      Array: selected stocks

    Returns:
      Dictionary: {
        'param1': 60-day windowed dataframes of the selected stocks,
        'param2': maximum scale for normalizing data}
    """
    dict_data = load_data()
    df = dict_data["prices"].astype("Float32")
    df = df[sel_stocks]
    log_df = np.log(df).diff().fillna(0)

    if stdout:
        plt.figure()
        for stock in sel_stocks:
            plt.plot(df.index, list(df[stock].values))
        plt.legend(sel_stocks)

        plt.figure()
        for stock in sel_stocks:
            plt.plot(log_df.index, list(log_df[stock].values))
        plt.legend(sel_stocks)

    # 60 days windowed dataframes

    num_of_past_dates = 60
    last_date_timestamp = dict_data["prices"].index[t]
    initial_date_timestamp = dict_data["prices"].index[t - 400]
    windowed_dfs = {}
    for stock in sel_stocks:
        windowed_stock_df = df_to_windowed_df(
            log_df[[stock]],
            initial_date_timestamp,  #'2017-12-29', # t_i = 1656
            last_date_timestamp,  #'2019-06-28', # t_f = 2031
            n=num_of_past_dates,
        )
        windowed_dfs[stock] = windowed_stock_df

    # Finds maximum for log returns to normalize data

    maxes = [
        i.drop(columns=["Target Date", "Target of Stock"]).max().max()
        for i in windowed_dfs.values()
    ]
    max_scale = np.max(maxes)

    if stdout:
        print("Maximum scale: ", max_scale)

    return {
        "windowed_dfs": windowed_dfs,
        "max_scale": max_scale,
    }


def mount_wallet(sel_stocks, dfs_dict, date, stdout=True, show_real_returns=True):
    """This function returns weights for mounting the stocks wallet.

    Args:
      Array: selected stocks
      Dictionary: {
        'param1': 60-windowed dataframes of the selected stocks,
        'param2': maximum scale for normalizing data
      }

    Returns:
      Dataframe: Wallet with weights, predicted returns and real returns
    """
    predicted_returns = []
    real_returns = []

    for stock in sel_stocks:
        predicted_return, real_return = train_model(stock, dfs_dict, stdout)
        predicted_returns.append(predicted_return)
        real_returns.append(real_return)

    weights = abs(
        np.array([i if i > 0 else 0 for i in predicted_returns])
        / np.sum(np.array([i if i > 0 else 0 for i in predicted_returns])),
    )
    weights = np.around(weights, decimals=4)

    weights_df = pd.DataFrame(
        {
            "ticker": sel_stocks,
            "weights": weights,
            "Predicted Returns": predicted_returns,
        },
    )
    if show_real_returns:
        weights_df["Real Returns"] = real_returns
    index = [date] * 10
    weights_df.index = index

    return weights_df


# Private Functions


def str_to_datetime(s):
    """This transforms date in string into datetime format.

    Args:
      String: date

    Returns:
      Datetime: date
    """
    split = s.split("-")
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)


# Transform data in a 60-day windowed dataframe


def df_to_windowed_df(dataframe, first_date, last_date, n=3):
    """Transform the stocks dataframes in 60-day windowed dataframes.

    Args:
      Dataframe: stock dataframe
      String: initial date
      String: final date

    Returns:
      Dataframe: 60-day windowed dataframe
    """
    if isinstance(first_date, str):
        first_date = str_to_datetime(first_date)

    if isinstance(last_date, str):
        last_date = str_to_datetime(last_date)

    target_date = first_date

    dates = []
    X, Y = [], []

    last_time = False
    col_name = list(dataframe.columns)[0]
    while True:
        df_subset = dataframe.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            print(f"Error: Window of size {n} is too large for date {target_date}")
            return None

        values = df_subset[col_name].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[
            target_date : target_date + datetime.timedelta(days=7)
        ]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split("T")[0]
        year_month_day = next_date_str.split("-")
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df["Target Date"] = dates

    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f"Target-{n-i}"] = X[:, i]

    ret_df["Target of Stock"] = Y

    return ret_df


# Helps separating data for training the model


def windowed_df_to_date_X_y(windowed_dataframe):
    """Transform the stocks dataframes in 60-day windowed dataframes.

    Args:
      Dataframe: 60-day windowed dataframe

    Returns:
      Array: index with dates
      Array: momentum input for the network (60 log-returns)
      Array: target value for the neural network
    """
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1)).astype(
        np.float32,
    )

    Y = df_as_np[:, -1].astype(np.float32)

    return dates, np.float32(X), np.float32(Y)


def optimize_model(scaled_X_train, scaled_y_train, scaled_X_val, scaled_y_val):
    """Choose the best number of Dense layers and neurons in each for the neural network.

    Args:
      Array: train input
      Array: train output
      Array: validation input
      Array: validation output

    Returns:
      Tuple: (number of dense layers, number of neurons in layer 1, number of neurons in other layers)
    """
    # Params for choosing best network
    num_dense_layers_list = [1, 2, 3]  # Number of Dense layers
    num_neurons_list = [4, 8, 16, 32]  # Number of neurons in every dense layer

    best_mse = float("inf")  # Best MSE initialization with infinity
    best_combination = None  # Best combination of hyperparameters initialized with None

    for num_dense_layers in num_dense_layers_list:
        for num_neurons1 in num_neurons_list:
            for num_neurons2 in [
                n for n in num_neurons_list if n <= num_neurons1
            ]:  # Iterate over num_neurons2 <= num_neurons1
                print(
                    f"Experimenting with {num_dense_layers} Dense layers: First layer {num_neurons1} neurons, Second layer {num_neurons2} neurons",
                )

                # Create model
                model = Sequential([layers.Input(shape=(60, 10)), layers.LSTM(96)])

                model.add(layers.Dense(num_neurons1, activation=activations.elu))

                if num_dense_layers >= 2:
                    model.add(layers.Dense(num_neurons2, activation=activations.elu))

                model.add(layers.Dense(1))

                # Compiling the model
                optimizer = Adam(learning_rate=0.0001, epsilon=1e-10)
                model.compile(
                    loss=Huber(delta=1.0),
                    optimizer=optimizer,
                    metrics=["mean_squared_error"],
                )

                # Adding Early Stopping for avoiding overfitting
                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=20,
                    restore_best_weights=True,
                )

                # Train the model
                model.fit(
                    scaled_X_train,
                    scaled_y_train,
                    validation_data=(scaled_X_val, scaled_y_val),
                    epochs=100,
                    callbacks=[early_stopping],
                    verbose=0,
                )

                # Evaluate the model
                train_predictions = model.predict(scaled_X_train).flatten()
                mse = mean_squared_error(scaled_y_train, train_predictions)
                print("MSE:", mse)
                print("\n")

                # Update the best combination if a better combination is found
                if mse < best_mse:
                    best_mse = mse
                    best_combination = (num_dense_layers, num_neurons1, num_neurons2)

    return best_combination


# Trains model based on best combination


def train_model(stock, dfs_dict, show_real_returns, stdout=True):
    """Choose the best number of Dense layers and neurons in each for the neural network.

    Args:
      String: stock
      Dictionary: {
        'param1': 60-windowed dataframes of the selected stocks,
        'param2': maximum scale for normalizing data
      }

    Returns:
      Tuple: (predicted returns, real returns)
    """
    max_scale = dfs_dict["max_scale"]
    stocks_df = copy.deepcopy(dfs_dict["windowed_dfs"])
    stocks_number = len(stocks_df.keys())
    dates, X1, y1 = windowed_df_to_date_X_y(stocks_df[stock])

    del stocks_df[stock]

    # Separating data in sets for training, validation and test

    q_80 = int(len(dates) * 0.8)
    q_90 = int(len(dates) * 0.9)

    X_train = X1[:q_80]
    y_train = y1[:q_80]

    X_val = X1[q_80:q_90]
    y_val = y1[q_80:q_90]

    X_test = X1[q_90:]
    y_test = y1[q_90:]

    dates_train = dates[:q_80]
    dates_val = dates[q_80:q_90]
    dates_test = dates[q_90:]

    best_combinations = {}

    # Gathering exogenous data:
    for stock_name in stocks_df:
        _, X, y = windowed_df_to_date_X_y(stocks_df[stock_name])
        X_train = np.concatenate((X_train, X[:q_80]), axis=2)
        y_train = np.concatenate((y_train, y[:q_80]))

        X_val = np.concatenate((X_val, X[q_80:q_90]), axis=2)
        y_val = np.concatenate((y_val, y[q_80:q_90]))

        X_test = np.concatenate((X_test, X[q_90:]), axis=2)
        y_test = np.concatenate((y_test, y[q_90:]))

    scaled_X_train = X_train / max_scale
    scaled_y1_train = y1[:q_80]

    scaled_X_val = X_val / max_scale
    scaled_y1_val = y1[q_80:q_90] / max_scale

    scaled_X_test = X_test / max_scale
    scaled_y1_test = y1[q_90:] / max_scale

    # best_combination = optimize_model(scaled_X_train, scaled_y1_train, scaled_X_val, scaled_y1_val)
    # best_combinations[stock] = best_combination

    # After running optimization, the best combinations for each stock are:
    best_combinations = {
        "DPZ": [1, 16, 4],
        "WST": [3, 32, 8],
        "ODFL": [1, 32, 32],
        "MKTX": [1, 16, 16],
        "TYL": [3, 8, 4],
        "AAPL": [1, 8, 8],
        "CPRT": [2, 32, 8],
        "MSCI": [3, 32, 16],
        "EXR": [1, 32, 8],
        "KR": [2, 32, 8],
    }

    # Create model
    model = Sequential([layers.Input(shape=(60, stocks_number)), layers.LSTM(96)])

    for _ in range(best_combinations[stock][0]):
        model.add(layers.Dense(best_combinations[stock][1], activation=activations.elu))

    if best_combinations[stock][0] >= 2:
        model.add(layers.Dense(best_combinations[stock][2], activation=activations.elu))

    model.add(layers.Dense(1))

    model.compile(
        loss=Huber(delta=1.0),
        optimizer=Adam(learning_rate=0.0001, epsilon=1e-10),
        metrics=["mean_squared_error"],
    )

    # Add Early Stopping to avoid overfitting
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )

    model.fit(
        scaled_X_train,
        scaled_y1_train,
        validation_data=(scaled_X_val, scaled_y1_val),
        epochs=100,
        callbacks=[early_stopping],
        verbose=0,
    )

    train_predictions = model.predict(scaled_X_train, verbose=0).flatten()
    mse = mean_squared_error(scaled_y1_train, train_predictions)

    if stdout:
        print("MSE:", mse)
    # print(f"'{stock}' stock best combination: {best_combinations[stock]}")

    val_predictions = model.predict(scaled_X_val, verbose=0).flatten()
    test_predictions = model.predict(scaled_X_test, verbose=0).flatten()

    # Show plots
    if stdout:
        plt.figure()
        plt.plot(
            dates_train,
            train_predictions * max_scale,
        )  # plotting denormalized data
        plt.plot(dates_train, scaled_y1_train * max_scale)
        plt.plot(dates_val, val_predictions * max_scale)
        plt.plot(dates_val, scaled_y1_val * max_scale)
        plt.plot(dates_test, test_predictions * max_scale)
        plt.plot(dates_test, scaled_y1_test * max_scale)
        plt.legend(
            [
                "Training Predictions",
                "Training Observations",
                "Validation Predictions",
                "Validation Observations",
                "Testing Predictions",
                "Testing Observations",
            ],
        )
        plt.title(f"Predictions of Stock {stock}")

    # Calculate the returns 1 day ahead:

    predicted_returns = np.exp(test_predictions[-1]) - 1
    real_returns = np.exp(y1[-1]) - 1 if show_real_returns else []

    return (predicted_returns, real_returns)


def lstm_strategy(dict_data, t, stdout=False, show_real_returns=False):
    """Executes the proposed low-volatility-momentum strategy on the given historical stocks data.

    Args:
      Dictionary: {
        'rate': daily rates compounded since February 2011 until March 2020,
        'sp': sp&500 stocks from 1996 until 2023,
        'prices': each stocks' prices dated from February 2011 until March 2023.
      }
      int: t, time value for calculation.
      boolean: stdout, whether to plot the stock analysis made throughout data pre-processing or not.

    Returns:
      DataFrame: portfolio weights DataFrame.
    """
    sel_stocks = initial_analysis(dict_data, stdout)
    dfs_dict = prepare_model_data(sel_stocks, t, stdout)
    weights = mount_wallet(
        sel_stocks,
        dfs_dict,
        dict_data["prices"].index[t + 1],
        stdout,
        show_real_returns,
    )
    return weights
