import numpy as np


def get_retornos_sp(data, t, window_size):
    """Calculate the returns of S&P 500 stocks at a specific time.

    Args:
        data (dict): Data dictionary containing 'sp' and 'prices' DataFrames.
        t (int): The desired time.
        window_size (int): Size of the window for calculations.

    Returns:
        DataFrame: Calculated returns based on the input data.
    """
    sp500 = data["sp"]
    prices = data["prices"]
    dates_prices = prices.index

    local_sp500 = dates_prices[t] > sp500["Date"]
    data_sp500 = sp500["Date"][local_sp500].tail(1).values[0]

    sp500_t = sp500["Ticker"].loc[sp500["Date"] == data_sp500]
    prices_t = prices[sp500_t].loc[dates_prices[t - window_size : t]].dropna(axis=1)
    returns_t = np.log(prices_t).diff().fillna(0)

    return returns_t
