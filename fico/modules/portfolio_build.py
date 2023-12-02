'''
    Provides the portfolio building based on PIN and delta (signal probability) estimations.

'''


import datetime as dt
import numpy as np
import pandas as pd


def build_portfolio(pin_model, eligible_stocks, pin_quantile=0.75, delta_quantile=0.8, weight_sum_limit=1,
                    leverage=True, fixed_income_weight=1):
    """
    Builds portfolios based on PINs and eligible stocks. If leverage is enabled, it creates a short portfolio.

    For long and long-only portfolios, stocks are selected based on the highest PINs and deltas.
    The short portfolio selects stocks based on the highest PINs and lowest deltas.
    The leveraged long portfolio allocates a percentage of `fixed_income_weight` to the SELIC.


    :param pin_model: Results from the previously executed PIN model.

    :param eligible_stocks: Binary dataframe indicating eligible stocks.

    :param pin_quantile: Quantile used as a minimum for stock selection based on PIN.

    :param delta_quantile: Quantile used as a minimum for stock selection based on Delta.
    (1 - delta_quantile) is used for the short portfolio.
    :param weight_sum_limit: Total weight of the portfolio.

    :param leverage: Leverage flag.

    :param fixed_income_weight: Percentage allocated to SELIC in case of leverage.

    :return: Dictionary containing 2 sub-dictionaries: one with weights of selected assets and one with their PINs.
    """
    dates = get_unique_dates(pin_model['period'])
    long_portfolio_pin = pd.DataFrame()
    short_portfolio_pin = pd.DataFrame()

    for date in eligible_stocks.index:
        if not eligible_stocks.loc[date].any():
            continue
        if date < dates[0]:
            continue
        if date > dt.datetime.now():
            continue
        pins = get_pins(pin_model, eligible_stocks, date)
        long, short = select_asset_by_delta(pins, q=delta_quantile)
        long = select_asset_by_pin(long, q=pin_quantile)
        short = select_asset_by_pin(short, q=pin_quantile)

        long_portfolio_pin = pd.concat([long_portfolio_pin, long], axis=0)

        if leverage:
            short_portfolio_pin = pd.concat([short_portfolio_pin, short], axis=0)

        long_weights = get_portfolio_weights(long_portfolio_pin, weight_sum_limit, leverage, fixed_income_weight)
        short_weights = get_portfolio_weights(short_portfolio_pin, fixed_income_weight, leverage=False,
                                              fixed_income_weight=0)
        longonly_weights = get_portfolio_weights(long_portfolio_pin, weight_sum_limit=1, leverage=False,
                                                 fixed_income_weight=0)
        portfolios = {
            'pins': {
                'long': long_portfolio_pin,
                'short': short_portfolio_pin,
            },
            'weights': {
                'long': long_weights,
                'short': short_weights,
                'longonly': longonly_weights,
            }
        }
    return portfolios


def get_pins(pin_model, eligible_stocks, date):
    """
    Retrieves the PINs of eligible stocks for a specific date.


    :param pin_model: Results from the previously executed PIN model.

    :param eligible_stocks: Binary dataframe indicating eligible stocks.

    :param date: Reference date.

    :return: Dataframe of PINs and other estimators (such as delta).
    """
    eligible_stocks_on_date = eligible_stocks.loc[date]
    stocks = eligible_stocks_on_date[eligible_stocks_on_date].index
    pins = get_pins_per_date(pin_model, date)
    pins = pins[pins['symbol'].isin(stocks)].copy()
    return pins


def get_pins_per_date(pin_model, date):
    """
    Selects the value of the reference PIN for a date.


    :param pin_model: Results from the previously executed PIN model.

    :param date: Reference date.

    :return: Dataframe of PINs and other estimators (such as delta).
    """
    date_found = False
    first_date = pin_model['period'].min()
    pin_on_date = None
    while not date_found and date >= first_date:
        pin_on_date = pin_model[pin_model['period'] == date]
        if pin_on_date.shape[0] > 0:
            date_found = True
        else:
            date = date - dt.timedelta(days=1)
    return pin_on_date


def select_asset_by_delta(pins, q):
    """
    Selects assets with deltas outside the [lower, upper] range defined by quantiles q and 1-q.


    :param pins: Results from the PIN model for a specific date.

    :param q: Quantile.

    :return: Selected assets for long and short portfolios.
    """
    delta_inf = np.quantile(pins['delta'], q=1 - q)
    delta_sup = np.quantile(pins['delta'], q=q)
    long = pins[pins['delta'] > delta_sup].copy()
    short = pins[pins['delta'] < delta_inf].copy()

    return long, short


def select_asset_by_pin(pins, q):
    """
    Selects assets with PIN above the threshold defined by quantile q.


    :param pins: Results from the PIN model for a specific date.

    :param q: Quantile.

    :return: Selected assets.
    """
    lim = np.quantile(pins['pin'], q=q)
    result = pins[pins['pin'] > lim].copy()
    return result


def get_unique_dates(dates):
    """
    Removes duplicates from the collection.
    :param dates: Collection of dates.
    :return: List of sorted and unique dates.
    """
    unique_dates = list(set(dates))
    unique_dates.sort()
    return unique_dates


def get_portfolio_weights(portfolio, weight_sum_limit=1, leverage=True, fixed_income_weight=1):
    """
    Calculates weights for portfolios.


    :param portfolio: Dictionary of selected stocks.

    :param weight_sum_limit: Maximum portfolio weight without leverage.

    :param leverage: Leverage flag.

    :param fixed_income_weight: Weight allocated to SELIC.

    :return: Portfolio weights.
    """
    portfolio_weights = dict()
    port_per_period = portfolio.groupby('period')
    for period in port_per_period.groups.keys():
        p = port_per_period.get_group(period)
        weights = get_weights(p, weight_sum_limit=weight_sum_limit)
        if leverage:
            weights['SELIC'] = fixed_income_weight
        portfolio_weights[period] = weights
    return portfolio_weights


def get_weights(portfolio, method='equal', weight_sum_limit=1):
    """
    Calculates weights for a single date. Currently, weights are distributed using the equal method.


    :param portfolio: Selected stocks for the portfolio.

    :param method: Weight distribution method. Currently, only equal.

    :param weight_sum_limit: Maximum portfolio weight.

    :return: Weights.
    """
    weights = dict()
    n = portfolio.shape[0]
    if method == 'equal':
        weights = {
            stock: 1 / n * weight_sum_limit
            for stock in portfolio['symbol']
        }
    return weights
