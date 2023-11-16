"""
Provide 4 stock filters:

    - Company's most traded stock filter
    - Minimum average volume filter
    - Minimum listed time filter
    - Logical AND of the three filters above
"""

import pandas as pd
import datetime as dt


def filter(volumes):
    '''
    Apply Asset Filter

    Eligibility criteria used by the Center for Financial Economics Research (NEFIN) at the Faculty of Economics, Administration, Accounting, and Actuarial Sciences of the University of São Paulo (FEA - USP).

    A stock traded on B3 will be eligible in a period if it meets the following three criteria:
    1. It is the most traded stock of the company in the previous period.
    2. The stock has an average daily trading volume of R$ 500,000 in the previous period.
    3. The stock has been listed for at least 2 years from the observation moment.


    :param volumes: Financial volumes traded for assets

    :return: Binary dataframe indicating whether a particular asset is eligible for a specific period
    '''

    mtsf = most_traded_stock_filter(volumes)
    mvf = minimum_volume_filter(volumes)
    mltf = minimum_listed_time_filter(volumes)

    eligible_stocks = mtsf & mvf & mltf
    return eligible_stocks



def most_traded_stock_filter(volumes, period='M'):
    '''
    Selects the most traded stock of the company in the previous period.


    :param volumes: Financial volumes traded for assets

    :param period: filtering period. Default='M', meaning monthly

    :return: Binary dataframe indicating whether a particular asset is eligible for a specific period
    '''
    stocks = stocks_per_firm(volumes.columns)
    volumes_periodically = volumes.resample(period).sum()
    result = pd.DataFrame(index=volumes_periodically.index, dtype=bool)
    for firm in stocks.keys():
        if len(stocks[firm]) == 1:
            mtpp = pd.Series([True for _ in result.index], index=result.index, name=stocks[firm][0])
        else:
            mtpp = most_traded_per_period(volumes_periodically[stocks[firm]])

        result = pd.concat([result, mtpp], axis=1)
    return result


def stocks_per_firm(stock_list):
    '''
    Helper function for most_traded_stock_filter, grouping stocks by company


    :param stock_list: list of stocks

    :return: Dictionary with the B3 company code as the key and lists of stocks as values
    '''
    firm_dict = dict()
    for stock in stock_list:
        if stock[0:4] not in firm_dict.keys():
            firm_dict[stock[0:4]] = list()
        firm_dict[stock[0:4]].append(stock)
    return firm_dict


def most_traded_per_period(volumes):
    '''
    Helper function for most_traded_stock_filter, checks which stock has the highest liquidity


    :param volumes: Financial volumes traded for assets of one company

    :return: Binary dataframe indicating whether a particular asset is eligible for a specific period
    '''
    result = pd.DataFrame(index=volumes.index, columns=volumes.columns, dtype=bool)
    for date in volumes.index:
        if volumes.loc[date].isna().all():
            result.loc[date] = [False for _ in range(len(volumes.loc[date]))]
        else:
            result.loc[date] = volumes.loc[date] == volumes.loc[date].max()
    return result



def minimum_volume_filter(volumes, period='M', limit=5e5):
    '''
    Minimum volume filter. Checks if stocks meet the required average volume limit


    :param volumes: Financial volumes traded for assets of one company

    :param period: filtering period. Default='M', meaning monthly

    :param limit: Minimum required limit. Default=5e5, meaning 500,000

    :return: Binary dataframe indicating whether a particular asset is eligible for a specific period
    '''
    return volumes.resample(period).mean() >= limit


def minimum_listed_time_filter(volumes, period='M', limit=500):
    '''
    Listing time filter. Checks if stocks meet the minimum listing time requirement


    :param volumes: Financial volumes traded for assets of one company

    :param period: filtering period. Default='M', meaning monthly

    :param limit: Minimum required limit. Default=500, meaning 500 business days or 2 years

    :return: Binary dataframe indicating whether a particular asset is eligible for a specific period
    '''
    result = pd.DataFrame(columns=volumes.columns, index=volumes.index)
    for stock in volumes.columns:
        first_trading_day = get_first_trading_day(volumes[stock])
        if first_trading_day:
            first_eligible_day = first_trading_day + dt.timedelta(days=limit)
        else:
            first_eligible_day = dt.datetime(2200, 1, 1)
        result[stock] = pd.Series(volumes[stock].index > first_eligible_day, index=volumes[stock].index)

    result = result.resample(period).apply(all)
    return result


def get_first_trading_day(volumes: pd.Series):
    '''
    Helper function for minimum_listed_time_filter. Finds the first trading day of the stock


    :param volumes: Financial volumes traded for assets of one company

    :return: Date of the first trading day
    '''
    trading_dates = volumes.dropna().index
    return trading_dates[0] if len(trading_dates) > 0 else None
