import pandas as pd
import datetime as dt


def calculate_all_portfolios_returns(portfolios_weights, prices, leverage=True):
    '''
    Calculate returns for all generated portfolios
    :param portfolios_weights: Portfolio weights
    :param prices: Price data
    :param leverage: Whether to calculate leveraged returns
    :return: DataFrame with portfolio returns
    '''
    returns = prices.pct_change()
    all_returns = {
        portfolio_name: calculate_portfolio_returns(weights, returns)
        for portfolio_name, weights in portfolios_weights.items()
    }
    if leverage:
        all_returns['long_short'] = all_returns['long'] - all_returns['short']
    return pd.DataFrame(all_returns)


def calculate_portfolio_returns(portfolio_weights, returns):
    '''
    Calculate returns for a single portfolio
    :param portfolio_weights: Portfolio weights
    :param returns: Returns data
    :return: Series with portfolio returns
    '''
    first_date = min(portfolio_weights.keys())
    selic = get_selic(first_date, dt.datetime.now())
    dates = pd.Series(portfolio_weights.keys(), index=portfolio_weights.keys())
    portfolio_returns = dict()
    for date in returns.index:
        if date <= first_date:
            continue
        portfolio = get_portfolio(portfolio_weights, dates, date)
        stocks_returns = get_stocks_returns(portfolio, returns.loc[date], selic.loc[date])
        portfolio_returns[date] = stocks_returns
    return pd.Series(portfolio_returns)


def get_portfolio(portfolio_weights, dates, date):
    '''
    Retrieve a portfolio for a specific date
    :param portfolio_weights: Portfolio weights
    :param dates: Dates associated with portfolio weights
    :param date: Date for which to retrieve the portfolio
    :return: Portfolio weights for the specified date
    '''
    i = dates.index.get_indexer([date], method='pad')[0]
    if date == dates.index[i]:
        # Portfolio assembled only at the closing
        date -= dt.timedelta(days=1)
        i = dates.index.get_indexer([date], method='pad')[0]
    target_date = dates.index[i]
    return portfolio_weights[target_date]


def get_stocks_returns(portfolio, returns, selic):
    '''
    Calculate stock returns
    :param portfolio: Portfolio weights
    :param returns: Stock returns
    :param selic: SELIC returns
    :return: Portfolio returns
    '''
    stocks_returns = 0
    for stock in portfolio.keys():
        weight = portfolio[stock]
        if stock == 'SELIC':
            r = selic
        else:
            r = returns[stock] if pd.notna(returns[stock]) else 0
        stocks_returns += r * weight
    return stocks_returns


def get_selic(start, end):
    '''
    Retrieve daily SELIC values
    :param start: Start date
    :param end: End date
    :return: SELIC data
    '''
    str_start = start.strftime('%d/%m/%Y')
    str_end = end.strftime('%d/%m/%Y')
    url = "http://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=csv&dataInicial=" + str_start + "&dataFinal=" + str_end
    selic = pd.read_csv(url, sep=";", decimal=',')
    selic.index = [dt.datetime.strptime(date, '%d/%m/%Y') for date in selic['data']]
    selic['valor'] = selic['valor'] / 100
    return selic['valor']
