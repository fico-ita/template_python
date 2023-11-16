import modules as md
import pandas as pd


def information_event_strategy(pin_window=60, verbose=False, persist_data=True, path='./data/results'):
    if verbose:
        print('Loading Data')
    quotes = md.dataset.load_cedro_quotes()
    eco_data = md.dataset.load_economatica_quotes()

    if verbose:
        print('Estimating PIN values')
    pins = md.pin_estimation.estimate_all_pins(quotes, window=pin_window, verbose=verbose)

    eligible_stocks = md.stock_selection.filter(eco_data['volumes'])
    portfolios = md.portfolio_build.portfolio_build(pins, eligible_stocks)
    returns = md.returns.calculate_all_portfolios_returns(portfolios['weights'], eco_data['closing_prices'])

    if persist_data:
        eligible_stocks.to_csv(f'{path}/eligible_stocks.csv')
        returns.to_csv(f'{path}/returns.csv')
        pins.to_csv(f'{path}/pins/pin_results.csv')
        for port_name in portfolios['weights'].keys():
            pd.DataFrame(portfolios['weights'][port_name]).T.to_csv(f'{path}/weights/{port_name}.csv')

    return returns, portfolios
