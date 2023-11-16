'''
    Provides the Probability of Insider Trading (PIN) estimation according to Easley et al. (1996)
'''


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

import warnings

warnings.filterwarnings('ignore')

all_likelihoods = list()
iteration_likelihood = None

def poisson_likelihood(params, buyers, sellers):
    '''
    Estimate the likelihood of the Poisson Mass Function, as described by Easley & O'Hara (1996)


    :param params: θ = (alpha, mu, delta, eps_b, eps_s)

    :param buyers: buying flux series

    :param sellers: selling flux series

    :return: negative value of likelihood
    '''

    probability_mass_function = poisson.pmf
    (alpha, mu, delta, eps_b, eps_s) = params

    # Poisson without the occurrence of an informational event (Pn)
    Pn_buy = probability_mass_function(buyers, eps_b)
    Pn_sell = probability_mass_function(sellers, eps_s)
    P_nothing = (1 - alpha) * Pn_buy * Pn_sell

    # Poisson with the occurrence of a positive signal informational event (P_plus)
    P_plus_buy = probability_mass_function(buyers, eps_b + mu)
    P_plus = alpha * delta * P_plus_buy * Pn_sell

    # Poisson with the occurrence of a negative signal informational event (P_minus)
    P_minus_sell = probability_mass_function(sellers, eps_s + mu)
    P_minus = alpha * (1 - delta) * Pn_buy * P_minus_sell
    likelihood = -np.sum(P_nothing + P_plus + P_minus)

    global iteration_likelihood
    iteration_likelihood = {
        'alpha': alpha,
        'mu': mu,
        'delta': delta,
        'eps_b': eps_b,
        'eps_s': eps_s,
        'likelihood': likelihood
    }
    return likelihood

def estimate_params(data, iterations=10):
    '''
    Estimate PIN params:

        alpha = Probability of the occurrence of an informational event

        delta = Conditional probability of an event with a positive signal

        mu = Informed agents flux

        eps_s = Uninformed agents selling flux

        eps_b = Uninformed agents buying flux


    :param data: pd.DataFrame with time series for the estimation

    :param iterations: number of maximum-likelihoods to calculate, to get the highest likelihood

    :return: params for the highest likelihood
    '''
    avg = data['comprador'].mean()
    bounds = (
        (0, 1),
        (None, None),
        (0, 1),
        (None, None),
        (None, None),
    )

    while iterations > 0:
        iterations -= 1
        initial_params = [np.random.rand(), avg * np.random.rand(), np.random.rand(), avg * np.random.rand(),
                          avg * np.random.rand()]

        results = minimize(poisson_likelihood, initial_params,
                           args=(data['comprador'].values, data['vendedor'].values),
                           bounds=bounds)
        all_likelihoods.append(iteration_likelihood)
    params = get_highest_likelihood_params()
    return params

def get_highest_likelihood_params():
    '''
    Select the highest likelihood


    :return estimations with highest likelihood:
    '''
    global all_likelihoods
    result_params = {'likelihood': -np.inf}
    for params in all_likelihoods:
        if params['likelihood'] > result_params['likelihood']:
            result_params = params
    result_params.pop('likelihood')
    all_likelihoods = list()
    return result_params

def pin_equation(params):
    '''
    PIN equation, according to Easley & O'Hara (1996)


    :param params: pd.DataFrame with the params alpha, mu, eps_b, and eps_s

    :return: float number representing the PIN estimation
    '''
    α = params['alpha']
    µ = params['mu']
    εb = params['eps_b']
    εs = params['eps_s']

    return (α * µ) / (α * µ + εb + εs)

def rolling_pin(data, window=60):
    '''
    Estimate PIN in a rolling window for a time series


    :param data: pd.DataFrame time series with `comprador` and `vendedor`

    :param window: integer indicating the window size (default = 60)

    :return: pd.DataFrame containing PIN and the estimated params
    '''
    data.sort_index(inplace=True)
    results = list()
    for i in range(window, data.shape[0]):
        params = {
            'symbol': data['symbol'].iloc[i],
            'period': data['bucket'].iloc[i],
        }
        frame = data.iloc[i:window + i]
        estimated_params = estimate_params(frame)
        try:
            params['pin'] = pin_equation(estimated_params)
        except KeyError:
            continue
        params.update(estimated_params)
        results.append(params)
    return pd.DataFrame(results)

def estimate_all_pins(data, window=60, verbose=False):
    '''
    Calculate PIN in a rolling window for all data


    :param data: pd.DataFrame with symbol (ticker), `comprador`, and `vendedor`

    :param window: integer indicating the window size (default = 60)

    :param verbose: if True, it enables prints during the execution

    :return:
    '''
    pin_params = pd.DataFrame()
    grouped_data = data.groupby(by='symbol')
    for stock in grouped_data.groups.keys():
        if verbose:
            print(stock)
        group = grouped_data.get_group(stock)
        pin_params = pd.concat([pin_params, rolling_pin(group.sort_index(), window)], ignore_index=True)
    return pin_params
