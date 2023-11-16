"""
    Provide the required data for the project development


"""


import json
import os
import pandas as pd
import datetime as dt


def load_economatica_quotes(path='./fico/data/economatica_data'):
    data = {
        'volumes': pd.read_excel(f'{path}/volumes.xlsx', index_col=0, header=3, na_values='-')
        .dropna(how='all', axis=1).dropna(how='all', axis=0),
        'ibov': pd.read_excel(f'{path}/ibov.xlsx', index_col=0, header=3, na_values='-')
        .dropna(how='all', axis=1).dropna(how='all', axis=0),
        'closing_prices': pd.read_excel(f'{path}/closing_prices.xlsx', index_col=0, header=3, na_values='-')
        .dropna(how='all', axis=1).dropna(how='all', axis=0),
    }
    for k, v in data.items():
        data[k].columns = treat_economatica_df_columns(v.columns)
    return data


def treat_economatica_df_columns(columns):
    return [col.split('\n')[-1] for col in columns]


def str_dates_to_dates(str_date_list):
    return [dt.datetime.strptime(date, '%Y-%m-%d') for date in str_date_list]


def load_results(path='./fico/data/results'):
    data = {
        'eligible_stocks': pd.read_csv(f'{path}/eligible_stocks.csv', index_col=0),
        'pin_results': pd.read_csv(f'{path}/pins/pin_results.csv', index_col=0),
    }
    data['eligible_stocks'].index = str_dates_to_dates(data['eligible_stocks'].index)
    data['pin_results']['period'] = str_dates_to_dates(data['pin_results']['period'])
    return data


def load_cedro_quotes(path='./fico/data/cedro_data'):
    '''
    Load quotations from Cedro (B3)
    Returns: DataFrame with all quotes available
    '''
    params = pd.DataFrame()
    for file in os.listdir(path):
        if file.endswith('.csv'):
            data = pd.read_csv(f'{path}/{file}', index_col=0)
            params = pd.concat([params, data], ignore_index=True)
    params['bucket'] = str_dates_to_dates(params['bucket'])
    return params


def date_keys_to_str(dict_):
    new_dict = {
        k.strftime('%Y-%m-%d'): v
        for k, v in dict_.items()
    }
    return new_dict


def save_json(dict_data, path='./fico/data/results', filename='*.json'):
    if isinstance(list(dict_data.keys())[0], dt.datetime):
        dict_data = date_keys_to_str(dict_data)
    with open(f'{path}/{filename}', 'w') as file:
        json.dump(dict_data, file)


def save_data(data: dict, path='./fico/data/results'):
    for k, v in data.items():
        if isinstance(v, dict):
            save_json(v, path, filename=f'{k}.json')
        else:
            v.to_csv(f'{path}/{k}.csv')
