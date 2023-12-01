# Load Data

import pandas as pd


def load_data():
    # load fed rate
    rate = pd.read_parquet("../../dataset/US/fed_rate.parquet")

    # load sp500_components
    sp = pd.read_parquet("../../dataset/US/sp_comp.parquet")

    # load prices_sp
    melt_prices = pd.read_parquet("../../dataset/US/prices_sp.parquet")
    df_prices = melt_prices.pivot_table(index="Date", columns="Ticker", values="value")

    # create a dictionary with data
    dict_data = {"rate": rate, "sp": sp, "prices": df_prices}
    return dict_data
