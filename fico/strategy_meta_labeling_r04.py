"""## Stocks Portfolio Construction Based on a Meta-Labeling ML Approach.

This module creates a portfolio of stocks with their
corresponding weights through the following pipeline steps:

Block-1: Inputs
    - This module requires FICO Load Data dictionary {'rate', 'sp', 'prices'} as input
      from which a time series of stocks' log returns is extracted.

Block-2: Individual Stock MetaLabeling Application
    - For each stock in the input time series, this block of the pipeline
      determines the next day's direction prediction (position side) and its accuracy
      probability. It uses Logistic Regression within a MetaLabeling framework to
      make predictions.
    - This block is comprised by function `A_prep_data` followed by function `B_stock_meta_labeling`.

Block-3: Position Sizing
    - For each stock, this block calculates the position size based on the accuracy
    probability. An Empirical Cummulative Distribution is build such that predictions
    with higher confidences are assigned higher position sizes, while predictions with 
    smaller confidences are assigned smaller positions.
    - This block is executed by function `C_position_sizing`.

Block-4: Portfolio Construction Strategy
    - Ranks all assessed stocks based on their defined position sizes and selects the
    first *n* stocks to compose the portfolio. The individual sizes are normalized so
    that they sum up to one.
    - This block is performed by function `D_strategy_meta_labeling`.

Block-5: Output
    - The module provides the resulting portfolio of *n* stocks and their respective
    weights.

This module is designed to create of portfolios based on the accuracy
of stock price predictions, allowing you to construct diversified investment portfolios.

For detailed function descriptions, refer to the individual function
docstrings described below. 

"""
import numpy as np
import pandas as pd
from typing import Tuple
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from modules.get_retornos_sp import get_retornos_sp

def A_prep_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for meta-labeling by transforming dependent and independent variables,
    adding a primary model's side forecast, and returning the meta_data DataFrame for training
    in the secondary model.

    Args:
        data: DataFrame of stocks' log returns to which to prepare.

    Returns:
        meta_data DataFrame.
        original data DataFrame transformed.

    """

    # Set the target variable. Binary classification
    data['target'] = data['rets'].apply(lambda x: 0 if x < 0 else 1).shift(-1)

    # Create the dataset
    data['target_rets'] = data['rets'].shift(-1)  # Add target returns for debugging
    data.dropna(inplace=True)

    # Auto-correlation trading rule: trade sign of the previous day.
    data['pmodel'] = data['rets'].apply(lambda x: 1 if x > 0.0 else 0)

    # Strategy daily returns. Lag by 1 to remove look-ahead and align dates
    data['prets'] = (data['pmodel'] * data['target_rets']).shift(1)
    data.dropna(inplace=True)

    # Add lag returns 2 and 3 for Logistic regression
    data['rets2'] = data['rets'].shift(1)
    data['rets3'] = data['rets'].shift(2)

    # Data used to train the model. Apply labels to the total dataset
    # In this setting, the target for the pmodel is the meta_labels when
    # you filter by only pmodel=1 (long-only strategy)
    model_data = data[data['pmodel'] == 1].copy()
    model_data.dropna(inplace=True)
    return model_data, data


def B_stock_meta_labeling(data_istock: pd.DataFrame) -> Tuple[float, str, pd.DataFrame, pd.DataFrame]:
    """Perform meta-labeling for a given stock's data, including data preparation,
    training a logistic regression model,and making predictions.

    Args:
        data_istock: Data for a single stock with columns 'rets' representing returns.

    Returns:
        Probability of the positive sign.
        forecast date.
        data_train_set.
        data_test_set.
    
    
    """

    # Create a single dataset
    data = data_istock

    # Prepare data, add primary model, get meta_labels
    model_data, data = A_prep_data(data=data)

    # Train-Test split
    # Train data (all but the last row)
    train = model_data[:-1]
    # Test data (only the last row)
    test = model_data.tail(1)

    # Features and Target variables
    # Train data
    x_train = train[['rets', 'rets2', 'rets3']]
    y_train = train['target']
    # Test data
    x_test = test[['rets', 'rets2', 'rets3']]
    y_test = test['target']

    # Scale Train data
    scaler = StandardScaler()
    x_train_info_scaled = scaler.fit_transform(x_train)

    # Retrieve means and stdevs used by scaler.fit_transform
    x_train_info_means = scaler.mean_
    x_train_info_stdevs = scaler.scale_

    # Scale Test data (using Train means/stDevs to avoid data leakage)
    x_test_info_scaled = (x_test - x_train_info_means) / x_train_info_stdevs

    # Train the model
    meta_model_info = LogisticRegression(random_state=0, penalty=None)
    meta_model_info.fit(x_train_info_scaled, y_train)

    # Create a copy of the DataFrame to avoid "SettingWithCopyWarning"
    train = train.copy()

    # Predicted Train's label and Probability for the positive class (class 1)
    train['pred_info'] = train_pred_info = meta_model_info.predict(x_train_info_scaled)
    train['prob_info'] = meta_model_info.predict_proba(x_train_info_scaled)[:, 1]

    # Set new columns
    data['pred_info'] = 0
    data['prob_info'] = 0

    # Add train data to the larger DataFrame
    data.loc[train.index, 'pred_info'] = train['pred_info']
    data.loc[train.index, 'prob_info'] = train['prob_info']

    # Complete the train dataset
    data_train_set = data.loc[train.index[0]:train.index[-1]]

    # --- Test Prediction ---

    # Create a copy of the DataFrame to avoid "SettingWithCopyWarning"
    test = test.copy()
    # Remove the header column to avoid "UserWarning: X has feature names, but
    # LogisticRegression was fitted without feature names"
    x_test_info_scaled.columns = range(x_test_info_scaled.shape[1])

    # Predicted Test's label and Probability for the positive class (class 1)
    test['pred_info'] = meta_model_info.predict(x_test_info_scaled)
    test['prob_info'] = meta_model_info.predict_proba(x_test_info_scaled)[:, 1]

    # Add test data to the larger DataFrame
    data.loc[test.index, 'pred_info'] = test['pred_info']
    data.loc[test.index, 'prob_info'] = test['prob_info']

    # Complete the test dataset
    data_test_set = data.loc[test.index[0]:test.index[-1]]

    # Get the current date and add one day (forecast for tomorrow)
    test_date = str(np.datetime_as_string(data_test_set.index.values[-1], unit='D'))
    test_date_plus1 = np.datetime64(test_date, 'D') + np.timedelta64(1, 'D')

    # Return the probability of the positive sign, forecast date and datasets for sizing
    return test['prob_info'][0], test_date_plus1, data_train_set, data_test_set

def C_position_sizing(data_train_set: pd.DataFrame, data_test_set: pd.DataFrame) -> float:
    """Perform position sizing based on the prob distribution of class 1 occurrence
    using logistic regression.

    Args:
        data_train_set: Training data set to fit the Empirical Cummulative Distribution.
        data_test_set: Test data set to determide the size based on the positive sign prob.

    Returns:
        Effective bet size.

    """

    # Get the probability distribution of class 1 occurrence using logistic regression.
    prob_train = data_train_set.loc[data_train_set['pred_info'] == 1, 'prob_info']
    prob_test = data_test_set.loc[data_test_set['pred_info'] == 1, 'prob_info']

    # ECDF Position Sizing
    try:
        ecdf = ECDF(prob_train)
        e_bet_sizes = prob_test.apply(lambda x: ecdf(x))
    except ZeroDivisionError:
        e_bet_sizes = 0

    # Create a copy of the DataFrame to avoid "SettingWithCopyWarning"
    data_test_set = data_test_set.copy()

    # Assign position sizes
    data_test_set['e_bet_size'] = 0
    data_test_set.loc[data_test_set['pred_info'] == 1, 'e_bet_size'] = e_bet_sizes

    # Return betting size
    return data_test_set['e_bet_size'][0]

def D_strategy_meta_labeling(data: pd.DataFrame, t: int, size: int, window_size: int) -> pd.DataFrame:
    """Execute the meta-labeling strategy for multiple stocks.

    Selected *n* Stocks are the ones with the highest positive class 
    probability (Predict_Proba).

    Final weights are the individual sizes normalized to sum up to one.
    
    Args:
        data: FICO Load Data dictionary {'rate', 'sp', 'prices'}
        t: Initical point in time of the dataset.
        size: Number of stocks to consider.
        window_size: size of the Training + Test dataset.

    Returns:
        Meta-labeling strategy results including date, tickers and weights.
        
    Examples:
        >>>#FICO Load Data module
        >>>from modules.load_data import load_data
        >>>dict_data = load_data()
        >>>
        >>>#Stocks Portfolio Construction Based on a Meta-Labeling ML Approach
        >>>from example.strategy_MetaLabeling import strategy_meta_labeling_r04
        >>>
        >>>Portifolio = strategy_meta_labeling_r04(dict_data, t = 2000, size = 10, window_size= 500)
        >>>Portifolio
                    ticker  weights        
        date                        
        2019-05-11  ALB     0.101264
        2019-05-11  REGN    0.101264
        2019-05-11  MYL     0.101264
        2019-05-11  BKNG    0.100882
        2019-05-11  ADI     0.100286
        2019-05-11  PHM     0.099938
        2019-05-11  AES     0.099651
        2019-05-11  PXD     0.098805
        2019-05-11  WELL    0.098347
        2019-05-11  FDX     0.098298

    """

    # Get Data containing log-returns for multiple stocks.
    returns = get_retornos_sp(data, t, window_size)

    # Run meta-labeling procedure for all available stock
    ticker, prob, sizing = [], [], []
    upper_bound = returns.shape[1] - 1

    for i in range(0, upper_bound, 1):

        # Returns for the current stock
        data_istock = returns.iloc[:, [i]].copy()
        i_ticker = data_istock.columns[0]

        # Run meta-labeling procedure for the current stock
        data_istock.columns = ['rets']
        i_prob, i_date, i_dtrain, i_dtest = B_stock_meta_labeling(data_istock)

        # Run position sizing procedure for the current stock
        i_size = C_position_sizing(i_dtrain, i_dtest)

        ticker.append(i_ticker)
        prob.append(i_prob)
        sizing.append(i_size)

    # Gather results into a single dictionary
    data_dict = {'date': i_date, 'ticker': ticker, 'prob': prob, 'bet_size': sizing}
    result_df = pd.DataFrame(data_dict)

    # Select the top 'size' stocks with the highest positive class (class 1) probability
    result_df = result_df.sort_values(by='bet_size', ascending=False).head(size)

    # Normalize weights to sum up to one
    sizing_sum = result_df['bet_size'].sum()
    result_df['weights'] = result_df['bet_size'].apply(lambda x: x / sizing_sum)
    result_df.reset_index(drop=True, inplace=True)

    result_df['date'] = pd.to_datetime(result_df['date'])
    result_df.set_index('date', inplace=True)

    return result_df[['ticker', 'weights']].copy()