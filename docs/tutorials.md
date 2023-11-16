# Tutorial

This section aims to help newcomers get started with the code in this project. It focuses on a hands-on, learning-oriented approach to guide users through practical examples. To refer to the main jupyter notebook used, click [here](./tutorials/main.ipynb)

## Getting Started

### Objective

In this tutorial, you'll learn how to utilize the LSTM neural network for stock selection based on low volatility and high momentum criteria.

### Steps

**Initial Setup**

   - Clone the repository from [GitHub link](https://github.com/fico-ita/po_245_2023_S2_T4/tree/dev) - **fico** folder.
!!! warning
    Ensure Python (version 3.9 or higher) and required packages are installed (`pandas`, `numpy`, `tensorflow`, `seaborn`, `matplotlib`).

**Understanding the Strategy**

   - Explore the provided code for the low-volatility-high-momentum investment strategy using LSTM.
   - Examine the modules and their functionalities for data loading, analysis, and model training.

**Executing the Strategy**

   - Use the provided functions like `initial_analysis` to select top stocks based on volatility and momentum thresholds.
   - Experiment with the `prepare_model_data` function to organize data for the LSTM model.
   - Try the `mount_wallet` function to calculate portfolio weights based on predicted returns.

## Code Example

### Objective

Demonstrate a simple example of using the provided functions for stock selection and portfolio construction.

```python
# Import necessary functions from the provided code
from lstm_strategy import initial_analysis, prepare_model_data, mount_wallet

# Load data
data_dict = load_data()  # Assuming load_data is imported

# Get the selected stocks
selected_stocks = initial_analysis(data_dict)

# Prepare data for the model
model_data = prepare_model_data(selected_stocks, t=2031) # t translates into a 2-year historical data range. t>= 400.

# Calculate portfolio weights, comparing predicted returns of the trained model with real returns if needed.
portfolio = mount_wallet(selected_stocks, model_data, show_real_returns = True)
```