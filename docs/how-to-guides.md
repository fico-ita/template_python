## How To Estimate PINs?
Download the code from this GitHub repository and place the `fico/` folder in the same directory as your Python script:

    your_project/
    │
    ├── fico/
    │   ├── __init__.py
    │   ├── strategy_simulator.py
    │   ├── modules/
    │   └── data/
    │
    └── your_script.py

In data folder, add CSVs (each one for a stock) containing two columns: vendedor and comprador (seller and buyer). Then run 

```python3
from fico.modules import pin_estimation
pin_estimation.estimate_all_pins()
```    

## How To Select Stocks?
Run the following code

```python3
from fico.modules import stock_selection
eligible_stocks = stock_selection.filter(data)
```

With `data` containing the volume per stock per day.

## How To Build a Portfolio?

Run the following code

```python3
from fico.modules import portfolio_build
portfolios = portfolio_build.build_portfolio(
    pins,
    eligible_stocks,
)
```

`pins` means the result of the PIN estimation and `eligible_stocks` means the result of the stock selection.

## How To Measure Returns? 

Run:
```python3
from fico.modules import returns
r = returns.calculate_all_portfolios_returns(weights, prices)
```
`weights` means the weight of an asset in the portfolio, and `prices` means the prices of the stocks in time series.
