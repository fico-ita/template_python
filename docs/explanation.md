## Objective 

Using the Probability of Insider Trading (PIN) estimates from Easley and O'Hara (1996) to construct long and short portfolios and generate a reasonable Sharpe ratio.


## Strategy

Asset Selection: The eligible assets for portfolio composition will be the stocks that meet the following asset filtering criteria:

- Higher Liquidity Criterion: To be the most traded stock of the company in the previous month.
- Daily Liquidity Criterion: To have an average daily trading volume of at least R$ 500,000 in the previous period.
- Listing Criterion: To be listed for at least 2 years from the observation moment.


PIN estimation: For each stock selected, we create a moving window of 60 days in order to estimate the following indicators:
- alpha = Probability of the occurrence of an informational event
- delta = Conditional probability of an event with a positive signal
- mu = Informed agents flux
- eps_s = Uninformed agents selling flux
- eps_b = Uninformed agents buying flux
- PIN = Probability of informed trading occurence


Sorting by Signal: Using the series of PIN models for eligible assets, two portfolios were constructed monthly: one for assets with positive informational events and another for assets with negative informational events. These assets were sorted by the probability 'delta' of presenting a positive signal, and the upper and lower quintiles were selected. The upper quintile formed the candidates for the positive portfolio, and the lower quintile formed the candidates for the negative portfolio.


Sorting by PIN: Next, within each portfolio, a sorting was performed based on the PIN value, and the upper quartile of each was selected. Thus, two portfolios are constructed: one with assets having the highest PINs and the highest probabilities of events being positive (long portfolio), and another with assets having the highest PINs and the lowest probabilities of events being positive (short portfolio).


Resulting Portfolio: With the assets from the two selected portfolios, the capital was allocated as follows:

- 100% of the capital was allocated to the Treasury SELIC as collateral for leverage.
- 100% of the capital was allocated to the long portfolio, distributed equally among the assets.
- 100% of the capital was used to sell the short portfolio, distributed equally among the assets.
