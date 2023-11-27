# Explanation

## Understanding the Meta-Labeling Approach in Financial Portfolio Construction

## 1. Introduction to Meta-Labeling

The evolution of the Meta-Labeling Approach in finance traces back to Lopez de Prado's 
foundational concepts **[1]**. This technique defines position signals using a primary strategy, 
often derived from machine learning algorithms or other decision processes. From the primary 
signal, a secondary model learns to predict the probability of this signal being a true 
positive (the target variable is this meta-label), which can then be used to define the bet sizing.

Hudson&Thames (H&T) quant firm expanded on these principles through a serie of comprehensive 
articles **[2-5]**, providing a deeper understanding of Meta-Labeling. Their articles not only 
elucidate the methodology but also offer practical implementation guides on GitHub.

This project was born with the goal of leveraging H&T's meta-labeling approach for 
real-world applications, particularly in constructing stock portfolios. For this purpose, 
pre-pandemic S&P500 stocks data were considered.


## 2. Primary and Secondary Models

At the core of the Meta-Labeling methodology are the primary and secondary models. The 
primary model serves the role of predicting the directional aspect of positions, determining 
whether to take a buy or sell stance. Complementing this, the secondary model validates 
these predictions, thus enabling informed decisions on position sizing based on the 
reliability of the primary model's output. The reliability factor plays a crucial role in 
optimizing portfolio performance. The picture below illustrates the interconnection between 
the primary and secondary models **[2]**.


<! ../docs/images/Meta-Labeling_Architecture.jpg"  >

<div style="text-align:center;">
    <img src="../images/Meta-Labeling_Architecture.jpg" alt="Meta-Labeling Architecture" width="80%">
</div>


For further details on H&T's Meta-Labeling Architecture refer to my white paper [Stocks Portfolio Construction Based on Meta-Labeling Machine Learning](materials/ITA_PO245_WhitePaper-IEEE_Access_LaTeX_template_AfonsoFeitoza.pdf).


## 3. Reproducing and Extending TH Firm's Approach

The journey began with replicating H&T's Meta-Labeling strategy, which uses synthetic 
stock return data. Its primary model is a simple momentum strategy (predicted direction 
is the same as the <ins>**previous**</ins> observed one). The target variable (the meta-label) 
of the secondary model is “buy” if the <ins>**next**</ins> training return is positive and its 
features are the last three stocks returns.

The resulting dataframe is used to train a logistic regression classifier (secondary model) 
from scikitlearn package, whose output are the predicted label and the corresponding true 
positive probability. When the predicted class is 1 it means that the primary model correctly 
predicted next day positive return, i.e., the primary model’s signal has a high confidence 
and can be followed. Once trained, the secondary model is applied over the test datapoints 
to predict next return signal and its probability.

In H&T's approach, the position size is derived from the probability related to the positive 
class. The idea is to assign larger position sizes for higher true positive probabilities. 
Although it does not reflect the optimal position sizing [3], the use of Empirical 
Cumulative Distribution Function (ECDF) serves this purpose in a simple manner (see picture below).

<! ../docs/images/Colab2-1.jpg" >

<div style="text-align:center;">
    <img src="../images/Colab2-1.jpg" alt="Empirical Cummulative Distribution Function" width="80%">
</div>

Following the procedures outlined by H&T in their [GitHub](https://github.com/hudson-and-thames/meta-labeling/tree/master/theory_and_framework), 
the same results as in reference **[2]** was obtained:

<! ../docs/images/Colab2-3.jpg >

<div style="text-align:center;">
    <img src="../images/Colab2-3.jpg" alt="Reproduction of articles results" width="110%">
</div>

This initial step aimed at not only comprehending the methodology but also 
ensuring accuracy in the implementation. Upon successfully replicating the results, confidence 
was established in the approach. This assurance paved the way for extending the methodology 
to a diversified stock portfolio, utilizing FICO stock data.

## 4. Raw Data Transformation

The transformation of raw data into the Meta-Labeling format involved converting time series 
stock data into features for model training. 

As mentioned in section [How-to Guides](how-to-guides.md) and detailed in  [Reference](reference.md), 
stocks log-returns are retrieved from FICO data base.

The primary model signal, "buy" or null, is derived from the current training return's 
positivity, while the secondary model signal is based on the next training return. 
The secondary model utilizes the last three returns as features, standardized across 
test and training sets. Notably, the training dataset for the secondary model consists 
solely of datapoints where the primary model signal is positive. This process results in 
the creation of the final training dataset tailored for the secondary model.

<! ../docs/images/Colab2-4.jpg >

<div style="text-align:center;">
    <img src="../images/Colab2-4.jpg" alt="Feature_Engineering" width="70%">
</div>


## 5.Portfolio Construction Pipeline

<! ../docs/images/PO245.Fluxograma-Horizontal.jpg >

<div style="text-align: center;">
    <img src="../images/PO245.Fluxograma-Horizontal.jpg" alt="Project Pipeline" style="margin-top: 20px;" />
</div>

As mentioned in section [Reference](reference.md), the project pipeline is organized 
into three main blocks:


### 5.1. Individual Stock MetaLabeling Application

This initial block involves applying the Meta-Labeling strategy to each available stock 
separately. The process starts by transforming the log-return time series into the secondary 
model dataframe, as explained in [**section 4**](#4-raw-data-transformation). Standardization 
of features across the entire dataset is performed using the means and standard deviations 
of the training features to prevent data leakage.

Subsequently, scikit-learn Logistic Regression is utilized to predict the direction of 
the next day and to derive associated probabilities.


<div style="text-align:center;">
    <img src="../images/Pipeline_01.jpg" alt="Individual Stock MetaLabeling Application" width="100%">
</div>


### 5.2. Position Sizing

In this phase, position sizes for each stock are calculated based on the accuracy probability. 
The process involves constructing an Empirical Cumulative Distribution (ECD) using the 
training dataset. Predictions with higher confidence levels are assigned larger position 
sizes (closer to one), whereas those with lower confidences receive smaller positions 
(closer to zero). The accuracy probability for test data is then applied to the ECD function 
to determine the position size.


<div style="text-align:center;">
    <img src="../images/Pipeline_02.jpg" alt="Position Sizing" width="100%">
</div>

### 5.3. Portfolio Construction Strategy

After evaluating stocks and defining their position sizes, this stage involves ranking 
the stocks by their position size and selecting the first *n* ones to form the portfolio. 
Individual sizes are normalized to ensure they collectively sum up to one.


<div style="text-align:center;">
    <img src="../images/Pipeline_03.jpg" alt="Portfolio Construction Strategy" width="100%">
</div>


## 6. Initial Results and Insights

Upon implementing the Meta-Labeling strategy, a backtest was performed using FICO's 
`strategy simulator` to get portfolio's historical weights and its next day returns. Then
`quantstats` package was used to calculate various performance metrics and create a 
comprehensive [TearSheet](materials/TearSheet.html).These results offered insights into 
the portfolio's performance, risk-adjusted returns, and model validation metrics. See the 
Figure below of an extract of this TearSheet

<! ../docs/images/Colab2-6.jpg" >

<div style="text-align:center;">
    <img src="../images/Colab2-6.jpg" alt="Initial Results" width="90%">
</div>


The portfolio's performance significantly lagged behind the benchmark across all metrics. 
This raised concerns regarding the practicality of applying H&T's approach, originally 
devised for synthetic data, to real stock data. Their methodology is structured as a 
controlled experiment chiefly aimed at simplifying comprehension of the technique itself. 
One key aspect contributing to the performance enhancement provided by their meta-labeling 
technique compared to the primary model is that the features used in the secondary model 
are the ones behind the generation of the synthetic return time series (namely, three 
lagged returns and regime-switching indicator). 

 
To verify this hypothesis, I replicated the same steps used to generate the results presented 
in [**section 3**](#3-reproducing-and-extending-th-firms-approach), but this time with real stock data instead of synthetic data. 
This includes using the same structures for the primary model (autocorrelation strategy) 
and secondary models (trained using the last three returns as features). The figure below 
illustrates the Sharpe Ratio outcomes using real stock data alongside the synthetic return 
data, reproduced from the figure in section 3 for convenience.

<div style="text-align:center;">
    <img src="../images/Explanation-06.01.jpg" alt="Initial Results" width="100%">
</div>

Unlike the behavior observed with synthetic data, where there was a clear improvement in 
the secondary model's performance compared to the primary one, there was no improvement 
when real stocks data were used. In order to ascertain whether that is due to limitations 
of the sencondary model (a simple logistic Linear regression) or from the features used 
(only the last three stocks' returns), the same experiment was extended using more elaborated 
secondary models (light gradient boosted machine; homogeneous ensembles with Random Forest; 
and heterogeneous ensembles with Logistic Regression, Decision Tree, Support Vector Machine, 
Naïve Bayes and Multilayer Perceptron) proposed by H&T in reference [5]. The results are 
presented in the picture below.


<div style="text-align:center;">
    <img src="../images/Explanation-06.02.jpg" alt="Initial Results" width="70%">
</div>

Despite employing more advanced models, the picture indicates no discernible improvement 
from the primary model to these secondary models when using real stock data. The suspicion 
arises that relying solely on the last three stocks' returns might not provide sufficient 
information for the secondary model to enhance directional predictions. This hypothesis 
will be verified in the next steps of the project.

## 7. Intended Improvements and Future Work

As with any evolving strategy, there exist opportunities for enhancement and refinement. 
Plans are underway to further optimize the Meta-Labeling strategy throught the following 
next steps:

 - Enhance the primary model (Random Forest and additional features) and the secondary 
   model (performance metrics of the primary model).
 - Incorporate Liquidity/Volume criteria for selecting portfolio assets.
 - Evaluate different sizing criteria.

Moreover, future works could leverage the insights gained from this project to apply the 
meta-labeling framework to various primary strategies."

## 8. References

**[1]** López de Prado, M. M. “The 10 Reasons Most Machine Learning Funds
Fail”. The Journal of Portfolio Management. v. 44, n. 6, p. 120–133, 2018a.

**[2]** López de Prado, M. M. Advances in Financial Machine Learning. New
Jersey: John Wiley & Sons, p. 50-55, 2018b.

**[3]** Joubert, J. F. “Meta-Labeling: Theory and Framework”. The Journal of
Financial Data Science. v. 4, n.3, p. 31-44, 2022.

**[4]** Meyer, M., Joubert, J. F., Messias, A. “Meta-Labeling Architecture”. The
Journal of Financial Data Science. v. 4, n. 4, p. 10–24, 2022.

**[5]** Thumm, D., Barucca, P., Joubert, J. F. “Ensemble Meta-Labeling”. The
Journal of Financial Data Science. v. 5, n. 1, p. 10-26, 2023.

**[6]** Meyer, M., Barziy, I., Joubert, J. F. “Meta-Labeling: Calibration and
Position Sizing”. The Journal of Financial Data Science. v. 5, n. 3, p. 23-
40, 2023.

