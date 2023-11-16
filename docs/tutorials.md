# Tutorials

## Building a Stock Portfolio Using Meta-Labeling ML Approach

Welcome to the Meta-Labeling Portifolio Strategy tutorial! 📈

In this guide, we'll walk you through the practical steps to construct a diversified 
stock portfolio using the Meta-Labeling approach developed within the FICO-ITA project.

### Step 0: Prerequisites
Ensure the required libraries are installed: numpy, pandas, statsmodels, scikit-learn and 
QuantStats.  If not installed, you can do so by using poetry:

```bash
poetry add numpy pandas statsmodels scikit-learn mQuantStats
```


### Step 1: Project Setup
Download the project and stocks data from GitHub:

```bash
poetry add git+https://github.com/fico-ita/po_245_2023_S2_T5.git
poetry add git+https://github.com/fico-ita/template_projetos.git
```

Verify that the directory structure matches the setup provided in [How-to Guides: Installation](how-to-guides.md).🔍

### Step 2: Meta-Labeling Strategy Overview
Explore the four crucial blocks of the [Meta-Labeling strategy](reference.md):

 - Inputs and Data Preparation📊
 - Individual Stock MetaLabeling Application📑
 - Position Sizing📏
 - Portfolio Construction Strategy🛠️

### Step 3: Executing the Strategy
Now, let's put this into action:

```python
#FICO Load Data module
from modules.load_data import load_data
dict_data = load_data()

#Stocks Portfolio Construction Based on a Meta-Labeling ML Approach
from fico.strategy_meta_labeling_r04 import D_strategy_meta_labeling

Portifolio = D_strategy_meta_labeling(dict_data, t = 2000, size = 10, window_size= 500)
```

### Step 4: Reviewing Results
Check out the resulting portfolio composition:

```bash
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
```
### Step 5: Further Exploration

Feel free to experiment with different parameters or extend the strategy. The other documentation 
sections like [How-To Guides](how-to-guides.md), [Reference](reference.md), and 
[Explanation](explanation.md) offer deeper insights into the inner workings and possibilities for improvement.🚀


### Step 6: Conclusion

This documentation serves as a gateway to the world of meta-Labeling computational finance. Whether you're 
here to learn, apply, or innovate, we welcome you to the FICO-ITA experience. May your 
exploration be insightful and your financial endeavors prosperous!🌟


