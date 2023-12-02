To get started with the **Stocks Portfolio Construction Based on a Meta-Labeling** project 
follow the steps described below.


### Installation

Download this project from the following GitHub repository:
```bash
poetry add git+https://github.com/fico-ita/po_245_2023_S2_T5.git
```

Get FICO's stocks data from the GitHub below. Copy folders `dataset` and `modules` to
directory `po_245_2023_S2_T5`.
```bash
poetry add git+https://github.com/fico-ita/template_projetos.git
```

The resulting directory structure must look like this:

    po_245_2023_S2_T5/
    │
    ├── docs/
    │   ├── images/
    │   ├── materials/
    │   ├── tutorials/
    │   ├── index.md
    │   ├── tutorials.md    
    │   ├── how-to-guides.md
    │   ├── reference.md
    │   └── explanation.md
    │
    ├── fico/
    │   ├── __init__.py
    │   └── strategy_meta_labeling_r04.py
    │
    ├── dataset/    
    │
    ├── modules/    
    │
    └── mkdocs.yml



### Usage

With the above structure in place, the package can be imported and executed as follows:

```python
from fico.strategy_meta_labeling_r04 import D_strategy_meta_labeling
Portfolio = D_strategy_meta_labeling(dict_data, t = 2000, size = 10, window_size= 500)
```

This way you should be able to get the following next-day[^1] portfolio of *10* stocks with their 
allocation weights:

```bash
        date        ticker  weights
        2019-05-11  ALB	    0.101264
        2019-05-11  REGN    0.101264
        2019-05-11  MYL	    0.101264
        2019-05-11  BKNG    0.100882
        2019-05-11  ADI	    0.100286
        2019-05-11  PHM	    0.099938
        2019-05-11  AES     0.099651
        2019-05-11  PXD     0.098805
        2019-05-11  WELL    0.098347
        2019-05-11  FDX     0.098298
```

[^1]:starting the dataset at `t=2000` and spaning it through `window_size= 500` days
yields `2019-05-10` as the dataset last entry day.