# FICO T3

FICO-ITA is a multidisciplinary interest group in Computational Finance and Systematic Investments
operating within the ITA, in Brazil. 

## Project

This project constructs a stocks portfolio based on the MetaLabeling approach proposed by López de Prado and as implemented by Joubert.

It is part of the requirements for the completion of the graduate course "PO-245. Aprendizado de Máquina em Finanças
Quantitativa" in 2023S2.

## Usage

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

### Requirements

Python 3.11 or higher is required. Ensure the required libraries are installed

- `numpy`
- `pandas`
- `statsmodels`
- `QuantStats`

### Example
    ```python
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
```


## Documentation

The documentation is available on [GitHub](
https://github.com/fico-ita/po_245_2023_S2_T5/tree/main/docs
)

## License

[Apache License 2.0](LICENSE)

## Citation

Since this project is research based, it is important to cite it in your work.
To cite this project, use the following reference:

### BibTeX
```bibtex
@misc{feitoza2023finance,
    author = {Feitoza, A. P.},
    title = {Stocks Portfolio Construction Based on a Meta-Labeling ML Approach},
    year = {2023},
    DOI = {10.5281/zenodo.9990001},
    publisher = {Zenodo},
    url = {https://doi.org/10.5281/zenodo.9990001} [TBD]
}
```
### APA
```text
Feitoza, A. P.(2023), Stocks Portfolio Construction Based on a Meta-Labeling ML Approach.
Zenodo. https://doi.org/10.5281/zenodo.9990001 [TBD]
```
