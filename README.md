`PyCPI` : A package for capturing, NLP preprocessing,
and Solr indexing of mandatory reports in the
Brazilian Capital Market
==============================================================================================
This package delivers a pipeline that captures
financial reports provided by the Brazilian Securities and Exchange Commission (CVM, 2023), performs preprocessing, including PDF-to-text conversion, tokenization, and tensor generation using the Transformer (Beltagy et al., 2020) architecture. It then indexes the metadata using `Apache Solr``
 for efficient and fast information retrieval. Another functionality offered by the PyCPI package is text comparison, which measures the similarity between two tensors. The package also includes features such text summarization, highlighting the most significant sentences in the text,
among others.
=======
Documentação do pacote Pipeline captura, pré-processamento e indexação de metadados (CPI)  no Github.
Este projeto faz parte do grupo de interesse em Finanças Computacionais e investimentos Sistemáticos multidisciplinar atuando dentro do Instituto Tecnológico de Aeronáutica (FICO-ITA).

## Documentation

- [Click here](docs/index.md) to access the documentation.

* [Click here](docs/PCI_package.pdf) to access the manuscript.
  

## How to install


Step 1: Install Poetry
```text
Make sure you have Poetry installed on your system. You can follow the installation instructions on the official Poetry website:
https://python-poetry.org/docs/#installation
```

Step 2: Clone the GitHub Repository

```bash
$ git clone https://github.com/fico-ita/po_245_2023_T4.git
```
Step 3: Navigate to the Cloned Directory


```bash
$ cd po_245_2023_T4
```

Step 4: Install Dependencies Using Poetry


```bash
$ poetry install
```

Step 5: Activate the Virtual Environment


```bash
$ poetry shell
```


## License

[Apache License 2.0](LICENSE)



## Citação

### APA
```text
C. N da Silva, T. Pipeline de disponibilização dos Relatórios Obrigatórios no mercado de Capitais Brasileiro [Computer software]. https://github.com/fico-ita/po_245_2023_T4.git
```

### BibTeX
```bibtex
@software{C_N_da_Silva_Pipeline_de_disponibilizacao,
author = {C. N da Silva, Thiago},
title = {{Pipeline de disponibilização dos Relatórios Obrigatórios no mercado de Capitais Brasileiro}},
url = {https://github.com/fico-ita/po_245_2023_T4.git}
}
```


