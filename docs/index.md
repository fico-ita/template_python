# Bem-vindo ao FICO

Nesta página, apresentaremos a documentação do pacote Pipeline para captura, pré-processamento e indexação de metadados (CPI).

Este projeto faz parte do grupo de interesse em Finanças Computacionais e Investimentos Sistemáticos multidisciplinar, atuando dentro do Instituto Tecnológico de Aeronáutica ([FICO-ITA](https://fico-ita.github.io/fico/)).

## Organização da documentação

Para a organização da documentação de usuário, baseie-se na proposta de
[Diátaxis](https://diataxis.fr/), que consiste em 4 classes de documentos:

1. Tutorial
1. Reference
1. Explanation


As documentações acima são pensadas para o leitor usuário da solução.O público alvo são os desenvolvedores e arquitetos da solução.

Nesta documentação o foco é a estrutura proposta para o pipeline e a organização da solução, apresentando um diagrama de componentes da solução e como eles se interagem. 

## Pipeline CPI - FICO

::: fico

### Arquitetura proposta:

<figure>
  <img src="componente_cpi-P%C3%A1gina-2.drawio.png" alt="componente cpi" width="800" height="500" />
  <figcaption>Pipeline de captura, pré-processamento e indexação de metadados</figcaption>
</figure>

Etapas

* `1`  Conecta ao portal de dados abertos CVM
* `2`  Faz o download da lsita de documentos, contendo nome da empresa, tipo de documento, link para download.
* `3` Cria a estrutura caso necessário, faz o download do relatorio economico financeiro armazenando o em pdf
* `4` Converte o pdf para texto
* `5` Gera tokens e tensores do respectivo documento, armazenando localmente no mesmo diretorio do relatório pré-processado
* `6` Gera o json contendo todos os metadados da etapa de captura e préprocessamento
* `7`  Inicializa o serviço solr (caso não esteja ativo), indexa os metadados





## Artigo

[Pipeline de disponibilização dos Relatórios Obrigatórios no mercado de Capitais Brasileiro](Pipeline_de_disponibilização_de_relatórios_obrigatórios_no_mercado_de_capitais_brasileiro.pdf)

## Licença

[Apache License 2.0](LICENSE)

## Agradecimento

<div style="display: flex; justify-content: center;">
  <div style="flex: 1; text-align: center;">
    <figure>
      <img src="Elton_Sbruzzi.png" alt="Elton Sbruzzi" width="150" height="150" />
      <figcaption>Elton Sbruzzi</figcaption>
    </figure>
  </div>

  <div style="flex: 1; text-align: center;">
    <figure>
      <img src="Michel_Leles.png" alt="Michel Leles" width="150" height="150" />
      <figcaption>Michel Leles</figcaption>
    </figure>
  </div>

  <div style="flex: 1; text-align: center;">
    <figure>
      <img src="vitor_curtis.png" alt="Vitor Curtis" width="150" height="150" />
      <figcaption>Vitor Curtis</figcaption>
    </figure>
  </div>
</div>

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