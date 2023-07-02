
# Pipeline (CPI)

Neste tutorial apresentaremos o Pipeline de Captura, Pré-processamento e Indexação de metadados (CPI), estrutura abrangente para a disponibilização de relatórios financeiros fornecidos pelas empresas à Comissão de Valores Mobiliários (CVM), com o objetivo de facilitar a análise e o acesso a informações financeiras relevantes. 
A estrutura proposta é composta por várias etapas interconectadas, que abrangem desde a coleta inicial dos dados financeiros (por meio de um pipeline de captura) até a sua disponibilização final para os usuários, utilizando a indexação de metadados por meio do Apache
Solr.


## Instalação do Apache Solr

O Apache Solr é a ferramenta utilizada par aindexação de metadados e busca de informações. É necessário fazer o download e executar os seguintes passos antes de utilizar o pipeline:

* Passo 1: Baixe o Apache solr: https://solr.apache.org/downloads.html

* Passo 2: Extraia o Apache solr para um diretório da sua escolha

* Passo 3: Abra um terminal ou prompt de comando

* Passo 4: Navegue até o diretório onde você extraiu o Solr, e execute o seguinte comando para inicaliza-lo

```bash
bin/solr start
```
!!! documentação 
    A documentação Oficial e os tutoriais da ferramentas seguem no site Apache Solr Reference Guide: https://solr.apache.org/guide/solr/latest/index.html

!!! warning
    Existem duas maneiras de trabalhar com o Solr e seus comandos diretem:
    modo standalone: O qual o serviço é instalado em uma instacia única.
    modo cloud: O qual o serviço é instalado em um cluster (mais de uma instância)
    Para este projeto trabalharemos no modo standalone.

## Instalação dos Pacotes 


Passo 1: Instale o Poetry
```text
Certifique-se de ter o Poetry instalado em seu sistema. Você pode seguir as instruções de instalação no site oficial do Poetry: 
https://python-poetry.org/docs/#installation
```
Passo 2: Clone o repositório do Github

```bash
$ git clone https://github.com/fico-ita/po_245_2023_T4.git
```
Paso 3: Navegue até o diretório clonado.
```bash
$ cd po_245_2023_T4
```

Passo 4: Instale as dependencias usando o Poetry
```bash
$ poetry install
```

Passo 5: Ative o ambiente virtual

```bash
$ poetry shell
```
## Utilização do componente CPI


