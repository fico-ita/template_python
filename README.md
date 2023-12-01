# Estrutura FICO

Documentação do pacote lstm_strategy no Github.
Utilize o README.md mais como um ponteiro para a documentação oficial e instruções
pertinentes.

Não esqueça de preencher as seções *Como citar* e *Agradecimento*, citando
explicitamente as empresas parceiras que apoiaram a solução.

# TODO

- Falta adicionar testes e log
- Mudar pasta padrão de código para src
- Testar empacotamento

## Estrutura

Esta estrutura utiliza basicamente

- Poetry como ferramenta de empacotamento e gerenciador de pacotes
- [Mkdocs](https://www.mkdocs.org/) para documentação, com template
  [Material](https://squidfunk.github.io/mkdocs-material/setup/), e
  [mkdocstrings](https://mkdocstrings.github.io/) para formatação do docstring no
  [formato Google](https://google.github.io/styleguide/pyguide.html)
- Ruff e Black são usados para estilo de código
- Pre-commit é utilizado para verificações antes de `commit`

### Por onde começar

O arquivo `pyproject.toml` é o arquivo principal de configuração do pacote. Altere a
seção `[tool.poetry]`.

A estrutura é configurada com um mínimo de funcionalidades para obter bons resultados.

A configuração de pacotes é feita com Poetry ao invés de pip.

```bash
# ative o virtual environment
cd pasta_projeto
poetry shell

# instalar pacotes do projeto
poetry install

# adicione uma biblioteca necessária
poetry add nome_biblioteca
# o pacote será adicionado em [tool.poetry.dependencies] de pyproject.toml

# caso queira adicionar uma biblioteca necessária para desenvolvimento e não para uso
# do pacote
poetry add add --group dev nome_biblioteca

# Dica: ative o ambiente virtual e depois chame o IDE (e.g. code .) para o encontrar
```

Ao contrário do tradicional, neste configuração, o código fonte pode ser encontrado
dentro da pasta (pacote python) `fico`. A documentação na pasta `docs`. Caso desejar,
exclua o `main.py`, que é apenas um script dummy.

Esta estrutura exige python 3.11. Talvez tenha que o instalar, assim como outras
ferramentas e pacotes. Adicionalmente, também já está incluso uma pequena configuração
para o vscode.

### Commits

Antes de começar a fazer commits, inicialize a configuração do `pre-commit`.

```bash
# Para fazer, use o comando
pre-commit install
```

Ao fazer commits, caso receba alguma mensagem de erro, pode-se executar apenas o id que
resultou em erro.
Por exemplo, se o id `fix-encoding-pragma` resultou em erro

```bash
# execute apenas esta verificação
pre-commit run fix-encoding-pragma

# analise as alterações feitas nos arquivos e mensagem de erro
# faça stage das alternações
git add file_name

# verifique novamente
pre-commit run fix-encoding-pragma

# se tudo der certo, pode tentar o commit novamente
```

### Documentação

Veja `docs` para servir a documentação.

## Licença

- Licença de código: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Fonte de dados: [Stock Market, Funds, Fixed Income and Asset Consolidation Data Analysis](https://www.comdinheiro.com.br/)
- Grupo de suporte: [FICO](https://fico-ita.github.io/)

## Agradecimento
Nós, integrantes do projeto LSTM & Low Volatility Quant, agradecemos:

- O suporte da plataforma **comdinheiro** da **Nelogica** em providenciar os dados de preços históricos das ações usados neste projeto.

![comdinheiro Logo](./comdinheiroinvest---versao-escura.png)

- O suporte prático e intelectual dado por todo os indivíduos-chave existentes no grupo **FICO-ITA**.

![FICO-ITA Logo](./FICO_ITA_logo.png)

## Como citar

Inclua nas citações de seu projeto o seguinte:
```python
authors:
- family-names: "Bustos"
  given-names: "Victor"
  LinkedIn: "https://www.linkedin.com/in/victoropb/"
- family-names: "Nunes"
  given-names: "Marcus"
  LinkedIn: "https://www.linkedin.com/in/marcusganunes/"
title: "Low Volatility and High Momentum Investment Strategy"
version: 1.0
url: "https://github.com/fico-ita/po_245_2023_S2_T4/tree/main"
```