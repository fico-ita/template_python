# FICO - Pipeline (CPI)

Documentação do pacote Pipeline captura, pré-processamento e indexação de metadados (CPI)  no Github.
Este projeto faz parte do grupo de interesse em Finanças Computacionais e investimentos Sistemáticos multidisciplinr atuando dentro do Instituto Tecnológico de Aeronáutica (FICO-ITA).


## Estrutura

Esta estrutura utiliza basicamente

- Poetry como ferramenta de empacotamento e gerenciador de pacotes
- [Mkdocs](https://www.mkdocs.org/) para documentação, com template
  [Material](https://squidfunk.github.io/mkdocs-material/setup/), e
  [mkdocstrings](https://mkdocstrings.github.io/) para formatação do docstring no
  [formato Google](https://google.github.io/styleguide/pyguide.html)
- Ruff e Black são usados para estilo de código
- Pre-commit é utilizado para verificações antes de `commit`

Lembre-se que o pacote deve conter a parte reproduzível de seu projeto. O uso deve ser
construído como um dos exemplos ou em outro repositório, que utiliza este pacote.

### Por onde começar

O arquivo `pyproject.toml` é o arquivo principal de configuração do pacote. Altere a
seção `[tool.poetry]`.

A estrutura é configurada com um mínimo de funcionalidades para obter bons resultados,
mas talvez você queira melhorar a configuração conforme compreender as ferramentas.

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

Note que não há testes nesta estrutura, o que é crucial, mas que não é exigido neste
trabalho acadêmico.

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

Esta documentação esta disponível em:

## License

[Apache License 2.0](LICENSE)

## Agradecimento

<div style="display: flex; justify-content: center;">
  <div style="flex: 1; text-align: center;">
    <figure>
      <img src=".src/Elton_Sbruzzi.png" alt="Elton Sbruzzi" width="150" height="150" />
      <figcaption>Elton Sbruzzi</figcaption>
    </figure>.src/
  </div>

  <div style="flex: 1; text-align: center;">
    <figure>
      <img src=".src/Michel_Leles.png" alt="Michel Leles" width="150" height="150" />
      <figcaption>Michel Leles</figcaption>
    </figure>
  </div>

  <div style="flex: 1; text-align: center;">
    <figure>
      <img src=".src/vitor_curtis.png" alt="Vitor Curtis" width="150" height="150" />
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


