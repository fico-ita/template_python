# Estrutura FICO

Documentação do pacote XXX no Github.
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

Um bom exemplo de Docstring no formato Google é o
[Napoleon's documentation](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

Veja as [seções docstrings na extensão Napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)

[Guia Khan](https://github.com/Khan/style-guides/blob/master/style/python.md#docstrings)
sobre Docstring Google.

## Git

É aconselhável o uso do git utilizando o fluxo de trabalho conhecido como [Trunk Based
Development (TBD)](https://cloud.google.com/architecture/devops/devops-tech-trunk-based-development),
i.e., pequenos incrementos, ao invés de gitflow.

## Licença

Escolha a licença Apache 2.0 e deixe o repositório como privado, enquanto atinge um
mínimo de qualidade. Inclua as licenças sobre os dados, quando houver, lembrando que
os dados não devem ter controle de versão, ou seja, não os adicione em uma pasta do
projeto. Informe seus links de acesso.

## Agradecimento

Escolha um bom nome de projeto e adicione os logos dos apoiadores para deixar sua
documentação com uma imagem mais profissional.

## Como citar

Copie aqui a forma de citação do software em formato de desejar e inclua o arquivo
[CITATION.cff](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) no projeto.

Caso tenha, adicione também a citação do paper conceitual sobre a solução.

Inclua também sua referência ao FICO.
