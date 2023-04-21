# Estrutura FICO

Documentação do pacote XXX no Github.
Utilize o README.md mais como um ponteiro para a documentação oficial e instruções
pertinentes.

Adicione as seções *Como citar* e *Apoio*, citando explicitamente as empresas parceiras
que utilizou.

## Estrutura

Esta estrutura utiliza basicamente

- Poetry como ferramenta de empacotamento e gerenciador de pacotes
- Mkdocs para documentação, com template Material, e mkdocstrings para formatação do
  docstring no [formato Google](https://google.github.io/styleguide/pyguide.html)
- Flake8 e Black são usados para estilo de código
- Pre-commit é utilizado para verificações antes de `commit`

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

Caso receba alguma mensagem de erro, pode executar apenas o id que resultou em erro.
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

## Licença e imagens

Escolha a licença Apache 2.0 e deixe o repositório como privado, enquanto atinge um
mínimo de qualidade.

Escolha um bom nome de projeto e adicione as imagens dos apoiadores para deixar sua
documentação com uma imagem mais profissional.
