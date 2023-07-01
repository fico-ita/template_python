"""Este módulo contém funções e bibliotecas relacionadas ao pipeline de captura,
pré-processamento e indexação de relatórios.
"""  # noqa: D205

import io
import json
import subprocess
import zipfile
from pathlib import Path

import PyPDF2
import requests
import torch
import unidecode
from transformers import (
    BertForQuestionAnswering,
    LongformerModel,
    LongformerTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

model_for_question = BertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad",
)
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096")
model = LongformerModel.from_pretrained("allenai/longformer-large-4096")
model_sum = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer_sum = T5Tokenizer.from_pretrained("t5-base")
# Caminho para o diretório bin do Solr
slrb = Path("C:\\Users\\thgcn\\OneDrive\\Academico\\PO-245\\Projeto\\solr-9.2.1\\bin\\")


def verificar_diretorio():
    """Verifica se o diretório informado pelo usuário é válido.

    Examples:
        >>> verificar_diretorio()
        Digite o caminho do diretório: /caminho/do/diretorio
        Diretório raiz para armazenamento dos documentos:
        '/caminho/do/diretorio'

    """
    root = Path(input("Digite o caminho do diretório: "))
    try:
        if root.is_dir():
            print("Diretório raiz para armazenamento dos documentos: " + str(root))
            return root
    except FileNotFoundError:
        print(
            "O diretório inserido não existe ou não é válido. \
            Certifique-se que o nome está correto"
        )


def busca_ipe(ano):
    """Obtém os dados do IPE (Informações Periódicas) de uma determinada empresa
    para o ano especificado.

    Args:
        ano (int): O ano para o qual se deseja obter os dados do IPE.

    Returns:
        list: Uma lista contendo as linhas dos dados do IPE da empresa.

    Examples:
        >>> busca_ipe(2022)
         ['Empresa A', '123456', '10.2', 'http:dados.cvm/ipe/123456'],
         ['Empresa B', '789012', '5.8', ''http:dados.cvm/ipe/789455']

    """  # noqa: D205
    url = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/"
    url += "ipe_cia_aberta_%d.zip" % ano
    file = "ipe_cia_aberta_%d.zip" % ano
    r = requests.get(url)
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    file = zf.namelist()
    zf = zf.open(file[0])
    lines = zf.readlines()
    lines = [i.strip().decode("ISO-8859-1") for i in lines]
    lines = [i.split(";") for i in lines]
    return lines


def search(lista, valor):
    """Retorna uma lista de elementos da lista que contêm o valor especificado.

    Args:
        lista (list): A lista de elementos para realizar a busca.
        valor (str): O valor a ser procurado nos elementos da lista.

    Returns:
        list: Uma lista dos elementos que contêm o valor especificado.

    Examples:
        >>> search(['apple', 'banana', 'orange'], 'an')
        ['banana', 'orange']

    """
    return [(lista[lista.index(x)]) for x in lista if valor in x]


def baixar_arquivo(url, endereco):
    """Faz o download de um arquivo a partir de uma URL e salva no caminho especificado.

    Args:
        url (str): A URL do arquivo a ser baixado.
        endereco (str): O caminho completo de destino para salvar o arquivo.

    Raises:
        Exception: Se ocorrer um erro ao fazer o download do arquivo.

    Examples:
        >>> baixar_arquivo('https://example.com/file.pdf', '/path/to/save/file.pdf')
        Download finalizado. Arquivo salvo em: /path/to/save/file.pdf

    """
    resposta = requests.get(url, allow_redirects=True, verify=False, stream=True)
    if resposta.status_code == requests.codes.OK:
        with endereco.open("wb") as novo_arquivo:
            novo_arquivo.write(resposta.content)
        print(f"Download finalizado. Arquivo salvo em: {endereco}")
    else:
        resposta.raise_for_status()


def transform_string(text):
    """Remove acentos, substitui espaços por underline, converte letras maiúsculas em
    minúsculas.

    Args:
        text (str): A string a ser transformada.

    Returns:
        str: A string transformada.

    Examples:
        >>> transform_string("Olá, Mundo!")
        'ola_mundo'

    """  # noqa: D205
    text = unidecode.unidecode(text)
    text = text.replace(" ", "_")
    text = text.replace(".", "")
    text = text.lower()
    return text


def download_def(empresa, year, root):  # noqa: C901, PLR0915
    """Realiza download de arquivos específicos com base na empresa e no ano fornecidos.

    Args:
        empresa (str): Nome da empresa para qual deseja fazer o download dos arquivos.
        year (int): O ano para o qual deseja-se fazer o download dos arquivos.
        root(str):  Caminho onde serao armazenados os documentos

    Returns:
        list: Uma lista de dicionários contendo os metadados dos arquivos baixados.

    Raises:
        ValueError: Se houver duplicidade de empresas encontradas.

    Examples:
        >>> download_def("Empresa XYZ", 2023)
        Empresa encontrada: Empresa XYZ | codigo cvm: 12345
        Ano: 2023
        nome do arquivo: 12345_empresa_xyz
        caminho dos arquivos: /caminho/do/arquivo/12345_empresa_xyz

    """
    lines = busca_ipe(year)
    defi = search(lines, "Dados Econômico-Financeiros")
    data = []
    empresa_name = ""
    num_cvm = ""
    for a in range(len(defi)):
        if empresa.upper() in defi[a][1]:
            empresa_name = defi[a][1]
            num_cvm = defi[a][2]
            print("Empresa encontrada: " + empresa_name + " | codigo cvm: " + num_cvm)
            print("Ano: " + str(year))
            sufix = transform_string(defi[a][2] + "_" + defi[a][1])
            company = root / sufix
            category = transform_string(defi[a][5])
            print("nome do arquivo: " + sufix)
            print("caminho dos arquivos: " + str(company))
            if not company.is_dir():
                print("diretorio não existe e será criado")
                company.mkdir(parents=True)
            diryear = company / str(year)
            if not diryear.is_dir():
                print("diretorio não existe e será criado")
                diryear.mkdir(parents=True)
            dircategory = diryear / category
            if not dircategory.is_dir():
                print("diretorio não existe e será criado")
                dircategory.mkdir(parents=True)
            url = defi[a][12]
            nome = (
                transform_string(defi[a][2])
                + "_"
                + defi[a][1]
                + "_"
                + defi[a][7][1:50]
                + "_"
                + defi[a][8]
            )[1:100]

            baixar_arquivo(url, dircategory / f"{nome}.pdf")

            title = transform_string(defi[a][6][0:50])
            if not title:
                if not defi[a][7]:
                    title = defi[a][7]
                elif not defi[a][5]:
                    title = defi[a][5]
                elif not defi[a][4]:
                    title = defi[a][4]
                else:
                    title = defi[a][1]
            company_name = transform_string(defi[a][1])
            cod_cvm = defi[a][2]
            date = defi[a][3]
            content = defi[a][7]
            if not content:
                content = defi[a][6]
            keyword = defi[a][4]
            file_path = dircategory / f"{nome}.pdf"
            size = file_path.stat().st_size
            file = str(file_path)
            data.append(
                {
                    "name": title,
                    "type": "string",
                    "company_name": company_name,
                    "cod_cvm": cod_cvm,
                    "content": content,
                    "year": year,
                    "date": date,
                    "keywords": keyword,
                    "size": size,
                    "tokens": "",
                    "tensor": "",
                    "file": file,
                }
            )
    return data


def convert_pdf(doc):
    """Converte um arquivo PDF em texto.

    Args:
        doc (str): O caminho do arquivo PDF a ser convertido.

    Returns:
        str: O texto extraído do arquivo PDF.

    Examples:
        >>> text = convert_pdf("path/to/file.pdf")
        >>> print(text)
        Text extracted from the PDF file.
    """
    with Path(doc).open("rb") as f:
        # Use a biblioteca PyPDF2 para ler o arquivo
        reader = PyPDF2.PdfReader(f)
        # Obtenha o número de páginas no arquivo
        num_pages = len(reader.pages)
        # String vazia para armazenar o texto extraído
        text = ""
        # Itere pelas páginas e extraia o texto
        for page in range(num_pages):
            # Obtenha o objeto da página
            # page_obj = reader.getPage(page)
            page_obj = reader.pages[page]
            # Extraia o texto da página
            # page_text = page_obj.extractText()
            page_text = page_obj.extract_text()
            # Adicione o texto extraído à string de texto
            text += page_text
    return text


def tokengen(text):
    """Gera uma lista de tokens a partir de um texto.

    Args:
        text (str): O texto a ser tokenizado.

    Returns:
        list: Uma lista de tokens.

    Examples:
        >>> text = "Hello, World! This is an example text."
        >>> tokens = tokengen(text)
        >>> print(tokens)
        ['[CLS]', 'hello', ',', 'world', '!', 'this',
        'is', 'an', 'example', 'text', '.', '[SEP]']
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return tokenizer.convert_ids_to_tokens(tokens)


def vector_one(text):
    """Gera um vetor de representação para um texto.

    Args:
        text (str): O texto a ser vetorizado.

    Returns:
        torch.Tensor: O vetor de representação do texto.

    Examples:
        >>> text = "This is a sample text."
        >>> text_vector = vector_one(text)
        >>> print(text_vector)
        torch.Tensor([[0.1, 0.2, 0.3, ...]])
        # Example tensor representing the vector for the text.
    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)

    max_length = 4096
    vectors = []
    # Divide the tokenized text into parts
    input_parts = [
        input_ids[:, i : i + max_length]
        for i in range(0, input_ids.shape[1], max_length)
    ]
    # Process each part of the tokenized text
    for part in input_parts:
        # Generate the embeddings vectors
        with torch.no_grad():
            outputs = model(part)
            last_hidden_states = outputs.last_hidden_state

        # Calculate the average vector
        average_vector = torch.mean(last_hidden_states, dim=1)

        # Add the average vector to the list
        vectors.append(average_vector)

    # Concatenate the generated vectors
    return torch.cat(vectors, dim=1)


def check_and_start_solr():
    """Verifica se o serviço do Solr está em execução e o inicia, se necessário.

    Returns:
        int: O código de retorno da execução do comando.
        str: A mensagem de erro, caso ocorra.

    Examples:
        >>> check_and_start_solr()
        Verificando o status do Solr...
        Solr já está em execução.
        >>> check_and_start_solr()
        Verificando o status do Solr...
        Iniciando o Solr...
        Solr iniciado com sucesso.
    """
    # Comando para verificar o status do serviço do Solr
    status_command = f"{slrb}\\solr status"

    # Comando para iniciar o serviço do Solr
    start_command = f"{slrb}\\solr start -port 8983"

    try:
        print("Verificando o status do Solr...")
        # Verifica se o Solr está em execução
        try:
            result = subprocess.run(
                status_command, shell=True, capture_output=True, text=True, check=True
            )
            if "no running" not in result.stdout.lower():
                print("Solr já está em execução.")
                return result.returncode, "Solr já está em execução."
        except subprocess.CalledProcessError as e:
            print(f"Erro ao verificar o status do Solr: {e.stderr.strip()}")
            return e.returncode, e.stderr.strip()

        print("Iniciando o Solr...")
        # Inicia o Solr em segundo plano
        process = subprocess.Popen(start_command, shell=True)
        process.communicate()  # Aguarda a conclusão do processo em segundo plano

        if process.returncode == 0:
            print("Solr iniciado com sucesso.")
            return process.returncode, "Solr iniciado com sucesso."
        print("Erro ao iniciar o Solr.")
        return process.returncode, "Erro ao iniciar o Solr."

    except Exception as e:
        print(f"Erro desconhecido: {str(e)}")
        return -1, str(e)


def add_document_to_solr(collection_name, document):
    """Adiciona um documento à coleção Solr especificada.

    Args:
        collection_name (str): O nome da coleção onde o documento será adicionado.
        document (dict): Dicionário contendo os metadados a serem indexados.

    Examples:
        >>> document = {"title": "Example Document",
            "content": "This is an example document.",
            "author": "John Doe"}
        >>> add_document_to_solr("my_collection", document)
        Documento adicionado com sucesso!

    """
    # Converter o caminho do arquivo em string
    document = {k: str(v) if isinstance(v, Path) else v for k, v in document.items()}

    # URL da API do Solr para adicionar documentos
    solr_url = f"http://localhost:8983/solr/{collection_name}/update?commit=true"

    # Envia a requisição POST para adicionar o documento
    response = requests.post(solr_url, json=[document])

    # Verifica o status da resposta
    status_ok = 200
    if response.status_code == status_ok:
        print("Documento adicionado com sucesso!")
    else:
        print("Erro ao adicionar o documento:", response.text)
        print(response.content.decode())


def pipeline(empresa, ano):
    """Executa o pipeline de captura,
    preprocessamento e indexação de documentos econômico-financeiros.

    Args:
        empresa (str): O nome da empresa para o qual deseja-se executar o pipeline.
        ano (int): O ano para o qual deseja-se executar o pipeline.


    Returns:
        str: Uma mensagem indicando o resultado da atualização do esquema no Solr.


    Examples:
        >>> pipeline("Empresa XYZ", 2023)
        Inicio do pipeline de captura *********
        Empresa encontrada: Empresa XYZ | codigo cvm: 123456
        Ano: 2023
        nome do arquivo: 123456_Empresa_XYZ
        caminho dos arquivos: /caminho/do/diretorio/123456_Empresa_XYZ
        diretorio não existe e será criado
        Download finalizado. Arquivo salvo em:
        /caminho/do/dir/123_Empresa_XYZ/2023/Economic_Fin_Data/123456_Empresa_XYZ.pdf
        geração de token e tensor.....
        Documento adicionado com sucesso!
    """  # noqa: D205
    # Faz o download do documento
    root = verificar_diretorio()
    collection_name = "dados_eco"
    print("Inicio do pipeline de captura *********")
    doc = download_def(empresa, ano, root)
    print("Inicio do pipeline de preprocessamento e indexacao *********")
    check_and_start_solr()
    for item in doc:
        # Define o tipo do item como "document"
        item["type"] = "document"
        # Define o nome da coleção como "dados_eco"
        collection_name = "dados_eco"
        # Obtém o diretório do arquivo
        diretorio = Path(item["file"]).parent
        # Obtém o nome do arquivo sem a extensão
        file_path = Path(item["file"])
        tokens_tensor = file_path.stem, file_path.suffix
        # Define o caminho do arquivo de tokens e tensor
        dataprep = Path(diretorio) / f"{tokens_tensor}_TOKENS_TENSOR.json"
        # Cria o output data com os tokens e tensor
        text = convert_pdf(item["file"])
        token = tokengen(text)
        vector = vector_one(text).tolist()
        output_data = [
            {
                "tokens": token,
                "tensor": vector,
            }
        ]

        # Salva o output data em um arquivo JSON
        with dataprep.open("w") as file:
            json.dump(output_data, file)

        # Atualiza os campos tokens e tensor do item com o caminho do arquivo
        item["tokens"] = dataprep
        item["tensor"] = dataprep
        # Verifica se algum campo está ausente e define como
        # "Descrição ausente" se necessário
        fields = [
            "name",
            "type",
            "company_name",
            "cod_cvm",
            "content",
            "year",
            "date",
            "keywords",
            "size",
            "tokens",
            "tensor",
            "file",
        ]
        for field in fields:
            if not item[field]:
                item[field] = "Descrição ausente"
        # Adiciona o documento ao Solr
        add_document_to_solr(collection_name, item)
