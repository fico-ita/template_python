"""Este módulo contém funções e bibliotecas relacionadas ao pipeline de captura,
pré-processamento e indexação de relatórios.
"""
root = os.path.dirname(
    "C:\\Users\\thgcn\\OneDrive\\Academico\\Mestrado - NLP - Finance\\datasets\\",
)

import io
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

import requests
import spacy
import torch
import unidecode
from sklearn.metrics.pairwise import cosine_similarity
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


def verificar_diretorio():
    """Verifica se o diretório informado pelo usuário é válido.

    Returns:
        str: O caminho absoluto do diretório válido.

    Example:
        >>> verificar_diretorio()
        Digite o caminho do diretório: /caminho/do/diretorio
        Diretório raiz para armazenamento dos documentos: /caminho/do/diretorio
        '/caminho/do/diretorio'

    """
    root = input("Digite o caminho do diretório: ")
    try:
        if os.path.isdir(root):
            print("Diretório raiz para armazenamento dos documentos: " + root)
            return root
        else:
            print("O diretório não existe ou não é válido.")
    except FileNotFoundError:
        print(
            "O diretório inserido não existe ou não é válido. Certifique-se que o nome está correto",
        )


def busca_ipe(ano):
    """Verifica se o diretório informado pelo usuário é válido.

    Returns:
        str: O caminho absoluto do diretório válido.

    Example:
        >>> verificar_diretorio()
        Digite o caminho do diretório: /caminho/do/diretorio
        Diretório raiz para armazenamento dos documentos: /caminho/do/diretorio
        '/caminho/do/diretorio'

    """
    url = (
        "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_%d.zip"
        % ano
    )
    file = "ipe_cia_aberta_%d.zip" % ano
    r = requests.get(url)
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    file = zf.namelist()
    zf = zf.open(file[0])
    lines = zf.readlines()
    lines = [i.strip().decode("ISO-8859-1") for i in lines]
    lines = [i.split(";") for i in lines]
    len(lines)
    return lines


def search(lista, valor):
    """Retorna uma lista de elementos da lista que contêm o valor especificado.

    Args:
        lista (list): A lista de elementos para realizar a busca.
        valor: O valor a ser procurado nos elementos da lista.

    Returns:
        list: Uma lista dos elementos que contêm o valor especificado.

    Example:
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

    Example:
        >>> baixar_arquivo('https://example.com/file.pdf', '/path/to/save/file.pdf')
        Download finalizado. Arquivo salvo em: /path/to/save/file.pdf

    """
    resposta = requests.get(url, allow_redirects=True, verify=False, stream=True)
    if resposta.status_code == requests.codes.OK:
        with open(endereco, "wb") as novo_arquivo:
            novo_arquivo.write(resposta.content)
        print(f"Download finalizado. Arquivo salvo em: {endereco}")
    else:
        resposta.raise_for_status()


def transform_string(text):
    """Transforma uma string removendo acentos, substituindo espaços por underscores e convertendo para letras minúsculas.

    Args:
        text (str): A string a ser transformada.

    Returns:
        str: A string transformada.

    Example:
        >>> transform_string("Olá, Mundo!")
        'ola_mundo'

    """
    text = unidecode.unidecode(text)
    text = text.replace(" ", "_")
    text = text.replace(".", "")
    text = text.lower()
    return text


def download_def(empresa, year):
    """Realiza o download de arquivos específicos com base na empresa e no ano fornecidos.

    Args:
        empresa (str): O nome da empresa para a qual deseja-se fazer o download dos arquivos.
        year (int): O ano para o qual deseja-se fazer o download dos arquivos.

    Returns:
        list: Uma lista de dicionários contendo os metadados dos arquivos baixados.

    Raises:
        ValueError: Se houver duplicidade de empresas encontradas nos dados econômico-financeiros.

    Example:
        >>> download_def("Empresa XYZ", 2023)
        Empresa encontrada: Empresa XYZ | codigo cvm: 12345
        Ano: 2023
        nome do arquivo: 12345_empresa_xyz
        caminho dos arquivos: /caminho/do/arquivo/12345_empresa_xyz
        ...

    """
    lines = busca_ipe(year)
    defi = search(lines, "Dados Econômico-Financeiros")
    year = year
    data = []
    for a in range(len(defi)):
        count = 0
        if empresa.upper() in defi[a][1]:
            count += 1
            if count > 1:
                raise ValueError("Duplicidade encontrada.")
            else:
                empresa_name = defi[a][1]
                num_cvm = defi[a][2]
    print("Empresa encontrada: " + empresa_name + " | codigo cvm: " + num_cvm)
    print("Ano: " + str(year))
    for a in range(len(defi)):
        if defi[a][2] in num_cvm:
            # sufix = re.sub(u'[^a-zA-Z0-9]', '_', defi[a][2] + "_" + defi[a][1])
            sufix = transform_string(defi[a][2] + "_" + defi[a][1])
            company = os.path.join(root, sufix)
            # category = re.sub(u'[^a-zA-Z0-9çãàáêéíõóú]', '_', defi[a][5])
            category = transform_string(defi[a][5])
            print("nome do arquivo: " + sufix)
            print("caminho dos arquivos: " + company)
            if not Path(root, sufix).is_dir():
                print("diretorio não existe e será criado")
                os.makedirs(os.path.join(root, sufix))
            if not Path(company, str(year)).is_dir():
                print("diretorio não existe e será criado")
                os.makedirs(os.path.join(company, str(year)))
            diryear = os.path.join(company, str(year))
            if not Path(diryear, category).is_dir():
                print("diretorio não existe e será criado")
                os.makedirs(os.path.join(diryear, category))
            dircategory = os.path.join(diryear, category)
            url = defi[a][12]
            nome = transform_string(
                defi[a][2]
                + "_"
                + defi[a][1]
                + "_"
                + defi[a][7][1:50]
                + "_"
                + defi[a][8],
            )[1:100]
            baixar_arquivo(url, os.path.join(dircategory, "%s.pdf" % nome))
            title = transform_string(defi[a][6][0:50])
            if not title:
                if not defi[a][7]:
                    title = defi[a][7]
                else:
                    if not defi[a][5]:
                        title = defi[a][5]
                    else:
                        if not defi[a][4]:
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
            size = os.path.getsize(os.path.join(dircategory, "%s.pdf" % nome))
            file = os.path.join(dircategory, "%s.pdf" % nome)
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
                },
            )
    return data


def convert_pdf(doc):
    """Converte um arquivo PDF em texto.

    Args:
        doc (str): O caminho do arquivo PDF a ser convertido.

    Returns:
        str: O texto extraído do arquivo PDF.

    Example:
        text = convert_pdf("path/to/file.pdf")
        print(text)
        # Output: Text extracted from the PDF file.

    """
    with open(doc, "rb") as f:
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


def remover_stopwords(tokens):
    """Remove as stopwords de uma lista de tokens.

    Args:
        tokens (list): A lista de tokens.

    Returns:
        list: A lista de tokens sem as stopwords.

    Example:
        token_list = ['This', 'is', 'a', 'sample', 'sentence']
        tokens_without_stopwords = remover_stopwords(token_list)
        print(tokens_without_stopwords)
        # Output: ['This', 'sample', 'sentence']

    """
    # Carrega o modelo do SpaCy para o idioma português
    nlp = spacy.load("pt_core_news_lg")

    # Cria uma lista para armazenar os tokens sem stopwords
    tokens_sem_stopwords = []

    # Percorre cada token na lista
    for token in tokens:
        # Verifica se o token não é uma stopword
        if not nlp.vocab[token].is_stop:
            tokens_sem_stopwords.append(token)
    return tokens_sem_stopwords


def normalize_text(text):
    """Normaliza um texto, convertendo-o para minúsculas e removendo caracteres especiais e acentuação.

    Args:
        text (str): O texto a ser normalizado.

    Returns:
        str: O texto normalizado.

    Example:
        text = "Hello, World! This is an example text."
        normalized_text = normalize_text(text)
        print(normalized_text)
        # Output: hello world this is an example text

    """
    # Converte o texto para minúsculas
    normalized_text = text.lower()
    # Remove caracteres especiais e acentuação
    normalized_text = re.sub(r"[^\w\s]", "", normalized_text)
    # Retorna o texto limpo
    return normalized_text


def tokengen(text):
    """Gera uma lista de tokens a partir de um texto.

    Args:
        text (str): O texto a ser tokenizado.

    Returns:
        list: Uma lista de tokens.

    Example:
        text = "Hello, World! This is an example text."
        tokens = tokengen(text)
        print(tokens)
        # Output: ['[CLS]', 'hello', ',', 'world', '!', 'this', 'is', 'an', 'example', 'text', '.', '[SEP]']

    """
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_list = tokenizer.convert_ids_to_tokens(tokens)
    return token_list


def vector_one(text):
    """Gera um vetor de representação para um texto.

    Args:
        text (str): O texto a ser vetorizado.

    Returns:
        torch.Tensor: O vetor de representação do texto.

    Example:
        text = "This is a sample text."
        text_vector = vector_one(text)
        print(text_vector)
        # Output: torch.Tensor representing the vector for the text.


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
    text_vector = torch.cat(vectors, dim=1)
    return text_vector


def similarity_vector(a, b):
    """Calcula a similaridade entre dois vetores.

    Args:
        a (torch.Tensor): O primeiro vetor.
        b (torch.Tensor): O segundo vetor.

    Returns:
        float: O valor da similaridade entre os vetores.

    Example:
        vector1 = torch.tensor([0.1, 0.2, 0.3])
        vector2 = torch.tensor([0.4, 0.5, 0.6])
        similarity = similarity_vector(vector1, vector2)
        print(similarity)
        # Output: Similarity score between the vectors.

    """
    size_a = a.size(0)
    size_b = b.size(0)
    if size_a == size_b:
        # similarity_score = cosine_similarity(a, b)
        similarity_score = cosine_similarity(a.float(), b.float())
        return similarity_score.item()
    if size_a > size_b:
        diff = size_a - size_b
        padded = F.pad(b, pad=(0, diff), mode="constant", value=0)
        similarity_score = cosine_similarity(a.float(), padded.float())
        return similarity_score.item()
    if size_a < size_b:
        diff = size_b - size_a
        padded = F.pad(a, pad=(0, diff), mode="constant", value=0)
        similarity_score = cosine_similarity(b.float(), padded.float())
        return similarity_score.item()
    return None


def question(text, question):
    """Calcula a similaridade entre dois vetores.

    Args:
        a (torch.Tensor): O primeiro vetor.
        b (torch.Tensor): O segundo vetor.

    Returns:
        float: O valor da similaridade entre os vetores.

    Example:
        vector1 = torch.tensor([0.1, 0.2, 0.3])
        vector2 = torch.tensor([0.4, 0.5, 0.6])
        similarity = similarity_vector(vector1, vector2)
        print(similarity)
        # Output: Similarity score between the vectors.

    """
    question = question
    max_answer_length = 512
    max_length = 512
    input_parts = []
    input_parts = [text[i : i + max_length] for i in range(0, len(text), max_length)]
    # Lista para armazenar as respostas
    answers = []
    # Processa cada parte do texto
    for part in input_parts:
        # Tokeniza a parte do texto
        # Tokeniza a pergunta e o texto menor
        encoding = tokenizer.encode_plus(
            question,
            part,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        # Obtém as respostas do modelo pré-treinado
        with torch.no_grad():
            outputs = model_forquestion(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
        # Obtém a resposta com maior probabilidade
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                input_ids[0][answer_start.item() : answer_end.item() + 1],
            ),
        )
        # Adiciona a resposta à lista
        answers.append(answer)
    # Combina as respostas
    combined_answer = " ".join(answers)
    # Limita o tamanho da resposta final
    combined_answer = combined_answer[:max_answer_length]
    # Formatação da resposta final
    formatted_answer = "RESPOSTA:\n\n"
    formatted_answer += combined_answer.replace(".", ".\n\n")
    # Apresenta a resposta final
    return formatted_answer


def summarization(text):
    """Gera um resumo do texto fornecido.

    Args:
        text (str): O texto a ser resumido.

    Returns:
        str: O resumo gerado do texto.

    Example:
        text = "This is a sample text. It contains multiple sentences. The goal is to generate a summary."
        summary = summarization(text)
        print(summary)
        # Output: A summary of the text.

    """
    num_sentences = 5
    # Tokenizar o texto em partes
    input_parts = text.split(". ")
    summaries = []
    for part in input_parts:
        length = len(part)
        m_length = int(length * 0.4)
        # Tokenizar a parte do texto
        inputs = tokenizersum.encode("summarize: " + part, return_tensors="pt")
        # Gerar o resumo da parte do texto usando o modelo T5
        outputs = modelsum.generate(
            inputs,
            max_length=m_length,
            num_beams=4,
            early_stopping=True,
        )
        summary = tokenizersum.decode(outputs[0], skip_special_tokens=True)
        # Adicionar o resumo à lista de sumários
        summaries.append(summary)

    # Concatenar os sumários em um único texto
    concatenated_summary = " ".join(summaries)

    # Extrair as N sentenças mais importantes
    sentences = concatenated_summary.split(". ")
    top_sentences = sorted(sentences, key=lambda x: len(x), reverse=True)[
        :num_sentences
    ]

    return ". ".join(top_sentences)


def check_collection_exists(collection_name):
    """Verifica se uma coleção existe no servidor Solr.

    Args:
        collection_name (str): O nome da coleção a ser verificada.

    Returns:
        bool: True se a coleção existe, False caso contrário.

    Example:
        >>> check_collection_exists("my_collection")
        True

    """
    url = f"http://localhost:8983/solr/{collection_name}/admin/ping"

    response = requests.get(url)
    if response.status_code == 200:
        return True
    elif response.status_code == 404:
        return False
    else:
        print(
            "Falha ao verificar a existência da coleção. Status:",
            response.status_code,
        )
        return False


def create_solr_collection(collection_name):
    """Cria uma coleção no servidor Solr e atualiza o esquema.

    Args:
        collection_name (str): O nome da coleção a ser criada.

    Example:
        >>> create_solr_collection("my_collection")
        Esquema atualizado com sucesso.

    """
    # Caminho para o diretório bin do Solr
    solr_bin_path = (
        "C:\\Users\\thgcn\\OneDrive\\Academico\\PO-245\\Projeto\\solr-9.2.1\\bin\\"
    )

    # Mude para o diretório bin do Solr
    os.chdir(solr_bin_path)

    # Comando para criar a coleção no modo standalone
    create_collection_command = f"solr.cmd create -c {collection_name} -s 2 -rf 2"

    # Execute o comando no terminal
    subprocess.run(create_collection_command, shell=True)

    # URL do endpoint Solr
    url = f"http://localhost:8983/solr/{collection_name}/schema"

    # Cabeçalhos da solicitação POST
    headers = {
        "Content-Type": "application/json",
    }

    # Corpo da solicitação POST
    data = {
        "add-field": [
            {"name": "name", "type": "text_general", "multiValued": False},
            {"name": "company_name", "type": "text_general", "multiValued": False},
            {"name": "cod_cvm", "type": "text_general", "multiValued": False},
            {"name": "content", "type": "text_general", "multiValued": True},
            {"name": "year", "type": "pint"},
            {"name": "date", "type": "pdate"},
            {"name": "keywords", "type": "text_general", "multiValued": True},
            {"name": "size", "type": "pint"},
            {"name": "tokens", "type": "text_general", "multiValued": False},
            {"name": "tensor", "type": "text_general", "multiValued": False},
            {"name": "file", "type": "text_general", "multiValued": False},
        ],
    }

    # Enviar solicitação POST para atualizar o esquema no Solr
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        print("Esquema atualizado com sucesso.")
    else:
        print("Falha ao atualizar o esquema. Status:", response.status_code)


def update_schema(data, collection_name):
    """Atualiza o esquema da coleção Solr com os campos fornecidos.

    Args:
        data (dict): Dicionário contendo os campos e configurações a serem adicionados ao esquema.
        collection_name (str): O nome da coleção a ser atualizada.

    Example:
        data = {
            "add-field": [
                {"name": "title", "type": "text_general", "multiValued": False},
                {"name": "content", "type": "text_general", "multiValued": True},
                {"name": "author", "type": "text_general", "multiValued": False}
            ]
        }
        update_schema(data, "my_collection")

    """
    # URL do endpoint Solr
    url = f"http://localhost:8983/solr/{collection_name}/schema"

    # Cabeçalhos da solicitação POST
    headers = {
        "Content-Type": "application/json",
    }

    # Converter tensores em listas
    data = json.loads(json.dumps(data, default=lambda x: x.tolist()))

    # Enviar solicitação POST para atualizar o esquema no Solr
    response = requests.post(url, json=data, headers=headers)

    upd = f"http://localhost:8983/solr/{core_name}/config"

    if response.status_code == 200:
        print("1/2 Esquema atualizado com sucesso.")
        data_up = {
            "set-property": {
                "updateHandler.autoCommit.maxTime": 15000,
            },
        }
        response = requests.post(upd, headers=headers, json=data_up)
        if response.status_code == 200:
            print("2/2 Commit realizado com sucesso.")
        else:
            print("Falha no commit. Status:", response.status_code)
            print(response.content.decode())
    else:
        print("Falha ao atualizar o esquema. Status:", response.status_code)
        print(response.content.decode())


def add_document_to_solr(collection_name, document):
    """Adiciona um documento à coleção Solr especificada.

    Args:
        collection_name (str): O nome da coleção onde o documento será adicionado.
        document (dict): Dicionário contendo os campos e valores do documento.

    Example:
        document = {
            "title": "Example Document",
            "content": "This is an example document.",
            "author": "John Doe"
        }
        add_document_to_solr("my_collection", document)

    """
    # URL da API do Solr para adicionar documentos
    solr_url = f"http://localhost:8983/solr/{collection_name}/update?commit=true"

    # Envia a requisição POST para adicionar o documento
    response = requests.post(solr_url, json=[document])

    # Verifica o status da resposta
    if response.status_code == 200:
        print("Documento adicionado com sucesso!")
    else:
        print("Erro ao adicionar o documento:", response.text)
        print(response.content.decode())


def pipeline(empresa, ano):
    """Executa o pipeline de captura, preprocessamento e indexação de documentos econômico-financeiros.

    Args:
        empresa (str): O nome da empresa para o qual deseja-se executar o pipeline.
        ano (int): O ano para o qual deseja-se executar o pipeline.

    Returns:
        str: Uma mensagem indicando o resultado da atualização do esquema no Solr.

    Example:
        >>> pipeline("Empresa XYZ", 2023)
        Inicio do pipeline de captura *********
        Empresa encontrada: Empresa XYZ | codigo cvm: 123456
        Ano: 2023
        nome do arquivo: 123456_Empresa_XYZ
        caminho dos arquivos: /caminho/do/diretorio/123456_Empresa_XYZ
        diretorio não existe e será criado
        diretorio não existe e será criado
        diretorio não existe e será criado
        Download finalizado. Arquivo salvo em: /caminho/do/diretorio/123456_Empresa_XYZ/2023/Economic_Financial_Data/123456_Empresa_XYZ.pdf
        geração de token e tensor.....
        Documento adicionado com sucesso!
        ...
        1/2 Esquema atualizado com sucesso.
        2/2 Commit realizado com sucesso.
    """
    # Faz o download do documento
    verificar_diretorio()
    print("Inicio do pipeline de captura *********")
    doc = download_def(empresa, ano)
    print("Inicio do pipeline de preprocessamento e indexacao *********")
    for item in doc:
        # Define o tipo do item como "document"
        item["type"] = "document"
        # Define o nome da coleção como "dados_eco"
        collection_name = "dados_eco"
        # Obtém o diretório do arquivo
        diretorio = os.path.dirname(item["file"])
        # Obtém o nome do arquivo sem a extensão
        tokens_tensor, ext = os.path.splitext(os.path.basename(item["file"]))
        # Define o caminho do arquivo de tokens e tensor
        dataprep = os.path.join(diretorio, f"{tokens_tensor}_TOKENS_TENSOR.txt")
        # Cria o output data com os tokens e tensor
        print("geracao de token e tensor.....")
        text = convert_pdf(item["file"])
        token = tokengen(text)
        vector = vector_one(text).tolist()

        output_data = [
            {
                "tokens": token,
                "tensor": vector,
            },
        ]
        # Salva o output data em um arquivo JSON
        with open(dataprep, "w") as file:
            json.dump(output_data, file)
        # Atualiza os campos tokens e tensor do item com o caminho do arquivo
        item["tokens"] = dataprep
        item["tensor"] = dataprep
        # Verifica se algum campo está ausente e define como "Descrição ausente" se necessário
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
        collection_name = "dados_eco"
        add_document_to_solr(collection_name, item)
    # Atualiza o schema fora do loop
    return update_schema(doc, collection_name)
