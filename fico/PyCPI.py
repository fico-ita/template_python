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


import spacy
import torch
import torch.nn.functional as fun
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
# Caminho para o diretório bin do Solr
slrb = Path("C:\\Users\\thgcn\\OneDrive\\Academico\\PO-245\\Projeto\\solr-9.2.1\\bin\\")

model_for_question = BertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad",
)








def check_directory():
    """Verifica se o diretório informado pelo usuário é válido.

    Examples:
        >>> check_directory()
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


def search_ipe(ano):
    """Obtém os dados do IPE (Informações Periódicas) de uma determinada empresa
    para o ano especificado.

    Args:
        ano (int): O ano para o qual se deseja obter os dados do IPE.

    Returns:
        list: Uma lista contendo as linhas dos dados do IPE da empresa.

    Examples:
        >>> search_ipe(2022)
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


def download_file(url, endereco):
    """Faz o download de um arquivo a partir de uma URL e salva no caminho especificado.

    Args:
        url (str): A URL do arquivo a ser baixado.
        endereco (str): O caminho completo de destino para salvar o arquivo.

    Raises:
        Exception: Se ocorrer um erro ao fazer o download do arquivo.

    Examples:
        >>> download_file('https://example.com/file.pdf', '/path/to/save/file.pdf')
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
    lines = search_ipe(year)
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

            download_file(url, dircategory / f"{nome}.pdf")

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


def pipeline_cpi(empresa, ano):
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
    root = check_directory()
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


def remove_stopwords(tokens):
    """Remove as stopwords de uma lista de tokens.

    Args:
        tokens (list): A lista de tokens.

    Returns:
        list: A lista de tokens sem as stopwords.

    Examples:
        >>> token_list = ['This', 'is', 'a', 'sample', 'sentence']
        >>> tokens_without_stopwords = remove_stopwords(token_list)
        >>> print(tokens_without_stopwords)
        ['This', 'sample', 'sentence']

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
    """Normaliza o texto, convertendo-o para minúsculas e removendo caracteres especiais
    e acentuação.

    Args:
        text (str): O texto a ser normalizado.

    Returns:
        str: O texto normalizado.

    Examples:
        >>> text = "Hello, World! This is an example text."
        >>> normalized_text = normalize_text(text)
        >>> print(normalized_text)
        hello world this is an example text

    """  # noqa: D205
    # Converte o texto para minúsculas
    normalized_text = text.lower()
    # Remove caracteres especiais e acentuação
    normalized_text = re.sub(r"[^\w\s]", "", normalized_text)
    # Retorna o texto limpo
    return normalized_text


def similarity_vector(a, b):
    """Calcula a similaridade entre dois vetores.

    Args:
        a (torch.Tensor): O primeiro vetor.
        b (torch.Tensor): O segundo vetor.

    Returns:
        float: O valor da similaridade entre os vetores.

    Examples:
        >>> vector1 = torch.tensor([0.1, 0.2, 0.3])
        >>> vector2 = torch.tensor([0.4, 0.5, 0.6])
        >>> similarity = similarity_vector(vector1, vector2)
        >>> print(similarity)
        0.9746318454742432

    """
    size_a = a.size(0)
    size_b = b.size(0)
    if size_a == size_b:
        # similarity_score = cosine_similarity(a, b)
        similarity_score = cosine_similarity(a.float(), b.float())
        return similarity_score.item()
    if size_a > size_b:
        diff = size_a - size_b
        padded = fun.pad(b, pad=(0, diff), mode="constant", value=0)
        similarity_score = cosine_similarity(a.float(), padded.float())
        return similarity_score.item()
    if size_a < size_b:
        diff = size_b - size_a
        padded = fun.pad(a, pad=(0, diff), mode="constant", value=0)
        similarity_score = cosine_similarity(b.float(), padded.float())
        return similarity_score.item()
    return None


def question(text, question):
    """Calcula a resposta para uma pergunta com base em um texto usando um modelo de
    similaridade.


    Args:
    text (str): O texto em que a pergunta será feita.
    question (str): A pergunta a ser respondida.

    Returns:
    str: A resposta para a pergunta.

    Examples:
        >>> text = "A inteligência artificial (IA) está se tornando cada vez mais
        presente em nossas vidas. Ela é aplicada em diversos setores, como saúde,
        finanças e transporte. A IA utiliza algoritmos e modelos de aprendizado de
        máquina para analisar dados e tomar decisões automatizadas."
        >>> question = "Quais são os setores em que a inteligência artificial é
        aplicada?"
        >>> answer = question(text, question)
        >>> print(answer)
        A IA é aplicada em diversos setores, como saúde, finanças e transporte.
"""  # noqa: D207, D205
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
            question, part, return_tensors="pt", max_length=512, truncation=True
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        # Obtém as respostas do modelo pré-treinado
        with torch.no_grad():
            outputs = model_for_question(
                input_ids=input_ids, attention_mask=attention_mask
            )
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
        # Obtém a resposta com maior probabilidade
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                input_ids[0][answer_start.item() : answer_end.item() + 1]
            )
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

    Examples:
        >>> text = "A inteligência artificial é um campo da ciência da computação que se
        dedica a criar máquinas capazes de simular a inteligência humana. Essas máquinas
        podem realizar tarefas complexas, como reconhecimento de padrões, processamento 
        de linguagem natural e tomada de decisões. A inteligência artificial tem sido 
        aplicada em diversos setores, como saúde, transporte, finanças e entretenimento,
        trazendo benefícios e transformando a maneira como vivemos e trabalhamos."
        >>> summary = summarization(text)
        >>> print(summary)
        A inteligência artificial é um campo da ciência da computação que se dedica
        a criar máquinas capazes de simular a inteligência humana. Essas máquinas
        podem realizar tarefas complexas, como reconhecimento de padrões, processamento
        de linguagem natural e tomada de decisões. A inteligência artificial tem sido
        aplicada em diversos setores, como saúde, transporte, finanças e entretenimento,
        trazendo benefícios e transformando a maneira como vivemos e trabalhamos.
    """
    num_sentences = 5
    # Tokenizar o texto em partes
    input_parts = text.split(". ")
    summaries = []
    for part in input_parts:
        length = len(part)
        m_length = int(length * 0.4)
        # Tokenizar a parte do texto
        inputs = tokenizer_sum.encode("summarize: " + part, return_tensors="pt")
        # Gerar o resumo da parte do texto usando o modelo T5
        outputs = model_sum.generate(
            inputs, max_length=m_length, num_beams=4, early_stopping=True
        )
        summary = tokenizer_sum.decode(outputs[0], skip_special_tokens=True)
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


def stop_solr_service():
    """Desliga o serviço do Solr.

    Returns:
        int: O código de retorno da execução do comando.
        str: A mensagem de erro, caso ocorra.

    Examples:
        >>> stop_solr_service()
       (0, 'Solr desligado com sucesso.')
        """
    command = f"{slrb}\\solr stop -port 8983"

    try:
        # Executa o comando no terminal
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            return result.returncode, "Solr desligado com sucesso."
        error_message = result.stderr.strip()
        return result.returncode, error_message

    except Exception as e:
        return -1, str(e)

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
    status_ok = 200
    status_not_found = 404
    if response.status_code == status_ok:
        return True
    elif response.status_code == status_not_found:  # noqa: RET505
        return False
    else:
        print(
            "Falha ao verificar a existência da coleção. Status:", response.status_code,
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
    # Mude para o diretório bin do Solr
    os.chdir(slrb)

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
    status_ok = 200

    if response.status_code == status_ok:
        print("Esquema atualizado com sucesso.")
    else:
        print("Falha ao atualizar o esquema. Status:", response.status_code)


def update_schema(data, collection_name):
    """Atualiza o esquema da coleção Solr com os campos fornecidos.

    Args:
        data (dict): Dicionário contendo os campos e configurações
        a serem adicionados ao esquema.
        collection_name (str): O nome da coleção a ser atualizada.

    Examples:
        data = {
            "add-field": [
                {"name": "title", "type": "text_general", "multiValued": False},
                {"name": "content", "type": "text_general", "multiValued": True},
                {"name": "author", "type": "text_general", "multiValued": False}
            ]
        }
        >>> update_schema(data, "my_collection")
        1/2 Esquema atualizado com sucesso.
        2/2 Commit realizado com sucesso.

    """
    # URL do endpoint Solr
    url = f"http://localhost:8983/solr/{collection_name}/schema"

    # Cabeçalhos da solicitação POST
    headers = {"Content-Type": "application/json"}

    # Converter tensores em listas
    data = json.loads(json.dumps(data, default=lambda x: x.tolist()))

    # Enviar solicitação POST para atualizar o esquema no Solr
    response = requests.post(url, json=data, headers=headers)

    upd = f"http://localhost:8983/solr/{collection_name}/config"
    status_ok = 200
    if response.status_code == status_ok:
        print("1/2 Esquema atualizado com sucesso.")
        data_up = {
            "set-property": {
                "updateHandler.autoCommit.maxTime": 15000,
            },
        }
        response = requests.post(upd, headers=headers, json=data_up)
        if response.status_code == status_ok:
            print("2/2 Commit realizado com sucesso.")
        else:
            print("Falha no commit. Status:", response.status_code)
            print(response.content.decode())
    else:
        print("Falha ao atualizar o esquema. Status:", response.status_code)
        print(response.content.decode())
