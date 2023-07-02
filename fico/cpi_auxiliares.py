"""Este módulo contém funções e bibliotecas auxiliares relacionadas ao pipeline de
pré-processamento e indexação de relatórios.contém funções:
- remoção de stopwords
- normalização de textos
- calculo de similaridade entre dois vetores
- resposta de perguntas referente ao texto
- resumo de texto
- parada do serviço solr
- criação de coleção no solr
- Atualiza estrutura da coleção (schema).
"""  # noqa: D205

import json
import os
import re
import subprocess
from pathlib import Path

import requests
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
    """Normaliza o texto, convertendo-o para minúsculas e removendo caracteres especiais
    e acentuação.

    Args:
        text (str): O texto a ser normalizado.

    Returns:
        str: O texto normalizado.

    Example:
        text = "Hello, World! This is an example text."
        normalized_text = normalize_text(text)
        print(normalized_text)
        # Output: hello world this is an example text

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
    """Calcula a resposta para uma pergunta com base em um texto usando
    um modelo de similaridade.

    Args:
        text (str): O texto em que a pergunta será feita.
        question (str): A pergunta a ser respondida.

    Returns:
        str: A resposta para a pergunta.

    Example:
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        question = "Qual é o significado de Lorem ipsum?"
        answer = question(text, question)
        print(answer)
        # Output: Resposta para a pergunta..

    """  # noqa: D205
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

    Example:
        text = "This is a sample text. It contains multiple sentences.
        The goal is to generate a summary."
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
    """
    # Comando para desligar o serviço do Solr
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
