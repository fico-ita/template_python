""" Faz a captura de dados do portal de dados abertos da Comissão de Valores Mobiliários (CVM), pré-processa esses documentos, gerando tokens e tensores e por último indexa os metadados no Solr Apache.

Módulos exportados por este pacote:

- `cpi`: Este módulo contém funções e bibliotecas relacionadas ao pipeline de captura,
pré-processamento e indexação de relatórios.
- `cpi_helpers`: Este módulo contém funções e bibliotecas auxiliares relacionadas ao pipeline de
pré-processamento e indexação de relatórios. Contém as seguintes funções: (remoção de stopwords, normalização de textos, cálculo de similaridade entre dois vetores, resposta de perguntas referente ao texto, resumo de texto, parada do serviço solr, criação de coleção no solr, Atualiza estrutura da coleção (schema).
"""
