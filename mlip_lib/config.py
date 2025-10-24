# -*- coding: utf-8 -*-
"""
Arquivo de configuração para o script orquestrador.
Substitui a necessidade de um arquivo .yaml.
"""

# Caminho para o executável de ativação do Conda
CONDA_ACTIVATE = "/opt/miniconda3/bin/activate"

# Ambiente base no qual os processos serão chamados
BASE_ENV = "menu"

#Benchmarks list
BENCHMARKS = ["surfaces", "eos"]

# Caminho para os modelos pré-treinados
MODELS_PATH = "/temp/"
# Para referência, o caminho original era:
# MODELS_PATH = "/home/p.zanineli/work/inference/pretrained_models"

# Dicionário de modelos a serem executados.
# A estrutura é:
# nome_do_ambiente_conda: {dicionário de modelos} ou None
FULL_MODELS = {
        "mace": "MACE-MP-0"
    
    # Modelos comentados para referência futura
    # "fair_chem": "fair_chem",
    # "sevenn": "7net-l3i5",
    # "orb-v3": "ORB-V2-MPTrj",
}
