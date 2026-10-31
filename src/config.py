"""Constantes do projeto: caminhos, semente e premissas de negócio."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO_BASE = RAIZ / 'Datasets' / 'transferencias.zip'
PASTA_MODELO = RAIZ / 'Model'
PASTA_REPORTS = RAIZ / 'reports'

SEMENTE = 42
ALVO = 'Target'

# Premissa de negócio: cada transação marcada pelo modelo é revisada por um analista.
# A revisão custa CUSTO_REVISAO reais; uma fraude não detectada custa o valor da própria transação.
CUSTO_REVISAO = 5.0
META_RECALL = 0.80   # alternativa de limiar: encontrar pelo menos 80% das fraudes

# Monitoramento em produção: variáveis acompanhadas pelo PSI e pelo teste K-S (as 4 mais importantes, o valor e a hora)
VARIAVEIS_MONITORADAS = ['crt', 'cep', 'reclamacoes', 'lim_crt', 'hist_saldo', 'hora_do_dia']

N_JOBS = 2   # núcleos em paralelo: limitado para não esgotar a memória da máquina
