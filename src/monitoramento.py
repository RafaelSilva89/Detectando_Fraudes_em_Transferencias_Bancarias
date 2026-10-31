"""Monitoramento do modelo em produção: data drift (PSI e teste K-S) e desempenho por lote contra um SLA."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score

from src.avaliacao import psi

# Faixas usuais do PSI (indústria financeira)
PSI_ATENCAO = 0.10
PSI_MUDANCA = 0.25


def leitura_psi(valor):
    """Estável (até 0,10), Atenção (até 0,25) ou Mudança (acima: investigar ou retreinar)."""
    if valor <= PSI_ATENCAO:
        return 'Estável'
    return 'Atenção' if valor <= PSI_MUDANCA else 'Mudança'


def relatorio_drift(referencia, atual, colunas):
    """PSI e teste K-S de cada coluna, comparando os dados `atual` com a `referencia` (DataFrames ou dicionários).

    Não precisa do gabarito. Com lotes grandes, o p-valor do K-S fica perto de 0 até para diferenças sem
    importância prática: a leitura usa o PSI e a distância do K-S (de 0 a 1).
    """
    linhas = {}
    for coluna in colunas:
        ks = ks_2samp(referencia[coluna], atual[coluna])
        valor_psi = psi(referencia[coluna], atual[coluna])
        linhas[coluna] = {'PSI': valor_psi, 'K-S (distância)': ks.statistic, 'K-S (p-valor)': ks.pvalue,
                          'Leitura': leitura_psi(valor_psi)}
    return pd.DataFrame.from_dict(linhas, orient='index')


def desempenho_lote(real, probabilidade, revisar):
    """Métricas de um lote depois que o gabarito chega: PR-AUC do score e recall/precisão da fila de revisão.

    Sem fraudes no lote, a PR-AUC e o recall não existem (NaN).
    """
    real, revisar = np.asarray(real), np.asarray(revisar, dtype=bool)
    fraudes, revisoes = int(real.sum()), int(revisar.sum())
    achadas = int((revisar & (real == 1)).sum())
    return {'Transações': len(real), 'Fraudes': fraudes,
            'PR-AUC': average_precision_score(real, probabilidade) if fraudes else np.nan,
            'Recall': achadas / fraudes if fraudes else np.nan,
            'Precisão da fila': achadas / revisoes if revisoes else 0.0, 'Revisões': revisoes}


def situacao_sla(pr_auc, recall, sla_pr_auc, sla_recall):
    if np.isnan(pr_auc) or np.isnan(recall):
        return '⚪ sem fraudes no lote'
    return '🚨 abaixo do SLA' if pr_auc < sla_pr_auc or recall < sla_recall else '✅ dentro do SLA'


def painel_lotes(real, probabilidade, revisar, lotes, sla_pr_auc, sla_recall):
    """Desempenho de cada lote (ex.: janelas de 4 horas) e a situação em relação ao SLA, na ordem dos lotes."""
    real, probabilidade, revisar = np.asarray(real), np.asarray(probabilidade), np.asarray(revisar, dtype=bool)
    lotes = pd.Categorical(lotes)
    linhas = {}
    for lote in lotes.categories:
        no_lote = np.asarray(lotes == lote)
        if no_lote.any():
            linhas[lote] = desempenho_lote(real[no_lote], probabilidade[no_lote], revisar[no_lote])
    painel = pd.DataFrame.from_dict(linhas, orient='index')
    painel['Situação'] = [situacao_sla(p, r, sla_pr_auc, sla_recall) for p, r in zip(painel['PR-AUC'], painel['Recall'])]
    return painel
