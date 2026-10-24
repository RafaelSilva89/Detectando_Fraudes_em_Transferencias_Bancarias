"""Funções de avaliação de classificadores: métricas completas, relatório de validação, limiar, custo e ganho.

Classe de interesse = 1 (fraude)."""
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from src.graficos import salvar

ROTULOS = ['Legítima (0)', 'Fraude (1)']


def calcular_metricas_completas(y_true, y_pred, y_proba):
    """Todas as métricas de um classificador binário (classe de interesse = 1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão': precision_score(y_true, y_pred, zero_division=0),
        'Recall (Sensibilidade)': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_true, y_proba),
        'PR-AUC': average_precision_score(y_true, y_proba),
        'Especificidade': tn / (tn + fp) if (tn + fp) else 0,
        'Valor Preditivo Negativo': tn / (tn + fn) if (tn + fn) else 0,
        'Verdadeiros Positivos': tp, 'Verdadeiros Negativos': tn,
        'Falsos Positivos': fp, 'Falsos Negativos': fn,
    }


def relatorio_validacao(nome, modelo, X_treino, y_treino, X_val, y_val, limiar=0.5, nome_arquivo=None):
    """Avalia um modelo já treinado na validação, no formato padrão do projeto.

    Mostra: amostra gabarito x probabilidade x previsão, métricas, AUC no treino x validação (sobreajuste),
    relatório de classificação e matriz de confusão. Devolve (métricas, probabilidades da validação).
    X_treino/y_treino devem ser os dados de treino ORIGINAIS (sem cópias do balanceamento).
    """
    proba = modelo.predict_proba(X_val)[:, 1]
    previsto = (proba >= limiar).astype(int)

    print('AMOSTRA: GABARITO x PROBABILIDADE x PREVISÃO (limiar = {:.2f})'.format(limiar))
    display(pd.DataFrame({'Gabarito': np.asarray(y_val), 'Probabilidade': proba.round(3),
                          'Previsão da máquina': previsto}).head(8))

    metricas = calcular_metricas_completas(y_val, previsto, proba)
    print('=' * 50)
    print(f'MÉTRICAS DE VALIDAÇÃO: {nome}')
    print('=' * 50)
    for chave in ['Acurácia', 'Precisão', 'Recall (Sensibilidade)', 'F1-Score', 'AUC-ROC', 'PR-AUC']:
        print(f'{chave:<24} {metricas[chave]:.4f}')
    print(f'{"":<24} (sorteio: PR-AUC = {np.mean(y_val):.4f}, AUC-ROC = 0,50)')

    proba_treino = modelo.predict_proba(X_treino)[:, 1]
    auc_treino, pr_auc_treino = roc_auc_score(y_treino, proba_treino), average_precision_score(y_treino, proba_treino)
    print('\nSOBREAJUSTE (treino x validação):')
    print(f'  PR-AUC : treino {pr_auc_treino:.4f} | validação {metricas["PR-AUC"]:.4f}'
          f' | diferença {pr_auc_treino - metricas["PR-AUC"]:+.4f}')
    print(f'  AUC-ROC: treino {auc_treino:.4f} | validação {metricas["AUC-ROC"]:.4f}'
          f' | diferença {auc_treino - metricas["AUC-ROC"]:+.4f}')
    metricas['AUC-ROC treino'], metricas['PR-AUC treino'] = auc_treino, pr_auc_treino

    print('\n' + classification_report(y_val, previsto, target_names=ROTULOS, digits=4, zero_division=0))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_val, previsto, display_labels=['Legítima', 'Fraude'],
                                            cmap='Blues', colorbar=False, ax=ax)
    ax.set(xlabel='Previsto pelo modelo', ylabel='Real')
    ax.set_title(f'Matriz de confusão: {nome}', fontsize=12, loc='left')
    if nome_arquivo:
        salvar(nome_arquivo)
    plt.show()
    return metricas, proba


def plot_comparacao_modelos(metricas, taxa_positivos, nome_arquivo):
    """Painel com uma barra por modelo em cada métrica e a linha de referência de um "modelo que chuta"."""
    modelos = list(metricas)
    cores = sns.color_palette('Set2', len(modelos))
    paineis = [('Acurácia', 1 - taxa_positivos, "sempre 'legítima'"), ('Precisão', taxa_positivos, 'sorteio'),
               ('Recall (Sensibilidade)', None, None), ('F1-Score', None, None), ('AUC-ROC', 0.5, 'sorteio'),
               ('PR-AUC', taxa_positivos, 'sorteio')]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (metrica, referencia, rotulo) in zip(axes.ravel(), paineis):
        valores = [metricas[m][metrica] for m in modelos]
        ax.bar(modelos, valores, color=cores, edgecolor='black', alpha=0.85)
        for i, v in enumerate(valores):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)
        if referencia is not None:
            ax.axhline(referencia, color='gray', ls='--', lw=1.2, label=f'{rotulo} = {referencia:.2f}')
            ax.legend(loc='upper right', fontsize=9)
        ax.set_title(metrica, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', labelsize=9, rotation=20)
    fig.suptitle('Comparação dos modelos na validação', fontsize=16, fontweight='bold', x=0.01, ha='left')
    plt.tight_layout()
    salvar(nome_arquivo)
    plt.show()


def limiar_otimo_f1(y, proba):
    """Limiar que maximiza o F1 (o 0,5 padrão raramente é o melhor corte operacional)."""
    precisao, revocacao, limiares = precision_recall_curve(y, proba)
    f1 = np.divide(2 * precisao * revocacao, precisao + revocacao, out=np.zeros_like(precisao),
                   where=(precisao + revocacao) > 0)
    i = int(np.argmax(f1[:-1]))
    return float(limiares[i]), float(f1[i])


def custo_financeiro(y, proba, valores, custo_revisao, limiares=np.linspace(0.01, 0.99, 99)):
    """Custo em reais de cada limiar: fraudes não detectadas custam o próprio valor da transação e cada
    transação marcada (fraude ou não) custa uma revisão manual de `custo_revisao` reais."""
    y, valores = np.asarray(y), np.asarray(valores)
    linhas = []
    for limiar in limiares:
        marcado = proba >= limiar
        perdidas = (~marcado) & (y == 1)
        linhas.append({'Limiar': limiar, 'Fraudes detectadas': int((marcado & (y == 1)).sum()),
                       'Fraudes perdidas': int(perdidas.sum()), 'Alarmes falsos': int((marcado & (y == 0)).sum()),
                       'Valor das fraudes perdidas': valores[perdidas].sum(),
                       'Custo das revisões': marcado.sum() * custo_revisao})
    tabela = pd.DataFrame(linhas)
    tabela['Custo total'] = tabela['Valor das fraudes perdidas'] + tabela['Custo das revisões']
    return tabela


def resumo_financeiro(y, previsto, valores, custo_revisao):
    """Resultado em reais de uma regra de decisão, comparado a não ter modelo (nenhuma fraude barrada)."""
    y, previsto, valores = np.asarray(y), np.asarray(previsto).astype(bool), np.asarray(valores)
    valor_fraudes = valores[y == 1].sum()
    evitado = valores[previsto & (y == 1)].sum()
    revisoes = previsto.sum() * custo_revisao
    return {'Valor total das fraudes': valor_fraudes, 'Valor das fraudes barradas': evitado,
            '% do valor barrado': evitado / valor_fraudes if valor_fraudes else 0,
            'Custo das revisões': revisoes, 'Economia líquida': evitado - revisoes,
            'Transações revisadas': int(previsto.sum())}


def aplicar_regra(proba, valores, limiar, regra_valor=None):
    """Transações enviadas para revisão: probabilidade >= limiar OU, se houver regra complementar,
    valor >= regra_valor['corte_valor'] com probabilidade >= regra_valor['limiar_valor']."""
    marcado = np.asarray(proba) >= limiar
    if regra_valor:
        marcado |= (np.asarray(valores) >= regra_valor['corte_valor']) & (np.asarray(proba) >= regra_valor['limiar_valor'])
    return marcado


def bootstrap_decisoes(y, valores, decisoes, custo_revisao, n=1000, semente=42):
    """Bootstrap estratificado (sorteia fraudes e legítimas separadamente) do resultado de cada regra.

    `decisoes` = {nome: vetor booleano de transações revisadas}. Todas as regras usam as MESMAS amostras,
    então a diferença entre elas pode ser medida amostra a amostra. Devolve um DataFrame longo.
    """
    y, valores = np.asarray(y), np.asarray(valores)
    rng = np.random.default_rng(semente)
    fraudes, legitimas = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    linhas = []
    for amostra in range(n):
        idx = np.concatenate([rng.choice(fraudes, len(fraudes)), rng.choice(legitimas, len(legitimas))])
        for nome, marcado in decisoes.items():
            r = resumo_financeiro(y[idx], np.asarray(marcado)[idx], valores[idx], custo_revisao)
            achadas = (np.asarray(marcado)[idx] & (y[idx] == 1)).sum()
            linhas.append({'amostra': amostra, 'regra': nome, 'Recall': achadas / len(fraudes),
                           'Precisão': achadas / max(np.asarray(marcado)[idx].sum(), 1),
                           '% do valor barrado': r['% do valor barrado'], 'Economia líquida': r['Economia líquida']})
    return pd.DataFrame(linhas)


def psi(esperado, atual, faixas=10):
    """Population Stability Index: o quanto a distribuição `atual` se afastou da `esperado`.

    Faixas pelos quantis da distribuição esperada. Leitura usual: < 0,10 estável; 0,10 a 0,25 atenção;
    > 0,25 mudança relevante (hora de investigar ou retreinar).
    """
    esperado, atual = np.asarray(esperado), np.asarray(atual)
    cortes = np.unique(np.quantile(esperado, np.linspace(0, 1, faixas + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p_esp = np.histogram(esperado, cortes)[0] / len(esperado)
    p_atu = np.histogram(atual, cortes)[0] / len(atual)
    p_esp, p_atu = np.clip(p_esp, 1e-6, None), np.clip(p_atu, 1e-6, None)
    return float(np.sum((p_atu - p_esp) * np.log(p_atu / p_esp)))


def ganho_vs_sorteio(y, proba):
    """Quantos positivos são encontrados lendo os N casos de maior score, contra a média de um sorteio."""
    y = np.asarray(y)[np.argsort(-proba)]
    lidos = np.arange(1, len(y) + 1)
    return pd.DataFrame({'Lidos': lidos, 'Modelo': np.cumsum(y), 'Sorteio': lidos * y.mean()})


def salvar_ficha_modelo(caminho, **campos):
    """Grava a ficha do modelo (JSON): tudo o que é preciso saber para usá-lo corretamente."""
    ficha = {'data_treino': datetime.now().strftime('%Y-%m-%d %H:%M'), **campos}
    with open(caminho, 'w', encoding='utf-8') as arquivo:
        json.dump(ficha, arquivo, ensure_ascii=False, indent=2, default=float)
    return ficha
