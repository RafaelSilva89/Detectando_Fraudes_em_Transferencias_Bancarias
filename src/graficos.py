"""Funções de gráficos usadas no notebook. Cada gráfico é salvo na pasta Imagens/ do projeto."""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

PASTA_IMAGENS = Path(__file__).resolve().parents[1] / 'Imagens'

COR_LEGITIMA = '#4C78A8'
COR_FRAUDE = '#D1495B'


def salvar(nome_arquivo):
    plt.savefig(PASTA_IMAGENS / f'{nome_arquivo}.png', bbox_inches='tight', dpi=110)


def titulo(texto, subtitulo=None):
    plt.suptitle(texto, fontsize=16, x=0.01, ha='left', y=1.02)
    if subtitulo:
        plt.figtext(0.01, 0.95, subtitulo, ha='left', fontsize=11, color='#656565', style='italic')


def plot_distribuicao_alvo(series, titulos, titulo_grafico, nome_arquivo, rotulos=('Legítima', 'Fraude')):
    """Um painel de barras (quantidade e %) por série do alvo, lado a lado (ex.: antes e depois do SMOTE)."""
    fig, axs = plt.subplots(1, len(series), figsize=(6.5 * len(series), 4.5), squeeze=False)
    for ax, serie, subtitulo in zip(axs[0], series, titulos):
        contagem = serie.value_counts().sort_index()
        percentual = serie.value_counts(normalize=True).sort_index() * 100
        barras = ax.bar(list(rotulos), contagem.values, color=[COR_LEGITIMA, COR_FRAUDE])
        for barra, qtd, pct in zip(barras, contagem.values, percentual.values):
            ax.text(barra.get_x() + barra.get_width() / 2, qtd, f'{qtd:,} ({pct:.2f}%)'.replace(',', '.'),
                    ha='center', va='bottom', fontsize=11)
        ax.set_title(subtitulo, loc='left')
        ax.set_ylabel('Transações')
        sns.despine(ax=ax, left=True)
    titulo(titulo_grafico)
    plt.tight_layout()
    salvar(nome_arquivo)
    plt.show()


def plot_barras(serie, titulo_grafico, nome_arquivo, xlabel='', ylabel='', cor=COR_LEGITIMA, horizontal=False,
                formato='{:.0f}', subtitulo=None):
    """Barras com o valor escrito em cada barra."""
    plt.figure(figsize=(10, 5) if not horizontal else (10, max(4, len(serie) * 0.35)))
    if horizontal:
        ax = sns.barplot(x=serie.values, y=serie.index.astype(str), color=cor)
    else:
        ax = sns.barplot(x=serie.index.astype(str), y=serie.values, color=cor)
    ax.bar_label(ax.containers[0], labels=[formato.format(v) for v in serie.values], padding=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not horizontal:
        plt.xticks(rotation=45 if len(serie) > 6 else 0, ha='right' if len(serie) > 6 else 'center')
    sns.despine(left=True)
    titulo(titulo_grafico, subtitulo)
    salvar(nome_arquivo)
    plt.show()
