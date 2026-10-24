"""Seleção de variáveis por dois métodos: importância do Random Forest e Informação Mútua.

Recebem a matriz de treino e os nomes das colunas à parte.
"""
from functools import partial

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def importancia_random_forest(X, y, nomes, semente=42, n_jobs=2):
    """Importância de cada coluna para um Random Forest (quanto ela ajudou a separar as classes)."""
    modelo = RandomForestClassifier(random_state=semente, n_jobs=n_jobs).fit(X, y)
    return pd.Series(modelo.feature_importances_, index=nomes).sort_values(ascending=False)


def selecao_features(X, y, nomes, k, semente=42):
    """Seleção das k melhores colunas por Informação Mútua (SelectKBest + mutual_info_classif).

    A informação mútua mede quanto saber o valor de uma coluna reduz a incerteza sobre o alvo (0 = nada),
    e capta relações não lineares. Devolve o ranking de todas as colunas (feature, score) e o seletor ajustado.
    """
    seletor = SelectKBest(score_func=partial(mutual_info_classif, random_state=semente), k=k).fit(X, y)
    df_scores = (pd.DataFrame({'feature': nomes, 'score': seletor.scores_})
                 .sort_values('score', ascending=False).reset_index(drop=True))
    return df_scores, seletor
