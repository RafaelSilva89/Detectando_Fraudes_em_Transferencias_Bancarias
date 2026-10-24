"""Explicação das previsões do XGBoost pelos valores SHAP (calculados pelo próprio XGBoost, sem a biblioteca shap)."""
import pandas as pd
import xgboost as xgb


def contribuicoes_xgboost(modelo, X):
    """Contribuição de cada variável para o score de cada transação (valores SHAP, na escala log-odds).

    Positivo = empurra para "fraude"; negativo = para "legítima". A coluna 'base' é o ponto de partida
    (a média do modelo); base + soma das contribuições = score (log-odds) da transação.
    """
    estimador = modelo.steps[-1][1] if hasattr(modelo, 'steps') else modelo
    valores = estimador.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(valores, index=X.index, columns=list(X.columns) + ['base'])
