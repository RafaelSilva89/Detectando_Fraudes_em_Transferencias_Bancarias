"""Comparação de métodos de ajuste de hiperparâmetros: Grid Search, Random Search e Bayesiana (Optuna, com poda)."""
import numpy as np
import optuna
from sklearn.base import clone


def _linhas(dados, indices):
    return dados.iloc[indices] if hasattr(dados, 'iloc') else dados[indices]


def objetivo_cv_com_poda(modelo, espaco, X, y, cv, metrica):
    """Cria a função objetivo do Optuna com poda (*pruning*): a validação cruzada roda dobra a dobra.

    `espaco` recebe o trial e devolve o dicionário de hiperparâmetros; `metrica(y_real, probabilidade)`
    é a métrica a maximizar (ex.: average_precision_score). Depois de cada dobra, a média até ali é
    informada ao Optuna (`trial.report`); se ela estiver abaixo da mediana das tentativas anteriores na
    mesma dobra, o pruner interrompe a tentativa e as dobras restantes não são treinadas. Na última dobra não
    há poda: todo o custo já foi pago, e a tentativa termina como completa.
    O desvio entre as dobras e o score no treino ficam guardados no trial, para a comparação com o sklearn.
    """
    dobras = list(cv.split(X, y))

    def objetivo(trial):
        estimador = clone(modelo).set_params(**espaco(trial))
        teste, treino = [], []
        for passo, (i_treino, i_teste) in enumerate(dobras):
            ajustado = clone(estimador).fit(_linhas(X, i_treino), _linhas(y, i_treino))
            teste.append(metrica(_linhas(y, i_teste), ajustado.predict_proba(_linhas(X, i_teste))[:, 1]))
            treino.append(metrica(_linhas(y, i_treino), ajustado.predict_proba(_linhas(X, i_treino))[:, 1]))
            trial.report(float(np.mean(teste)), passo)
            if passo < len(dobras) - 1 and trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr('desvio', float(np.std(teste)))
        trial.set_user_attr('treino', float(np.mean(treino)))
        return float(np.mean(teste))
    return objetivo


def modelos_treinados_optuna(study, n_splits):
    """Quantos modelos o estudo treinou: as tentativas podadas param no meio da validação cruzada."""
    podada = optuna.trial.TrialState.PRUNED
    return int(sum(len(t.intermediate_values) if t.state == podada else n_splits for t in study.trials))


def resumo_sklearn(nome, busca, tempo, sinal=1):
    """Linha da tabela de comparação a partir de um GridSearchCV/RandomizedSearchCV (com return_train_score=True).

    `sinal=-1` desfaz o sinal negativo das métricas de erro do sklearn (ex.: neg_mean_absolute_error -> MAE).
    """
    i = busca.best_index_
    r = busca.cv_results_
    return {'Método': nome, 'Melhor score (CV)': sinal * r['mean_test_score'][i], 'Desvio entre dobras': r['std_test_score'][i],
            'Score no treino': sinal * r['mean_train_score'][i], 'Combinações testadas': len(r['params']),
            'Modelos treinados': len(r['params']) * busca.n_splits_, 'Tempo (s)': tempo,
            'Melhores hiperparâmetros': {k.replace('modelo__', ''): v for k, v in busca.best_params_.items()}}


def resumo_optuna(nome, study, tempo, n_splits, sinal=1):
    """Linha da tabela de comparação a partir de um estudo do Optuna criado com `objetivo_cv_com_poda`."""
    melhor = study.best_trial
    return {'Método': nome, 'Melhor score (CV)': sinal * melhor.value, 'Desvio entre dobras': melhor.user_attrs['desvio'],
            'Score no treino': sinal * melhor.user_attrs['treino'], 'Combinações testadas': len(study.trials),
            'Modelos treinados': modelos_treinados_optuna(study, n_splits), 'Tempo (s)': tempo,
            'Melhores hiperparâmetros': {k.replace('modelo__', ''): v for k, v in melhor.params.items()}}


def historico_optuna(study, maior_melhor=True):
    """Score de cada trial (NaN nas podadas) e o melhor score acumulado até ele (para o gráfico do histórico).

    No Optuna 5, a tentativa podada guarda como `value` a média até a poda: por isso o filtro é pelo estado.
    """
    completa = optuna.trial.TrialState.COMPLETE
    valores = np.array([t.value if t.state == completa else np.nan for t in study.trials], dtype=float)
    acumulado = np.fmax.accumulate(valores) if maior_melhor else np.fmin.accumulate(valores)
    return valores, acumulado
