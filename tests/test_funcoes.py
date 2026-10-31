"""Testes das funções de src/ (dados, avaliação e estatística). Rodar na raiz do projeto com: pytest"""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.avaliacao import (aplicar_regra, bootstrap_decisoes, calcular_metricas_completas, custo_financeiro,
                           ganho_vs_sorteio, limiar_otimo_f1, psi, resumo_financeiro)
from src.explicacao import contribuicoes_xgboost
from src.dados import COLUNAS_ORIGINAIS, COMPONENTES_PCA, hora_do_dia, preparar_variaveis, remover_duplicatas
from src.estatistica import odds_ratio, teste_mann_whitney


def base_minima():
    return pd.DataFrame({'Timestamp': [0.0, 3600.0, 90000.0], 'hist_saldo': [10.0, 0.0, 99.9], 'Target': [0, 1, 0]})


def test_dicionario_cobre_as_31_colunas_e_28_componentes():
    assert len(COLUNAS_ORIGINAIS) == 31
    assert len(COMPONENTES_PCA) == 28
    assert COLUNAS_ORIGINAIS['hist_saldo'] == 'Amount' and COLUNAS_ORIGINAIS['valor'] == 'V10'


def test_hora_do_dia_volta_a_zero_no_segundo_dia():
    assert hora_do_dia(pd.Series([0, 3600, 86400, 90000])).tolist() == pytest.approx([0, 1, 0, 1])


def test_preparar_variaveis_troca_o_timestamp_pela_hora():
    X, y = preparar_variaveis(base_minima())
    assert 'Timestamp' not in X and 'Target' not in X
    assert X['hora_do_dia'].tolist() == pytest.approx([0, 1, 1])
    assert y.tolist() == [0, 1, 0]


def test_remover_duplicatas_conta_as_linhas_removidas():
    df = pd.concat([base_minima(), base_minima().iloc[[0]]])
    sem, removidas = remover_duplicatas(df)
    assert removidas == 1 and len(sem) == 3


def test_metricas_da_classe_fraude():
    m = calcular_metricas_completas([0, 0, 1, 1], [0, 1, 1, 0], [0.1, 0.6, 0.9, 0.4])
    assert m['Precisão'] == pytest.approx(0.5) and m['Recall (Sensibilidade)'] == pytest.approx(0.5)
    assert m['Falsos Positivos'] == 1 and m['Falsos Negativos'] == 1


def test_limiar_otimo_f1_separa_classes_perfeitamente():
    limiar, f1 = limiar_otimo_f1(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert f1 == pytest.approx(1.0) and 0.2 < limiar <= 0.8


def test_custo_financeiro_soma_fraudes_perdidas_e_revisoes():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.2, 0.6, 0.1])
    valores = np.array([100.0, 50.0, 10.0, 10.0])
    tabela = custo_financeiro(y, proba, valores, custo_revisao=5, limiares=[0.5])
    linha = tabela.iloc[0]
    # marcadas: 0,9 (fraude) e 0,6 (legítima) -> 2 revisões = R$ 10; perdida: a fraude de R$ 50
    assert linha['Fraudes perdidas'] == 1 and linha['Alarmes falsos'] == 1
    assert linha['Custo total'] == pytest.approx(60)


def test_resumo_financeiro_economia_liquida():
    r = resumo_financeiro([1, 1, 0], [1, 0, 1], [100.0, 50.0, 10.0], custo_revisao=5)
    assert r['Valor das fraudes barradas'] == pytest.approx(100)
    assert r['% do valor barrado'] == pytest.approx(100 / 150)
    assert r['Economia líquida'] == pytest.approx(90)   # 100 barrados - 2 revisões x R$ 5


def test_ganho_vs_sorteio_ordena_pelo_score():
    ganho = ganho_vs_sorteio(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert ganho['Modelo'].tolist()[:2] == [1, 2]
    assert ganho['Sorteio'].iloc[-1] == pytest.approx(2)


def test_odds_ratio_e_mann_whitney():
    condicao = [True, True, False, False, True, False]
    alvo = [1, 0, 0, 0, 1, 1]
    assert odds_ratio(condicao, alvo) == pytest.approx((2 * 2) / (1 * 1))
    efeito = teste_mann_whitney([5, 6, 7], [1, 2, 3])['tamanho do efeito']
    assert efeito == pytest.approx(1.0)


def test_aplicar_regra_com_regra_complementar_de_valor():
    proba = np.array([0.5, 0.01, 0.01, 0.001])
    valores = np.array([10.0, 300.0, 50.0, 300.0])
    assert aplicar_regra(proba, valores, 0.11).tolist() == [True, False, False, False]
    regra = {'corte_valor': 200, 'limiar_valor': 0.005}
    assert aplicar_regra(proba, valores, 0.11, regra).tolist() == [True, True, False, False]


def test_bootstrap_e_reprodutivel_e_mantem_as_fraudes():
    y = np.array([1] * 10 + [0] * 90)
    marcado = np.array([True] * 8 + [False] * 2 + [True] * 5 + [False] * 85)
    a = bootstrap_decisoes(y, np.ones(100), {'r': marcado}, 1, n=50, semente=3)
    b = bootstrap_decisoes(y, np.ones(100), {'r': marcado}, 1, n=50, semente=3)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 50 and a['Recall'].between(0, 1).all()


def test_psi_zero_para_mesma_distribuicao_e_alto_para_deslocada():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 20_000)
    assert psi(base, rng.normal(0, 1, 20_000)) < 0.01
    assert psi(base, rng.normal(1.5, 1, 20_000)) > 0.25


def test_contribuicoes_xgboost_somam_o_score():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=['a', 'b', 'c'])
    y = (X['a'] + rng.normal(scale=0.5, size=300) > 0).astype(int)
    modelo = xgb.XGBClassifier(n_estimators=20, max_depth=2, random_state=0).fit(X, y)
    contrib = contribuicoes_xgboost(modelo, X.head(5))
    margem = modelo.predict(X.head(5), output_margin=True)
    assert np.allclose(contrib.sum(axis=1).values, margem, atol=1e-4)


def test_objetivo_com_poda_guarda_desvio_e_treino():
    import optuna
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold
    from src.otimizacao import objetivo_cv_com_poda, resumo_optuna

    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=['a', 'b'])
    y = pd.Series((X['a'] + rng.normal(scale=0.5, size=200) > 0).astype(int))
    objetivo = objetivo_cv_com_poda(LogisticRegression(), lambda t: {'C': t.suggest_float('C', 0.01, 10, log=True)},
                                    X, y, StratifiedKFold(3, shuffle=True, random_state=0), average_precision_score)
    estudo = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    estudo.optimize(objetivo, n_trials=3)
    linha = resumo_optuna('Optuna', estudo, 1.0, n_splits=3)
    assert 0.5 < linha['Melhor score (CV)'] <= 1
    assert linha['Desvio entre dobras'] >= 0 and linha['Score no treino'] > 0.5
    assert linha['Modelos treinados'] == 9   # 3 tentativas completas x 3 dobras


def test_modelos_treinados_conta_as_dobras_das_podadas():
    import optuna
    from src.otimizacao import historico_optuna, modelos_treinados_optuna

    estudo = optuna.create_study(direction='maximize')
    estudo.add_trial(optuna.trial.create_trial(value=0.8, intermediate_values={0: 0.8, 1: 0.8, 2: 0.8}))
    estudo.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED, intermediate_values={0: 0.5}))
    estudo.add_trial(optuna.trial.create_trial(value=0.9, intermediate_values={0: 0.9, 1: 0.9, 2: 0.9}))
    assert modelos_treinados_optuna(estudo, 3) == 7
    valores, melhor = historico_optuna(estudo)
    assert np.isnan(valores[1]) and list(melhor) == [0.8, 0.8, 0.9]   # a podada não interrompe o acumulado


def test_poda_nunca_acontece_na_ultima_dobra():
    import optuna
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import PredefinedSplit, StratifiedKFold
    from src.otimizacao import modelos_treinados_optuna, objetivo_cv_com_poda

    class PodaSempre(optuna.pruners.BasePruner):
        def prune(self, study, trial):
            return True

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(120, 2)), columns=['a', 'b'])
    y = pd.Series((X['a'] > 0).astype(int))
    espaco = lambda t: {'C': t.suggest_float('C', 0.1, 1)}
    uma_dobra = PredefinedSplit([-1] * 80 + [0] * 40)
    for cv, n_dobras, estado in [(StratifiedKFold(3, shuffle=True, random_state=0), 3, optuna.trial.TrialState.PRUNED),
                                 (uma_dobra, 1, optuna.trial.TrialState.COMPLETE)]:
        estudo = optuna.create_study(direction='maximize', pruner=PodaSempre())
        estudo.optimize(objetivo_cv_com_poda(LogisticRegression(), espaco, X, y, cv, average_precision_score), n_trials=1)
        assert estudo.trials[0].state == estado
        assert modelos_treinados_optuna(estudo, n_dobras) == 1   # podada na 1ª dobra, ou completa com 1 dobra só


def test_historico_ignora_o_valor_guardado_nas_podadas():
    import optuna
    from src.otimizacao import historico_optuna

    estudo = optuna.create_study(direction='maximize')
    estudo.add_trial(optuna.trial.create_trial(value=0.8, intermediate_values={0: 0.8, 1: 0.8}))
    estudo.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED, value=0.95, intermediate_values={0: 0.95}))
    valores, melhor = historico_optuna(estudo)
    assert np.isnan(valores[1]) and list(melhor) == [0.8, 0.8]   # no Optuna 5 a podada tem value, mas não é resultado


def test_relatorio_drift_separa_estavel_de_mudanca():
    from src.monitoramento import relatorio_drift

    rng = np.random.default_rng(4)
    referencia = pd.DataFrame({'igual': rng.normal(size=5000), 'deslocada': rng.normal(size=5000)})
    atual = pd.DataFrame({'igual': rng.normal(size=5000), 'deslocada': rng.normal(1, 1, size=5000)})
    drift = relatorio_drift(referencia, atual, ['igual', 'deslocada'])
    assert drift.loc['igual', 'Leitura'] == 'Estável' and drift.loc['deslocada', 'Leitura'] == 'Mudança'
    assert drift.loc['deslocada', 'K-S (distância)'] > 0.3 > drift.loc['igual', 'K-S (distância)']


def test_desempenho_lote_bate_com_o_sklearn():
    from sklearn.metrics import average_precision_score, precision_score, recall_score
    from src.monitoramento import desempenho_lote

    real = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    prob = np.array([0.1, 0.4, 0.8, 0.3, 0.7, 0.9, 0.05, 0.2])
    revisar = prob >= 0.5
    r = desempenho_lote(real, prob, revisar)
    assert r['Fraudes'] == 3 and r['Revisões'] == 3
    assert r['Recall'] == recall_score(real, revisar) and r['Precisão da fila'] == precision_score(real, revisar)
    assert r['PR-AUC'] == average_precision_score(real, prob)


def test_painel_lotes_marca_o_sla_e_o_lote_sem_fraudes():
    from src.monitoramento import painel_lotes

    real = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    prob = np.array([0.9, 0.8, 0.1, 0.2, 0.1, 0.6, 0.2, 0.1, 0.3, 0.2])
    lotes = ['a'] * 4 + ['b'] * 3 + ['c'] * 3
    painel = painel_lotes(real, prob, prob >= 0.5, lotes, sla_pr_auc=0.7, sla_recall=0.7)
    assert list(painel.index) == ['a', 'b', 'c']
    assert list(painel['Situação']) == ['✅ dentro do SLA', '🚨 abaixo do SLA', '⚪ sem fraudes no lote']
