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
