"""Teste do app Streamlit: abre sem erro, mostra os indicadores, reage ao limiar e monitora os lotes."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def abrir():
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_abre_e_mostra_os_indicadores():
    app = abrir()
    assert not app.exception
    assert len(app.metric) == 10         # 4 da regra de decisão + 3 da transação + 3 do monitoramento
    assert app.metric[0].label == 'Fraudes barradas'


def test_limiar_mais_alto_gera_menos_alarmes_falsos():
    app = abrir()
    app.sidebar.slider[0].set_value(0.02).run()
    alarmes_baixo = int(app.metric[1].value)
    app.sidebar.slider[0].set_value(0.95).run()
    alarmes_alto = int(app.metric[1].value)
    assert not app.exception
    assert alarmes_alto <= alarmes_baixo


def test_aba_monitoramento_mostra_lotes_e_drift():
    app = abrir()
    assert app.metric[9].label == 'Lotes abaixo do SLA'
    assert app.metric[9].value.endswith('de 3')     # lotes de 4 horas nas últimas 12 horas
    assert len(app.dataframe) == 2                   # painel dos lotes + drift do lote escolhido
    app.select_slider[0].set_value(12).run()
    assert not app.exception
    assert app.metric[9].value.endswith('de 1')
