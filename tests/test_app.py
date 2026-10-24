"""Teste do app Streamlit: abre sem erro, mostra os indicadores e reage ao limiar."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def abrir():
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_abre_e_mostra_os_indicadores():
    app = abrir()
    assert not app.exception
    assert len(app.metric) == 7          # 4 da regra de decisão + 3 da transação
    assert app.metric[0].label == 'Fraudes barradas'


def test_limiar_mais_alto_gera_menos_alarmes_falsos():
    app = abrir()
    app.sidebar.slider[0].set_value(0.02).run()
    alarmes_baixo = int(app.metric[1].value)
    app.sidebar.slider[0].set_value(0.95).run()
    alarmes_alto = int(app.metric[1].value)
    assert not app.exception
    assert alarmes_alto <= alarmes_baixo
