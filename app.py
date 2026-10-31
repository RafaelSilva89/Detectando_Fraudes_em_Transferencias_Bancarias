import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.avaliacao import aplicar_regra, resumo_financeiro
from src.config import CUSTO_REVISAO, PASTA_MODELO, PASTA_REPORTS, VARIAVEIS_MONITORADAS
from src.explicacao import contribuicoes_xgboost
from src.monitoramento import painel_lotes, relatorio_drift

st.set_page_config(page_title="Detecção de Fraudes 🕵️", page_icon="🕵️", layout="wide")
st.title("🕵️ Detecção de Fraudes em Transferências")
st.markdown("Teste a **regra de decisão** do modelo: a partir de qual probabilidade uma transação vai para "
            "**revisão manual**, e quanto isso barra de fraudes e custa em revisões.")

CAMINHO_MODELO = PASTA_MODELO / "modelo_fraude.pkl"


### Carregamento
@st.cache_resource
def carregar_modelo():
    return joblib.load(CAMINHO_MODELO)


@st.cache_data
def carregar_dados():
    previsoes = pd.read_csv(PASTA_REPORTS / "previsoes_teste.csv")
    amostra = pd.read_csv(PASTA_REPORTS / "amostra_transacoes.csv", index_col="id")
    return previsoes, amostra


@st.cache_data
def carregar_monitoramento():
    with open(PASTA_MODELO / "modelo_fraude.json", encoding="utf-8") as arquivo:
        sla = json.load(arquivo)["monitoramento"]
    lotes = pd.read_csv(PASTA_REPORTS / "lotes_monitoramento.csv.gz")
    referencia = pd.read_csv(PASTA_REPORTS / "referencia_monitoramento.csv")
    return sla, lotes, referencia


def reais(valor):
    return f"R$ {valor:,.0f}".replace(",", ".")


FORMATO_LOTES = {"Transações": "{:,.0f}", "Fraudes": "{:.0f}", "Revisões": "{:.0f}", "PR-AUC": "{:.3f}",
                 "Recall": "{:.3f}", "Precisão da fila": "{:.3f}"}
# A hora do dia fica fora do drift por lote: um lote de poucas horas cobre só uma parte do dia e sempre "mudaria"
VARIAVEIS_LOTE = [coluna for coluna in VARIAVEIS_MONITORADAS if coluna != "hora_do_dia"]
FORMATO_DRIFT = {"PSI": "{:.3f}", "K-S (distância)": "{:.3f}", "K-S (p-valor)": "{:.3g}"}

if not CAMINHO_MODELO.exists():
    st.error("Modelo não encontrado. Execute o notebook v2 para gerar Model/modelo_fraude.pkl.")
    st.stop()

modelo = carregar_modelo()
previsoes, amostra = carregar_dados()

### Parâmetros
with st.sidebar:
    st.header("⚙️ Regra de decisão")
    limiar = st.slider("Limiar (probabilidade mínima para revisar)", 0.01, 0.99, round(modelo["limiar"], 2), step=0.01)
    custo_revisao = st.number_input("Custo de revisar uma transação (R$)", 0.0, 100.0, float(CUSTO_REVISAO), step=1.0)
    st.caption(f"Limiar recomendado pelo notebook: **{modelo['limiar']:.2f}** (custo mínimo na validação, "
               f"com revisão a R$ {CUSTO_REVISAO:.0f}).")
    regra_valor = modelo.get("regra_valor")
    usar_regra = False
    if regra_valor:
        usar_regra = st.checkbox(f"Regra complementar: revisar também valores ≥ {reais(regra_valor['corte_valor'])} "
                                 f"com probabilidade ≥ {regra_valor['limiar_valor']:.2%}", value=True)
    regra_ativa = regra_valor if usar_regra else None

aba_regra, aba_transacao, aba_monitoramento = st.tabs(["📊 Regra de decisão", "🔍 Analisar uma transação",
                                                       "📡 Monitoramento"])

### Regra de decisão, nas transações do teste
with aba_regra:
    real = previsoes["real"].to_numpy()
    marcado = aplicar_regra(previsoes["probabilidade"], previsoes["valor"], limiar, regra_ativa)
    resultado = resumo_financeiro(real, marcado, previsoes["valor"], custo_revisao)
    encontradas, total = int((marcado & (real == 1)).sum()), int(real.sum())
    alarmes = int((marcado & (real == 0)).sum())

    st.caption(f"Calculado nas {len(previsoes):,} transações do teste, que o modelo não viu no treino.".replace(",", "."))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fraudes barradas", f"{encontradas} de {total}", f"{encontradas / total:.0%} das fraudes", delta_color="off")
    col2.metric("Alarmes falsos", f"{alarmes}", f"{alarmes / max(marcado.sum(), 1):.0%} das revisões", delta_color="off")
    col3.metric("Valor das fraudes barradas", reais(resultado["Valor das fraudes barradas"]),
                f"{resultado['% do valor barrado']:.0%} do valor fraudado", delta_color="off")
    col4.metric("Economia líquida", reais(resultado["Economia líquida"]),
                f"revisões: {reais(resultado['Custo das revisões'])}", delta_color="off")

    st.markdown("#### Precisão e recall para cada limiar")
    grade = np.round(np.arange(0.01, 1.0, 0.01), 2)
    curva = pd.DataFrame({
        "Recall (fraudes encontradas)": [((previsoes["probabilidade"] >= g) & (real == 1)).sum() / total for g in grade],
        "Precisão (revisões que eram fraude)": [
            ((previsoes["probabilidade"] >= g) & (real == 1)).sum() / max((previsoes["probabilidade"] >= g).sum(), 1)
            for g in grade],
    }, index=pd.Index(grade, name="Limiar"))
    st.line_chart(curva)
    st.caption("Limiar baixo: mais fraudes encontradas e mais alarmes falsos. Limiar alto: o contrário.")

### Uma transação da amostra
with aba_transacao:
    st.markdown("Escolha uma transação da amostra do teste (15 fraudes e 35 legítimas). As 28 componentes PCA são "
                "anônimas, por isso não há campos para digitar uma transação nova.")
    escolhida = st.selectbox("Transação", amostra.index, format_func=lambda i: f"#{i} – {reais(amostra.loc[i, 'hist_saldo'])}")
    linha = amostra.loc[[escolhida]]
    probabilidade = float(modelo["pipeline"].predict_proba(linha[modelo["colunas"]])[:, 1][0])
    decisao = "🚨 Revisar" if aplicar_regra([probabilidade], linha["hist_saldo"], limiar, regra_ativa)[0] else "✅ Liberar"

    col1, col2, col3 = st.columns(3)
    col1.metric("Probabilidade de fraude", f"{probabilidade:.2%}")
    col2.metric("Decisão com o limiar escolhido", decisao)
    col3.metric("Na verdade era", "Fraude" if linha["Target"].iloc[0] == 1 else "Legítima")
    st.caption(f"Valor: {reais(linha['hist_saldo'].iloc[0])} · hora do dia: {linha['hora_do_dia'].iloc[0]:.1f} h")

    st.markdown("#### Por que essa probabilidade? (valores SHAP)")
    efeito = contribuicoes_xgboost(modelo["pipeline"], linha[modelo["colunas"]]).iloc[0].drop("base")
    principais = efeito.reindex(efeito.abs().sort_values(ascending=False).index[:8])
    st.bar_chart(principais.rename("Efeito no score (positivo = fraude)"), horizontal=True)
    st.caption("As componentes PCA (crt, cep, reclamacoes etc.) são anônimas: o gráfico mostra quais pesaram, "
               "mas não o que significam no negócio.")

### Monitoramento em produção: desempenho por lote (SLA) e data drift
with aba_monitoramento:
    sla, lotes, referencia = carregar_monitoramento()
    st.markdown("Simulação de produção: o modelo treinado nas **primeiras 36 horas** da base decide as **últimas 12 horas**, "
                "em lotes, como se o gabarito de cada lote chegasse depois (contestações dos clientes). "
                "A regra de decisão é a escolhida ao lado.")
    horas_lote = st.select_slider("Tamanho do lote (horas)", options=[2, 3, 4, 6, 12], value=4)
    inicio = 36 + (lotes["hora"] - 36) // horas_lote * horas_lote
    rotulos = inicio.astype(int).map(lambda h: f"{h} a {h + horas_lote} h")
    revisar = aplicar_regra(lotes["probabilidade"], lotes["hist_saldo"], limiar, regra_ativa)
    painel = painel_lotes(lotes["real"], lotes["probabilidade"], revisar, rotulos,
                          sla["sla_pr_auc_minima"], sla["sla_recall_minimo"])
    abaixo = int((painel["Situação"] == "🚨 abaixo do SLA").sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("SLA de PR-AUC", f"≥ {sla['sla_pr_auc_minima']:.3f}")
    col2.metric("SLA de recall da regra", f"≥ {sla['sla_recall_minimo']:.3f}")
    col3.metric("Lotes abaixo do SLA", f"{abaixo} de {len(painel)}")
    st.dataframe(painel.style.format(FORMATO_LOTES, na_rep="–"))
    st.caption("SLA = limites inferiores do intervalo de 95% do bootstrap do teste. Lotes pequenos têm poucas "
               "fraudes, e uma fraude a mais ou a menos muda bastante a PR-AUC: um alerta isolado pede investigação, "
               "não retreino automático.")

    st.markdown("#### Data drift de um lote (não precisa do gabarito)")
    lote = st.selectbox("Lote", painel.index, index=len(painel) - 1)
    drift = relatorio_drift(referencia, lotes[(rotulos == lote).to_numpy()], VARIAVEIS_LOTE)
    st.dataframe(drift.style.format(FORMATO_DRIFT))
    st.caption(f"Referência: amostra de {len(referencia):,} transações das primeiras 36 horas. ".replace(",", ".") +
               f"PSI até 0,10: estável; até 0,25: atenção; acima de {sla['psi_alerta']:.2f}: mudança (investigar ou "
               "retreinar). Com lotes grandes, o p-valor do K-S fica perto de 0 até para diferenças pequenas: leia a distância. "
               "A hora do dia fica de fora: um lote de poucas horas sempre cobre só uma parte do dia (no notebook, ela é "
               "comparada dia contra dia).")

    st.markdown("#### Avaliar um lote novo")
    arquivo = st.file_uploader("CSV com as 30 colunas do modelo (e a coluna Target, se o gabarito já tiver chegado)",
                               type="csv")
    if arquivo is not None:
        novo = pd.read_csv(arquivo)
        faltando = [coluna for coluna in modelo["colunas"] if coluna not in novo.columns]
        if faltando:
            st.error(f"Faltam {len(faltando)} colunas no arquivo, por exemplo: {', '.join(faltando[:5])}.")
        else:
            prob_novo = modelo["pipeline"].predict_proba(novo[modelo["colunas"]])[:, 1]
            revisar_novo = aplicar_regra(prob_novo, novo["hist_saldo"], limiar, regra_ativa)
            st.metric("Transações enviadas para revisão", f"{int(revisar_novo.sum())} de {len(novo)}")
            st.dataframe(relatorio_drift(referencia, novo, VARIAVEIS_LOTE).style.format(FORMATO_DRIFT))
            if "Target" in novo.columns:
                st.dataframe(painel_lotes(novo["Target"], prob_novo, revisar_novo, ["lote enviado"] * len(novo),
                                          sla["sla_pr_auc_minima"], sla["sla_recall_minimo"])
                             .style.format(FORMATO_LOTES, na_rep="–"))

st.info("ℹ️ Premissas: uma fraude não barrada custa o valor da transação; cada transação revisada custa o valor "
        "definido ao lado. O modelo foi treinado com 2 dias de transações e deve ser acompanhado em produção.")
