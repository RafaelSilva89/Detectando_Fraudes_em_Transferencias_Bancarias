# 🕵️ Detectando Fraudes em Transferências Bancárias

**Um modelo que coloca as transações suspeitas no topo da fila, e uma regra de decisão escolhida pelo custo em reais.**

<div align="center">
<img src="https://media4.giphy.com/media/RDZo7znAdn2u7sAcWH/giphy.gif" width="300px">
</div>

**Python · scikit-learn · XGBoost · LightGBM · imbalanced-learn · Streamlit · CRISP-DM**

| 📊 Destaques | |
|---|---|
| Base | **283.726 transações** em 2 dias (sem duplicatas), só **473 fraudes (0,17%)** |
| Resultado no teste | Revisando **132** das 56.746 transações (0,23%), o modelo barra **78 das 95 fraudes (82%)** e **82,3% do valor fraudado** |
| Em reais | **R$ 12.154,24** barrados com R$ 660 em revisões: economia líquida de **R$ 11.494,24** (intervalo de 95%: R$ 6.255 a R$ 17.940) |
| Modelo | **XGBoost** com peso da classe, PR-AUC de **0,817** no teste (o acaso daria 0,002). Regra de decisão escolhida pelo menor custo: limiar de **0,11**, ou de **0,005** para transações acima de **R$ 200** |
| Entrega | Notebook CRISP-DM + modelo e ficha em `Model/` + **app Streamlit** com explicação SHAP + **16 testes automatizados** (com CI) |

---

## 📑 Sumário

1. [Contexto do problema](#1-contexto-do-problema)
2. [Objetivo](#2-objetivo)
3. [Base de dados](#3-base-de-dados)
4. [Estrutura do projeto](#4-estrutura-do-projeto)
5. [Como executar](#5-como-executar)
6. [Etapas do projeto (CRISP-DM)](#6-etapas-do-projeto-crisp-dm)
7. [Principais descobertas da EDA](#7-principais-descobertas-da-eda)
8. [Construção da solução](#8-construção-da-solução)
9. [Resultado do modelo](#9-resultado-do-modelo)
10. [App interativo (Streamlit)](#10-app-interativo-streamlit)
11. [O que mudou da v1 para a v2](#11-o-que-mudou-da-v1-para-a-v2)
12. [Insights](#12-insights)
13. [Limites](#13-limites)
14. [Próximos passos](#14-próximos-passos)

---

## 1. Contexto do problema

Golpes em transações pela internet crescem todo ano no Brasil. Para um banco, cada fraude que passa é **dinheiro perdido**, e cada transação legítima bloqueada por engano é **custo de atendimento** e atrito com o cliente. Como as fraudes são raríssimas (menos de 2 em cada mil transações), revisar tudo é inviável, e revisar ao acaso não encontra quase nada.

## 2. Objetivo

> **Pergunta de negócio:** *Quantas fraudes, e quanto dinheiro, o banco consegue barrar revisando só as transações que o modelo marcar? Qual o custo dessas revisões?*

## 3. Base de dados

**284.807 transações** feitas em **48 horas**, com **492 fraudes**. A base é o conjunto público *Credit Card Fraud Detection* (Universidade Livre de Bruxelas, no Kaggle) com as colunas renomeadas, e fica em `Datasets/transferencias.zip` (68,5 MB).

> ⚠️ **As colunas não são o que os nomes dizem.** "valor", "antecedentes", "empréstimos", "país" etc. são **28 componentes PCA anônimas** (V1 a V28): combinações matemáticas das variáveis originais, que o banco não divulgou. As únicas variáveis com significado são:
>
> | Coluna | O que é |
> |---|---|
> | `hist_saldo` | **Valor da transação** |
> | `Timestamp` | Segundos desde a 1ª transação (janela de 48 h) |
> | `Target` | 1 = fraude, 0 = legítima |

## 4. Estrutura do projeto

```
Detectando_Fraudes_em_Transferencias_Bancarias/
├── .github/workflows/testes.yml           # roda os testes a cada push (GitHub Actions)
├── Datasets/
│   └── transferencias.zip                 # base completa (CSV compactado)
├── Imagens/                               # gráficos gerados pelos notebooks
├── Model/
│   ├── modelo_fraude.pkl                  # pipeline final (XGBoost) + limiar + colunas
│   ├── modelo_fraude.json                 # ficha do modelo: limiar, métricas, hiperparâmetros, avisos
│   └── Modelo_Random_Forest_v01.pkl       # modelo da versão original
├── Notebooks/
│   ├── Detectando Fraudes em Transferências Bancárias.ipynb        # versão original
│   └── Detectando Fraudes em Transferências Bancárias - v2.ipynb   # versão reorganizada (CRISP-DM)
├── docs/
│   └── decisoes_analiticas.md             # registro das decisões do projeto (DA-01 a DA-12)
├── reports/
│   ├── comparacao_modelos.csv             # métricas de todos os modelos na validação
│   ├── previsoes_teste.csv                # probabilidades do teste (usadas pelo app)
│   └── amostra_transacoes.csv             # 50 transações do teste para o app
├── src/
│   ├── config.py                          # caminhos, semente e premissas de custo
│   ├── dados.py                           # leitura, significado das colunas, duplicatas e hora do dia
│   ├── estatistica.py                     # testes das hipóteses (com tamanho de efeito) e regra do IQR
│   ├── selecao.py                         # Random Forest e Informação Mútua
│   ├── avaliacao.py                       # métricas, limiar, regra de valor, custo em reais, bootstrap, PSI e ficha do modelo
│   ├── explicacao.py                      # valores SHAP de cada previsão (pelo próprio XGBoost)
│   └── graficos.py                        # funções de gráficos
├── tests/
│   ├── test_funcoes.py                    # testes das funções de src/
│   └── test_app.py                        # abre o app e confere os indicadores (AppTest)
├── app.py                                 # app Streamlit
├── pytest.ini
├── requirements.txt
└── README.md
```

## 5. Como executar

```bash
# 1. Clonar o repositório
git clone https://github.com/RafaelSilva89/Detectando_Fraudes_em_Transferencias_Bancarias.git
cd Detectando_Fraudes_em_Transferencias_Bancarias

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Abrir o notebook (a execução completa leva cerca de 40 minutos)
jupyter notebook "Notebooks/Detectando Fraudes em Transferências Bancárias - v2.ipynb"

# 4. Abrir o app
streamlit run app.py

# 5. Rodar os testes
pytest
```

## 6. Etapas do projeto (CRISP-DM)

| Fase | O que foi feito |
|---|---|
| **01. Entendimento do negócio** | Pergunta, métrica (PR-AUC) e custo de cada tipo de erro |
| **02. Preparação dos dados** | Significado real das colunas, remoção de 1.081 duplicatas, outliers pelo IQR, 6 hipóteses com testes estatísticos e criação da hora do dia |
| **03. Modelagem** | Separação 60/20/20 estratificada, peso da classe x SMOTE, seleção de variáveis, 4 algoritmos, validação cruzada, ajuste fino pela PR-AUC, **limiar pelo custo em reais**, calibração, **regra complementar para valores altos** e teste anti-vazamento |
| **04. Avaliação** | Teste aberto uma única vez: métricas, resultado em reais, **intervalos de confiança (bootstrap)**, **validação temporal**, erros por valor e horário, importância por permutação e **SHAP** |
| **05. Implantação** | Modelo e ficha em `Model/`, simulação, curva "fraudes encontradas x transações revisadas" e **monitoramento de deriva (PSI)** |

## 7. Principais descobertas da EDA

**A fraude é mais comum de madrugada**

<div align="center">
<img src="Imagens/Fraude%20por%20hora.png" width="850px">
</div>

- Entre 0 h e 6 h, a taxa de fraude é de **0,482%**, contra **0,138%** no resto do dia: a chance é **3,5 vezes** maior.

**Algumas componentes separam fortemente as fraudes**

<div align="center">
<img src="Imagens/Top%203%20componentes.png" width="850px">
</div>

- **13 das 28** componentes separam as fraudes com efeito grande; nas melhores (`crt`, `cep`, `reclamacoes`), as duas classes quase não se sobrepõem.

**Outros achados**
- **Valor zero é suspeito:** 1,38% das transações de R$ 0 são fraudes, contra 0,16% das demais (chance **8,8 vezes** maior). Pode ser um teste para ver se a conta funciona antes do golpe.
- **As fraudes têm valor mediano menor** (R$ 9,82 x R$ 22,00), mas média maior (R$ 123,87 x R$ 88,41): muitas pequenas e algumas grandes.

> 🧪 **Testes estatísticos** (o que importa é o tamanho do efeito, não o p):
> - **H1 (valor):** Mann-Whitney com correlação de postos de −0,11 (efeito pequeno).
> - **H2 (madrugada):** qui-quadrado com *odds ratio* de 3,51.
> - **H3 (valor zero):** qui-quadrado com *odds ratio* de 8,81.
> - **H4 (componentes):** Mann-Whitney com efeito de até 0,89 (`crt`).
> - **H5 (correlação entre componentes):** máximo de 0,019; não há multicolinearidade.
> - **H6 (dois dias):** V de Cramér de 0,005, diferença desprezível; por isso o `Timestamp` cru saiu do modelo.

## 8. Construção da solução

**1️⃣ Limpeza:** remoção de **1.081 duplicatas** (com elas, 344 transações do teste teriam uma cópia no treino) e troca do `Timestamp` pela **hora do dia**.

**2️⃣ Separação:** treino (60%), validação (20%) e teste (20%), todos com 0,17% de fraudes. O teste só é usado no fim.

**3️⃣ Desbalanceamento:** com 0,17% de fraudes, o modelo precisa de ajuda para não ignorar a classe rara. O **peso da classe** (fazer cada erro numa fraude "custar" mais) empatou com o SMOTE (criar fraudes sintéticas), sem dados falsos:

| PR-AUC na validação | Sem balanceamento | Peso da classe | SMOTE |
|---|---|---|---|
| XGBoost | 0,711 | **0,891** | 0,893 |
| LightGBM | 0,092 | **0,855** | 0,853 |
| Regressão Logística | 0,800 | 0,788 | 0,788 |

**4️⃣ Modelos:** XGBoost, Random Forest, LightGBM e Regressão Logística, comparados pela **PR-AUC**, que mede o quanto o modelo coloca as fraudes no topo da lista. A acurácia não serve: "nunca é fraude" acerta 99,83%.

| Validação cruzada (5 dobras) | PR-AUC (média ± desvio) |
|---|---|
| **XGBoost** | **0,848 ± 0,028** |
| Random Forest | 0,840 ± 0,034 (cerca de 8 vezes mais lento) |
| LightGBM | 0,824 ± 0,032 |
| Regressão Logística | 0,755 ± 0,027 |

**5️⃣ Ajuste fino:** busca aleatória de hiperparâmetros otimizando a PR-AUC. O XGBoost ajustado (0,841) superou o padrão (0,820) nas mesmas dobras.

**6️⃣ Limiar pelo custo em reais:** uma fraude não detectada custa o **valor da transação**, e cada revisão custa **R$ 5** (premissa). Na validação, o limiar de menor custo é **0,11**, e ele se mantém para revisões de R$ 1 a R$ 10.

<div align="center">
<img src="Imagens/Curva%20de%20custo.png" width="750px">
</div>

**7️⃣ Regra complementar para valores altos:** as fraudes que mais custam são as de valor alto. Na validação, revisar também as transações **acima de R$ 200 com probabilidade a partir de 0,5%** aumentou a economia em R$ 122. Pesar cada fraude pelo valor no treino foi testado e não ajudou. A calibração isotônica também foi testada e não mudou nada mensurável.

## 9. Resultado do modelo

No **teste** (56.746 transações e 95 fraudes nunca vistas), com a regra final (limiar de 0,11 + regra de valor):

| | Resultado |
|---|---|
| **Fraudes barradas** | **78 de 95** (recall de 82,1%) |
| **Precisão** | 59,1% das transações revisadas eram fraude |
| **Alarmes falsos** | 54 (0,1% das legítimas) |
| **Valor barrado** | **R$ 12.154,24** de R$ 14.766,31 (**82,3%**) |
| **Custo das revisões** | 132 x R$ 5 = R$ 660 |
| **Economia líquida** | **R$ 11.494,24** (intervalo de 95% por bootstrap: R$ 6.255 a R$ 17.940) |
| PR-AUC / AUC-ROC | 0,817 (intervalo de 0,736 a 0,889) / 0,975 |

<div align="center">
<img src="Imagens/Ganho%20do%20modelo%20x%20sorteio.png" width="750px">
</div>

- Revisando as **50** transações de maior probabilidade, o time encontra **49 fraudes**; num sorteio, encontraria 0,08.
- 🔎 A regra de valor rendeu **+R$ 989** no teste em relação a usar só o limiar (2 fraudes a mais, de valor alto, com 37 revisões a mais). No bootstrap, a regra final economiza mais que o limiar de 0,5 em **88%** das amostras.
- ⚠️ **O que escapa:** 17 fraudes (R$ 2.612,07). Só **3** delas, acima de R$ 500, respondem por **83%** desse valor, e 13 têm probabilidade de até 0,1%: nas componentes, parecem legítimas.
- ⚠️ **Em dados futuros, a fila cresce:** treinando nas primeiras 36 horas e avaliando nas últimas 12, a PR-AUC se mantém (**0,798**), mas a precisão cai de 0,59 para **0,27** (cerca de 3 alarmes falsos por fraude). A componente `lim_crt` muda bastante entre os dois dias (PSI de 1,05), e o monitoramento de deriva faz parte da entrega.

## 10. App interativo (Streamlit)

O app permite que a área de fraudes **teste a regra de decisão** sem abrir o notebook:

1. **Regra de decisão:** escolha o **limiar** e o **custo da revisão**, ligue ou desligue a **regra complementar para valores altos** e veja, nas 56.746 transações do teste, quantas fraudes seriam barradas, os alarmes falsos, o valor barrado e a economia líquida, além da curva de precisão e recall.
2. **Analisar uma transação:** escolha uma das 50 transações da amostra e veja a probabilidade de fraude, a decisão e **por que** o modelo chegou a ela (valores SHAP).

```bash
streamlit run app.py
```

## 11. O que mudou da v1 para a v2

| Tema | v1 (original) | v2 |
|---|---|---|
| Base | Lida de `Dataset/transferencias.csv`, fora do repositório | `Datasets/transferencias.zip` versionada; conferida com a base pública |
| Colunas | Interpretadas pelos nomes ("valor médio das fraudes = −5", "−7 empréstimos") | Identificadas como **componentes PCA**; valor real em `hist_saldo` |
| Duplicatas | Mantidas (1.081) | **Removidas** antes da separação |
| `Timestamp` | Usado cru | Trocado pela **hora do dia** |
| Balanceamento | SMOTE calculado, mas **nunca usado** no treino | Peso da classe x SMOTE comparados; **peso da classe** escolhido |
| Validação | Só a 1ª divisão estratificada | 60/20/20 estratificado + **validação cruzada** (média ± desvio) |
| Métrica | Acurácia (99,9% para todos) | **PR-AUC**, recall, precisão e **custo em reais** |
| Seleção de variáveis | `mutual_info_regression` (de regressão) | `mutual_info_classif` + Random Forest |
| Ajuste fino | Otimizava a acurácia; `max_features='auto'` (removido do scikit-learn) | `RandomizedSearchCV` otimizando a PR-AUC |
| Avaliação dos modelos ajustados | **Nos dados em que treinaram** ("LightGBM com 100% em tudo") | Na validação nunca vista; o item 9 reproduz o erro (tudo = 1,000) |
| Limiar | Fixo em 0,5 | **0,11**, de menor custo na validação, com análise de sensibilidade, + regra complementar para valores altos |
| Incerteza e futuro | Não medidos | Intervalos por bootstrap, validação temporal e monitoramento de deriva (PSI) |
| Explicação | Importância de variáveis lidas pelos nomes fictícios | Importância por permutação + **SHAP** de cada transação |
| Resultado no teste | RF: F1 de 86,03%, recall de 78,57% | XGBoost: recall de **82,1%**, **82,3%** do valor barrado, economia de **R$ 11.494,24** |
| Entrega | Notebook + `.pkl` | Notebook + pipeline e ficha JSON + app + 16 testes com CI + registro de decisões |

## 12. Insights

1. **A acurácia engana:** "nunca é fraude" acerta 99,83%. A PR-AUC e o resultado em reais são as medidas certas.
2. **Poucas componentes fazem o trabalho:** `crt` (V14) e `cep` (V4) lideram a importância por permutação e o SHAP.
3. **Peso da classe vale o mesmo que o SMOTE**, sem dados falsos. Sem balanceamento, o LightGBM falha.
4. **Validação pequena engana:** com 94 fraudes, a validação cruzada é que decidiu o modelo.
5. **A regra certa é a de menor custo, não o limiar de 0,5:** limiar de 0,11 mais uma regra para valores acima de R$ 200, robusta para revisões de até R$ 10.
6. **O que escapa é pouco, mas caro, e invisível:** 3 fraudes grandes respondem por 83% do valor perdido, todas com probabilidade de até 0,1%.
7. **O futuro é mais difícil que o teste aleatório:** em dados posteriores ao treino, a precisão cai para 0,27. Monitorar é parte da solução.
8. **Os "100%" da versão original eram vazamento.**
9. **Sinais simples ajudam o analista:** madrugada (3,5 vezes mais fraude) e valor zero (8,8 vezes).

## 13. Limites

- **Só 2 dias de dados:** os padrões de fraude mudam com o tempo. A validação temporal cobre só 12 horas e já mostrou a precisão caindo.
- **Componentes anônimas:** não dá para explicar a decisão ao cliente em termos de negócio (o SHAP aponta componentes, não motivos), nem aplicar o modelo a outra base sem o mesmo PCA. O PCA foi feito pelo banco com a base inteira, incluindo o nosso teste.
- **Premissas de custo:** o custo de R$ 5 por revisão é uma suposição, e os valores estão originalmente em euros na base pública.
- **A regra de valor foi escolhida entre 28 combinações:** na validação, o ganho foi pequeno (R$ 122).
- **Só 95 fraudes no teste:** a economia tem intervalo de 95% de R$ 6.255 a R$ 17.940.

## 14. Próximos passos

1. **Mais dias de dados e validação temporal em janelas deslizantes**, para entender a queda de precisão e definir de quanto em quanto tempo retreinar.
2. **Alertas de deriva automáticos**, começando pela investigação de `lim_crt`.
3. **Variáveis de comportamento do cliente** (histórico, frequência, dispositivo), o único caminho para as fraudes que o modelo não enxerga.
4. **Fila priorizada pelo valor esperado** (probabilidade × valor), com teto de revisões por dia.
5. **Publicar o app** no Streamlit Community Cloud e expor o modelo numa API.

---

**Autor:** Rafael R. Silva · [GitHub](https://github.com/RafaelSilva89)
