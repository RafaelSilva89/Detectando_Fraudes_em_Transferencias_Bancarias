# Registro de decisões analíticas

Este documento registra as principais decisões do projeto, no formato
**Contexto → Decisão → Alternativas consideradas → Consequência → Onde está**.
Cada decisão tem um código (DA-XX) para ser citada em commits e pull requests.

Os números vêm das saídas do notebook [`Notebooks/Detectando Fraudes em Transferências Bancárias - v2.ipynb`](../Notebooks/).

| Código | Decisão |
|---|---|
| [DA-01](#da-01--base-no-repositório-e-significado-real-das-colunas) | Base no repositório e significado real das colunas |
| [DA-02](#da-02--remoção-das-duplicatas-antes-da-separação) | Remoção das duplicatas antes da separação |
| [DA-03](#da-03--hora-do-dia-no-lugar-do-timestamp) | Hora do dia no lugar do `Timestamp` |
| [DA-04](#da-04--pr-auc-como-métrica-principal) | PR-AUC como métrica principal |
| [DA-05](#da-05--peso-da-classe-em-vez-de-smote) | Peso da classe em vez de SMOTE |
| [DA-06](#da-06--todas-as-30-variáveis) | Todas as 30 variáveis |
| [DA-07](#da-07--xgboost-escolhido-pela-validação-cruzada) | XGBoost escolhido pela validação cruzada |
| [DA-08](#da-08--limiar-pelo-custo-em-reais-escolhido-na-validação) | Limiar pelo custo em reais, escolhido na validação |
| [DA-09](#da-09--avaliação-honesta-e-teste-anti-vazamento) | Avaliação honesta e teste anti-vazamento |
| [DA-10](#da-10--regra-complementar-para-valores-altos) | Regra complementar para valores altos |
| [DA-11](#da-11--probabilidades-sem-calibração-isotônica) | Probabilidades sem calibração isotônica |
| [DA-12](#da-12--validação-temporal-e-monitoramento-de-deriva) | Validação temporal e monitoramento de deriva |

---

## DA-01 — Base no repositório e significado real das colunas

- **Contexto:** a versão original lia `Dataset/transferencias.csv`, que não estava no repositório, e descrevia as colunas como "país", "valor", "antecedentes criminais", "empréstimos" etc. A base é idêntica, valor por valor, ao `creditcard.csv` público (ULB/Kaggle), com as colunas renomeadas.
- **Decisão:** versionar a base compactada (`Datasets/transferencias.zip`, 68,5 MB) e manter os nomes da base, mas tratá-los como o que são: 28 componentes PCA anônimas (V1 a V28), o tempo (`Timestamp`) e o valor da transação (`hist_saldo`).
- **Alternativas consideradas:** renomear para V1…V28 (perderia a comparação com a versão original); interpretar os nomes fictícios (o que gerou conclusões sem sentido na versão original, como "média de −5 nas fraudes").
- **Consequência:** o notebook é reproduzível, e a EDA usa o valor e a hora reais; as componentes são analisadas só pela capacidade de separar as classes.
- **Onde está:** `COLUNAS_ORIGINAIS` em `src/dados.py`; itens 1.2 e 2.1 da Fase 02.

## DA-02 — Remoção das duplicatas antes da separação

- **Contexto:** 1.081 linhas (0,38%) são cópias idênticas, 19 delas fraudes. Separando com elas, 344 transações do teste teriam uma cópia no treino, 8 delas fraudes (8,2% das fraudes do teste).
- **Decisão:** remover as duplicatas exatas antes de separar treino, validação e teste.
- **Alternativas consideradas:** separar mantendo as cópias juntas (por grupo); manter as duplicatas (versão original).
- **Consequência:** 283.726 transações e 473 fraudes; o teste não tem transações "já vistas".
- **Onde está:** `remover_duplicatas` em `src/dados.py`; item 2.3 da Fase 02 e item 9 da Fase 03.

## DA-03 — Hora do dia no lugar do `Timestamp`

- **Contexto:** o `Timestamp` são os segundos desde a 1ª transação, numa janela de 48 h: a posição na coleta, que não se repete no futuro. A fraude é 3,5 vezes mais provável de madrugada (H2), e a taxa muda um pouco entre os dois dias (H6).
- **Decisão:** trocar o `Timestamp` pela hora do dia (segundos módulo 86.400 ÷ 3.600), supondo que a contagem começa à meia-noite, o que o volume da madrugada sustenta.
- **Alternativas consideradas:** manter o `Timestamp` cru (versão original).
- **Consequência:** a variável tem significado e se repete em qualquer dia; no modelo final, ela contribui pouco (fora das 15 mais importantes), porque as componentes já carregam a informação.
- **Onde está:** `hora_do_dia` e `preparar_variaveis` em `src/dados.py`; H2, H6 e item 4 da Fase 02.

## DA-04 — PR-AUC como métrica principal

- **Contexto:** com 0,17% de fraudes, "nunca é fraude" acerta 99,83%. A versão original comparava e ajustava os modelos pela acurácia (99,92% a 99,95% para todos).
- **Decisão:** usar a **PR-AUC** (área sob a curva precisão x recall) para comparar, validar e ajustar os modelos, com recall, precisão, F1 e o custo em reais como apoio.
- **Alternativas consideradas:** acurácia (não discrimina); AUC-ROC (fica perto de 1 para todos, porque as legítimas dominam); F1 (depende do limiar).
- **Consequência:** o "modelo que chuta" tem PR-AUC de 0,002; o XGBoost, de 0,891 na validação e 0,817 no teste.
- **Onde está:** `calcular_metricas_completas` em `src/avaliacao.py`; itens 5 a 7 da Fase 03.

## DA-05 — Peso da classe em vez de SMOTE

- **Contexto:** a versão original gerava um treino 50/50 com SMOTE, mas treinava os modelos com o treino original. Sem balanceamento, o LightGBM falha (PR-AUC de 0,092) e o XGBoost cai para 0,711.
- **Decisão:** usar **peso da classe** (`class_weight='balanced'` ou `scale_pos_weight`) dentro do próprio modelo.
- **Alternativas consideradas:** SMOTE dentro de um Pipeline do `imblearn` (testado: empata, com 0,893 x 0,891 no XGBoost, mas cria cerca de 170 mil fraudes sintéticas e dobra o tempo de treino); nenhum balanceamento.
- **Consequência:** mesma qualidade, sem dados falsos e com treino mais rápido.
- **Onde está:** item 4.2 da Fase 03.

## DA-06 — Todas as 30 variáveis

- **Contexto:** Random Forest e Informação Mútua concordam no topo (9 das 10 primeiras), e 6 variáveis têm informação mútua praticamente nula. A versão original usava `mutual_info_regression` num problema de classificação.
- **Decisão:** manter as 30 variáveis, usando `mutual_info_classif` só para análise.
- **Alternativas consideradas:** as 10 melhores pela informação mútua (a PR-AUC do XGBoost cai de 0,891 para 0,833).
- **Consequência:** modelo mais simples de manter (as mesmas colunas da base), sem perda de desempenho.
- **Onde está:** `selecao_features` e `importancia_random_forest` em `src/selecao.py`; item 4.3 da Fase 03.

## DA-07 — XGBoost escolhido pela validação cruzada

- **Contexto:** a validação tem só 94 fraudes; uma fraude a mais ou a menos muda as métricas. Na validação, o XGBoost padrão (0,891) e o ajustado (0,889) empatavam.
- **Decisão:** escolher o algoritmo pela validação cruzada de 5 dobras (XGBoost: 0,848 ± 0,028; Random Forest: 0,840 ± 0,034, cerca de 8 vezes mais lento) e a versão (padrão x ajustada) pelas mesmas 3 dobras da busca (ajustado: 0,841; padrão: 0,820). A busca otimiza a PR-AUC.
- **Alternativas consideradas:** escolher pela validação simples; Random Forest (a versão original escolheu o RF depois de compará-los com vazamento).
- **Consequência:** XGBoost ajustado (454 árvores, profundidade 4, peso da fraude de 24,5), treinado de novo com treino + validação.
- **Onde está:** itens 5 a 7 da Fase 03.

## DA-08 — Limiar pelo custo em reais, escolhido na validação

- **Contexto:** o limiar padrão de 0,5 não leva em conta o custo dos erros. Uma fraude não detectada custa o valor da transação; uma revisão custa o tempo de um analista (premissa de R$ 5).
- **Decisão:** escolher, **na validação**, o limiar de menor custo total: **0,11**. Ele se mantém para revisões de R$ 1 a R$ 10.
- **Alternativas consideradas:** 0,5; F1 máximo (0,897); meta de recall de 80% (0,921).
- **Consequência:** na validação, economia de R$ 7.916 contra R$ 7.717 com 0,5. No teste, só com o limiar, 76 de 95 fraudes e 74,4% do valor barrado, com economia de R$ 10.504,76, ou R$ 69,95 a menos que com 0,5. O limiar não foi trocado depois de ver o teste. A regra final soma a regra de valor (DA-10).
- **Onde está:** `custo_financeiro` e `resumo_financeiro` em `src/avaliacao.py`; `CUSTO_REVISAO` em `src/config.py`; item 8 da Fase 03.

## DA-09 — Avaliação honesta e teste anti-vazamento

- **Contexto:** a versão original treinou os modelos "ajustados" com treino + validação e os avaliou na própria validação. Daí veio o "LightGBM com 100% em tudo", que pesou na escolha do modelo.
- **Decisão:** separação estratificada 60/20/20; modelo e limiar escolhidos na validação e na validação cruzada; teste aberto uma única vez; e uma seção que reproduz o erro para medir o seu tamanho.
- **Alternativas consideradas:** usar a validação temporal como separação principal (com só 48 h, sobraria pouco treino; ela entrou como conferência, DA-12).
- **Consequência:** avaliando nos dados já vistos, todas as métricas dão 1,000; no teste, F1 de 0,854 e PR-AUC de 0,817 (limiar de 0,5).
- **Onde está:** itens 4.1 e 9 da Fase 03.

## DA-10 — Regra complementar para valores altos

- **Contexto:** as fraudes que escapam com o maior prejuízo são as de valor alto, e o limiar único trata uma transação de R$ 1 como uma de R$ 1.000.
- **Decisão:** além do limiar de 0,11, revisar as transações **acima de R$ 200 com probabilidade a partir de 0,005**. Os dois parâmetros foram escolhidos na validação, numa grade de 28 combinações, pelo menor custo total.
- **Alternativas consideradas:** só o limiar (economia de R$ 7.915,75 na validação, contra R$ 8.037,66 com a regra); pesar cada fraude pelo valor no treino (`sample_weight`: PR-AUC de 0,883 e economia de R$ 7.886,75, pior).
- **Consequência:** no teste, 78 de 95 fraudes, 82,3% do valor barrado e economia de R$ 11.494,24 (+R$ 989,48 sobre só o limiar), com 54 alarmes falsos em vez de 19. No bootstrap, a regra final supera o limiar de 0,5 em 88% das amostras (diferença entre −R$ 276 e +R$ 2.647, intervalo de 95%).
- **Onde está:** `aplicar_regra` e `bootstrap_decisoes` em `src/avaliacao.py`; item 8.5 da Fase 03 e item 1.1 da Fase 04; `regra_complementar_de_valor` na ficha do modelo.

## DA-11 — Probabilidades sem calibração isotônica

- **Contexto:** para comunicar o risco ("30% de chance de fraude"), as probabilidades deveriam bater com a frequência real.
- **Decisão:** manter as probabilidades do XGBoost, sem calibração.
- **Alternativas consideradas:** calibração isotônica com 3 dobras (Brier de 0,00043 nos dois casos; PR-AUC de 0,8169 → 0,8172; nas faixas intermediárias, de 18 a 21 transações, troca uma superestimativa por uma subestimativa).
- **Consequência:** modelo mais simples (um treino em vez de quatro), sem perda mensurável. Nas faixas extremas, que concentram quase todas as transações, as probabilidades já batem com a taxa real.
- **Onde está:** item 8.4 da Fase 03.

## DA-12 — Validação temporal e monitoramento de deriva

- **Contexto:** em produção, o modelo prevê transações futuras, e a separação aleatória mistura os dois dias.
- **Decisão:** conferir o modelo treinando nas primeiras 36 h e avaliando nas últimas 12 h (sem escolher nada ali) e monitorar o PSI do score e das variáveis mais importantes, com investigação acima de 0,25.
- **Alternativas consideradas:** confiar só no teste aleatório.
- **Consequência:** a PR-AUC se mantém (0,798 contra 0,817), mas a precisão cai de 0,591 para 0,268: a fila de revisão pode ser mais que o dobro em produção. O score está estável entre os dias (PSI de 0,035), mas `lim_crt` muda muito (PSI de 1,05; KS de 0,36).
- **Onde está:** `psi` em `src/avaliacao.py`; item 1.2 da Fase 04 e monitoramento de deriva na Fase 05.
