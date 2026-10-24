"""Leitura da base, significado real das colunas e preparação das variáveis."""
import pandas as pd

from src.config import ALVO

# A base é o conjunto público "Credit Card Fraud Detection" (ULB/Kaggle) com as colunas renomeadas.
# Os nomes "fictícios" foram mantidos, mas 28 deles são componentes PCA anônimas (V1 a V28):
# não medem país, cidade, valor etc. O valor real da transação está em `hist_saldo` (Amount).
COLUNAS_ORIGINAIS = {
    'Timestamp': 'Time', 'pais': 'V1', 'cidade': 'V2', 'bairro': 'V3', 'cep': 'V4', 'ip': 'V5', 'dia': 'V6',
    'hora': 'V7', 'minuto': 'V8', 'so': 'V9', 'valor': 'V10', 'antecedentes': 'V11', 'reclamacoes': 'V12',
    'qtd_trans': 'V13', 'crt': 'V14', 'limite_global': 'V15', 'tipo_crt': 'V16', 'estabelecimento': 'V17',
    'contas': 'V18', 'emprestimos': 'V19', 'navegador': 'V20', 'android': 'V21', 'ios': 'V22', 'compras': 'V23',
    'hist_navegacao': 'V24', 'relacionamento': 'V25', 'ind_seguranca': 'V26', 'tempo_transacao': 'V27',
    'lim_crt': 'V28', 'hist_saldo': 'Amount', 'Target': 'Class',
}
COMPONENTES_PCA = [nome for nome, original in COLUNAS_ORIGINAIS.items() if original.startswith('V')]
COLUNA_VALOR = 'hist_saldo'          # valor da transação (a única variável com unidade real, além do tempo)
SEGUNDOS_POR_DIA = 24 * 60 * 60


def carregar_base(caminho):
    """Lê a base (CSV dentro do zip)."""
    return pd.read_csv(caminho)


def remover_duplicatas(df):
    """Remove linhas idênticas em todas as colunas. Devolve (base sem duplicatas, quantidade removida)."""
    sem_duplicatas = df.drop_duplicates().reset_index(drop=True)
    return sem_duplicatas, len(df) - len(sem_duplicatas)


def hora_do_dia(timestamp):
    """Converte os segundos desde a 1ª transação (janela de 48 h) na hora do dia, de 0 a 24.

    Supõe que a contagem começa perto da meia-noite: o volume de transações despenca entre 1 h e 7 h dessa
    escala, como numa madrugada (conferido na EDA, H2).
    """
    return (timestamp % SEGUNDOS_POR_DIA) / 3600


def preparar_variaveis(df):
    """Separa X e y: troca o `Timestamp` cru (posição na janela de coleta, que não se repete no futuro)
    pela hora do dia, que é um padrão que pode se repetir."""
    X = df.drop(columns=[ALVO, 'Timestamp']).assign(hora_do_dia=hora_do_dia(df['Timestamp']))
    return X, df[ALVO]
