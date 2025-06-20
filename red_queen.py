# PROJETO RED QUEEN - VERSÃO FINAL (IA ESPECIALISTA + UPLOAD)

# IMPORTAÇÕES NECESSÁRIAS
import streamlit as st                   # Biblioteca principal para a interface web
import pandas as pd                      # Manipulação de dados
import time                              # Pausas temporais na simulação
import shap                              # Explicabilidade do modelo de ML
from matplotlib import pyplot as plt     # Visualizações com gráficos
from xgboost import XGBClassifier        # Algoritmo de aprendizado supervisionado baseado em árvores
from sklearn.model_selection import train_test_split     # Separação de dados para treino/teste

# CONFIGURAÇÃO DA PÁGINA
# Define o layout, título da aba e a barra lateral expandida por padrão
st.set_page_config(layout="wide",page_title="Red Queen Threat Control",initial_sidebar_state="expanded")

# FUNÇÕES
@st.cache_data
def load_data(path):
    """Carrega o dataset a partir de um caminho especificado."""
    return pd.read_csv(path, encoding='utf-8')


@st.cache_resource
def train_model(df):
    """
    Treina o modelo XGBoost com balanceamento (via scale_pos_weight)
    Retorna o modelo treinado.
    """
    X = df.drop('Anomaly', axis=1)
    y = df['Anomaly']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    if 1 not in y_train.value_counts():
        ratio = 1
    else:
        ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

    model = XGBClassifier(scale_pos_weight=ratio, use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    return model


@st.cache_resource
def create_explainer(_model):
    """Cria e retorna um objeto explainer do SHAP para o modelo."""
    if _model:
        return shap.TreeExplainer(_model)
    return None


# PREPARAÇÃO DE DADOS E TREINAMENTO
# Carrega o dataset original
df_original = load_data("Red_Queen_Synthetic_Dataset.csv")

# Simula uma anomalia silenciosa (somente o T-Virus elevado)
novas_anomalias = {
    'T_Virus_Level':      [95.0],  # Apenas o T-Virus está em nível crítico
    'Room_Temperature':   [21.5],  # O resto está normal
    'Humidity':           [48.0],
    'Gas_Leak_Level':     [0.0],
    'Security_Clearance': [2],
    'AI_Override_Attempts':[0],
    'Proximity_To_Core':  [20.0],
    'Anomaly':            [1]      # E isso é uma anomalia
}

# Replica a anomalia 10 vezes para reforçar no treinamento
df_novas_anomalias = pd.DataFrame(novas_anomalias)
df_novas_anomalias_reforcado = pd.concat([df_novas_anomalias] * 10, ignore_index=True)

# Junta os dados originais com os dados reforçados
df_enriquecido = pd.concat([df_original, df_novas_anomalias_reforcado], ignore_index=True)

# Treina o modelo especialista e cria o explicador SHAP
model = train_model(df_enriquecido)
explainer = create_explainer(model)

# LÓGICA DA INTERFACE E APLICAÇÃO

# Título e introdução do painel
st.title("🚨 Red Queen: Painel de Controle de Ameaças Biológicas")
st.markdown("Use os controles na barra lateral para rodar a simulação ou envie um novo dataset para análise.")

# Inicialização do estado da sessão

# Índice atual da linha sendo analisada
if 'current_index' not in st.session_state: st.session_state.current_index = 0
# Histórico dos dados analisados
if 'data_history' not in st.session_state: st.session_state.data_history = pd.DataFrame()
# Estado da simulação (ativa ou pausada)
if 'running' not in st.session_state: st.session_state.running = False

# Dataset ativo (default = enriquecido com anomalia reforçada)
if 'active_df' not in st.session_state:
    st.session_state.active_df = df_enriquecido

# SIDEBAR: Upload, Sobre, e Controles da Simulação

# Informações sobre o projeto
with st.sidebar:
    with st.expander("ℹ️ Sobre o Projeto"):
        st.markdown("""
        Este painel é uma simulação do **Projeto Red Queen**, inspirado no universo de Resident Evil. 

        Um modelo de Inteligência Artificial (XGBoost) monitora os sensores de uma instalação de biorrisco para detectar anomalias em tempo real, como vazamentos do T-Virus.

        **Funcionalidades:**
        - **Simulação Ativa:** Acompanhe os dados de exemplo sendo analisados em tempo real.
        - **Análise de Novos Datasets:** Envie seu próprio arquivo CSV para que a Red Queen o analise.
        - **Análise de Causa Raiz:** Quando uma anomalia é detectada, um modelo SHAP explica os fatores que levaram ao alerta.
        """)

# Upload de novos dados CSV para análise
with st.sidebar:
    st.header("Analisar Novo Dataset")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV para simulação", type="csv")

    # Botão para simular com novo arquivo enviado
    if st.button("Simulação com Novo Dataset", type="primary"):
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                # Verificação se o arquivo possui as colunas esperadas
                expected_columns = ['T_Virus_Level', 'Room_Temperature', 'Humidity', 'Gas_Leak_Level',
                                    'Security_Clearance', 'AI_Override_Attempts', 'Proximity_To_Core']
                if all(col in new_df.columns for col in expected_columns):
                    # Troca o dataset ativo e reinicia simulação
                    st.session_state.active_df = new_df
                    st.session_state.current_index = 0
                    st.session_state.data_history = pd.DataFrame()
                    st.session_state.running = True
                    st.rerun()
                else:
                    st.error("O arquivo enviado não contém as colunas esperadas.")
            except Exception as e:
                st.error(f"Erro ao ler o arquivo: {e}")
        else:
            st.warning("Por favor, envie um arquivo primeiro.")

    # Botões de controle da simulação
    st.divider()
    st.header("Controles da Simulação Ativa")
    if st.button("Inicia Dataset de Treino ⏯️"):
        st.session_state.active_df = df_enriquecido
        st.session_state.running = True
        st.rerun()
    if st.button("Pausar Simulação ⏸️"):
        st.session_state.running = False
        st.rerun()
    if st.button("Resetar Simulação 🔄"):
        st.session_state.active_df = df_enriquecido
        st.session_state.current_index = 0
        st.session_state.data_history = pd.DataFrame()
        st.session_state.running = False
        st.rerun()


# --- PAINEL PRINCIPAL ---
st.header("Resultado da Simulação")
col1, col2 = st.columns([3, 1])

# Pega o dataset ativo (pode ser o original ou o enviado)
active_dataframe = st.session_state.active_df

# Se passar do fim do dataset, volta ao início
if st.session_state.current_index >= len(active_dataframe):
    st.session_state.current_index = 0

# Captura a linha atual
current_data_row = active_dataframe.iloc[[st.session_state.current_index]]
X_current = current_data_row.drop('Anomaly', axis=1, errors='ignore')

# Realiza a previsão da IA
if model:
    feature_names = model.get_booster().feature_names
    X_current = X_current[feature_names]
    prediction = model.predict(X_current)[0]
else:
    prediction = 0

# Armazena a linha no histórico
st.session_state.data_history = pd.concat([st.session_state.data_history, current_data_row])

# COLUNA 2: STATUS DA SIMULAÇÃO
with col2:
    st.subheader("Status da Simulação")
    status_placeholder = st.empty()
    if prediction == 0:
        status_placeholder.success(f"✅ SEGURO (Índice: {st.session_state.current_index})")
    else:
        # Em caso de anomalia, pausa a simulação
        st.session_state.running = False
        prediction_proba = model.predict_proba(X_current)[0]
        probability_of_anomaly = prediction_proba[1] * 100
        status_placeholder.error(
            f"🚨 ALERTA DE ANOMALIA! (Índice: {st.session_state.current_index})\n\nConfiança: {probability_of_anomaly:.2f}%")
        # Botão para seguir após alerta
        if st.button("Continuar Simulação ➡️"):
            st.session_state.current_index += 1
            st.session_state.running = True
            st.rerun()

# COLUNA 1: DADOS E HISTÓRICO DOS SENSORES
with col1:
    st.subheader("Leituras Atuais dos Sensores")
    st.dataframe(current_data_row, use_container_width=True)
    st.header("Histórico de Sensores Críticos")
    chart_col1, chart_col2 = st.columns(2)

    # Gráficos de linha com últimos valores
    with chart_col1:
        st.subheader("Nível do T-Virus")
        st.line_chart(st.session_state.data_history['T_Virus_Level'].tail(100))
        st.subheader("Umidade")
        st.line_chart(st.session_state.data_history['Humidity'].tail(100))
    with chart_col2:
        st.subheader("Temperatura da Sala")
        st.line_chart(st.session_state.data_history['Room_Temperature'].tail(100))
        st.subheader("Nível de Vazamento de Gás")
        st.line_chart(st.session_state.data_history['Gas_Leak_Level'].tail(100))

    # Se for anomalia, mostra explicação com SHAP
    if prediction == 1 and explainer is not None:
        st.header("Análise da Causa Raiz (SHAP)")

        # O waterfall_plot é mais fácil de ler para uma única previsão
        shap_values_obj = explainer(X_current)

        # Precisamos criar uma figura Matplotlib para exibir no Streamlit
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values_obj[0], show=False)
        st.pyplot(plt.gcf())
        plt.clf()  # Limpa a figura para a próxima execução

# CONTROLE DO LOOP AUTOMÁTICO
if st.session_state.running:

    # Avança para a próxima linha a cada segundo
    st.session_state.current_index = (st.session_state.current_index + 1) % len(active_dataframe)

    # Zera o histórico ao reiniciar
    if st.session_state.current_index == 0:
        st.session_state.data_history = pd.DataFrame()

    # Espera 1 segundo e reinicia a interface
    time.sleep(1)
    st.rerun()

