import pandas as pd         # Manipulação e análise de dados (para carregar o CSV)
import numpy as np          # Operações numéricas (para calcular os 'bins' dos histogramas)

# PARÂMETROS DA GERAÇÃO
NUM_AMOSTRAS = 50
PCT_ANOMALIAS = 0.20

# DEFINIÇÃO DA "NORMALIDADE"
NORMAL_TEMP_MEDIA = 21.5
NORMAL_TEMP_STD = 1.5
NORMAL_HUM_MEDIA = 48.0
NORMAL_HUM_STD = 5.0

def gerar_dados_normais(n_amostras):
    """Gera um DataFrame com dados de operação normal."""

    # MUDANÇA NA GERAÇÃO DE PROXIMIDADE
    # 90% dos dados estarão na faixa "dia a dia" (10m a 40m)
    n_dia_a_dia = int(n_amostras * 0.9)

    # 10% estarão na faixa "visita ao núcleo" (0m a 10m)
    n_perto_nucleo = n_amostras - n_dia_a_dia

    # Geração dos dados de proximidade: principal faixa usa distribuição normal
    prox_dia_a_dia = np.random.normal(22, 5, n_dia_a_dia)
    prox_perto_nucleo = np.random.uniform(0, 10, n_perto_nucleo)

    # Junta os dois tipos de distância
    prox_completa = np.concatenate([prox_dia_a_dia, prox_perto_nucleo])
    np.random.shuffle(prox_completa)
    prox_completa[prox_completa < 0] = 0

    # Geração de todas as demais variáveis em operação normal
    data = {
        'T_Virus_Level': np.random.uniform(0, 8, n_amostras).round(4),
        'Room_Temperature': np.random.normal(NORMAL_TEMP_MEDIA, NORMAL_TEMP_STD, n_amostras).round(4),
        'Humidity': np.random.normal(NORMAL_HUM_MEDIA, NORMAL_HUM_STD, n_amostras).round(4),
        'Gas_Leak_Level': np.random.uniform(0, 0.2, n_amostras).round(4),
        'Security_Clearance': np.random.randint(1, 5, n_amostras),
        'AI_Override_Attempts': np.random.randint(0, 2, n_amostras),
        'Proximity_To_Core': prox_completa.round(4),  # Usando a nova proximidade realista
        'Anomaly': np.zeros(n_amostras, dtype=int)
    }
    return pd.DataFrame(data)


# DEFINIÇÃO DAS "RECEITAS DAS CRISES"
def criar_anomalia_caotica():
    """Gera uma anomalia caótica, que ocorre perto do núcleo."""
    return {
        'T_Virus_Level': np.random.uniform(85, 100),        # Níveis extremamente altos
        'Room_Temperature': np.random.uniform(45, 60),      # Superaquecimento
        'Humidity': np.random.uniform(90, 100),             # Umidade quase saturada
        'Gas_Leak_Level': np.random.uniform(3, 5),          # Vazamento intenso de gás
        'AI_Override_Attempts': np.random.randint(5, 15),   # Muitas tentativas de controle
        'Proximity_To_Core': np.random.uniform(0, 10)       # Próximo ao núcleo
    }


def criar_anomalia_silenciosa_t_virus():
    """Gera uma anomalia de T-Virus, que ocorre perto do núcleo."""
    return {
        'T_Virus_Level': np.random.uniform(13, 100),   # Foco apenas no T-Virus
        'Proximity_To_Core': np.random.uniform(0, 15)  # Também acontece perto do núcleo
    }


# GERAR A BASE DE DADOS "NORMAL"
print("Gerando base de dados com operações normais e realistas...")
df = gerar_dados_normais(NUM_AMOSTRAS)

# INJETAR O CAOS (ADICIONAR AS ANOMALIAS)
num_anomalias = int(NUM_AMOSTRAS * PCT_ANOMALIAS) # Quantidade de anomalias a serem inseridas
print(f"Gerando e injetando {num_anomalias} anomalias em locais de risco...")

# Escolhe aleatoriamente os índices das linhas onde as anomalias serão inseridas
indices_anomalia = np.random.choice(df.index, num_anomalias, replace=False)

# Itera pelos índices sorteados e aplica anomalias caóticas ou silenciosas
for i in indices_anomalia:
    if np.random.rand() > 0.4:
        anomalia = criar_anomalia_caotica()# 60% de chance de ser caótica
    else:
        anomalia = criar_anomalia_silenciosa_t_virus() # 40% de chance de ser silenciosa de T-Virus

    # Aplica os valores da anomalia nas colunas correspondentes
    for chave, valor in anomalia.items():
        df.loc[i, chave] = valor

    # Marca a linha como anomalia (1)
    df.loc[i, 'Anomaly'] = 1

# MISTURAR E SALVAR
print("Misturando o dataset final...")
df_final = df.sample(frac=1).reset_index(drop=True) # Embaralha as linhas para não ficar previsível

# Salva em arquivo CSV
nome_arquivo = 'Red_Queen_Dataset_Realista_v3.csv'
df_final.to_csv(nome_arquivo, index=False)  # Mantendo o padrão de vírgula

# Resumo no terminal
print(f"\nDataset '{nome_arquivo}' gerado com sucesso!")
print(f"Total de linhas: {len(df_final)}")
print("Distribuição de anomalias:")
print(df_final['Anomaly'].value_counts())