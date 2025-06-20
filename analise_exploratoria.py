# Importação das bibliotecas necessárias
import pandas as pd               # Manipulação e análise de dados (para carregar o CSV)
import matplotlib.pyplot as plt   # Visualização de gráficos (para criar histogramas)
import numpy as np                # Operações numéricas (para calcular os 'bins' dos histogramas)

# Tenta carregar o dataset a partir do caminho especificado
try:
    df = pd.read_csv("Red_Queen_Synthetic_Dataset.csv", encoding='utf-8')
except FileNotFoundError:
    # Mensagem de erro se o arquivo não for encontrado
    print("Erro: O arquivo 'Red_Queen_Synthetic_Dataset.csv' não foi encontrado.")
    print("Verifique se o caminho do arquivo está correto ou coloque o arquivo na mesma pasta do script.")
    exit()  # Encerra a execução do script

# Lista de colunas numéricas contínuas
colunas_numericas = ['T_Virus_Level', 'Room_Temperature', 'Humidity', 'Gas_Leak_Level', 'Proximity_To_Core']

print("Gerando histogramas para colunas numéricas contínuas...")

# Para cada coluna da lista acima, cria um histograma
for nome_coluna in colunas_numericas:
    plt.figure(figsize=(8, 6))  # Cria uma nova figura de 8x6 polegadas

    # Cria histograma com 30 'bins' (intervalos)
    plt.hist(df[nome_coluna], bins=30, color='skyblue', edgecolor='black')

    # Define título e rótulos dos eixos
    plt.title(f'Distribuição de {nome_coluna}')
    plt.xlabel(nome_coluna)
    plt.ylabel('Frequência')

    # Ajusta o layout para que os rótulos não fiquem cortados
    plt.tight_layout()

    # Exibe o gráfico
    plt.show()

print("Histogramas das colunas contínuas gerados com sucesso.\n")

# Lista de colunas com valores inteiros (variáveis discretas)
colunas_inteiras = ['Security_Clearance', 'AI_Override_Attempts', 'Anomaly']

print("Gerando histogramas para colunas de valores inteiros...")

# Para cada coluna inteira, cria um histograma com tratamento especial
for nome_coluna in colunas_inteiras:
    plt.figure(figsize=(8, 6))  # Cria uma nova figura

    dados_coluna = df[nome_coluna]          # Pega os dados da coluna
    valores_unicos = dados_coluna.nunique() # Conta quantos valores diferentes existem

    # Se tiver poucos valores únicos, ajusta os 'bins' para que cada barra represente um número inteiro
    if valores_unicos <= 10:
        min_val = dados_coluna.min()
        max_val = dados_coluna.max()
        bins_config = np.arange(min_val - 0.5, max_val + 1.5, 1)  # Centraliza os inteiros nos bins
    else:
        bins_config = 20  # Para colunas com muitos valores únicos, usa 20 bins fixos

    # Cria o histograma com a configuração de bins escolhida
    plt.hist(dados_coluna, bins=bins_config, color='lightcoral', edgecolor='black', rwidth=0.8)

    # Títulos e rótulos
    plt.title(f'Distribuição de {nome_coluna}')
    plt.xlabel(nome_coluna)
    plt.ylabel('Frequência')

    # Se os valores forem inteiros e poucos, força os ticks no eixo X a serem inteiros
    if valores_unicos <= 10:
        plt.xticks(np.arange(min_val, max_val + 1, 1))

    # Ajusta o layout
    plt.tight_layout()

    # Exibe o histograma
    plt.show()

print("Histogramas das colunas de inteiros gerados com sucesso.")
print("\nAnálise concluída!")
