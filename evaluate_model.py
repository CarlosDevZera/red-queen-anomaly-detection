# IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
import pandas as pd                        # Para leitura e manipulação de dados
import xgboost as xgb                      # Para usar o classificador XGBoost
import seaborn as sns                     # Para visualização de gráficos avançados (como heatmaps)
import matplotlib.pyplot as plt           # Para criação de gráficos
from sklearn.model_selection import train_test_split    # Para dividir os dados em treino e teste
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score  # Para avaliação do modelo

# DEFINIÇÃO DA FUNÇÃO PRINCIPAL
def evaluate_red_queen_model():

    print("Iniciando o protocolo de avaliação do modelo Red Queen...")

    # CARGA E PREPARAÇÃO DOS DADOS
    print("\n[PASSO 1/5] Carregando e preparando o dataset...")
    try:
        df_original = pd.read_csv("Red_Queen_Synthetic_Dataset.csv")
    except FileNotFoundError:
        # Se o arquivo não for encontrado, exibe erro e encerra
        print("\nERRO: Arquivo 'Red_Queen_Synthetic_Dataset.csv' não encontrado.")
        print("Certifique-se de que o arquivo está na mesma pasta que este script.")
        return

    # INSERÇÃO DE CASOS DE ANOMALIA SILENCIOSA
    # Cria um pequeno DataFrame com um padrão de anomalia sutil (apenas T_Virus_Level elevado)
    novas_anomalias = {
        'T_Virus_Level': [95.0],
        'Room_Temperature': [21.5],
        'Humidity': [48.0],
        'Gas_Leak_Level': [0.0],
        'Security_Clearance': [2],
        'AI_Override_Attempts': [0],
        'Proximity_To_Core': [20.0],
        'Anomaly': [1]
    }

    # Cria 10 cópias da mesma anomalia para reforçar esse padrão no treinamento
    df_novas_anomalias = pd.DataFrame(novas_anomalias)
    df_novas_anomalias_reforcado = pd.concat([df_novas_anomalias] * 10, ignore_index=True)

    # Une os dados originais com os novos dados reforçados
    df_enriquecido = pd.concat([df_original, df_novas_anomalias_reforcado], ignore_index=True)
    print(f"Dataset carregado e enriquecido. Total de {len(df_enriquecido)} registros.")

    # DIVISÃO EM TREINO E TESTE
    print("\n[PASSO 2/5] Dividindo os dados em conjuntos de treino e teste...")

    # Separa variáveis independentes (X) e o alvo (y)
    X = df_enriquecido.drop('Anomaly', axis=1)
    y = df_enriquecido['Anomaly']

    # Divide os dados garantindo a mesma proporção de anomalias em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print("Dados divididos: 75% para treino, 25% para teste.")

    # TREINAMENTO DO MODELO
    print("\n[PASSO 3/5] Treinando o modelo XGBoost...")

    # Calcula a razão entre classes para lidar com desbalanceamento
    if 1 in y_train.value_counts():
        ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
    else:
        ratio = 1  # Caso não tenha nenhuma anomalia (evita divisão por zero)

    # Inicializa o classificador com o peso ajustado para a classe minoritária
    model = xgb.XGBClassifier(
        scale_pos_weight=ratio,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    # Treina o modelo com os dados de treino
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso.")

    # PREVISÕES NO CONJUNTO DE TESTE
    print("\n[PASSO 4/5] Realizando previsões no conjunto de teste (dados não vistos)...")
    y_pred = model.predict(X_test)                          # Predição das classes
    y_pred_proba = model.predict_proba(X_test)[:, 1]        # Probabilidade da classe positiva (Anomaly)

    # AVALIAÇÃO E EXIBIÇÃO DOS RESULTADOS
    print("\n[PASSO 5/5] Exibindo as métricas de performance do modelo:")

    # Exibe o relatório de classificação (precision, recall, f1-score)
    print("\n" + "=" * 40)
    print("      RELATÓRIO DE CLASSIFICAÇÃO")
    print("=" * 40)
    print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Anomalia (1)']))

    # AUC-ROC: métrica que avalia a capacidade do modelo em separar as classes
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print("=" * 40)
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print("=" * 40)
    print("(Quanto mais próximo de 1.0, melhor o modelo em distinguir as classes)")

    # Exibe a matriz de confusão no terminal
    print("\n" + "=" * 40)
    print("         MATRIZ DE CONFUSÃO")
    print("=" * 40)
    cm = confusion_matrix(y_test, y_pred)
    print(f"                    Previsto:")
    print(f"                  Normal | Anomalia")
    print(f"Verdadeiro Normal:   {cm[0][0]:<5} | {cm[0][1]:<5}")
    print(f"Verdadeiro Anomalia: {cm[1][0]:<5} | {cm[1][1]:<5}")
    print("=" * 40)

    # GERAÇÃO DO GRÁFICO DA MATRIZ DE CONFUSÃO
    print("\nGerando gráfico da Matriz de Confusão (salvo como 'matriz_confusao.png')...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Normal', 'Anomalia'],
                yticklabels=['Normal', 'Anomalia'])
    plt.xlabel('Classe Prevista pelo Modelo')
    plt.ylabel('Classe Verdadeira')
    plt.title('Matriz de Confusão - Desempenho em Dados de Teste')
    plt.savefig('matriz_confusao.png')  # Salva a imagem como arquivo PNG
    # plt.show()  # Caso queira abrir a imagem em uma janela, descomente

    print("\nProtocolo de avaliação concluído.")

# EXECUÇÃO DA FUNÇÃO
evaluate_red_queen_model()
