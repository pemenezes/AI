#==============================================================
# Treinamento de um modelo de rede neural multicamadas (MLP)
#===============================================================

import pandas as pd

df = pd.read_csv("manutencao_preditiva.csv", encoding="utf-8")
print(df)

#transformação das informações do tipo de falha:
map_falha = {
    "No failure": 0,
    "Heat Dissipation Failure": 1,
    "Power Failure": 2,
    "Overstrain Failure": 3,
    "Tool Wear Failure": 4,
    "Random Failures": 5
}

#Criando nova coluna com valores nuuméricos
df["Tipo da Falha Num"] = df["Tipo da Falha"].map(map_falha)

#Transformação das informações da coluna tipo
map_falha = {
    "H" : 0,
    "L" : 1,
    "M" : 2
}

#Criando uma nova coluna com valores numéricos
df["Tipo Num"] = df["Tipo"].map(map_falha)

#Separação dos dados para treino e teste
x = df[['Tipo Num',
'Temperatura Ar [K]',
'Temperatura Processo [K]',
'Velocidade Rotacao [rpm]',
'Torque [Nm]']]

y = df[['Tipo da Falha']]

#Separação dos dados para treino e teste
from sklearn.model_selection import train_test_split

X_treino, X_teste, Y_treino, Y_teste = train_test_split(x,y,test_size = 0.3, random_state=42)

# Preparação dos parâmetros do modelo de rede neural multicamadas (MLP)
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(300,150,50), max_iter = 1000, activation='relu', 
                    random_state = 42, verbose=True)

# Avaliação do modelo utilizando os dados de treino e teste
from sklearn.metrics import accuracy_score, confusion_matrix


# 2. TREINAMENTO (A linha que estava faltando!)
# É aqui que o modelo aprende a relação entre os dados e as falhas
mlp.fit(X_treino, Y_treino)

Y_pred = mlp.predict(X_teste)
acuracia = accuracy_score(Y_teste, Y_pred)
matriz_confusao = confusion_matrix(Y_teste, Y_pred)
print("Acurácia: ", acuracia)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Matriz de confusão
cm = confusion_matrix(Y_teste, Y_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y_treino), 
            yticklabels=np.unique(Y_treino))
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()


# Teste do modelo para uma amostra 
# OBSERVAÇÃO: MODELO ERROU AO FAZER A PREVISÃO - Tool Wear Failure
# O que aconteceu com o modelo? Pq não fez a previsão correta?

amostra = np.array([[1, 298.8, 308.9, 1455, 41.3]])

# Fazer previsão
previsao = mlp.predict(amostra)
print("Previsão para a amostra: ", previsao)
previsao 