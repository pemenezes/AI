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
df["Tipo de falha num"] = df["Tipo de falha"].map(map_falha)

#Transformação das informações da coluna tipo
map_falha = {
    "H" : 0,
    "L" : 1,
    "M" : 2
}

#Criando uma nova coluna com valores numéricos
df["Tipo num"] = df["Tipo"].map(map_falha)

#Separação dos dados para treino e teste
x = df[['Tipo num',
        'Temperatura ar [K]',
        'Temperatura processo [K]',
        'Velocidade rotação [rpm]',
        'Torque [Nm]',
        ]]

y = df[['Tipo de falha']]

#Separação dos dados para treino e teste
from sklearn.model_selection import train_test_split

X_treino, X_teste, Y_treino, Y_teste = train_test_split(x,y,test_size = 0.3, random_state=42)