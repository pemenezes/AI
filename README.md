# Manutenção Preditiva com Redes Neurais (MLP)

Este projeto utiliza uma rede neural multicamadas (Multi-Layer Perceptron) para classificar tipos de falhas em equipamentos industriais com base em sensores de temperatura, torque e rotação.

## 🚀 Funcionalidades

*   **Processamento de Dados:** Mapeamento de variáveis categóricas (Tipos de Falha e Tipos de Produto) para formatos numéricos.
*   **Treinamento de Rede Neural:** Implementação de um classificador MLP com arquitetura de três camadas ocultas (300, 150, 50).
*   **Análise de Performance:** Avaliação de acurácia e geração de matriz de confusão para identificar padrões de erro.
*   **Inferência:** Teste do modelo com amostras específicas para validação de cenários reais.

## 🛠️ Tecnologias Utilizadas

*   **Python 3**
*   **Pandas:** Manipulação e limpeza dos dados do dataset `manutencao_preditiva.csv`.
*   **Scikit-Learn:** Divisão de dados, implementação do `MLPClassifier` e métricas de avaliação.
*   **Matplotlib & Seaborn:** Visualização da matriz de confusão e comportamento do modelo.
*   **NumPy:** Operações matemáticas e formatação de arrays para predição.

## 📊 Estrutura do Modelo

O modelo foi configurado com os seguintes parâmetros:
*   **Camadas Ocultas:** (300, 150, 50).
*   **Função de Ativação:** `relu`.
*   **Máximo de Iterações:** 1000.
*   **Otimização:** Ativação do modo `verbose` para acompanhar a redução da perda durante o treino.

## 📂 Como Usar

1.  Certifique-se de que o arquivo `manutencao_preditiva.csv` está no mesmo diretório do script.
2.  Instale as bibliotecas necessárias:
```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```
