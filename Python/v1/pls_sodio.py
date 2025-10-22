import pandas as pd
import numpy as np
import sys

from pretrat import savgol_, snv_, mean_center_
from pca_model import pca_
from pls_model import pls_
from splits import split_data_



# ---- Leitura dos dados ----
dados_brutos = pd.read_excel("dados.xlsx", sheet_name='original', header=None)


X = dados_brutos.iloc[1:, 6:].to_numpy(dtype=float)     
y = dados_brutos.iloc[1:, 2].to_numpy(dtype=float)   
Classe_Faixa = dados_brutos.iloc[1:, 3].to_numpy(dtype=float)  
Amostras = dados_brutos.iloc[1:, 1].to_numpy()    
Variaveis = dados_brutos.iloc[0, 6:].to_numpy(dtype=float)      
Diluida = dados_brutos.iloc[1:, 4].to_numpy()       

# ---- Filtro: manter apenas "Não" em Diluida ----
#mask = (Diluida == "Não")
#X2 = X[mask, :]
#y2 = y[mask]
#Classe_Faixa2 = Classe_Faixa[mask]
#Amostras2 = Amostras[mask]
X2 = X
y2 = y

# ---- Remoção de variáveis ----
X2 = X2[:, 1:1460]             
Variaveis2 = Variaveis[:1459]  

#Pré-processamento dos dados

#Aplica Savitzky-Golay, SNV e centramento na média
X_p1 = savgol_(X2, width=15, order=2, deriv=1, plotar=False)
X_p1 = snv_(X_p1, plotar=False)
X_p1 = mean_center_(X_p1, plotar=False)




##Aplicação de PLS 

grupos = np.array(Amostras)
#Divisão de treino e teste (Com réplicas) #28
train_idx, test_idx = split_data_(X_p1, y2, percentual=0.3, random_state=28, rep=grupos)

# Dividindo os conjuntos de treino e teste
X_train = X_p1[train_idx]
X_test = X_p1[test_idx]
y_train = y2[train_idx]
y_test = y2[test_idx]

# Dividindo os nomes das amostras entre treino e teste
Amostras_treino = Amostras[train_idx]
Amostras_teste = Amostras[test_idx]


# Chamada da função pls_ com os conjuntos de treino e teste
coef, metrics, df_results = pls_(
    X_cal=X_train, 
    X_test=X_test, 
    y_cal=y_train, 
    y_test=y_test, 
    n_components=10,
    unidade_medida='mEq/L',
    var_select=None, 
    plotar=True
)


print("Métricas do Modelo PLS:")
print(metrics)
print("\nResultados (Previsto vs. Real):")
print(df_results)