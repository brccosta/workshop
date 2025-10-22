import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from chemotools.baseline import AirPls



#Pré-processamento SNV
def snv_(X, plotar=False):
    """
    Equivalente no MATLAB:
    X3 = (X2 - mean(X2')' * ones(1, size(X2,2))) ./ (std(X2')' * ones(1, size(X2,2)));
    """
    # Calcula a média de cada LINHA (cada espectro)
    media_linha = np.mean(X, axis=1, keepdims=True)
    #media_linha = np.mean(X_Corte, axis=1, keepdims=True)


    # Calcula o desvio padrão de cada LINHA  
    dp_linha = np.std(X, axis=1, keepdims=True, ddof=1)
    #dp_linha = np.std(X_Corte, axis=1, keepdims=True)


    # Aplica o SNV: (X - média) / desvio_padrão para cada espectro
    X_SNV = (X - media_linha) / dp_linha
    #X_p1 = (X_Corte - media_linha) / dp_linha
    if plotar == True:
        plotar_(X, X_SNV, 'SNV')
    return X_SNV

#Pré-processamento Savitzky-Golay
def savgol_(X, width=None, order=None, deriv=None, plotar=False):
    
    X_smooth = savgol_filter(X, width, order, deriv)
    if plotar == True:
        plotar_(X, X_smooth, f'Savitzky-Golay (width={width}, order={order}, deriv={deriv})')
    return X_smooth

#Pré-processamento MSC
def msc_(X, reference=None, plotar=False):
    """
    Equivalente no MATLAB:
    X3 = (X2 - mean(X2')' * ones(1, size(X2,2))) ./ (std(X2')' * ones(1, size(X2,2)));
    """
    X = np.array(X, dtype=np.float64)
    
    if reference is None:
        reference = np.mean(X, axis=0)
    
    # Centralizar os dados
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    reference_centered = reference - np.mean(reference)
    
    # Calcular os coeficientes de regressão
    coeffs = np.linalg.lstsq(reference_centered[:, np.newaxis], X_centered.T, rcond=None)[0]
    
    # Ajustar os dados
    X_msc = (X_centered - coeffs.T * reference_centered) + np.mean(X, axis=1, keepdims=True)
    if plotar == True:
        plotar_(X, X_msc, 'MSC')
    return X_msc

#Pré-processamento Centrar na média
def mean_center_(X, plotar=False):
    """
    Equivalente no MATLAB:
    X3 = (X2 - mean(X2')' * ones(1, size(X2,2))) ./ (std(X2')' * ones(1, size(X2,2)));
    """
    X = np.array(X, dtype=np.float64)
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    if plotar == True:
        plotar_(X, X_centered, 'Mean Centering')
    return X_centered


#Pré-processamento EMSC
def emsc_(X, reference=None, order=2, plotar=False):
    """
    Implementação do EMSC (Extended Multiplicative Signal Correction).
    
    Parâmetros:
    X : array-like, shape (n_samples, n_features)
        Matriz de dados espectrais.
    reference : array-like, shape (n_features,), optional
        Espectro de referência. Se None, a média dos espectros será usada.
    order : int, optional
        Ordem do polinômio para modelar a variação de base. Padrão é 2.
    
    Retorna:
    X_emsc : array-like, shape (n_samples, n_features)
        Matriz de dados após aplicação do EMSC.
    """
    X = np.array(X, dtype=np.float64)
    
    if reference is None:
        reference = np.mean(X, axis=0)
    
    n_samples, n_features = X.shape
    
    # Construir a matriz de design
    P = np.vander(np.linspace(-1, 1, n_features), N=order + 1, increasing=True)
    P = np.hstack((np.ones((n_features, 1)), P))  # Adicionar coluna de uns para o intercepto
    P = np.hstack((P, reference[:, np.newaxis]))  # Adicionar espectro de referência
    
    # Inicializar a matriz de saída
    X_emsc = np.zeros_like(X)
    
    for i in range(n_samples):
        # Resolver o sistema linear para cada espectro
        coeffs = np.linalg.lstsq(P, X[i, :], rcond=None)[0]
        
        # Reconstruir o espectro corrigido
        fitted = P @ coeffs
        X_emsc[i, :] = X[i, :] - (fitted - coeffs[-1] * reference)
    if plotar == True:
        plotar_(X, X_emsc, 'EMSC')
    return X_emsc


#Pré-processamento Autoescalamento
def autoscale_(X, plotar=False):
    """
    Equivalente no MATLAB:
    X3 = (X2 - mean(X2')' * ones(1, size(X2,2))) ./ (std(X2')' * ones(1, size(X2,2)));
    """
    X = np.array(X, dtype=np.float64)
    X_autoscaled = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True, ddof=1)
    if plotar == True:
        plotar_(X, X_autoscaled, 'Autoscaling')
    return X_autoscaled


def airPLS_(X, lambda_=10e7, order=2, wep=0.1, p=0.05, itermax=20, plotar=False):
    '''
    airPLS.py Copyright 2014 Renato Lombardo - renato.lombardo@unipa.it
    Baseline correction using adaptive iteratively reweighted penalized least squares

    This program is a translation in python of the R source code of airPLS version 2.0
    by Yizeng Liang and Zhang Zhimin - https://code.google.com/p/airpls
    Reference:
    Z.-M. Zhang, S. Chen, and Y.-Z. Liang, Baseline correction using adaptive iteratively reweighted penalized least squares. Analyst 135 (5), 1138-1146 (2010).

    Description from the original documentation:

    Baseline drift always blurs or even swamps signals and deteriorates analytical results, particularly in multivariate analysis.  It is necessary to correct baseline drift to perform further data analysis. Simple or modified polynomial fitting has been found to be effective in some extent. However, this method requires user intervention and prone to variability especially in low signal-to-noise ratio environments. The proposed adaptive iteratively reweighted Penalized Least Squares (airPLS) algorithm doesn't require any user intervention and prior information, such as detected peaks. It iteratively changes weights of sum squares errors (SSE) between the fitted baseline and original signals, and the weights of SSE are obtained adaptively using between previously fitted baseline and original signals. This baseline estimator is general, fast and flexible in fitting baseline.


    LICENCE
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>
    '''

    
    """
    Baseline correction using adaptive iteratively reweighted Penalized Least Squares
    Equivalente à função airPLS do MATLAB
    
    Input:
        X: matriz de espectros (size m*n, m é amostra e n é variável)
        lambda_: parâmetro ajustável. Quanto maior, mais suave será a baseline
        order: ordem da diferença das penalidades
        wep: proporção de exceção de peso no início e fim
        p: parâmetro de assimetria para início e fim
        itermax: número máximo de iterações
        
    Output:
        Xc: espectros corrigidos (size m*n)
        Z: baseline estimada (size m*n)
    """
    from scipy.sparse import eye, diags, csc_matrix
    from scipy.sparse.linalg import spsolve
    
    X = np.array(X, dtype=np.float64)
    
    # Se for um vetor 1D, converter para matriz 2D
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    m, n = X.shape
    
    # Calcular índices das extremidades (equivalente ao MATLAB)
    first_part = np.arange(0, int(np.ceil(n * wep)))
    second_part = np.arange(int(np.floor(n - n * wep)), n)
    wi = np.concatenate([first_part, second_part])
    
    # Criar matriz de diferenças D (equivalente ao diff(speye(n), order) do MATLAB)
    E = eye(n, format='csc')
    for i in range(order):
        E = E[1:] - E[:-1]
    D = E
    
    # Calcular DD = lambda * D' * D
    DD = lambda_ * (D.T * D)
    
    # Inicializar matriz de baseline
    Z = np.zeros_like(X)
    
    # Processar cada espectro (equivalente ao loop do MATLAB)
    for i in range(m):
        w = np.ones(n)
        x = X[i, :]
        
        for j in range(1, itermax + 1):
            # Criar matriz diagonal de pesos W
            W = diags(w, 0, shape=(n, n), format='csc')
            
            # Resolver sistema linear: (W + DD) * z = W * x
            A = W + DD
            b = W * x
            z = spsolve(A, b)
            
            # Calcular diferença
            d = x - z
            dssn = np.abs(np.sum(d[d < 0]))
            
            # Critério de convergência (equivalente ao MATLAB)
            if dssn < 0.001 * np.sum(np.abs(x)):
                break
            
            # Atualizar pesos (equivalente ao MATLAB)
            w[d >= 0] = 0                                    # Pontos acima da baseline
            w[wi] = p                                        # Extremidades com peso p
            w[d < 0] = j * np.exp(np.abs(d[d < 0]) / dssn)  # Pontos abaixo da baseline
        
        Z[i, :] = z
    
    # Calcular espectros corrigidos
    Xc = X - Z
    
    if plotar:
        plotar_(X, Xc, f'airPLS (λ={lambda_:.0e})')
    
    return Xc, Z


#Função para plotar os vetores originais e processados
def plotar_(X_original, X_processado, titulo_processamento):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6), dpi=300)
    plt.subplot(2, 1, 1)
    plt.plot(X_original[1, :].T, color='black')
    plt.title('Vetor original')
    #plt.xlabel('Comprimento de Onda')
    #plt.ylabel('Absorbância')
    plt.subplot(2, 1, 2)
    plt.plot(X_processado[1, :].T, color='green')
    plt.title(f'Processamento {titulo_processamento}')
    #plt.xlabel('Comprimento de Onda')
    #plt.ylabel('Absorbância')
    plt.tight_layout()
    plt.show()