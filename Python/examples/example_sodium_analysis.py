"""
Exemplo de análise de sódio usando a biblioteca Chemometrics.

Este exemplo demonstra como usar a biblioteca para:
1. Carregar dados espectrais
2. Aplicar pré-processamento
3. Dividir os dados
4. Treinar modelo PLS
5. Avaliar resultados
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemometrics import (
    SNVPreprocessor, 
    SavitzkyGolayPreprocessor, 
    MeanCenterPreprocessor,
    PLSRegressor,
    DataSplitter,
    MetricsCalculator
)

# Configurar logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: str = "data/dados.xlsx") -> tuple:
    """
    Carrega os dados do arquivo Excel.
    
    Parameters
    ----------
    file_path : str
        Caminho para o arquivo de dados.
        
    Returns
    -------
    tuple
        X, y, amostras, variaveis
    """
    logger.info("Carregando dados...")
    
    dados_brutos = pd.read_excel(file_path, sheet_name='original', header=None)
    
    # Extrair dados
    X = dados_brutos.iloc[1:, 6:].to_numpy(dtype=float)     
    y = dados_brutos.iloc[1:, 2].to_numpy(dtype=float)   
    amostras = dados_brutos.iloc[1:, 1].to_numpy()    
    variaveis = dados_brutos.iloc[0, 6:].to_numpy(dtype=float)      
    
    # Remover variáveis (manter apenas 1459)
    X = X[:, 1:1460]             
    variaveis = variaveis[:1459]  
    
    logger.info(f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} variáveis")
    
    return X, y, amostras, variaveis


def preprocess_data(X: np.ndarray, plot: bool = False) -> np.ndarray:
    """
    Aplica pré-processamento aos dados espectrais.
    
    Parameters
    ----------
    X : np.ndarray
        Dados espectrais.
    plot : bool, default=False
        Se True, plota os resultados.
        
    Returns
    -------
    np.ndarray
        Dados pré-processados.
    """
    logger.info("Aplicando pré-processamento...")
    
    # 1. Savitzky-Golay (suavização e derivação)
    savgol = SavitzkyGolayPreprocessor(
        window_length=15, 
        polyorder=2, 
        deriv=1, 
        plot=plot
    )
    X_processed = savgol.fit_transform(X)
    
    # 2. SNV (Standard Normal Variate)
    snv = SNVPreprocessor(plot=plot)
    X_processed = snv.fit_transform(X_processed)
    
    # 3. Mean Centering
    mean_center = MeanCenterPreprocessor(plot=plot)
    X_processed = mean_center.fit_transform(X_processed)
    
    logger.info("Pré-processamento concluído")
    
    return X_processed


def split_data(X: np.ndarray, y: np.ndarray, amostras: np.ndarray) -> tuple:
    """
    Divide os dados em treino e teste.
    
    Parameters
    ----------
    X : np.ndarray
        Dados de entrada.
    y : np.ndarray
        Variável resposta.
    amostras : np.ndarray
        IDs das amostras.
        
    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, train_idx, test_idx
    """
    logger.info("Dividindo dados...")
    
    # Usar DataSplitter com grupos (réplicas)
    splitter = DataSplitter(test_size=0.3, random_state=28)
    train_idx, test_idx = splitter.split_with_groups(X, y, groups=amostras)
    
    # Dividir os dados
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    logger.info(f"Dados divididos: {len(train_idx)} treino, {len(test_idx)} teste")
    
    return X_train, X_test, y_train, y_test, train_idx, test_idx


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> PLSRegressor:
    """
    Treina o modelo PLS.
    
    Parameters
    ----------
    X_train : np.ndarray
        Dados de treino.
    y_train : np.ndarray
        Resposta de treino.
        
    Returns
    -------
    PLSRegressor
        Modelo treinado.
    """
    logger.info("Treinando modelo PLS...")
    
    # Criar e treinar modelo
    model = PLSRegressor()
    model.fit(X_train, y_train, cv_folds=5, optimize_components=True)
    
    logger.info("Modelo treinado com sucesso")
    
    return model


def evaluate_model(model: PLSRegressor, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Avalia o modelo e exibe resultados.
    
    Parameters
    ----------
    model : PLSRegressor
        Modelo treinado.
    X_train : np.ndarray
        Dados de treino.
    y_train : np.ndarray
        Resposta de treino.
    X_test : np.ndarray
        Dados de teste.
    y_test : np.ndarray
        Resposta de teste.
        
    Returns
    -------
    dict
        Métricas de avaliação.
    """
    logger.info("Avaliando modelo...")
    
    # Avaliar modelo
    metrics = model.evaluate(X_train, y_train, X_test, y_test, unit='mEq/L')
    
    # Exibir métricas
    MetricsCalculator.print_metrics(metrics, "Métricas do Modelo PLS")
    
    # Plotar resultados
    model.plot_results(X_train, y_train, X_test, y_test, unit='mEq/L')
    
    # Plotar VIP scores
    model.plot_vip_scores(threshold=1.0)
    
    # Criar DataFrame com resultados
    df_results = model.get_results_dataframe(X_train, y_train, X_test, y_test)
    print("\nPrimeiras linhas dos resultados:")
    print(df_results.head())
    
    return metrics


def main():
    """Função principal do exemplo."""
    print("=== ANÁLISE DE SÓDIO COM CHEMOMETRICS ===\n")
    
    try:
        # 1. Carregar dados
        X, y, amostras, variaveis = load_data()
        
        # 2. Pré-processamento
        X_processed = preprocess_data(X, plot=False)
        
        # 3. Divisão dos dados
        X_train, X_test, y_train, y_test, train_idx, test_idx = split_data(
            X_processed, y, amostras
        )
        
        # 4. Treinar modelo
        model = train_model(X_train, y_train)
        
        # 5. Avaliar modelo
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        print("\n=== ANÁLISE CONCLUÍDA COM SUCESSO ===")
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        raise


if __name__ == "__main__":
    main()
