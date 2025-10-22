"""
Análise principal de sódio usando a biblioteca Chemometrics reorganizada.

Este script demonstra como usar a nova estrutura modular para análise quimiométrica.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chemometrics import (
    SNVPreprocessor, 
    SavitzkyGolayPreprocessor, 
    MeanCenterPreprocessor,
    PLSRegressor,
    DataSplitter,
    MetricsCalculator
)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Função principal da análise."""
    print("=== ANÁLISE DE SÓDIO - BIBLIOTECA CHEMOMETRICS ===\n")
    
    try:
        # 1. Carregar dados
        logger.info("Carregando dados...")
        dados_brutos = pd.read_excel("data/dados.xlsx", sheet_name='original', header=None)
        
        # Extrair dados
        X = dados_brutos.iloc[1:, 6:].to_numpy(dtype=float)     
        y = dados_brutos.iloc[1:, 2].to_numpy(dtype=float)   
        amostras = dados_brutos.iloc[1:, 1].to_numpy()    
        variaveis = dados_brutos.iloc[0, 6:].to_numpy(dtype=float)      
        
        # Remover variáveis (manter apenas 1459)
        X = X[:, 1:1460]             
        variaveis = variaveis[:1459]  
        
        logger.info(f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} variáveis")
        
        # 2. Pré-processamento
        logger.info("Aplicando pré-processamento...")
        
        # Savitzky-Golay
        savgol = SavitzkyGolayPreprocessor(window_length=15, polyorder=2, deriv=1, plot=False)
        X_processed = savgol.fit_transform(X)
        
        # SNV
        snv = SNVPreprocessor(plot=False)
        X_processed = snv.fit_transform(X_processed)
        
        # Mean Centering
        mean_center = MeanCenterPreprocessor(plot=False)
        X_processed = mean_center.fit_transform(X_processed)
        
        logger.info("Pré-processamento concluído")
        
        # 3. Divisão dos dados
        logger.info("Dividindo dados...")
        splitter = DataSplitter(test_size=0.3, random_state=28)
        train_idx, test_idx = splitter.split_with_groups(X_processed, y, groups=amostras)
        
        # Dividir os dados
        X_train = X_processed[train_idx]
        X_test = X_processed[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        logger.info(f"Dados divididos: {len(train_idx)} treino, {len(test_idx)} teste")
        
        # 4. Treinar modelo PLS
        logger.info("Treinando modelo PLS...")
        model = PLSRegressor()
        model.fit(X_train, y_train, cv_folds=5, optimize_components=True)
        
        # 5. Avaliar modelo
        logger.info("Avaliando modelo...")
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
        
        print("\n=== ANÁLISE CONCLUÍDA COM SUCESSO ===")
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        raise


if __name__ == "__main__":
    main()
