"""
Demonstração simples da biblioteca Chemometrics (sem plots).
"""

import sys
import os
import numpy as np
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


def demonstrate_preprocessing():
    """Demonstra as técnicas de pré-processamento."""
    print("\n=== DEMONSTRAÇÃO DE PRÉ-PROCESSAMENTO ===")
    
    # Criar dados sintéticos para demonstração
    np.random.seed(42)
    n_samples, n_features = 50, 100
    X = np.random.randn(n_samples, n_features) + np.linspace(0, 10, n_features)
    
    print(f"Dados sintéticos: {X.shape[0]} amostras, {X.shape[1]} variáveis")
    
    # 1. Savitzky-Golay
    print("\n1. Aplicando Savitzky-Golay...")
    savgol = SavitzkyGolayPreprocessor(window_length=15, polyorder=2, deriv=1, plot=False)
    X_savgol = savgol.fit_transform(X)
    print(f"   Dados após Savitzky-Golay: {X_savgol.shape}")
    
    # 2. SNV
    print("\n2. Aplicando SNV...")
    snv = SNVPreprocessor(plot=False)
    X_snv = snv.fit_transform(X_savgol)
    print(f"   Dados após SNV: {X_snv.shape}")
    
    # 3. Mean Centering
    print("\n3. Aplicando Mean Centering...")
    mean_center = MeanCenterPreprocessor(plot=False)
    X_final = mean_center.fit_transform(X_snv)
    print(f"   Dados finais: {X_final.shape}")
    
    return X_final


def demonstrate_data_splitting():
    """Demonstra as técnicas de divisão de dados."""
    print("\n=== DEMONSTRAÇÃO DE DIVISÃO DE DADOS ===")
    
    # Criar dados sintéticos
    np.random.seed(42)
    n_samples, n_features = 100, 50
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    groups = np.repeat(range(20), 5)  # 20 grupos com 5 amostras cada
    
    print(f"Dados: {X.shape[0]} amostras, {X.shape[1]} variáveis")
    print(f"Grupos: {len(np.unique(groups))} grupos")
    
    splitter = DataSplitter(test_size=0.3, random_state=42)
    
    # 1. Divisão com grupos
    print("\n1. Divisão com grupos (réplicas)...")
    train_idx, test_idx = splitter.split_with_groups(X, y, groups=groups)
    print(f"   Treino: {len(train_idx)}, Teste: {len(test_idx)}")
    
    # 2. Kennard-Stone
    print("\n2. Algoritmo Kennard-Stone...")
    train_idx, test_idx = splitter.split_kennard_stone(X, y)
    print(f"   Treino: {len(train_idx)}, Teste: {len(test_idx)}")
    
    # 3. SPXY
    print("\n3. Algoritmo SPXY...")
    train_idx, test_idx = splitter.split_spxy(X, y)
    print(f"   Treino: {len(train_idx)}, Teste: {len(test_idx)}")


def demonstrate_modeling():
    """Demonstra a modelagem PLS."""
    print("\n=== DEMONSTRAÇÃO DE MODELAGEM PLS ===")
    
    # Criar dados sintéticos
    np.random.seed(42)
    n_samples, n_features = 80, 30
    X = np.random.randn(n_samples, n_features)
    y = 2 * X[:, 0] + 1.5 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.1
    
    print(f"Dados: {X.shape[0]} amostras, {X.shape[1]} variáveis")
    
    # Dividir dados
    splitter = DataSplitter(test_size=0.3, random_state=42)
    # Criar grupos sintéticos para demonstração
    groups = np.repeat(range(20), 4)  # 20 grupos com 4 amostras cada
    train_idx, test_idx = splitter.split_with_groups(X, y, groups=groups)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Divisão: {len(train_idx)} treino, {len(test_idx)} teste")
    
    # Treinar modelo
    print("\nTreinando modelo PLS...")
    model = PLSRegressor()
    model.fit(X_train, y_train, cv_folds=5, optimize_components=True)
    
    # Avaliar modelo
    print("\nAvaliando modelo...")
    metrics = model.evaluate(X_train, y_train, X_test, y_test, unit='units')
    
    # Exibir métricas
    MetricsCalculator.print_metrics(metrics, "Métricas do Modelo PLS")
    
    # VIP scores
    vip_scores = model.vip_scores_
    print(f"\nVIP Scores - Top 5 variáveis mais importantes:")
    top_vars = np.argsort(vip_scores)[-5:][::-1]
    for i, var_idx in enumerate(top_vars):
        print(f"   {i+1}. Variável {var_idx}: VIP = {vip_scores[var_idx]:.3f}")


def demonstrate_metrics():
    """Demonstra o cálculo de métricas."""
    print("\n=== DEMONSTRAÇÃO DE MÉTRICAS ===")
    
    # Criar dados sintéticos
    np.random.seed(42)
    y_true = np.random.randn(50) * 10 + 5
    y_pred = y_true + np.random.randn(50) * 2
    
    # Calcular métricas
    metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred)
    
    # Exibir métricas
    MetricsCalculator.print_metrics(metrics, "Métricas de Exemplo")
    
    # Comparar modelos
    print("\nComparação de modelos:")
    metrics_list = [
        {'R2': 0.85, 'RMSE': 2.1, 'MAE': 1.8, 'MAPE': 15.2},
        {'R2': 0.78, 'RMSE': 2.8, 'MAE': 2.3, 'MAPE': 18.5},
        {'R2': 0.92, 'RMSE': 1.5, 'MAE': 1.2, 'MAPE': 12.1}
    ]
    model_names = ['Modelo A', 'Modelo B', 'Modelo C']
    
    MetricsCalculator.compare_models(metrics_list, model_names)


def main():
    """Função principal da demonstração."""
    print("=== DEMONSTRAÇÃO DA BIBLIOTECA CHEMOMETRICS ===\n")
    
    try:
        # Demonstrações
        demonstrate_preprocessing()
        demonstrate_data_splitting()
        demonstrate_modeling()
        demonstrate_metrics()
        
        print("\n=== DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO ===")
        print("\nPara usar com seus dados reais, execute:")
        print("python main_analysis.py")
        
    except Exception as e:
        logger.error(f"Erro na demonstração: {e}")
        raise


if __name__ == "__main__":
    main()
