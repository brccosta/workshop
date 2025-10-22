"""
Chemometrics Library - Análise Quimiométrica de Dados Espectrais
================================================================

Uma biblioteca Python para análise quimiométrica focada em espectroscopia.

Módulos:
- preprocessing: Técnicas de pré-processamento espectroscópico
- modeling: Modelos de regressão PLS
- utils: Utilitários para divisão de dados e métricas

Exemplo de uso:
    from chemometrics import PLSRegressor, SNVPreprocessor, DataSplitter
    
    # Pré-processamento
    preprocessor = SNVPreprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    # Divisão dos dados
    splitter = DataSplitter(test_size=0.3)
    train_idx, test_idx = splitter.split_with_groups(X, y, groups)
    
    # Modelagem
    model = PLSRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
"""

__version__ = "1.0.0"
__author__ = "Chemometrics Team"

# Imports principais
from .preprocessing import *
from .modeling import *
from .utils import *

__all__ = [
    # Preprocessing
    'SNVPreprocessor',
    'SavitzkyGolayPreprocessor', 
    'MSCPreprocessor',
    'MeanCenterPreprocessor',
    'AutoScalePreprocessor',
    
    # Modeling
    'PLSRegressor',
    
    # Utils
    'DataSplitter',
    'MetricsCalculator'
]
