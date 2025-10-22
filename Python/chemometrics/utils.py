"""
Utilitários para análise quimiométrica.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Classe para divisão de dados em conjuntos de treino e teste.
    """
    
    def __init__(self, test_size: float = 0.3, random_state: int = 42):
        """
        Inicializa o divisor de dados.
        
        Parameters
        ----------
        test_size : float, default=0.3
            Proporção dos dados para teste.
        random_state : int, default=42
            Semente para reprodutibilidade.
        """
        self.test_size = test_size
        self.random_state = random_state
        
    def split_with_groups(self, X: np.ndarray, y: np.ndarray, 
                        groups: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Divide os dados considerando grupos (ex: réplicas).
        
        Parameters
        ----------
        X : np.ndarray
            Dados de entrada.
        y : np.ndarray
            Variável resposta.
        groups : np.ndarray, optional
            Grupos para divisão (ex: IDs das amostras).
            
        Returns
        -------
        train_idx : np.ndarray
            Índices de treino.
        test_idx : np.ndarray
            Índices de teste.
        """
        gss = GroupShuffleSplit(
            n_splits=1, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        
        logger.info(f"Dados divididos: {len(train_idx)} treino, {len(test_idx)} teste")
        
        return train_idx, test_idx
    
    def split_kennard_stone(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Divide os dados usando algoritmo Kennard-Stone.
        
        Parameters
        ----------
        X : np.ndarray
            Dados de entrada.
        y : np.ndarray
            Variável resposta.
            
        Returns
        -------
        train_idx : np.ndarray
            Índices de treino.
        test_idx : np.ndarray
            Índices de teste.
        """
        X_arr = np.array(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
            
        n_samples = X_arr.shape[0]
        n_cal = n_samples - int(n_samples * self.test_size)
        
        # Calcular matriz de distância
        dist_matrix = np.linalg.norm(X_arr[:, np.newaxis] - X_arr, axis=2)
        
        # Selecionar primeira amostra (maior distância)
        cal_idx = list(np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape))
        rem_idx = [i for i in range(n_samples) if i not in cal_idx]
        
        # Selecionar amostras de calibração
        for _ in range(n_cal - 2):
            min_dists = [np.min(dist_matrix[i, cal_idx]) for i in rem_idx]
            new_idx = rem_idx.pop(np.argmax(min_dists))
            cal_idx.append(new_idx)
        
        train_idx = np.array(cal_idx)
        test_idx = np.array(rem_idx)
        
        logger.info(f"Kennard-Stone: {len(train_idx)} treino, {len(test_idx)} teste")
        
        return train_idx, test_idx
    
    def split_spxy(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Divide os dados usando algoritmo SPXY.
        
        Parameters
        ----------
        X : np.ndarray
            Dados de entrada.
        y : np.ndarray
            Variável resposta.
            
        Returns
        -------
        train_idx : np.ndarray
            Índices de treino.
        test_idx : np.ndarray
            Índices de teste.
        """
        X_arr = np.array(X)
        y_arr = np.array(y).reshape(-1, 1)
        
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
            
        # Normalizar dados
        X_s = minmax_scale(X_arr)
        y_s = minmax_scale(y_arr)
        
        # Calcular distâncias
        dist_x = np.linalg.norm(X_s[:, np.newaxis] - X_s, axis=2)
        dist_y = np.linalg.norm(y_s[:, np.newaxis] - y_s, axis=2)
        
        # Combinar distâncias
        dist_matrix = (dist_x / np.max(dist_x)) + (dist_y / np.max(dist_y))
        
        n_samples = X_arr.shape[0]
        n_cal = n_samples - int(n_samples * self.test_size)
        
        # Selecionar primeira amostra
        cal_idx = list(np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape))
        rem_idx = [i for i in range(n_samples) if i not in cal_idx]
        
        # Selecionar amostras de calibração
        for _ in range(n_cal - 2):
            min_dists = [np.min(dist_matrix[i, cal_idx]) for i in rem_idx]
            new_idx = rem_idx.pop(np.argmax(min_dists))
            cal_idx.append(new_idx)
        
        train_idx = np.array(cal_idx)
        test_idx = np.array(rem_idx)
        
        logger.info(f"SPXY: {len(train_idx)} treino, {len(test_idx)} teste")
        
        return train_idx, test_idx


class MetricsCalculator:
    """
    Classe para cálculo de métricas de avaliação.
    """
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula todas as métricas de avaliação.
        
        Parameters
        ----------
        y_true : np.ndarray
            Valores reais.
        y_pred : np.ndarray
            Valores preditos.
            
        Returns
        -------
        metrics : dict
            Dicionário com as métricas.
        """
        # Métricas básicas
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Bias
        bias = np.mean(y_pred - y_true)
        
        # RPD (Ratio of Performance to Deviation)
        rpd = np.std(y_true) / rmse
        
        # RER (Range Error Ratio)
        rer = (np.max(y_true) - np.min(y_true)) / rmse
        
        return {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Bias': bias,
            'RPD': rpd,
            'RER': rer
        }
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], title: str = "Métricas") -> None:
        """
        Imprime as métricas de forma formatada.
        
        Parameters
        ----------
        metrics : dict
            Dicionário com as métricas.
        title : str, default="Métricas"
            Título para exibição.
        """
        print(f"\n=== {title.upper()} ===")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    @staticmethod
    def compare_models(metrics_list: list, model_names: list) -> None:
        """
        Compara métricas de diferentes modelos.
        
        Parameters
        ----------
        metrics_list : list
            Lista de dicionários com métricas.
        model_names : list
            Nomes dos modelos.
        """
        if len(metrics_list) != len(model_names):
            raise ValueError("Número de métricas deve ser igual ao número de nomes.")
        
        print("\n=== COMPARAÇÃO DE MODELOS ===")
        print(f"{'Modelo':<15} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'MAPE':<8}")
        print("-" * 50)
        
        for metrics, name in zip(metrics_list, model_names):
            r2 = metrics.get('R2', metrics.get('R2c', metrics.get('R2p', 0)))
            rmse = metrics.get('RMSE', metrics.get('RMSEC', metrics.get('RMSEP', 0)))
            mae = metrics.get('MAE', metrics.get('MAEc', metrics.get('MAEp', 0)))
            mape = metrics.get('MAPE', 0)
            
            print(f"{name:<15} {r2:<8.3f} {rmse:<8.3f} {mae:<8.3f} {mape:<8.1f}")
