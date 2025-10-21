"""
Módulo de modelagem quimiométrica.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PLSRegressor:
    """
    Regressor PLS com funcionalidades para análise quimiométrica.
    """
    
    def __init__(self, n_components: Optional[int] = None, max_iter: int = 500,
                 tol: float = 1e-06, copy: bool = True, scale: bool = True):
        """
        Inicializa o regressor PLS.
        
        Parameters
        ----------
        n_components : int, optional
            Número de componentes latentes. Se None, será determinado por validação cruzada.
        max_iter : int, default=500
            Número máximo de iterações.
        tol : float, default=1e-06
            Tolerância para convergência.
        copy : bool, default=True
            Se True, faz uma cópia dos dados.
        scale : bool, default=True
            Se True, escala os dados.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self.scale = scale
        
        self.model_ = None
        self.is_fitted = False
        self.metrics_ = {}
        self.vip_scores_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            cv_folds: int = 5, optimize_components: bool = True) -> 'PLSRegressor':
        """
        Ajusta o modelo PLS aos dados.
        
        Parameters
        ----------
        X : np.ndarray
            Dados de entrada.
        y : np.ndarray
            Variável resposta.
        cv_folds : int, default=5
            Número de folds para validação cruzada.
        optimize_components : bool, default=True
            Se True, otimiza o número de componentes por validação cruzada.
            
        Returns
        -------
        self : PLSRegressor
            Instância ajustada.
        """
        X, y = np.array(X), np.array(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        # Otimizar número de componentes se necessário
        if optimize_components and self.n_components is None:
            self.n_components = self._optimize_components(X, y, cv_folds)
            
        # Ajustar modelo final
        self.model_ = PLSRegression(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            copy=self.copy,
            scale=self.scale
        )
        
        self.model_.fit(X, y)
        self.is_fitted = True
        
        # Calcular VIP scores
        self.vip_scores_ = self._calculate_vip_scores(X)
        
        logger.info(f"Modelo PLS ajustado com {self.n_components} componentes")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições usando o modelo ajustado."""
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado. Chame fit() primeiro.")
            
        return self.model_.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calcula o R² score."""
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado. Chame fit() primeiro.")
            
        return self.model_.score(X, y)
    
    def evaluate(self, X_cal: np.ndarray, y_cal: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                unit: str = 'units') -> Dict[str, float]:
        """
        Avalia o modelo com métricas completas.
        
        Parameters
        ----------
        X_cal : np.ndarray
            Dados de calibração.
        y_cal : np.ndarray
            Resposta de calibração.
        X_test : np.ndarray
            Dados de teste.
        y_test : np.ndarray
            Resposta de teste.
        unit : str, default='units'
            Unidade de medida.
            
        Returns
        -------
        metrics : dict
            Métricas de avaliação.
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado. Chame fit() primeiro.")
            
        # Predições
        y_cal_pred = self.predict(X_cal)
        y_test_pred = self.predict(X_test)
        
        # Métricas de calibração
        r2c = r2_score(y_cal, y_cal_pred)
        rmsec = np.sqrt(mean_squared_error(y_cal, y_cal_pred))
        
        # Métricas de predição
        r2p = r2_score(y_test, y_test_pred)
        rmsep = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Validação cruzada
        cv_scores = cross_val_score(self.model_, X_cal, y_cal, cv=5, scoring='r2')
        rmsecv = np.sqrt(mean_squared_error(y_cal, self.predict(X_cal)))
        
        self.metrics_ = {
            'R2c': r2c,
            'R2p': r2p,
            'RMSEC': rmsec,
            'RMSEP': rmsep,
            'RMSECV': rmsecv,
            'n_components': self.n_components,
            'unit': unit
        }
        
        return self.metrics_
    
    def plot_results(self, X_cal: np.ndarray, y_cal: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    unit: str = 'units', figsize: Tuple[int, int] = (15, 6)) -> None:
        """
        Plota os resultados do modelo.
        
        Parameters
        ----------
        X_cal : np.ndarray
            Dados de calibração.
        y_cal : np.ndarray
            Resposta de calibração.
        X_test : np.ndarray
            Dados de teste.
        y_test : np.ndarray
            Resposta de teste.
        unit : str, default='units'
            Unidade de medida.
        figsize : tuple, default=(15, 6)
            Tamanho da figura.
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado. Chame fit() primeiro.")
            
        # Predições
        y_cal_pred = self.predict(X_cal)
        y_test_pred = self.predict(X_test)
        
        # Métricas
        r2c = r2_score(y_cal, y_cal_pred)
        r2p = r2_score(y_test, y_test_pred)
        rmsec = np.sqrt(mean_squared_error(y_cal, y_cal_pred))
        rmsep = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Criar figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Gráfico Predito vs Real
        ax1.scatter(y_cal, y_cal_pred, alpha=0.7, label='Calibração', color='blue', s=50)
        ax1.scatter(y_test, y_test_pred, alpha=0.7, label='Teste', color='red', s=50)
        
        # Linha de referência
        min_val = min(np.min(y_cal), np.min(y_test), np.min(y_cal_pred), np.min(y_test_pred))
        max_val = max(np.max(y_cal), np.max(y_test), np.max(y_cal_pred), np.max(y_test_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
        
        ax1.set_xlabel(f'Real ({unit})')
        ax1.set_ylabel(f'Predito ({unit})')
        ax1.legend()
        ax1.set_title(f'Predito vs Real\nR²c = {r2c:.3f}, R²p = {r2p:.3f}')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de Resíduos
        residuals_cal = y_cal - y_cal_pred
        residuals_test = y_test - y_test_pred
        
        ax2.scatter(y_cal_pred, residuals_cal, alpha=0.7, label='Calibração', color='blue', s=50)
        ax2.scatter(y_test_pred, residuals_test, alpha=0.7, label='Teste', color='red', s=50)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.8, linewidth=2)
        
        ax2.set_xlabel(f'Predito ({unit})')
        ax2.set_ylabel(f'Resíduos ({unit})')
        ax2.legend()
        ax2.set_title(f'Resíduos\nRMSEC = {rmsec:.3f}, RMSEP = {rmsep:.3f}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_vip_scores(self, feature_names: Optional[List[str]] = None,
                       threshold: float = 1.0, figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plota os VIP scores.
        
        Parameters
        ----------
        feature_names : list, optional
            Nomes das variáveis.
        threshold : float, default=1.0
            Limiar para destacar variáveis importantes.
        figsize : tuple, default=(12, 6)
            Tamanho da figura.
        """
        if self.vip_scores_ is None:
            raise ValueError("VIP scores não foram calculados. Ajuste o modelo primeiro.")
            
        plt.figure(figsize=figsize)
        
        if feature_names is None:
            feature_names = [f'Var_{i}' for i in range(len(self.vip_scores_))]
        
        # Plotar VIP scores
        bars = plt.bar(range(len(self.vip_scores_)), self.vip_scores_, 
                      color=['red' if score > threshold else 'blue' for score in self.vip_scores_])
        
        # Linha de referência
        plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Limiar = {threshold}')
        
        plt.xlabel('Variáveis')
        plt.ylabel('VIP Score')
        plt.title('VIP Scores - Importância das Variáveis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rotacionar labels se necessário
        if len(feature_names) > 20:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def get_results_dataframe(self, X_cal: np.ndarray, y_cal: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Cria DataFrame com resultados detalhados.
        
        Parameters
        ----------
        X_cal : np.ndarray
            Dados de calibração.
        y_cal : np.ndarray
            Resposta de calibração.
        X_test : np.ndarray
            Dados de teste.
        y_test : np.ndarray
            Resposta de teste.
            
        Returns
        -------
        df_results : pd.DataFrame
            DataFrame com resultados.
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi ajustado. Chame fit() primeiro.")
            
        # Predições
        y_cal_pred = self.predict(X_cal)
        y_test_pred = self.predict(X_test)
        
        # Criar DataFrame
        df_results = pd.DataFrame({
            'conjunto': ['calibração'] * len(y_cal) + ['teste'] * len(y_test),
            'y_real': np.concatenate([y_cal, y_test]),
            'y_predito': np.concatenate([y_cal_pred, y_test_pred]),
            'erro_absoluto': np.abs(np.concatenate([
                y_cal - y_cal_pred, y_test - y_test_pred
            ])),
            'erro_percentual': np.abs(np.concatenate([
                np.where(y_cal != 0, (y_cal - y_cal_pred) / y_cal * 100, 0),
                np.where(y_test != 0, (y_test - y_test_pred) / y_test * 100, 0)
            ]))
        })
        
        return df_results
    
    def _optimize_components(self, X: np.ndarray, y: np.ndarray, 
                           cv_folds: int = 5) -> int:
        """Otimiza o número de componentes por validação cruzada."""
        n_comp_max = min(10, X.shape[0] - 1, X.shape[1])
        best_score = -np.inf
        best_n_comp = 1
        
        for n_comp in range(1, n_comp_max + 1):
            try:
                pls_temp = PLSRegression(n_components=n_comp)
                scores = cross_val_score(pls_temp, X, y, cv=cv_folds, scoring='r2')
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_n_comp = n_comp
                    
            except Exception as e:
                logger.warning(f"Erro ao testar {n_comp} componentes: {e}")
                continue
                
        logger.info(f"Número ótimo de componentes: {best_n_comp} (R² = {best_score:.3f})")
        return best_n_comp
    
    def _calculate_vip_scores(self, X: np.ndarray) -> np.ndarray:
        """Calcula VIP scores para seleção de variáveis."""
        if not self.is_fitted:
            return None
            
        t = self.model_.x_scores_
        w = self.model_.x_weights_
        q = self.model_.y_loadings_
        
        p = X.shape[1]
        vip = np.zeros(p)
        
        s = np.diag(t.T @ t @ q.T @ q).reshape(-1, 1)
        total_s = np.sum(s)
        
        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 
                             for j in range(self.model_.n_components)])
            vip[i] = np.sqrt(p * (s.T @ weight) / total_s)
        
        return vip.ravel()
