"""
Módulo de pré-processamento espectroscópico. Mensagem 1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.sparse import eye, diags, csc_matrix
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BasePreprocessor:
    """Classe base para pré-processadores."""
    
    def __init__(self, plot: bool = False):
        self.plot = plot
        self.is_fitted = False
        
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Valida e prepara os dados de entrada."""
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        if X.size == 0:
            raise ValueError("Dados de entrada não podem estar vazios.")
            
        return X
    
    def _plot_results(self, X_original: np.ndarray, X_processed: np.ndarray, 
                     title: str, sample_idx: int = 1) -> None:
        """Plota os resultados do pré-processamento."""
        if not self.plot:
            return
            
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi=300)
            
            # Espectro original
            ax1.plot(X_original[sample_idx, :].T, color='black', linewidth=1)
            ax1.set_title('Espectro Original')
            ax1.set_ylabel('Absorbância')
            ax1.grid(True, alpha=0.3)
            
            # Espectro processado
            ax2.plot(X_processed[sample_idx, :].T, color='green', linewidth=1)
            ax2.set_title(f'Espectro Processado - {title}')
            ax2.set_xlabel('Comprimento de Onda')
            ax2.set_ylabel('Absorbância')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.warning(f"Erro ao plotar resultados: {e}")


class SNVPreprocessor(BasePreprocessor):
    """
    Standard Normal Variate (SNV) preprocessing.
    
    Normaliza cada espectro subtraindo a média e dividindo pelo desvio padrão.
    """
    
    def __init__(self, ddof: int = 1, plot: bool = False):
        super().__init__(plot)
        self.ddof = ddof
        
    def fit(self, X: np.ndarray) -> 'SNVPreprocessor':
        """Ajusta o pré-processador (SNV não precisa de ajuste)."""
        self._validate_input(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica SNV aos dados."""
        if not self.is_fitted:
            self.fit(X)
            
        X = self._validate_input(X)
        
        # Calcula média e desvio padrão de cada linha (espectro)
        mean_spectrum = np.mean(X, axis=1, keepdims=True)
        std_spectrum = np.std(X, axis=1, keepdims=True, ddof=self.ddof)
        
        # Evita divisão por zero
        std_spectrum = np.where(std_spectrum == 0, 1, std_spectrum)
        
        # Aplica SNV
        X_snv = (X - mean_spectrum) / std_spectrum
        
        if self.plot:
            self._plot_results(X, X_snv, 'SNV')
            
        return X_snv
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajusta e transforma os dados."""
        return self.fit(X).transform(X)


class SavitzkyGolayPreprocessor(BasePreprocessor):
    """
    Savitzky-Golay filter preprocessing.
    """
    
    def __init__(self, window_length: int = 15, polyorder: int = 2, 
                 deriv: int = 0, plot: bool = False):
        super().__init__(plot)
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        
    def fit(self, X: np.ndarray) -> 'SavitzkyGolayPreprocessor':
        """Ajusta o pré-processador."""
        self._validate_input(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica filtro Savitzky-Golay aos dados."""
        if not self.is_fitted:
            self.fit(X)
            
        X = self._validate_input(X)
        
        try:
            X_filtered = savgol_filter(X, self.window_length, self.polyorder, 
                                     deriv=self.deriv, axis=1)
        except Exception as e:
            logger.error(f"Erro no filtro Savitzky-Golay: {e}")
            return X
            
        if self.plot:
            title = f'Savitzky-Golay (window={self.window_length}, order={self.polyorder}, deriv={self.deriv})'
            self._plot_results(X, X_filtered, title)
            
        return X_filtered
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajusta e transforma os dados."""
        return self.fit(X).transform(X)


class MSCPreprocessor(BasePreprocessor):
    """
    Multiplicative Scatter Correction (MSC) preprocessing.
    """
    
    def __init__(self, reference: Optional[np.ndarray] = None, plot: bool = False):
        super().__init__(plot)
        self.reference = reference
        self.reference_spectrum = None
        
    def fit(self, X: np.ndarray) -> 'MSCPreprocessor':
        """Ajusta o pré-processador MSC."""
        X = self._validate_input(X)
        
        if self.reference is None:
            self.reference_spectrum = np.mean(X, axis=0)
        else:
            self.reference_spectrum = np.array(self.reference)
            
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica MSC aos dados."""
        if not self.is_fitted:
            raise ValueError("Pré-processador não foi ajustado. Chame fit() primeiro.")
            
        X = self._validate_input(X)
        
        # Centralizar os dados
        X_centered = X - np.mean(X, axis=1, keepdims=True)
        reference_centered = self.reference_spectrum - np.mean(self.reference_spectrum)
        
        # Calcular coeficientes de regressão
        try:
            coeffs = np.linalg.lstsq(reference_centered[:, np.newaxis], X_centered.T, rcond=None)[0]
            
            # Ajustar os dados
            X_msc = (X_centered - coeffs.T * reference_centered) + np.mean(X, axis=1, keepdims=True)
            
        except np.linalg.LinAlgError:
            logger.warning("Erro na decomposição QR. Retornando dados originais.")
            X_msc = X
            
        if self.plot:
            self._plot_results(X, X_msc, 'MSC')
            
        return X_msc
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajusta e transforma os dados."""
        return self.fit(X).transform(X)


class MeanCenterPreprocessor(BasePreprocessor):
    """
    Mean Centering preprocessing.
    """
    
    def __init__(self, plot: bool = False):
        super().__init__(plot)
        self.mean_ = None
        
    def fit(self, X: np.ndarray) -> 'MeanCenterPreprocessor':
        """Ajusta o pré-processador."""
        X = self._validate_input(X)
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica centramento na média aos dados."""
        if not self.is_fitted:
            raise ValueError("Pré-processador não foi ajustado. Chame fit() primeiro.")
            
        X = self._validate_input(X)
        X_centered = X - self.mean_
        
        if self.plot:
            self._plot_results(X, X_centered, 'Mean Centering')
            
        return X_centered
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajusta e transforma os dados."""
        return self.fit(X).transform(X)


class AutoScalePreprocessor(BasePreprocessor):
    """
    Autoscaling preprocessing.
    """
    
    def __init__(self, ddof: int = 1, plot: bool = False):
        super().__init__(plot)
        self.ddof = ddof
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X: np.ndarray) -> 'AutoScalePreprocessor':
        """Ajusta o pré-processador."""
        X = self._validate_input(X)
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.std_ = np.std(X, axis=0, keepdims=True, ddof=self.ddof)
        
        # Evita divisão por zero
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica autoscaling aos dados."""
        if not self.is_fitted:
            raise ValueError("Pré-processador não foi ajustado. Chame fit() primeiro.")
            
        X = self._validate_input(X)
        X_scaled = (X - self.mean_) / self.std_
        
        if self.plot:
            self._plot_results(X, X_scaled, 'Autoscaling')
            
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Ajusta e transforma os dados."""
        return self.fit(X).transform(X)
