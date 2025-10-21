"""
Configurações simples do projeto Chemometrics.
"""

# Configurações de logging
LOGGING_LEVEL = 'INFO'
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Configurações padrão para pré-processamento
DEFAULT_PREPROCESSING = {
    'savgol': {
        'window_length': 15,
        'polyorder': 2,
        'deriv': 1
    },
    'snv': {
        'ddof': 1
    },
    'msc': {
        'reference': None
    }
}

# Configurações padrão para modelagem
DEFAULT_MODELING = {
    'pls': {
        'n_components': None,
        'max_iter': 500,
        'tol': 1e-06,
        'copy': True
    },
    'validation': {
        'cv_folds': 5,
        'test_size': 0.3,
        'random_state': 42
    }
}

# Configurações de plotagem
PLOT_CONFIG = {
    'figsize': (10, 6),
    'dpi': 300,
    'style': 'seaborn-v0_8'
}
