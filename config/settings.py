"""
Configurações do projeto Chemometrics.
"""

import os
from pathlib import Path

# Diretórios do projeto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Configurações de logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'chemometrics': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

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

# Criar diretórios se não existirem
for directory in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)
