from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import minmax_scale

def split_data_(X, y, percentual=0.3, random_state=28, rep=None):
    gss = GroupShuffleSplit(n_splits=1, test_size=percentual, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=rep))
    return train_idx, test_idx

def split_method_(X, y, test_size=0.30, method='ks'):
    """
    Divide X e y usando uma estrutura 'case' para selecionar o algoritmo.
    """
    X_arr, y_arr = np.array(X), np.array(y).reshape(-1, 1)
    if X_arr.ndim == 1: X_arr = X_arr.reshape(-1, 1)

    # Estrutura 'case' para cálculo da matriz de distância
    match method.lower():
        case 'ks':
            dist_matrix = np.linalg.norm(X_arr[:, np.newaxis] - X_arr, axis=2)
        case 'spxy':
            X_s = minmax_scale(X_arr)
            y_s = minmax_scale(y_arr)
            dist_x = np.linalg.norm(X_s[:, np.newaxis] - X_s, axis=2)
            dist_y = np.linalg.norm(y_s[:, np.newaxis] - y_s, axis=2)
            dist_matrix = (dist_x / np.max(dist_x)) + (dist_y / np.max(dist_y))
        case _:
            raise ValueError(f"Método '{method}' não suportado.")

    # Lógica de seleção (inalterada)
    n_samples = X_arr.shape[0]
    n_cal = n_samples - int(n_samples * test_size)
    
    cal_idx = list(np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape))
    rem_idx = [i for i in range(n_samples) if i not in cal_idx]

    for _ in range(n_cal - 2):
        min_dists = [np.min(dist_matrix[i, cal_idx]) for i in rem_idx]
        new_idx = rem_idx.pop(np.argmax(min_dists))
        cal_idx.append(new_idx)

    test_idx = rem_idx
    return X_arr[cal_idx], X_arr[test_idx], y_arr[cal_idx].ravel(), y_arr[test_idx].ravel()

