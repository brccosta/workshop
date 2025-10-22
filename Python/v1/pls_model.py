import numpy as np
from sklearn.preprocessing import minmax_scale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')



# ========== FUNÇÕES AUXILIARES PARA SELEÇÃO DE VARIÁVEIS ==========

def vip_scores(pls_model, X):
    """Calcula VIP scores para seleção de variáveis"""
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    
    p = X.shape[1]
    vip = np.zeros((p,))
    
    s = np.diag(t.T @ t @ q.T @ q).reshape(-1, 1)
    total_s = np.sum(s)
    
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(pls_model.n_components)])
        vip[i] = np.sqrt(p * (s.T @ weight) / total_s)
    
    return vip.ravel()

def apply_vip(X_cal, X_test, y_cal, vip_threshold=1.0):
    """Aplica seleção baseada em VIP scores"""
    pls_temp = PLSRegression(n_components=min(10, X_cal.shape[1]))
    pls_temp.fit(X_cal, y_cal)
    
    vip = vip_scores(pls_temp, X_cal)
    selected_idx = np.where(vip > vip_threshold)[0]
    
    return X_cal[:, selected_idx], X_test[:, selected_idx], selected_idx

def apply_spa(X_cal, X_test, y_cal, n_variables=10):
    """Implementação do Successive Projections Algorithm"""
    def column_select(X, selected):
        if len(selected) == 0:
            return np.argmax(np.var(X, axis=0))
        
        X_proj = X.copy()
        for col in selected:
            x_j = X[:, col].reshape(-1, 1)
            for k in range(X.shape[1]):
                if k not in selected:
                    x_k = X[:, k].reshape(-1, 1)
                    proj = (x_k.T @ x_j) / (x_j.T @ x_j) * x_j
                    X_proj[:, k] = (x_k - proj).ravel()
        
        X_proj[:, selected] = 0
        return np.argmax(np.linalg.norm(X_proj, axis=0))
    
    selected = []
    for _ in range(min(n_variables, X_cal.shape[1])):
        new_idx = column_select(X_cal, selected)
        selected.append(new_idx)
    
    return X_cal[:, selected], X_test[:, selected], selected

def apply_ga_pls(X_cal, X_test, y_cal, population_size=50, generations=100, n_variables=10):
    """Algoritmo Genético simplificado para seleção de variáveis"""
    n_features = X_cal.shape[1]
    
    def fitness(individual):
        selected = np.where(individual)[0]
        if len(selected) == 0 or len(selected) > n_variables * 2:
            return -np.inf
        
        try:
            pls_temp = PLSRegression(n_components=min(5, len(selected)))
            pls_temp.fit(X_cal[:, selected], y_cal)
            y_pred = pls_temp.predict(X_cal[:, selected]).ravel()
            return -mean_squared_error(y_cal, y_pred)
        except:
            return -np.inf
    
    # População inicial
    population = np.zeros((population_size, n_features), dtype=bool)
    for i in range(population_size):
        n_selected = np.random.randint(1, n_variables + 1)
        selected = np.random.choice(n_features, n_selected, replace=False)
        population[i, selected] = True
    
    # Evolução
    for gen in range(generations):
        scores = np.array([fitness(ind) for ind in population])
        best_idx = np.argsort(scores)[-population_size//2:]
        
        new_population = []
        for idx in best_idx:
            new_population.append(population[idx].copy())
            
            # Crossover e mutação
            parent2 = population[np.random.choice(best_idx)]
            child = population[idx].copy()
            crossover_point = np.random.randint(n_features)
            child[crossover_point:] = parent2[crossover_point:]
            
            # Mutação
            mutation_point = np.random.randint(n_features)
            child[mutation_point] = not child[mutation_point]
            new_population.append(child)
        
        population = np.array(new_population)
    
    # Melhor indivíduo
    scores = np.array([fitness(ind) for ind in population])
    best_individual = population[np.argmax(scores)]
    selected_idx = np.where(best_individual)[0]
    
    return X_cal[:, selected_idx], X_test[:, selected_idx], selected_idx

def apply_bpls(X_cal, X_test, y_cal, n_variables=10):
    """Backward PLS - Remove variáveis iterativamente"""
    n_features = X_cal.shape[1]
    selected = list(range(n_features))
    
    while len(selected) > n_variables:
        pls_temp = PLSRegression(n_components=min(5, len(selected)))
        pls_temp.fit(X_cal[:, selected], y_cal)
        
        # Remove variável com menor coeficiente absoluto
        coef = np.abs(pls_temp.coef_.ravel())
        remove_idx = np.argmin(coef)
        selected.pop(remove_idx)
    
    return X_cal[:, selected], X_test[:, selected], selected

def apply_sr(X_cal, X_test, y_cal, n_variables=10):
    """Regressão por Espectros - Seleção forward baseada em correlação"""
    selected = []
    remaining = list(range(X_cal.shape[1]))
    
    for _ in range(n_variables):
        best_score = -np.inf
        best_feature = None
        
        for feature in remaining:
            trial_selected = selected + [feature]
            if len(trial_selected) == 1:
                corr = np.abs(np.corrcoef(X_cal[:, feature], y_cal)[0, 1])
                score = corr
            else:
                pls_temp = PLSRegression(n_components=min(5, len(trial_selected)))
                pls_temp.fit(X_cal[:, trial_selected], y_cal)
                y_pred = pls_temp.predict(X_cal[:, trial_selected]).ravel()
                score = r2_score(y_cal, y_pred)
            
            if score > best_score:
                best_score = score
                best_feature = feature
        
        selected.append(best_feature)
        remaining.remove(best_feature)
    
    return X_cal[:, selected], X_test[:, selected], selected

def apply_ipls(X_cal, X_test, y_cal, n_intervals=10):
    """Interval PLS - Seleciona melhor intervalo espectral"""
    n_features = X_cal.shape[1]
    interval_size = n_features // n_intervals
    
    best_score = -np.inf
    best_interval = None
    
    for i in range(n_intervals):
        start = i * interval_size
        end = min((i + 1) * interval_size, n_features)
        
        if end - start < 2:
            continue
            
        pls_temp = PLSRegression(n_components=min(5, end - start))
        pls_temp.fit(X_cal[:, start:end], y_cal)
        y_pred = pls_temp.predict(X_cal[:, start:end]).ravel()
        score = r2_score(y_cal, y_pred)
        
        if score > best_score:
            best_score = score
            best_interval = (start, end)
    
    start, end = best_interval
    selected_idx = list(range(start, end))
    return X_cal[:, start:end], X_test[:, start:end], selected_idx

# ========== FUNÇÃO PRINCIPAL PLS ==========

def pls_(X_cal, X_test, y_cal, y_test, unidade_medida, var_select=None, plotar=False, **kwargs):
    """
    Função PLS com múltiplos métodos de seleção de variáveis
    
    Parâmetros:
    -----------
    X_cal, X_test: arrays - Conjuntos de calibração e teste
    y_cal, y_test: arrays - Variáveis resposta
    unidade_medida: str - Unidade de medida para labels
    var_select: str - Método de seleção ('vip', 'spa', 'ga', 'bpls', 'sr', 'ipls')
    plotar: bool - Se True, plota gráficos de resultados
    **kwargs: parâmetros adicionais para métodos específicos
    
    Retorna:
    --------
    coef: coeficientes do modelo
    metrics: dicionário com métricas
    df_results: DataFrame com resultados detalhados
    """
    
    X_cal_orig, X_test_orig = X_cal.copy(), X_test.copy()
    selected_idx = None
    
    # Aplicar seleção de variáveis se especificado
    if var_select:
        if var_select.lower() == 'vip':
            X_cal, X_test, selected_idx = apply_vip(X_cal, X_test, y_cal, 
                                                   kwargs.get('vip_threshold', 1.0))
        elif var_select.lower() == 'spa':
            X_cal, X_test, selected_idx = apply_spa(X_cal, X_test, y_cal,
                                                   kwargs.get('n_variables', 10))
        elif var_select.lower() == 'ga':
            X_cal, X_test, selected_idx = apply_ga_pls(X_cal, X_test, y_cal,
                                                      kwargs.get('population_size', 50),
                                                      kwargs.get('generations', 100),
                                                      kwargs.get('n_variables', 10))
        elif var_select.lower() == 'bpls':
            X_cal, X_test, selected_idx = apply_bpls(X_cal, X_test, y_cal,
                                                    kwargs.get('n_variables', 10))
        elif var_select.lower() == 'sr':
            X_cal, X_test, selected_idx = apply_sr(X_cal, X_test, y_cal,
                                                  kwargs.get('n_variables', 10))
        elif var_select.lower() == 'ipls':
            X_cal, X_test, selected_idx = apply_ipls(X_cal, X_test, y_cal,
                                                    kwargs.get('n_intervals', 10))
        else:
            raise ValueError(f"Método {var_select} não reconhecido")
    
    # Determinar número ótimo de componentes por validação cruzada
    n_comp_max = min(10, X_cal.shape[0] - 1, X_cal.shape[1])
    best_comp = 1
    best_rmsecv = np.inf
    
    for n_comp in range(1, n_comp_max + 1):
        try:
            pls_temp = PLSRegression(n_components=n_comp)
            
            # Validação cruzada simples (leave-one-out)
            rmsecv_vals = []
            for i in range(len(X_cal)):
                X_train = np.delete(X_cal, i, axis=0)
                y_train = np.delete(y_cal, i)
                X_val = X_cal[i:i+1]
                y_val = y_cal[i:i+1]
                
                pls_temp.fit(X_train, y_train)
                y_pred = pls_temp.predict(X_val).ravel()
                rmsecv_vals.append((y_val[0] - y_pred[0])**2)
            
            rmsecv = np.sqrt(np.mean(rmsecv_vals))
            
            if rmsecv < best_rmsecv:
                best_rmsecv = rmsecv
                best_comp = n_comp
        except:
            continue
    
    # Modelo PLS final
    pls_final = PLSRegression(n_components=best_comp)
    pls_final.fit(X_cal, y_cal)
    
    # Previsões
    y_cal_pred = pls_final.predict(X_cal).ravel()
    y_test_pred = pls_final.predict(X_test).ravel()
    
    # Métricas
    R2c = r2_score(y_cal, y_cal_pred)
    R2p = r2_score(y_test, y_test_pred)
    RMSEC = np.sqrt(mean_squared_error(y_cal, y_cal_pred))
    RMSEP = np.sqrt(mean_squared_error(y_test, y_test_pred))
    RMSECV = best_rmsecv
    
    # Coeficientes (mapeados de volta para variáveis originais se houve seleção)
    if selected_idx is not None:
        full_coef = np.zeros(X_cal_orig.shape[1])
        full_coef[selected_idx] = pls_final.coef_.ravel()
        coef = full_coef
    else:
        coef = pls_final.coef_.ravel()
    
    # DataFrame de resultados
    df_results = pd.DataFrame({
        'conjunto': ['calibração'] * len(y_cal) + ['teste'] * len(y_test),
        'y_real': np.concatenate([y_cal, y_test]),
        'y_predito': np.concatenate([y_cal_pred, y_test_pred]),
        'erro_absoluto': np.abs(np.concatenate([y_cal - y_cal_pred, y_test - y_test_pred])),
        'erro_percentual': np.abs(np.concatenate([
            np.where(y_cal != 0, (y_cal - y_cal_pred) / y_cal * 100, 0),
            np.where(y_test != 0, (y_test - y_test_pred) / y_test * 100, 0)
        ]))
    })
    
    # Plot se solicitado
    if plotar:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico Predito vs Real
        ax1.scatter(y_cal, y_cal_pred, alpha=0.7, label='Calibração', color='blue')
        ax1.scatter(y_test, y_test_pred, alpha=0.7, label='Teste', color='red')
        min_val = min(np.min(y_cal), np.min(y_test), np.min(y_cal_pred), np.min(y_test_pred))
        max_val = max(np.max(y_cal), np.max(y_test), np.max(y_cal_pred), np.max(y_test_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
        ax1.set_xlabel(f'Real ({unidade_medida})')
        ax1.set_ylabel(f'Predito ({unidade_medida})')
        ax1.legend()
        ax1.set_title(f'Predito vs Real\nR²c = {R2c:.3f}, R²p = {R2p:.3f}')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de Resíduos
        residuals_cal = y_cal - y_cal_pred
        residuals_test = y_test - y_test_pred
        ax2.scatter(y_cal_pred, residuals_cal, alpha=0.7, label='Calibração', color='blue')
        ax2.scatter(y_test_pred, residuals_test, alpha=0.7, label='Teste', color='red')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.8)
        ax2.set_xlabel(f'Predito ({unidade_medida})')
        ax2.set_ylabel(f'Resíduos ({unidade_medida})')
        ax2.legend()
        ax2.set_title(f'Resíduos\nRMSEC = {RMSEC:.3f}, RMSEP = {RMSEP:.3f}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print das métricas
        print(f"=== MÉTRICAS DO MODELO PLS ===")
        print(f"Componentes Latentes: {best_comp}")
        print(f"R² Calibração: {R2c:.4f}")
        print(f"R² Predição: {R2p:.4f}")
        print(f"RMSEC: {RMSEC:.4f} {unidade_medida}")
        print(f"RMSECV: {RMSECV:.4f} {unidade_medida}")
        print(f"RMSEP: {RMSEP:.4f} {unidade_medida}")
        if selected_idx is not None:
            print(f"Variáveis selecionadas ({var_select}): {len(selected_idx)}/{X_cal_orig.shape[1]}")
    
    metrics = {
        'R2c': R2c, 'R2p': R2p, 
        'RMSEC': RMSEC, 'RMSECV': RMSECV, 'RMSEP': RMSEP,
        'n_components': best_comp,
        'selected_variables': selected_idx if selected_idx is not None else list(range(X_cal_orig.shape[1]))
    }
    
    return coef, metrics, df_results

