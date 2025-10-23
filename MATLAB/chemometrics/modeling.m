function model = modeling(X, y, method, varargin)
%MODELING Treina modelos de regressão quimiométrica
%
% Sintaxe:
%   model = modeling(X, y, method)
%   model = modeling(X, y, method, 'Name__', Value)
%
% Parâmetros:
%   X - Dados de entrada (amostras x variáveis)
%   y - Variável resposta (amostras x 1)
%   method - Método de modelagem:
%           'pls' - Partial Least Squares
%
% Propriedades (Name-Value pairs):
%   'n_components' - número de componentes (default: auto)
%   'cv_folds' - número de folds para validação cruzada (default: 5)
%   'optimize_components' - otimizar componentes automaticamente (default: true)
%
% Exemplo:
%   model = modeling(X, y, 'pls', 'n_components', 5);

    % Configurar parser de argumentos
    p = inputParser;
    addRequired(p, 'X', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addRequired(p, 'method', @ischar);
    addParameter(p, 'n_components', [], @isnumeric);
    addParameter(p, 'cv_folds', 5, @isnumeric);
    addParameter(p, 'optimize_components', true, @islogical);
    
    parse(p, X, y, method, varargin{:});
    
    % Validar entrada
    if isempty(X) || isempty(y)
        error('Dados de entrada não podem estar vazios.');
    end
    
    if size(X, 1) ~= size(y, 1)
        error('Número de amostras em X e y deve ser igual.');
    end
    
    % Aplicar método de modelagem
    switch lower(method)
        case 'pls'
            model = train_pls(X, y, p.Results.n_components, ...
                            p.Results.cv_folds, p.Results.optimize_components);
        otherwise
            error('Método de modelagem não reconhecido: %s', method);
    end
end

function pls_model = train_pls(X, y, n_components, cv_folds, optimize_components)
% Treina modelo PLS
    % Otimizar número de componentes se necessário
    if optimize_components && isempty(n_components)
        n_components = optimize_pls_components(X, y, cv_folds);
    elseif isempty(n_components)
        n_components = min(10, size(X, 1) - 1, size(X, 2));
    end
    
    % Treinar modelo PLS
    try
        [XL, YL, XS, YS, BETA, PCTVAR, MSE, stats] = plsregress(X, y, n_components);
        
        % Criar estrutura do modelo
        pls_model = struct();
        pls_model.method = 'pls';
        pls_model.n_components = n_components;
        pls_model.XL = XL;
        pls_model.YL = YL;
        pls_model.XS = XS;
        pls_model.YS = YS;
        pls_model.BETA = BETA;
        pls_model.PCTVAR = PCTVAR;
        pls_model.MSE = MSE;
        pls_model.stats = stats;
        pls_model.is_fitted = true;
        
        % Calcular VIP scores
        pls_model.vip_scores = calculate_vip_scores(X, XL, YL);
        
        fprintf('Modelo PLS treinado com %d componentes\n', n_components);
        
    catch ME
        error('Erro ao treinar modelo PLS: %s', ME.message);
    end
end

function n_comp_opt = optimize_pls_components(X, y, cv_folds)
% Otimiza o número de componentes por validação cruzada
    n_comp_max = min(10, size(X, 1) - 1, size(X, 2));
    best_score = -inf;
    best_n_comp = 1;
    
    fprintf('Otimizando número de componentes...\n');
    
    for n_comp = 1:n_comp_max
        try
            % Validação cruzada
            cv_scores = zeros(cv_folds, 1);
            indices = crossvalind('Kfold', size(X, 1), cv_folds);
            
            for fold = 1:cv_folds
                test_idx = (indices == fold);
                train_idx = ~test_idx;
                
                % Treinar modelo
                [XL, YL, XS, YS, BETA] = plsregress(X(train_idx, :), y(train_idx), n_comp);
                
                % Predizer
                y_pred = [ones(sum(test_idx), 1), X(test_idx, :)] * BETA;
                
                % Calcular R²
                cv_scores(fold) = calculate_r2(y(test_idx), y_pred);
            end
            
            mean_score = mean(cv_scores);
            
            if mean_score > best_score
                best_score = mean_score;
                best_n_comp = n_comp;
            end
            
        catch
            continue;
        end
    end
    
    fprintf('Número ótimo de componentes: %d (R² = %.3f)\n', best_n_comp, best_score);
    n_comp_opt = best_n_comp;
end

function vip = calculate_vip_scores(X, XL, YL)
% Calcula VIP scores para seleção de variáveis
    p = size(X, 2);
    n_comp = size(XL, 2);
    
    vip = zeros(p, 1);
    
    % Calcular VIP scores
    for i = 1:p
        weight_sum = 0;
        total_sum = 0;
        
        for j = 1:n_comp
            weight = (XL(i, j) / norm(XL(:, j)))^2;
            weight_sum = weight_sum + weight * YL(j)^2;
            total_sum = total_sum + YL(j)^2;
        end
        
        vip(i) = sqrt(p * weight_sum / total_sum);
    end
end

function y_pred = predict(model, X)
% Faz predições usando o modelo ajustado
    if ~model.is_fitted
        error('Modelo não foi ajustado. Chame modeling() primeiro.');
    end
    
    switch model.method
        case 'pls'
            y_pred = [ones(size(X, 1), 1), X] * model.BETA;
        otherwise
            error('Método de modelo não reconhecido: %s', model.method);
    end
end

function metrics = evaluate_model(model, X_cal, y_cal, X_test, y_test, unit)
% Avalia o modelo com métricas completas
    if nargin < 6
        unit = 'units';
    end
    
    if ~model.is_fitted
        error('Modelo não foi ajustado. Chame modeling() primeiro.');
    end
    
    % Predições
    y_cal_pred = predict(model, X_cal);
    y_test_pred = predict(model, X_test);
    
    % Métricas de calibração
    r2c = calculate_r2(y_cal, y_cal_pred);
    rmsec = calculate_rmse(y_cal, y_cal_pred);
    
    % Métricas de predição
    r2p = calculate_r2(y_test, y_test_pred);
    rmsep = calculate_rmse(y_test, y_test_pred);
    
    % Criar estrutura de métricas
    metrics = struct();
    metrics.R2c = r2c;
    metrics.R2p = r2p;
    metrics.RMSEC = rmsec;
    metrics.RMSEP = rmsep;
    metrics.n_components = model.n_components;
    metrics.unit = unit;
    
    % Exibir métricas
    print_metrics(metrics, 'Métricas do Modelo PLS');
end

function r2 = calculate_r2(y_true, y_pred)
% Calcula coeficiente de determinação R²
    ss_res = sum((y_true - y_pred).^2);
    ss_tot = sum((y_true - mean(y_true)).^2);
    r2 = 1 - (ss_res / ss_tot);
end

function rmse = calculate_rmse(y_true, y_pred)
% Calcula Root Mean Square Error
    rmse = sqrt(mean((y_true - y_pred).^2));
end

function print_metrics(metrics, title)
% Imprime as métricas de forma formatada
    fprintf('\n=== %s ===\n', upper(title));
    fprintf('R²c: %.4f\n', metrics.R2c);
    fprintf('R²p: %.4f\n', metrics.R2p);
    fprintf('RMSEC: %.4f %s\n', metrics.RMSEC, metrics.unit);
    fprintf('RMSEP: %.4f %s\n', metrics.RMSEP, metrics.unit);
    fprintf('Componentes: %d\n', metrics.n_components);
end

function plot_results(model, X_cal, y_cal, X_test, y_test, unit)
% Plota os resultados do modelo
    if nargin < 6
        unit = 'units';
    end
    
    % Predições
    y_cal_pred = predict(model, X_cal);
    y_test_pred = predict(model, X_test);
    
    % Métricas
    r2c = calculate_r2(y_cal, y_cal_pred);
    r2p = calculate_r2(y_test, y_test_pred);
    rmsec = calculate_rmse(y_cal, y_cal_pred);
    rmsep = calculate_rmse(y_test, y_test_pred);
    
    % Criar figura
    figure('Position', [100, 100, 1500, 600]);
    
    % Gráfico Predito vs Real
    subplot(1, 2, 1);
    scatter(y_cal, y_cal_pred, 50, 'b', 'filled', 'Alpha', 0.7);
    hold on;
    scatter(y_test, y_test_pred, 50, 'r', 'filled', 'Alpha', 0.7);
    
    % Linha de referência
    min_val = min([y_cal; y_test; y_cal_pred; y_test_pred]);
    max_val = max([y_cal; y_test; y_cal_pred; y_test_pred]);
    plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 2);
    
    xlabel(sprintf('Real (%s)', unit));
    ylabel(sprintf('Predito (%s)', unit));
    legend('Calibração', 'Teste', 'Location', 'best');
    title(sprintf('Predito vs Real\nR²c = %.3f, R²p = %.3f', r2c, r2p));
    grid on;
    
    % Gráfico de Resíduos
    subplot(1, 2, 2);
    residuals_cal = y_cal - y_cal_pred;
    residuals_test = y_test - y_test_pred;
    
    scatter(y_cal_pred, residuals_cal, 50, 'b', 'filled', 'Alpha', 0.7);
    hold on;
    scatter(y_test_pred, residuals_test, 50, 'r', 'filled', 'Alpha', 0.7);
    yline(0, 'k--', 'LineWidth', 2);
    
    xlabel(sprintf('Predito (%s)', unit));
    ylabel(sprintf('Resíduos (%s)', unit));
    legend('Calibração', 'Teste', 'Location', 'best');
    title(sprintf('Resíduos\nRMSEC = %.3f, RMSEP = %.3f', rmsec, rmsep));
    grid on;
    
    sgtitle('Resultados do Modelo PLS');
end

function plot_vip_scores(model, threshold)
% Plota os VIP scores
    if nargin < 2
        threshold = 1.0;
    end
    
    if isempty(model.vip_scores)
        error('VIP scores não foram calculados. Ajuste o modelo primeiro.');
    end
    
    figure('Position', [100, 100, 1200, 600]);
    
    % Plotar VIP scores
    colors = repmat({'blue'}, length(model.vip_scores), 1);
    colors(model.vip_scores > threshold) = {'red'};
    
    bar(model.vip_scores, 'FaceColor', 'flat', 'CData', ...
        [0 0 1; 1 0 0] * (model.vip_scores > threshold)' + ...
        [0 0 1; 0 0 1] * (model.vip_scores <= threshold)');
    
    % Linha de referência
    yline(threshold, 'r--', 'LineWidth', 2, ...
          'Label', sprintf('Limiar = %.1f', threshold));
    
    xlabel('Variáveis');
    ylabel('VIP Score');
    title('VIP Scores - Importância das Variáveis');
    grid on;
    
    % Rotacionar labels se necessário
    if length(model.vip_scores) > 20
        xtickangle(45);
    end
end

