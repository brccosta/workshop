function [train_idx, test_idx] = utils(X, y, method, varargin)
%UTILS Utilitários para análise quimiométrica
%
% Sintaxe:
%   [train_idx, test_idx] = utils(X, y, method)
%   [train_idx, test_idx] = utils(X, y, method, 'Name', Value)
%
% Parâmetros:
%   X - Dados de entrada (amostras x variáveis)
%   y - Variável resposta (amostras x 1)
%   method - Método de divisão:
%           'groups' - Divisão por grupos (réplicas)
%           'kennard_stone' - Algoritmo Kennard-Stone
%           'spxy' - Algoritmo SPXY
%
% Propriedades (Name-Value pairs):
%   'test_size' - proporção para teste (default: 0.3)
%   'random_state' - semente para reprodutibilidade (default: 42)
%   'groups' - grupos para divisão (default: [])
%
% Exemplo:
%   [train_idx, test_idx] = utils(X, y, 'groups', 'groups', sample_ids);

    % Configurar parser de argumentos
    p = inputParser;
    addRequired(p, 'X', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addRequired(p, 'method', @ischar);
    addParameter(p, 'test_size', 0.3, @isnumeric);
    addParameter(p, 'random_state', 42, @isnumeric);
    addParameter(p, 'groups', [], @isnumeric);
    
    parse(p, X, y, method, varargin{:});
    
    % Validar entrada
    if isempty(X) || isempty(y)
        error('Dados de entrada não podem estar vazios.');
    end
    
    if size(X, 1) ~= size(y, 1)
        error('Número de amostras em X e y deve ser igual.');
    end
    
    % Aplicar método de divisão
    switch lower(method)
        case 'groups'
            [train_idx, test_idx] = split_with_groups(X, y, p.Results.groups, ...
                                                    p.Results.test_size, p.Results.random_state);
        case 'kennard_stone'
            [train_idx, test_idx] = split_kennard_stone(X, y, p.Results.test_size);
        case 'spxy'
            [train_idx, test_idx] = split_spxy(X, y, p.Results.test_size);
        otherwise
            error('Método de divisão não reconhecido: %s', method);
    end
end

function [train_idx, test_idx] = split_with_groups(X, y, groups, test_size, random_state)
% Divide os dados considerando grupos (ex: réplicas)
    if isempty(groups)
        % Divisão aleatória simples
        rng(random_state);
        n_samples = size(X, 1);
        n_test = round(n_samples * test_size);
        test_idx = randperm(n_samples, n_test);
        train_idx = setdiff(1:n_samples, test_idx);
    else
        % Divisão por grupos
        unique_groups = unique(groups);
        n_groups = length(unique_groups);
        n_test_groups = round(n_groups * test_size);
        
        rng(random_state);
        test_groups = unique_groups(randperm(n_groups, n_test_groups));
        
        test_idx = ismember(groups, test_groups);
        train_idx = ~test_idx;
        
        train_idx = find(train_idx);
        test_idx = find(test_idx);
    end
    
    fprintf('Dados divididos: %d treino, %d teste\n', length(train_idx), length(test_idx));
end

function [train_idx, test_idx] = split_kennard_stone(X, y, test_size)
% Divide os dados usando algoritmo Kennard-Stone
    n_samples = size(X, 1);
    n_cal = n_samples - round(n_samples * test_size);
    
    % Calcular matriz de distância
    dist_matrix = pdist2(X, X);
    
    % Selecionar primeira amostra (maior distância)
    [~, max_idx] = max(dist_matrix(:));
    [i, j] = ind2sub(size(dist_matrix), max_idx);
    
    cal_idx = [i, j];
    rem_idx = setdiff(1:n_samples, cal_idx);
    
    % Selecionar amostras de calibração
    for k = 3:n_cal
        min_dists = min(dist_matrix(rem_idx, cal_idx), [], 2);
        [~, new_idx] = max(min_dists);
        cal_idx = [cal_idx, rem_idx(new_idx)];
        rem_idx(new_idx) = [];
    end
    
    train_idx = cal_idx;
    test_idx = rem_idx;
    
    fprintf('Kennard-Stone: %d treino, %d teste\n', length(train_idx), length(test_idx));
end

function [train_idx, test_idx] = split_spxy(X, y, test_size)
% Divide os dados usando algoritmo SPXY
    % Normalizar dados
    X_s = normalize_data(X);
    y_s = normalize_data(y);
    
    % Calcular distâncias
    dist_x = pdist2(X_s, X_s);
    dist_y = pdist2(y_s, y_s);
    
    % Combinar distâncias
    dist_matrix = (dist_x / max(dist_x(:))) + (dist_y / max(dist_y(:)));
    
    n_samples = size(X, 1);
    n_cal = n_samples - round(n_samples * test_size);
    
    % Selecionar primeira amostra
    [~, max_idx] = max(dist_matrix(:));
    [i, j] = ind2sub(size(dist_matrix), max_idx);
    
    cal_idx = [i, j];
    rem_idx = setdiff(1:n_samples, cal_idx);
    
    % Selecionar amostras de calibração
    for k = 3:n_cal
        min_dists = min(dist_matrix(rem_idx, cal_idx), [], 2);
        [~, new_idx] = max(min_dists);
        cal_idx = [cal_idx, rem_idx(new_idx)];
        rem_idx(new_idx) = [];
    end
    
    train_idx = cal_idx;
    test_idx = rem_idx;
    
    fprintf('SPXY: %d treino, %d teste\n', length(train_idx), length(test_idx));
end

function X_norm = normalize_data(X)
% Normaliza os dados para o intervalo [0, 1]
    X_min = min(X, [], 1);
    X_max = max(X, [], 1);
    X_norm = (X - X_min) ./ (X_max - X_min);
    
    % Evitar divisão por zero
    X_norm(isnan(X_norm)) = 0;
end

function metrics = calculate_metrics(y_true, y_pred)
% Calcula métricas de avaliação
    % Métricas básicas
    r2 = calculate_r2(y_true, y_pred);
    rmse = calculate_rmse(y_true, y_pred);
    mae = calculate_mae(y_true, y_pred);
    
    % MAPE (Mean Absolute Percentage Error)
    mape = mean(abs((y_true - y_pred) ./ y_true)) * 100;
    
    % Bias
    bias = mean(y_pred - y_true);
    
    % RPD (Ratio of Performance to Deviation)
    rpd = std(y_true) / rmse;
    
    % RER (Range Error Ratio)
    rer = (max(y_true) - min(y_true)) / rmse;
    
    % Criar estrutura de métricas
    metrics = struct();
    metrics.R2 = r2;
    metrics.RMSE = rmse;
    metrics.MAE = mae;
    metrics.MAPE = mape;
    metrics.Bias = bias;
    metrics.RPD = rpd;
    metrics.RER = rer;
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

function mae = calculate_mae(y_true, y_pred)
% Calcula Mean Absolute Error
    mae = mean(abs(y_true - y_pred));
end

function print_metrics(metrics, title)
% Imprime as métricas de forma formatada
    if nargin < 2
        title = 'Métricas';
    end
    
    fprintf('\n=== %s ===\n', upper(title));
    
    fields = fieldnames(metrics);
    for i = 1:length(fields)
        value = metrics.(fields{i});
        if isnumeric(value)
            fprintf('%s: %.4f\n', fields{i}, value);
        else
            fprintf('%s: %s\n', fields{i}, value);
        end
    end
end

function compare_models(metrics_list, model_names)
% Compara métricas de diferentes modelos
    if length(metrics_list) ~= length(model_names)
        error('Número de métricas deve ser igual ao número de nomes.');
    end
    
    fprintf('\n=== COMPARAÇÃO DE MODELOS ===\n');
    fprintf('%-15s %-8s %-8s %-8s %-8s\n', 'Modelo', 'R²', 'RMSE', 'MAE', 'MAPE');
    fprintf('%-15s %-8s %-8s %-8s %-8s\n', '---------------', '--------', '--------', '--------', '--------');
    
    for i = 1:length(metrics_list)
        metrics = metrics_list{i};
        name = model_names{i};
        
        r2 = get_metric_value(metrics, 'R2');
        rmse = get_metric_value(metrics, 'RMSE');
        mae = get_metric_value(metrics, 'MAE');
        mape = get_metric_value(metrics, 'MAPE');
        
        fprintf('%-15s %-8.3f %-8.3f %-8.3f %-8.1f\n', name, r2, rmse, mae, mape);
    end
end

function value = get_metric_value(metrics, field)
% Obtém valor de métrica com fallback
    if isfield(metrics, field)
        value = metrics.(field);
    else
        % Tentar variações do nome
        variations = {field, [field 'c'], [field 'p']};
        for i = 1:length(variations)
            if isfield(metrics, variations{i})
                value = metrics.(variations{i});
                return;
            end
        end
        value = 0;
    end
end

