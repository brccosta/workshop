function sodium_analysis()
%SODIUM_ANALYSIS Análise de sódio usando biblioteca Chemometrics
%
% Este script demonstra como usar a biblioteca para:
% 1. Carregar dados espectrais
% 2. Aplicar pré-processamento
% 3. Dividir os dados
% 4. Treinar modelo PLS
% 5. Avaliar resultados
%
% Exemplo:
%   sodium_analysis();

    fprintf('=== ANÁLISE DE SÓDIO COM CHEMOMETRICS ===\n\n');
    
    try
        % 1. Carregar dados
        [X, y, amostras, variaveis] = load_data();
        
        % 2. Pré-processamento
        X_processed = preprocess_data(X);
        
        % 3. Divisão dos dados
        [X_train, X_test, y_train, y_test, train_idx, test_idx] = split_data(...
            X_processed, y, amostras);
        
        % 4. Treinar modelo
        model = train_model(X_train, y_train);
        
        % 5. Avaliar modelo
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test);
        
        fprintf('\n=== ANÁLISE CONCLUÍDA COM SUCESSO ===\n');
        
    catch ME
        fprintf('Erro na análise: %s\n', ME.message);
        rethrow(ME);
    end
end

function [X, y, amostras, variaveis] = load_data(file_path)
%LOAD_DATA Carrega os dados do arquivo Excel
%
% Parâmetros:
%   file_path - Caminho para o arquivo de dados (default: 'data/dados.xlsx')
%
% Retorna:
%   X - Dados espectrais (amostras x variáveis)
%   y - Variável resposta (amostras x 1)
%   amostras - IDs das amostras
%   variaveis - Comprimentos de onda

    if nargin < 1
        file_path = 'data/dados.xlsx';
    end
    
    fprintf('Carregando dados...\n');
    
    try
        % Ler dados do Excel
        dados_brutos = readtable(file_path, 'Sheet', 'original', 'ReadVariableNames', false);
        
        % Extrair dados
        X = table2array(dados_brutos(2:end, 7:end));      % Dados espectrais
        y = table2array(dados_brutos(2:end, 3));          % Variável resposta
        amostras = table2array(dados_brutos(2:end, 2));   % IDs das amostras
        variaveis = table2array(dados_brutos(1, 7:end));   % Comprimentos de onda
        
        % Remover variáveis (manter apenas 1459)
        X = X(:, 1:1459);
        variaveis = variaveis(1:1459);
        
        fprintf('Dados carregados: %d amostras, %d variáveis\n', size(X, 1), size(X, 2));
        
    catch ME
        error('Erro ao carregar dados: %s', ME.message);
    end
end

function X_processed = preprocess_data(X, plot)
%PREPROCESS_DATA Aplica pré-processamento aos dados espectrais
%
% Parâmetros:
%   X - Dados espectrais
%   plot - Se true, plota os resultados (default: false)
%
% Retorna:
%   X_processed - Dados pré-processados

    if nargin < 2
        plot = false;
    end
    
    fprintf('Aplicando pré-processamento...\n');
    
    % 1. Savitzky-Golay (suavização e derivação)
    X_processed = preprocessing(X, 'savgol', 'window_length', 15, ...
                              'polyorder', 2, 'deriv', 1, 'plot', plot);
    
    % 2. SNV (Standard Normal Variate)
    X_processed = preprocessing(X_processed, 'snv', 'plot', plot);
    
    % 3. Mean Centering
    X_processed = preprocessing(X_processed, 'mean_center', 'plot', plot);
    
    fprintf('Pré-processamento concluído\n');
end

function [X_train, X_test, y_train, y_test, train_idx, test_idx] = split_data(X, y, amostras)
%SPLIT_DATA Divide os dados em treino e teste
%
% Parâmetros:
%   X - Dados de entrada
%   y - Variável resposta
%   amostras - IDs das amostras
%
% Retorna:
%   X_train, X_test - Dados de treino e teste
%   y_train, y_test - Resposta de treino e teste
%   train_idx, test_idx - Índices de treino e teste

    fprintf('Dividindo dados...\n');
    
    % Usar divisão por grupos (réplicas)
    [train_idx, test_idx] = utils(X, y, 'groups', 'test_size', 0.3, ...
                                 'random_state', 28, 'groups', amostras);
    
    % Dividir os dados
    X_train = X(train_idx, :);
    X_test = X(test_idx, :);
    y_train = y(train_idx);
    y_test = y(test_idx);
    
    fprintf('Dados divididos: %d treino, %d teste\n', length(train_idx), length(test_idx));
end

function model = train_model(X_train, y_train)
%TRAIN_MODEL Treina o modelo PLS
%
% Parâmetros:
%   X_train - Dados de treino
%   y_train - Resposta de treino
%
% Retorna:
%   model - Modelo treinado

    fprintf('Treinando modelo PLS...\n');
    
    % Criar e treinar modelo
    model = modeling(X_train, y_train, 'pls', 'cv_folds', 5, ...
                   'optimize_components', true);
    
    fprintf('Modelo treinado com sucesso\n');
end

function metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
%EVALUATE_MODEL Avalia o modelo e exibe resultados
%
% Parâmetros:
%   model - Modelo treinado
%   X_train, y_train - Dados de treino
%   X_test, y_test - Dados de teste
%
% Retorna:
%   metrics - Métricas de avaliação

    fprintf('Avaliando modelo...\n');
    
    % Avaliar modelo
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, 'mEq/L');
    
    % Plotar resultados
    plot_results(model, X_train, y_train, X_test, y_test, 'mEq/L');
    
    % Plotar VIP scores
    plot_vip_scores(model, 1.0);
    
    % Criar tabela com resultados
    results_table = create_results_table(model, X_train, y_train, X_test, y_test);
    fprintf('\nPrimeiras linhas dos resultados:\n');
    disp(head(results_table, 10));
end

function results_table = create_results_table(model, X_cal, y_cal, X_test, y_test)
%CREATE_RESULTS_TABLE Cria tabela com resultados detalhados
%
% Parâmetros:
%   model - Modelo treinado
%   X_cal, y_cal - Dados de calibração
%   X_test, y_test - Dados de teste
%
% Retorna:
%   results_table - Tabela com resultados

    % Predições
    y_cal_pred = predict(model, X_cal);
    y_test_pred = predict(model, X_test);
    
    % Criar tabela
    conjunto = [repmat({'calibração'}, length(y_cal), 1); ...
                repmat({'teste'}, length(y_test), 1)];
    
    y_real = [y_cal; y_test];
    y_predito = [y_cal_pred; y_test_pred];
    
    erro_absoluto = abs(y_real - y_predito);
    erro_percentual = abs((y_real - y_predito) ./ y_real) * 100;
    erro_percentual(isnan(erro_percentual)) = 0;
    
    results_table = table(conjunto, y_real, y_predito, erro_absoluto, erro_percentual, ...
                         'VariableNames', {'conjunto', 'y_real', 'y_predito', ...
                                         'erro_absoluto', 'erro_percentual'});
end

