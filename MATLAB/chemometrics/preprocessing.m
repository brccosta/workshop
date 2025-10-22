function X_processed = preprocessing(X, method, varargin)
%PREPROCESSING Aplica pré-processamento aos dados espectrais
%
% Sintaxe:
%   X_processed = preprocessing(X, method)
%   X_processed = preprocessing(X, method, 'Name', Value)
%
% Parâmetros:
%   X - Dados espectrais (amostras x variáveis)
%   method - Método de pré-processamento:
%           'snv' - Standard Normal Variate
%           'savgol' - Savitzky-Golay
%           'msc' - Multiplicative Scatter Correction
%           'mean_center' - Mean Centering
%           'autoscale' - Autoscaling
%
% Propriedades (Name-Value pairs):
%   'plot' - true/false para plotar resultados (default: false)
%   'window_length' - comprimento da janela para Savitzky-Golay (default: 15)
%   'polyorder' - ordem do polinômio para Savitzky-Golay (default: 2)
%   'deriv' - ordem da derivada para Savitzky-Golay (default: 0)
%   'reference' - espectro de referência para MSC (default: média)
%
% Exemplo:
%   X_processed = preprocessing(X, 'snv', 'plot', true);
%   X_processed = preprocessing(X, 'savgol', 'window_length', 21, 'deriv', 1);

    % Configurar parser de argumentos
    p = inputParser;
    addRequired(p, 'X', @isnumeric);
    addRequired(p, 'method', @ischar);
    addParameter(p, 'plot', false, @islogical);
    addParameter(p, 'window_length', 15, @isnumeric);
    addParameter(p, 'polyorder', 2, @isnumeric);
    addParameter(p, 'deriv', 0, @isnumeric);
    addParameter(p, 'reference', [], @isnumeric);
    
    parse(p, X, method, varargin{:});
    
    % Validar entrada
    if isempty(X)
        error('Dados de entrada não podem estar vazios.');
    end
    
    if size(X, 1) == 1
        X = X'; % Transpor se necessário
    end
    
    % Aplicar método de pré-processamento
    switch lower(method)
        case 'snv'
            X_processed = apply_snv(X);
        case 'savgol'
            X_processed = apply_savgol(X, p.Results.window_length, ...
                                     p.Results.polyorder, p.Results.deriv);
        case 'msc'
            X_processed = apply_msc(X, p.Results.reference);
        case 'mean_center'
            X_processed = apply_mean_center(X);
        case 'autoscale'
            X_processed = apply_autoscale(X);
        otherwise
            error('Método de pré-processamento não reconhecido: %s', method);
    end
    
    % Plotar resultados se solicitado
    if p.Results.plot
        plot_preprocessing_results(X, X_processed, method);
    end
end

function X_snv = apply_snv(X)
% Aplica Standard Normal Variate (SNV)
    % Calcular média e desvio padrão de cada espectro
    mean_spectrum = mean(X, 2);
    std_spectrum = std(X, 0, 2);
    
    % Evitar divisão por zero
    std_spectrum(std_spectrum == 0) = 1;
    
    % Aplicar SNV
    X_snv = (X - mean_spectrum) ./ std_spectrum;
end

function X_savgol = apply_savgol(X, window_length, polyorder, deriv)
% Aplica filtro Savitzky-Golay
    try
        X_savgol = zeros(size(X));
        for i = 1:size(X, 1)
            X_savgol(i, :) = sgolayfilt(X(i, :), polyorder, window_length, deriv);
        end
    catch ME
        warning('Erro no filtro Savitzky-Golay: %s', ME.message);
        X_savgol = X;
    end
end

function X_msc = apply_msc(X, reference)
% Aplica Multiplicative Scatter Correction (MSC)
    if isempty(reference)
        reference_spectrum = mean(X, 1);
    else
        reference_spectrum = reference;
    end
    
    % Centralizar os dados
    X_centered = X - mean(X, 2);
    reference_centered = reference_spectrum - mean(reference_spectrum);
    
    % Calcular coeficientes de regressão
    try
        coeffs = reference_centered' \ X_centered';
        
        % Ajustar os dados
        X_msc = (X_centered - coeffs' * reference_centered) + mean(X, 2);
    catch
        warning('Erro na decomposição QR. Retornando dados originais.');
        X_msc = X;
    end
end

function X_centered = apply_mean_center(X)
% Aplica centramento na média
    X_centered = X - mean(X, 1);
end

function X_scaled = apply_autoscale(X)
% Aplica autoscaling
    X_scaled = (X - mean(X, 1)) ./ std(X, 0, 1);
    
    % Evitar divisão por zero
    X_scaled(isnan(X_scaled)) = 0;
    X_scaled(isinf(X_scaled)) = 0;
end

function plot_preprocessing_results(X_original, X_processed, method)
% Plota os resultados do pré-processamento
    figure('Position', [100, 100, 1200, 600]);
    
    % Espectro original
    subplot(2, 1, 1);
    plot(X_original(1, :)', 'k-', 'LineWidth', 1);
    title('Espectro Original');
    ylabel('Absorbância');
    grid on;
    
    % Espectro processado
    subplot(2, 1, 2);
    plot(X_processed(1, :)', 'g-', 'LineWidth', 1);
    title(sprintf('Espectro Processado - %s', upper(method)));
    xlabel('Comprimento de Onda');
    ylabel('Absorbância');
    grid on;
    
    sgtitle(sprintf('Pré-processamento: %s', upper(method)));
end

