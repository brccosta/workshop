# Análise de Sódio - MATLAB

Este projeto implementa uma análise quimiométrica de sódio usando MATLAB, correspondente ao projeto Python/v2.

## Estrutura do Projeto

```
MATLAB/
├── chemometrics/           # Módulos de quimiometria
│   ├── preprocessing.m    # Pré-processamento espectroscópico
│   ├── modeling.m        # Modelagem quimiométrica
│   └── utils.m           # Utilitários para análise
├── data/                  # Dados
│   └── dados.xlsx        # Arquivo de dados espectrais
├── sodium_analysis.m     # Script principal de análise
└── README.md             # Este arquivo
```

## Funcionalidades

### Pré-processamento (`preprocessing.m`)
- **SNV (Standard Normal Variate)**: Normalização por espectro
- **Savitzky-Golay**: Filtro de suavização e derivação
- **MSC (Multiplicative Scatter Correction)**: Correção de espalhamento
- **Mean Centering**: Centramento na média
- **Autoscaling**: Escalamento automático

### Modelagem (`modeling.m`)
- **PLS (Partial Least Squares)**: Regressão por mínimos quadrados parciais
- Otimização automática do número de componentes
- Cálculo de VIP scores
- Validação cruzada

### Utilitários (`utils.m`)
- Divisão de dados por grupos (réplicas)
- Algoritmo Kennard-Stone
- Algoritmo SPXY
- Cálculo de métricas de avaliação
- Comparação de modelos

## Uso

### Configuração Inicial
```matlab
% Adicionar diretórios ao path do MATLAB
addpath('chemometrics');
addpath('data');

% Verificar se o arquivo de dados existe
if exist('data/dados.xlsx', 'file')
    fprintf('Arquivo de dados encontrado!\n');
else
    error('Arquivo data/dados.xlsx não encontrado!');
end
```

### Execução Completa
```matlab
% Executar análise completa
sodium_analysis();
```

### Uso Individual dos Módulos

#### Pré-processamento
```matlab
% Aplicar SNV
X_snv = preprocessing(X, 'snv', 'plot', true);

% Aplicar Savitzky-Golay
X_savgol = preprocessing(X, 'savgol', 'window_length', 21, 'deriv', 1);

% Aplicar MSC
X_msc = preprocessing(X, 'msc', 'reference', reference_spectrum);
```

#### Modelagem
```matlab
% Treinar modelo PLS
model = modeling(X, y, 'pls', 'n_components', 5);

% Fazer predições
y_pred = predict(model, X_test);

% Avaliar modelo
metrics = evaluate_model(model, X_cal, y_cal, X_test, y_test, 'mEq/L');
```

#### Divisão de Dados
```matlab
% Divisão por grupos
[train_idx, test_idx] = utils(X, y, 'groups', 'groups', sample_ids);

% Algoritmo Kennard-Stone
[train_idx, test_idx] = utils(X, y, 'kennard_stone');

% Algoritmo SPXY
[train_idx, test_idx] = utils(X, y, 'spxy');
```

## Requisitos

- MATLAB R2018b ou superior
- Statistics and Machine Learning Toolbox
- Signal Processing Toolbox (para Savitzky-Golay)

## Dados

O arquivo `data/dados.xlsx` deve estar localizado no diretório `data/` e conter:
- **Sheet 'original'**: Dados espectrais brutos
- **Coluna 2**: IDs das amostras
- **Coluna 3**: Concentração de sódio (mEq/L)
- **Colunas 7+**: Dados espectrais (1459 variáveis)

### Estrutura do Diretório de Dados
```
data/
├── dados.xlsx        # Arquivo principal de dados
└── README.md         # Documentação dos dados
```

## Métricas de Avaliação

- **R²**: Coeficiente de determinação
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Bias**: Viés
- **RPD**: Ratio of Performance to Deviation
- **RER**: Range Error Ratio

## Exemplo de Saída

```
=== ANÁLISE DE SÓDIO COM CHEMOMETRICS ===

Carregando dados...
Dados carregados: 120 amostras, 1459 variáveis

Aplicando pré-processamento...
Pré-processamento concluído

Dividindo dados...
Dados divididos: 84 treino, 36 teste

Treinando modelo PLS...
Otimizando número de componentes...
Número ótimo de componentes: 5 (R² = 0.892)
Modelo PLS treinado com 5 componentes
Modelo treinado com sucesso

Avaliando modelo...

=== MÉTRICAS DO MODELO PLS ===
R²c: 0.9234
R²p: 0.8756
RMSEC: 0.1234 mEq/L
RMSEP: 0.1567 mEq/L
Componentes: 5

=== ANÁLISE CONCLUÍDA COM SUCESSO ===
```

## Boas Práticas

1. **Estrutura de Código**: Use funções modulares e bem documentadas
2. **Tratamento de Erros**: Implemente validação de entrada e tratamento de exceções
3. **Documentação**: Documente todas as funções com help e exemplos
4. **Reprodutibilidade**: Use sementes aleatórias para resultados reproduzíveis
5. **Visualização**: Inclua plots informativos para análise dos resultados

## Diferenças do Python

- **Sintaxe**: MATLAB usa sintaxe diferente (sem classes, mais funcional)
- **Estruturas**: Uso de structs em vez de classes
- **Plotting**: Funções de plot integradas do MATLAB
- **Tratamento de Dados**: Uso de tabelas e arrays nativos do MATLAB

## Troubleshooting

### Erro: "Function not found"
- Verifique se todos os arquivos estão no path do MATLAB
- Use `addpath('chemometrics')` para adicionar o diretório
- Execute `which sodium_analysis` para verificar se a função está no path

### Erro: "Toolbox not found"
- Instale as toolboxes necessárias (Statistics and Machine Learning, Signal Processing)
- Verifique com `ver` se as toolboxes estão instaladas

### Erro: "File not found"
- Verifique se o arquivo `data/dados.xlsx` existe no diretório correto
- Execute `ls data/` para listar os arquivos no diretório
- Verifique se o arquivo tem as permissões corretas de leitura

### Erro: "Sheet not found"
- Verifique se o arquivo Excel tem a sheet 'original'
- Use `xlsfinfo('data/dados.xlsx')` para listar as sheets disponíveis

### Erro: "Invalid data format"
- Verifique se as colunas estão nas posições corretas (2, 3, 7+)
- Execute `readtable('data/dados.xlsx', 'Sheet', 'original', 'ReadVariableNames', false)` para inspecionar os dados

## Contribuição

Para contribuir com o projeto:
1. Mantenha a compatibilidade com versões anteriores
2. Documente novas funcionalidades
3. Teste com diferentes conjuntos de dados
4. Siga as convenções de nomenclatura do MATLAB
