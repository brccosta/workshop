# Chemometrics Library

Uma biblioteca Python para análise quimiométrica de dados espectrais, desenvolvida seguindo as melhores práticas de Engenharia de Software e Arquitetura de Software.

## Visão Geral

Esta biblioteca fornece ferramentas para:
- **Pré-processamento espectroscópico**: SNV, Savitzky-Golay, MSC, Mean Centering, Autoscaling
- **Modelagem quimiométrica**: Regressão PLS com otimização automática
- **Divisão de dados**: Kennard-Stone, SPXY, divisão com grupos
- **Avaliação**: Métricas completas e visualizações

## Arquitetura do Projeto

```
workshop/
├── chemometrics/              # Biblioteca principal
│   ├── __init__.py           # Interface pública da biblioteca
│   ├── preprocessing.py      # Técnicas de pré-processamento espectroscópico
│   ├── modeling.py          # Modelos de regressão PLS
│   └── utils.py             # Utilitários (divisão de dados, métricas)
├── config/                   # Configurações
│   └── settings.py          # Configurações do projeto
├── examples/                 # Exemplos de uso
│   └── example_sodium_analysis.py
├── data/                     # Dados
│   ├── dados.xlsx          # Dados espectrais originais
│   └── README.md           # Documentação dos dados
├── main_analysis.py         # Script principal de análise
├── demo_usage.py            # Demonstração da biblioteca
├── config.py                # Configurações simples
├── requirements.txt         # Dependências
├── setup.py                # Instalação
└── README.md               # Documentação
```

### Arquivos Principais

**Biblioteca Chemometrics**
- `chemometrics/__init__.py`: Interface pública com todos os imports
- `chemometrics/preprocessing.py`: SNV, Savitzky-Golay, MSC, Mean Centering, Autoscaling
- `chemometrics/modeling.py`: Regressor PLS com otimização automática
- `chemometrics/utils.py`: Divisão de dados e cálculo de métricas

**Scripts de Uso**
- `main_analysis.py`: Análise principal com dados reais
- `demo_usage.py`: Demonstração com dados sintéticos
- `examples/example_sodium_analysis.py`: Exemplo completo

**Configuração**
- `config.py`: Configurações simples
- `config/settings.py`: Configurações avançadas
- `requirements.txt`: Dependências Python
- `setup.py`: Instalação da biblioteca

## Instalação

```bash
# Instalar dependências
pip install -r requirements.txt

# Instalar a biblioteca em modo desenvolvimento
pip install -e .
```

## Uso Básico

### Carregamento e Pré-processamento

```python
from chemometrics import SNVPreprocessor, SavitzkyGolayPreprocessor, MeanCenterPreprocessor

# Carregar dados
X, y = load_your_spectral_data()

# Pré-processamento sequencial
savgol = SavitzkyGolayPreprocessor(window_length=15, polyorder=2, deriv=1)
X_processed = savgol.fit_transform(X)

snv = SNVPreprocessor()
X_processed = snv.fit_transform(X_processed)

mean_center = MeanCenterPreprocessor()
X_processed = mean_center.fit_transform(X_processed)
```

### Modelagem PLS

```python
from chemometrics import PLSRegressor, DataSplitter

# Dividir dados
splitter = DataSplitter(test_size=0.3, random_state=42)
train_idx, test_idx = splitter.split_with_groups(X, y, groups=sample_ids)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Treinar modelo
model = PLSRegressor()
model.fit(X_train, y_train, cv_folds=5, optimize_components=True)

# Fazer predições
y_pred = model.predict(X_test)

# Avaliar modelo
metrics = model.evaluate(X_train, y_train, X_test, y_test, unit='mEq/L')
model.plot_results(X_train, y_train, X_test, y_test, unit='mEq/L')
```

### Divisão de Dados

```python
from chemometrics import DataSplitter

splitter = DataSplitter(test_size=0.3, random_state=42)

# Divisão com grupos (réplicas)
train_idx, test_idx = splitter.split_with_groups(X, y, groups=sample_ids)

# Algoritmo Kennard-Stone
train_idx, test_idx = splitter.split_kennard_stone(X, y)

# Algoritmo SPXY
train_idx, test_idx = splitter.split_spxy(X, y)
```

## Funcionalidades

### Pré-processamento Espectroscópico

- **SNV (Standard Normal Variate)**: Normalização por espectro
- **Savitzky-Golay**: Filtro de suavização e derivação
- **MSC (Multiplicative Scatter Correction)**: Correção de espalhamento
- **Mean Centering**: Centramento na média
- **Autoscaling**: Normalização por desvio padrão

### Modelagem

- **PLS Regressor**: Regressão PLS com otimização automática
- **Validação cruzada**: Otimização do número de componentes
- **VIP Scores**: Importância das variáveis
- **Métricas completas**: R², RMSE, MAE, MAPE, Bias, RPD, RER

### Divisão de Dados

- **Divisão com grupos**: Considera réplicas de amostras
- **Kennard-Stone**: Seleção baseada em distância euclidiana
- **SPXY**: Combina distâncias em X e Y

## Exemplo Completo

Execute o exemplo completo de análise de sódio:

```bash
python examples/example_sodium_analysis.py
```

Este exemplo demonstra:
1. Carregamento de dados espectrais
2. Pré-processamento sequencial
3. Divisão considerando réplicas
4. Treinamento de modelo PLS
5. Avaliação e visualização

## Princípios de Arquitetura

### Separação de Responsabilidades
- Cada módulo tem uma responsabilidade específica
- Código organizado por funcionalidade
- `preprocessing.py`: Técnicas de pré-processamento
- `modeling.py`: Modelos de regressão
- `utils.py`: Utilitários auxiliares

### Interface Consistente
- Todas as classes seguem o padrão `fit()`, `transform()`, `fit_transform()`
- Métodos padronizados para avaliação e visualização
- API uniforme para todos os componentes

### Configuração Centralizada
- Configurações em `config/settings.py`
- Parâmetros padrão para todos os métodos
- Fácil customização de parâmetros

### Logging e Tratamento de Erros
- Sistema de logging configurável
- Tratamento robusto de exceções
- Mensagens informativas para debugging

### Documentação Clara
- Docstrings detalhadas em todas as funções
- Exemplos práticos de uso
- README abrangente com guias de uso

### Benefícios para Aplicação Acadêmica
1. **Modular**: Fácil de entender e modificar
2. **Reutilizável**: Classes independentes
3. **Extensível**: Fácil adicionar funcionalidades
4. **Testável**: Estrutura permite testes
5. **Documentada**: Código bem documentado
6. **Profissional**: Segue boas práticas de engenharia de software

## Dependências

- `numpy>=1.21.0`: Computação numérica
- `pandas>=1.3.0`: Manipulação de dados
- `scikit-learn>=1.0.0`: Machine learning
- `scipy>=1.7.0`: Computação científica
- `matplotlib>=3.4.0`: Visualização
- `openpyxl>=3.0.0`: Leitura de Excel

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para detalhes.

## Contribuição

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Contato

Para dúvidas ou sugestões, entre em contato com a equipe de desenvolvimento.
