# Análise Quimiométrica de Dados Espectrais

Este programa realiza análise quimiométrica de dados espectrais para determinação de sódio em amostras, utilizando técnicas de pré-processamento espectroscópico e modelagem PLS (Partial Least Squares).

## Estrutura do Projeto

### V1 - Implementação Inicial

```
Python/v1/
├── dados.xlsx              # Dados espectrais
├── pls_sodio.py           # Script principal (análise completa)
├── pretrat.py             # Funções de pré-processamento
├── pls_model.py           # Funções de modelagem PLS
└── splits.py              # Funções de divisão de dados
```

**Características da V1:**
- **Estrutura procedural** com funções soltas
- **Código monolítico** no arquivo principal (`pls_sodio.py`)
- **Acoplamento forte** entre módulos
- **Parâmetros hardcoded** em todo o código
- **Sem documentação** estruturada
- **Difícil reutilização** de componentes

### V2 - Implementação Refatorada

```
Python/v2/
├── sodium_analysis.py     # Script principal modular
├── chemometrics/          # Biblioteca quimiométrica
│   ├── __init__.py        # Interface da biblioteca
│   ├── preprocessing.py   # Pré-processamento espectroscópico
│   ├── modeling.py        # Modelagem PLS
│   └── utils.py           # Utilitários (divisão, métricas)
├── data/                  # Dados do projeto
│   ├── dados.xlsx
│   └── README.md
├── requirements.txt       # Dependências
└── README.md             # Documentação
```

**Características da V2:**
- **Arquitetura modular** com separação clara de responsabilidades
  ```python
  # Separação clara: cada módulo tem responsabilidade específica
  from chemometrics.preprocessing import SNVPreprocessor
  from chemometrics.modeling import PLSRegressor
  from chemometrics.utils import DataSplitter
  ```
- **Classes orientadas a objetos** com interfaces consistentes
  ```python
  # Interface consistente: todos os preprocessadores seguem o mesmo padrão
  snv = SNVPreprocessor(plot=True)
  X_processed = snv.fit_transform(X)
  
  savgol = SavitzkyGolayPreprocessor(window_length=15, polyorder=2, deriv=1)
  X_processed = savgol.fit_transform(X_processed)
  ```
- **Baixo acoplamento** entre módulos
  ```python
  # Módulos independentes: mudança em um não afeta outros
  from chemometrics import PLSRegressor, SNVPreprocessor, DataSplitter
  # Cada classe é independente e pode ser usada isoladamente
  ```
- **Parâmetros configuráveis** e flexíveis
  ```python
  # Flexibilidade total: parâmetros configuráveis em tempo de execução
  preprocessor = SavitzkyGolayPreprocessor(
      window_length=21,    # Configurável
      polyorder=3,         # Configurável
      deriv=2,             # Configurável
      plot=True           # Configurável
  )
  ```
- **Documentação completa** com docstrings
  ```python
  def load_data(file_path: str = "data/dados.xlsx") -> tuple:
      """
      Carrega os dados do arquivo Excel.
      
      Parameters
      ----------
      file_path : str
          Caminho para o arquivo de dados.
          
      Returns
      -------
      tuple
          X, y, amostras, variaveis
      """
  ```
- **Alta reutilização** de componentes
  ```python
  # Reutilização: mesma biblioteca para diferentes projetos
  # Projeto de proteínas
  from chemometrics import SNVPreprocessor, PLSRegressor
  
  # Projeto de óleos  
  from chemometrics import SavitzkyGolayPreprocessor, MSCPreprocessor
  
  # Projeto de alimentos
  from chemometrics import MeanCenterPreprocessor, DataSplitter
  ```

## Comparação Arquitetural

### Organization

| Aspecto | V1 | V2 |
|---------|----|----|
| **Estrutura** | Arquivos soltos sem organização | Hierarquia clara com separação de responsabilidades |
| **Nomenclatura** | Inconsistente (`savgol_`, `snv_`) | Padronizada (`SavitzkyGolayPreprocessor`) |
| **Documentação** | Mínima | Completa com docstrings e README |

### Folder and Directory Structure

**V1 - Estrutura Plana:**
```
v1/
├── pls_sodio.py    # Tudo misturado
├── pretrat.py      # Funções soltas
├── pls_model.py    # Funções soltas
└── splits.py       # Funções soltas
```

**V2 - Estrutura Hierárquica:**
```
v2/
├── sodium_analysis.py    # Orquestração
├── chemometrics/        # Biblioteca reutilizável
│   ├── preprocessing/   # Camada de pré-processamento
│   ├── modeling/        # Camada de modelagem
│   └── utils/          # Camada de utilitários
└── data/               # Camada de dados
```

### Dependency Flow and Communication

**V1 - Acoplamento Forte:**
```python
# Dependências circulares e acoplamento forte
from pretrat import savgol_, snv_, mean_center_
from pls_model import pls_
from splits import split_data_

# Uso direto sem abstração
X_p1 = savgol_(X2, width=15, order=2, deriv=1, plotar=False)
X_p1 = snv_(X_p1, plotar=False)
```

**V2 - Baixo Acoplamento:**
```python
# Interface limpa e desacoplada
from chemometrics import (
    SNVPreprocessor, 
    SavitzkyGolayPreprocessor, 
    MeanCenterPreprocessor,
    PLSRegressor,
    DataSplitter
)

# Uso através de classes com interface consistente
savgol = SavitzkyGolayPreprocessor(window_length=15, polyorder=2, deriv=1)
X_processed = savgol.fit_transform(X)
```

### Layer Separation

**V1 - Sem Separação de Camadas:**
- **Apresentação, lógica e dados** misturados no mesmo arquivo
- **Hardcoded** em todo lugar
- **Sem abstração** entre camadas

**V2 - Separação Clara de Camadas:**

| Camada | Responsabilidade | Componentes |
|--------|------------------|-------------|
| **Presentation** | Interface e orquestração | `sodium_analysis.py` |
| **Business Logic** | Regras de negócio | `chemometrics/modeling.py` |
| **Data Processing** | Pré-processamento | `chemometrics/preprocessing.py` |
| **Utilities** | Serviços auxiliares | `chemometrics/utils.py` |
| **Data** | Armazenamento | `data/` |

### Code Modularization and Reuse

**V1 - Modularização Limitada:**
```python
# Funções soltas sem padrão
def savgol_(X, width=None, order=None, deriv=None, plotar=False):
    X_smooth = savgol_filter(X, width, order, deriv)
    return X_smooth

def snv_(X, plotar=False):
    media_linha = np.mean(X, axis=1, keepdims=True)
    dp_linha = np.std(X, axis=1, keepdims=True, ddof=1)
    X_SNV = (X - media_linha) / dp_linha
    return X_SNV
```

**V2 - Modularização Avançada:**
```python
# Classes com interface consistente
class SavitzkyGolayPreprocessor(BasePreprocessor):
    def __init__(self, window_length=15, polyorder=2, deriv=1, plot=False):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        super().__init__(plot=plot)
    
    def fit_transform(self, X):
        X = self._validate_input(X)
        X_processed = savgol_filter(X, self.window_length, self.polyorder, self.deriv)
        self._plot_results(X, X_processed, "Savitzky-Golay")
        return X_processed
```

## Benefícios da Arquitetura V2

### Comprehensibility
- **Estrutura clara** que qualquer desenvolvedor pode entender
- **Nomenclatura consistente** e descritiva
- **Documentação completa** com exemplos de uso
- **Separação lógica** de responsabilidades

### Maintainability
- **Mudanças localizadas** não afetam o sistema inteiro
- **Testes unitários** possíveis para cada componente
- **Debugging facilitado** com responsabilidades isoladas
- **Evolução incremental** sem quebrar funcionalidades existentes

### Scalability and Evolution
- **Adição de novos preprocessadores** sem modificar código existente
- **Extensão da biblioteca** com novos algoritmos
- **Reutilização** em outros projetos de análise espectroscópica
- **Integração** com outras bibliotecas científicas
