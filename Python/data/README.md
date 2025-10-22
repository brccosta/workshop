# Pasta de Dados

Esta pasta contém os dados utilizados no projeto Chemometrics.

## Arquivos de Dados

### `dados.xlsx`
- **Descrição**: Dados espectrais para análise de concentração de sódio
- **Formato**: Excel com múltiplas abas
- **Aba principal**: 'original'
- **Estrutura**:
  - Coluna 1: IDs das amostras
  - Coluna 2: Concentração de sódio (mEq/L) - variável resposta
  - Coluna 3: Classe/Faixa
  - Coluna 4: Status de diluição
  - Colunas 6+: Dados espectrais (1459 variáveis)

## Como Usar

Os dados são carregados automaticamente pelos scripts:

```python
# Em main_analysis.py
dados_brutos = pd.read_excel("data/dados.xlsx", sheet_name='original', header=None)

# Em examples/example_sodium_analysis.py
X, y, amostras, variaveis = load_data()  # Usa "data/dados.xlsx" por padrão
```

## Estrutura dos Dados

- **Amostras**: Dados espectrais de diferentes amostras
- **Variáveis**: 1459 comprimentos de onda
- **Resposta**: Concentração de sódio em mEq/L
- **Grupos**: IDs das amostras para divisão considerando réplicas

## Objetivo

Predição da concentração de sódio a partir de dados espectrais usando técnicas quimiométricas (PLS).
