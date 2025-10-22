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

## Estrutura dos Dados

- **Amostras**: Dados espectrais de diferentes amostras
- **Variáveis**: 1459 comprimentos de onda
- **Resposta**: Concentração de sódio em mEq/L
- **Grupos**: IDs das amostras para divisão considerando réplicas

## Objetivo

Predição da concentração de sódio a partir de dados espectrais usando técnicas quimiométricas (PLS).
