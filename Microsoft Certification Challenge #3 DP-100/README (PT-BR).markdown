# Gelato Mágico: Previsão de Vendas de Sorvete 🍦

Este projeto utiliza Machine Learning para prever as vendas de sorvete com base na temperatura diária para a Gelato Mágico, uma sorveteria fictícia. Com um modelo de Regressão Linear, MLflow para rastreamento de experimentos e um pipeline estruturado, o projeto visa otimizar a produção, reduzir desperdícios e maximizar lucros.

## Estrutura do Projeto
- `inputs/ice_cream_sales.csv`: Conjunto de dados sintético com temperatura e vendas.
- `train_model.py`: Script Python para treinamento e registro do modelo com MLflow.
- `README.md`: Documentação do projeto.

## Conjunto de Dados
O conjunto de dados (`inputs/ice_cream_sales.csv`) contém 100 linhas com:
- `temperature`: Temperatura diária em Celsius (15°C a 35°C).
- `sales`: Número de sorvetes vendidos (correlacionado linearmente com a temperatura, com ruído).

Exemplo:
```
temperature,sales
15.2,152.3
16.8,168.7
...
```

## Instruções de Configuração
1. **Clonar o repositório**:
   ```bash
   git clone <sua-url-do-repositorio>
   cd <nome-do-repositorio>
   ```
2. **Instalar dependências**:
   ```bash
   pip install pandas numpy scikit-learn mlflow
   ```
3. **Executar o script de treinamento**:
   ```bash
   python train_model.py
   ```
4. **Visualizar a interface do MLflow**:
   ```bash
   mlflow ui
   ```
   Abra `http://localhost:5000` no navegador para ver os resultados dos experimentos.

## Processo
1. **Preparação dos Dados**: Carregou o conjunto de dados e dividiu em conjuntos de treinamento (80%) e teste (20%).
2. **Pipeline**: Criou um pipeline do scikit-learn com:
   - `StandardScaler`: Para normalizar a feature de temperatura.
   - `LinearRegression`: Para prever as vendas.
3. **Treinamento**: Treinou o modelo e avaliou com Erro Quadrático Médio (MSE) e R².
4. **MLflow**: Registrou parâmetros (`test_size`, `random_state`), métricas (`mse`, `r2`) e o modelo.
5. **Resultados**: O modelo obteve um MSE baixo e um R² alto, indicando bom desempenho preditivo.

## Insights
- **Relação Linear**: Os dados sintéticos mostram uma forte correlação linear entre temperatura e vendas, tornando a Regressão Linear uma escolha adequada.
- **Benefícios do MLflow**: O MLflow facilitou o rastreamento de experimentos, a comparação de métricas e a reprodutibilidade, essencial para projetos reais de ML.
- **Escalabilidade**: O pipeline garante reprodutibilidade e pode ser estendido com mais features (e.g., dia da semana, feriados).
- **Potencial na Nuvem**: Implantar este modelo no Azure ML poderia permitir previsões em tempo real via APIs, ajudando a sorveteria a ajustar a produção diariamente.

## Possibilidades de Melhoria
- **Features Adicionais**: Incluir fatores como umidade, dia da semana ou feriados para melhorar as previsões.
- **Modelos Avançados**: Experimentar regressão polinomial ou modelos baseados em árvores se a relação for não linear.
- **Implantação em Tempo Real**: Usar o Azure ML para implantar o modelo como um endpoint para previsões em tempo real.
- **Coleta de Dados**: Substituir dados sintéticos por dados reais de vendas para maior precisão.

## Capturas de Tela
*(Placeholder: Inclua capturas de tela no seu repositório GitHub)*
- **Interface do MLflow**: Captura da interface do MLflow mostrando execuções, métricas e parâmetros registrados.
- **Gráfico de Previsão**: Gráfico de dispersão de temperatura vs. vendas com a linha de regressão.
- **Diagrama do Pipeline**: Diagrama do pipeline do scikit-learn (escalonador → regressor).

## Tecnologias Utilizadas
- Python, pandas, numpy
- scikit-learn (LinearRegression, Pipeline, StandardScaler)
- MLflow
- GitHub para controle de versão

## Conclusão
Este projeto demonstra uma aplicação prática de Machine Learning para resolver um problema de negócios. Ao prever as vendas de sorvete, a Gelato Mágico pode otimizar a produção e reduzir desperdícios. O uso do MLflow e de um pipeline garante reprodutibilidade, tornando esta uma solução robusta para implantação no mundo real.

Bons estudos e aproveite um sorvete! 🍦