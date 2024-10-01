# Plataforma de Planejamento de Saúde - PIPS

Este projeto é uma API desenvolvida em **Flask** com suporte a **Flask-RESTful**, que fornece endpoints para análise e previsão de diversos indicadores de saúde pública. A API utiliza dados CSVs como base para análises e modelos de aprendizado de máquina para gerar previsões.

## Funcionalidades

A API fornece as seguintes funcionalidades:
- Previsão de mortalidade por CID-10
- Análise e previsão de expectativa de vida
- Gráficos de análise de mortalidade, atendimentos, e capacidade de infraestrutura de saúde
- Previsão de causas de morte em Inhumas e outras cidades
- Previsão e evolução de casos de dengue, SRAG, nascimentos e atendimentos

## Tecnologias Utilizadas

- **Flask**: Framework web para Python.
- **Flask-RESTful**: Extensão do Flask para construção de APIs RESTful.
- **SQLAlchemy**: ORM para trabalhar com banco de dados.
- **Pandas**: Para manipulação e análise de dados em CSV.
- **Scikit-learn**: Para treinamento de modelos de machine learning.
- **Joblib**: Para salvar e carregar modelos treinados.
- **Plotly**: Para visualização de dados em gráficos interativos.

## Estrutura do Projeto

```
/my_flask_app
│
├── /app
│   ├── /data                  # Dados em csv
│   ├── /routes                # Definição das rotas da API
│   ├── /services              # Funções de carregamento de dados
│   ├── /config                # Configurações da aplicação
│   ├── __init__.py            # Inicialização da aplicação (Factory Pattern)
│   └── extensions.py          # Inicialização das extensões, como SQLAlchemy
├── /tests                     # Testes unitários
├── run.py                     # Ponto de entrada da aplicação
├── requirements.txt           # Dependências do projeto
└── README.md                  # Documentação do projeto
```

## Instalação

Siga os passos abaixo para configurar o ambiente do projeto localmente:

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/seu-repositorio.git
   cd seu-repositorio
   ```

2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Rodar a aplicação**:
   ```bash
   python run.py
   ```

4. Acesse a API no navegador ou em uma ferramenta como o Postman via `http://127.0.0.1:5000`.

## Endpoints Disponíveis

Aqui estão alguns dos principais endpoints da API:

- **Previsão de Atendimento por CID-10**  
  `GET /predicao/<string:cid_10>`
  
- **Análise de Mortalidade**  
  `GET /grafico/analise-mortalidade`

- **Previsão de Expectativa de Vida**  
  `GET /previsao/expectativa_vida`

- **Previsão de Causas de Morte em Inhumas**  
  `GET /previsao/causas_morte_inhumas`

- **Gráficos de Evolução de Atendimentos**  
  `GET /grafico/evolucao-atendimentos`

- **Gráficos de Capacidade de Infraestrutura**  
  `GET /grafico/capacidade-infraestrutura`

Para ver a lista completa de endpoints, confira o arquivo `routes.py` ou acesse a documentação da API (se aplicável).

## Estrutura dos Dados

A API utiliza arquivos CSV para carregar os dados. As funções de carregamento estão organizadas em `app/services/data_loader.py`. Esses dados são processados e transformados antes de serem usados nos modelos preditivos.

## Fontes de dados
- **CNES SUS**
- **DataSUS**
- **IBGE**