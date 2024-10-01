import logging
import os
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
from flask_cors import CORS
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor
import joblib 
from app.services.data_loader import (
    carregar_dados,
    carregar_dados_srag,
    carregar_dados_nascimentos,
    carregar_dados_mortalidade,
    carregar_dados_expectativa_vida,
    carregar_evolucao_atendimentos,
    carregar_atendimentos_por_ano,
    carregar_dados_atendimento_clientela,
    carregar_dados_capacidade_consultorios_leitos,
    carregar_dados_unidades_saude,
    carregar_dados_servicos_saude,
    carregar_dados_infraestrutura,
    carregar_dados_prever_mortalidade,
    carregar_dados_filtrados,
    carregar_dados_inhumas_prever_causas,
    carregar_dados_analise_expectativa_vida,
    carregar_modelo,
    carregar_dados_mortalide_especialidades,
    carregar_dados_prever_aumento_atendimento
)

app = Flask(__name__)
CORS(app)

# Configuração do banco de dados
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
api = Api(app)

# Caminho para o arquivo CSV
caminho_arquivo_taxas_mortalidade = r"app\data\dadosinhumas_2024-08-29_141327.csv"
caminho_arquivo = r'app\data\casosdengue_2024-08-29_141327.csv' # ok
caminho_arquivo_srag = r"app\data\sindromerespiratoria_2024-08-29_141327.csv" # ok
caminho_arquivo_nascimentos = r"app\data\dadosnascimento_2024-08-29_141327.csv" # ok
caminho_arquivo_descricao = r"app\data\mortalidadeinhumas_2024-08-29_141327.csv" # ok
caminho_arquivo_mortalidade = r"app\data\mortalidade_com_descricao.csv" # ok
caminho_arquivo_expectativa_vida = r"app\data\expectativa_de_vida_inhumas.csv" # ok

df = pd.read_csv(caminho_arquivo)
df['data_sintomas'] = pd.to_datetime(df['data_sintomas'], format='%Y%m%d')
df['data_notificacao'] = pd.to_datetime(df['data_notificacao'], format='%Y%m%d')

class UserLoginResource(Resource):
    def post(self):
        if not request.is_json:
            return {'message': 'Content-Type must be application/json'}, 400

        username = request.json.get('username', None)
        password = request.json.get('password', None)
     
        if username != 'user' or password != 'pass':
            return {'message': 'Invalid credentials'}, 401
     
        # Simplesmente retorna uma mensagem de sucesso
        return {'message': 'Login successful'}, 200
    
def preparar_dados_evolucao(df):
    # Filtrar os dados para o município de Inhumas
    df_inhumas = df[df['municipio_residencia'] == 'INHUMAS']

    # Agrupar os dados por ano epidemiológico e contar os casos
    casos_por_ano = df_inhumas.groupby('ano_epidemiologica').size()

    # Preparar os dados para o frontend
    data = {
        'labels': casos_por_ano.index.tolist(),
        'datasets': [{
            'label': 'Número de Casos de Dengue',
            'data': casos_por_ano.values.tolist(),
            'backgroundColor': 'rgba(75, 192, 192, 0.2)',
            'borderColor': 'rgba(75, 192, 192, 1)',
            'borderWidth': 1,
            'fill': False
        }]
    }
    
    return data

# Rota para enviar os dados do gráfico
class GraficoEvolucaoDengueResource(Resource):
    def get(self):
        # Carregar os dados
        df = carregar_dados(caminho_arquivo)

        # Preparar os dados de evolução
        data = preparar_dados_evolucao(df)

        # Retornar os dados em formato JSON
        return jsonify(data)

# Função para preparar os dados de evolução dos casos de SRAG em Inhumas
def preparar_dados_evolucao_srag(df):
    # Filtrar para o município de Inhumas (código IBGE 521000)
    df_inhumas = df[df['municipio_residencia'] == 'INHUMAS']

    # Agrupar os dados por ano de sintomas e contar os casos
    casos_por_ano = df_inhumas.groupby('ano_sintomas').size()

    # Preparar os dados para o frontend
    data = {
        'labels': casos_por_ano.index.tolist(),
        'datasets': [{
            'label': 'Número de Casos de SRAG',
            'data': casos_por_ano.values.tolist(),
            'backgroundColor': 'rgba(255, 99, 132, 0.2)',
            'borderColor': 'rgba(255, 99, 132, 1)',
            'borderWidth': 1,
            'fill': False
        }]
    }

    return data

# Rota para enviar os dados de evolução de SRAG
class GraficoEvolucaoSRAGResource(Resource):
    def get(self):
        # Carregar os dados de SRAG
        df = carregar_dados_srag(caminho_arquivo_srag)

        # Preparar os dados de evolução dos casos de SRAG
        data = preparar_dados_evolucao_srag(df)

        # Retornar os dados em formato JSON
        return jsonify(data)

# Função para preparar os dados de evolução dos nascimentos em Inhumas
def preparar_dados_evolucao_nascimentos(df):
    # Filtrar para o município de Inhumas (código IBGE 521000)
    df_inhumas = df[df['municipio_ocorrencia'] == 'INHUMAS']

    # Agrupar os dados por ano de nascimento e contar os nascimentos
    nascimentos_por_ano = df_inhumas.groupby('ano_nascimento').size()

    # Preparar os dados para o frontend
    data = {
        'labels': nascimentos_por_ano.index.tolist(),
        'datasets': [{
            'label': 'Número de Nascimentos',
            'data': nascimentos_por_ano.values.tolist(),
            'backgroundColor': 'rgba(54, 162, 235, 0.2)',
            'borderColor': 'rgba(54, 162, 235, 1)',
            'borderWidth': 1,
            'fill': False
        }]
    }

    return data

# Rota para enviar os dados de evolução dos nascimentos
class GraficoEvolucaoNascimentosResource(Resource):
    def get(self):
        # Carregar os dados de nascimentos
        df = carregar_dados_nascimentos(caminho_arquivo_nascimentos)

        # Preparar os dados de evolução dos nascimentos
        data = preparar_dados_evolucao_nascimentos(df)

        # Retornar os dados em formato JSON
        return jsonify(data)

# Função para preparar os dados de óbitos por faixa etária e sexo
def preparar_dados_obitos_faixa_etaria_sexo(df):
    # Agrupar os dados por faixa etária e sexo, contando os óbitos
    obitos_por_faixa_etaria_sexo = df.groupby(['faixa_etaria', 'sexo']).size().unstack(fill_value=0)

    # Preparar os dados para o frontend
    data = {
        'labels': obitos_por_faixa_etaria_sexo.index.tolist(),
        'datasets': [
            {
                'label': 'Feminino',
                'data': obitos_por_faixa_etaria_sexo['FEMININO'].tolist(),
                'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                'borderColor': 'rgba(255, 99, 132, 1)',
                'borderWidth': 1,
                'fill': False
            },
            {
                'label': 'Masculino',
                'data': obitos_por_faixa_etaria_sexo['MASCULINO'].tolist(),
                'backgroundColor': 'rgba(54, 162, 235, 0.2)',
                'borderColor': 'rgba(54, 162, 235, 1)',
                'borderWidth': 1,
                'fill': False
            }
        ]
    }

    return data

# Rota para enviar os dados de distribuição de óbitos por faixa etária e sexo
class GraficoDistribuicaoObitosFaixaEtariaSexoResource(Resource):
    def get(self):
        # Carregar os dados de mortalidade
        df = carregar_dados_mortalidade(caminho_arquivo_mortalidade)

        # Preparar os dados de óbitos por faixa etária e sexo
        data = preparar_dados_obitos_faixa_etaria_sexo(df)

        # Retornar os dados em formato JSON
        return jsonify(data)

# Função para preparar os dados de causas de morte mais comuns (DESCRICAO)
def preparar_dados_causas_morte(df_mortalidade):
    # Padronizar os códigos CID-10 (pegar os 3 primeiros caracteres do código CID-10)
    df_mortalidade['Cod_CID_10_padronizado'] = df_mortalidade['Cod_CID_10'].apply(lambda x: str(x)[:3])

    # Agrupar os dados por descrição (DESCRICAO) e contar o número de óbitos
    causas_comuns = df_mortalidade.groupby('DESCRICAO').size().sort_values(ascending=False).head(10)

    # Preparar os dados para o frontend
    data = {
        'labels': causas_comuns.index.tolist(),  # As causas mais comuns (descrições)
        'datasets': [{
            'label': 'Número de Óbitos',
            'data': causas_comuns.values.tolist(),  # O número de óbitos por causa
            'backgroundColor': 'rgba(75, 192, 192, 0.2)',
            'borderColor': 'rgba(75, 192, 192, 1)',
            'borderWidth': 1,
            'fill': False
        }]
    }

    return data

# Rota para enviar os dados de causas de morte mais comuns (DESCRICAO)
class GraficoCausasMorteCID10Resource(Resource):
    def get(self):
        # Carregar os dados de mortalidade (já inclui a descrição)
        df_mortalidade = carregar_dados_mortalidade(caminho_arquivo_mortalidade)

        # Preparar os dados de causas de morte mais comuns
        data = preparar_dados_causas_morte(df_mortalidade)

        # Retornar os dados em formato JSON
        return jsonify(data)

# Função para preparar os dados de evolução de óbitos ao longo dos anos
def preparar_dados_evolucao_obitos(df_mortalidade):
    # Agrupar os dados por ano e contar o número de óbitos
    obitos_por_ano = df_mortalidade.groupby('ano').size()

    # Preparar os dados para o frontend
    data = {
        'labels': obitos_por_ano.index.tolist(),  # Os anos
        'datasets': [{
            'label': 'Número de Óbitos',
            'data': obitos_por_ano.values.tolist(),  # O número de óbitos por ano
            'backgroundColor': 'rgba(153, 102, 255, 0.2)',
            'borderColor': 'rgba(153, 102, 255, 1)',
            'borderWidth': 2,
            'fill': False,
            'lineTension': 0.1
        }]
    }

    return data

# Rota para enviar os dados de evolução de óbitos ao longo dos anos
class GraficoEvolucaoObitosResource(Resource):
    def get(self):
        # Carregar os dados de mortalidade
        df_mortalidade = carregar_dados_mortalidade(caminho_arquivo_mortalidade)

        # Preparar os dados de evolução de óbitos
        data = preparar_dados_evolucao_obitos(df_mortalidade)

        # Retornar os dados em formato JSON
        return jsonify(data)

# Função para preparar os dados de taxas de mortalidade
def preparar_dados_taxas_mortalidade(df):
    # Coleta de todas as taxas ao longo dos anos
    anos = [2013, 2014, 2015, 2016, 2017]
    taxas_mortalidade = {
        "Taxa Mortalidade Infantil": df[
            ["Taxa_mortalidade_infantil_2013", "Taxa_mortalidade_infantil_2014", "Taxa_mortalidade_infantil_2015", "Taxa_mortalidade_infantil_2016", "Taxa_mortalidade_infantil_2017"]
        ].mean().tolist(),
        "Taxa Bruta de Mortalidade": df[
            ["Taxa_bruta_mortalidade_2013", "Taxa_bruta_mortalidade_2014", "Taxa_bruta_mortalidade_2015", "Taxa_bruta_mortalidade_2016", "Taxa_bruta_mortalidade_2017"]
        ].mean().tolist(),
        "Taxa Mortalidade por Homicídios": df[
            ["Taxa_mortalidade_homicidios_2013", "Taxa_mortalidade_homicidios_2014", "Taxa_mortalidade_homicidios_2015", "Taxa_mortalidade_homicidios_2016", "Taxa_mortalidade_homicidios_2017"]
        ].mean().tolist(),
        "Taxa Mortalidade Doenças Não Transmissíveis": df[
            ["Taxa_mortalidade_doencas_nao_transmissiveis_2013", "Taxa_mortalidade_doencas_nao_transmissiveis_2014", "Taxa_mortalidade_doencas_nao_transmissiveis_2015", "Taxa_mortalidade_doencas_nao_transmissiveis_2016", "Taxa_mortalidade_doencas_nao_transmissiveis_2017"]
        ].mean().tolist(),
        "Taxa Mortalidade por Câncer de Mama": df[
            ["Taxa_mortalidade_cancer_mama_2013", "Taxa_mortalidade_cancer_mama_2014", "Taxa_mortalidade_cancer_mama_2015", "Taxa_mortalidade_cancer_mama_2016", "Taxa_mortalidade_cancer_mama_2017"]
        ].mean().tolist(),
        "Taxa Mortalidade por Câncer de Próstata": df[
            ["Taxa_mortalidade_cancer_prostata_2013", "Taxa_mortalidade_cancer_prostata_2014", "Taxa_mortalidade_cancer_prostata_2015", "Taxa_mortalidade_cancer_prostata_2016", "Taxa_mortalidade_cancer_prostata_2017"]
        ].mean().tolist(),
        "Taxa Mortalidade por Acidentes de Trânsito": df[
            ["Taxa_mortalidade_acidente_transito_2013", "Taxa_mortalidade_acidente_transito_2014", "Taxa_mortalidade_acidente_transito_2015", "Taxa_mortalidade_acidente_transito_2016", "Taxa_mortalidade_acidente_transito_2017"]
        ].mean().tolist(),
        "Taxa Mortalidade por Suicídio": df[
            ["Taxa_mortalidade_suicidio_2013", "Taxa_mortalidade_suicidio_2014", "Taxa_mortalidade_suicidio_2015", "Taxa_mortalidade_suicidio_2016", "Taxa_mortalidade_suicidio_2017"]
        ].mean().tolist()
    }

    # Preparar os dados para o frontend
    data = {
        "labels": anos,
        "datasets": [
            {
                "label": key,
                "data": value,
                "fill": False,
                "borderWidth": 2
            }
            for key, value in taxas_mortalidade.items()
        ]
    }

    return data

# Rota para enviar os dados de taxas de mortalidade
class GraficoTaxasMortalidadeResource(Resource):
    def get(self):
        # Carregar os dados de mortalidade
        df_mortalidade = carregar_dados_mortalidade(caminho_arquivo_taxas_mortalidade)

        # Preparar os dados das taxas de mortalidade
        data = preparar_dados_taxas_mortalidade(df_mortalidade)

        # Retornar os dados em formato JSON
        return jsonify(data)

# Rota para retornar os dados de expectativa de vida
class GraficoExpectativaVidaResource(Resource):
    def get(self):
        # Carregar os dados e calcular a expectativa de vida
        dados = carregar_dados_expectativa_vida(caminho_arquivo_expectativa_vida)

        # Converter os dados para formato JSON
        dados_json = dados.to_dict(orient='records')

        return jsonify(dados_json)

# Rota para retornar os dados de evolução de atendimentos
class EvolucaoAtendimentosResource(Resource):
    def get(self):
        # Carregar os dados
        caminho_arquivo_clientela = r'app\data\clientela_consulta.csv'
        df = carregar_evolucao_atendimentos(caminho_arquivo_clientela)

        # Retornar os dados em formato JSON
        return jsonify(df.to_dict(orient='records'))

# Rota para retornar os dados de evolução de atendimentos por ano
class EvolucaoAtendimentosAnoResource(Resource):
    def get(self):
        # Obter o parâmetro de ano da query string
        ano = request.args.get('ano')

        # Carregar os dados
        caminho_arquivo_clientela = r'app\data\clientela_consulta.csv'
        df = carregar_atendimentos_por_ano(caminho_arquivo_clientela, ano)

        # Retornar os dados em formato JSON
        return jsonify(df.to_dict(orient='records'))
    
# Rota para retornar os dados de tipos de atendimento e fluxo de clientela
class GraficoAtendimentoClientelaResource(Resource):
    def get(self):
        try:
            # Carregar os dados
            caminho_arquivo = r'app\data\ubs_info.csv'
            df = carregar_dados_atendimento_clientela(caminho_arquivo)

            if df.empty:
                return {"error": "Nenhum dado encontrado"}, 500

            dados_json = df.to_dict(orient='records')
            return jsonify(dados_json)
        except Exception as e:
            print(f"Erro ao processar a requisição: {e}")
            return {"error": "Erro interno ao processar a requisição"}, 500

# Rota para retornar os dados de capacidade de consultórios e leitos
class GraficoCapacidadeConsultoriosLeitosResource(Resource):
    def get(self):
        try:
            # Carregar os dados
            caminho_arquivo = r'app\data\ubs_info.csv'
            df = carregar_dados_capacidade_consultorios_leitos(caminho_arquivo)

            # Verificar se os dados foram carregados corretamente
            if df.empty:
                return {"error": "Nenhum dado encontrado"}, 500

            # Converter os dados para JSON
            dados_json = df.to_dict(orient='records')
            return jsonify(dados_json)  # Retornar como JSON

        except Exception as e:
            print(f"Erro ao processar a requisição: {e}")
            return {"error": "Erro interno ao processar a requisição"}, 500

# Classe do recurso para retornar os dados do gráfico em formato JSON
class GraficoUnidadesSaudeResource(Resource):
    def get(self):
        try:
            # Caminho do arquivo CSV
            caminho_arquivo_unidades_saude = r'app\data\atendimentos_prestados.csv'
            df = carregar_dados_unidades_saude(caminho_arquivo_unidades_saude)

            data = {
                'labels': df['Descricao'].tolist(),
                'datasets': [{
                    'label': 'Total de Unidades de Saúde',
                    'data': df['Total'].tolist(),
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                    'borderColor': 'rgba(75, 192, 192, 1)',
                    'borderWidth': 1
                }]
            }

            return jsonify(data)

        except Exception as e:
            return jsonify({"error": f"Erro ao processar a requisição: {e}"}), 500

# Classe do recurso para retornar os dados do gráfico em formato JSON
class GraficoServicosSaudeResource(Resource):
    def get(self):
        try:
            caminho_arquivo_servicos_saude = r'app\data\estabelecimentos.csv'
            df = carregar_dados_servicos_saude(caminho_arquivo_servicos_saude)

            data = {
                'labels': df['Descrição'].tolist(),
                'datasets': [{
                    'label': 'Total de Serviços de Saúde em Inhumas',
                    'data': df['Total'].tolist(),
                    'backgroundColor': 'rgba(153, 102, 255, 0.2)',
                    'borderColor': 'rgba(153, 102, 255, 1)',
                    'borderWidth': 1
                }]
            }

            return jsonify(data)

        except Exception as e:
            return jsonify({"error": f"Erro ao processar a requisição: {e}"}), 500

# Rota para retornar os dados de infraestrutura
class GraficoCapacidadeInfraestruturaResource(Resource):
    def get(self):
        caminho_arquivo = r'app\data\ubs_info.csv'
        infraestrutura = carregar_dados_infraestrutura(caminho_arquivo)
        
        if "error" in infraestrutura:
            return jsonify({"error": infraestrutura["error"]})

        chart_data = {
            'labels': list(infraestrutura.keys()),
            'datasets': [{
                'label': 'Quantidade',
                'data': list(infraestrutura.values()),
                'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                'borderColor': 'rgba(75, 192, 192, 1)',
                'borderWidth': 1
            }]
        }
        return jsonify(chart_data)

# Função para treinar o modelo de regressão linear
def treinar_modelo_prever_mortalidade(caminho_csv):
    try:
        X, y = carregar_dados_prever_mortalidade(caminho_csv)

        # Dividindo os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinando o modelo
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        # Avaliação do modelo
        previsoes = modelo.predict(X_test)
        mse = mean_squared_error(y_test, previsoes)
        print(f"Erro quadrático médio no conjunto de teste: {mse}")

        return modelo
    except Exception as e:
        return {"error": str(e)}

# Função para garantir que o valor da previsão esteja entre 0 e 1
def limitar_previsao(previsao):
    return max(0, min(1, previsao))

# Classe que processa a predição de mortalidade via POST
class PredicaoMortalidadeResource(Resource):
    def post(self):
        # Caminho para o arquivo CSV de mortalidade
        caminho_arquivo = r'app\data\mortalidade_com_descricao.csv'

        # Carregar o arquivo CSV
        df = pd.read_csv(caminho_arquivo)

        # Carregar e treinar o modelo
        modelo = treinar_modelo_prever_mortalidade(caminho_arquivo)
        if isinstance(modelo, dict) and "error" in modelo:
            return jsonify({"error": modelo["error"]})

        # Pegar os dados enviados no corpo da requisição
        dados_entrada = request.json
        faixa_etaria = dados_entrada.get('faixa_etaria')
        sexo = dados_entrada.get('sexo')
        cod_cid_10 = dados_entrada.get('Cod_CID_10')

        # Validar os dados
        if not faixa_etaria or not sexo or not cod_cid_10:
            return jsonify({"error": "Faltam parâmetros para a predição."}), 400

        # Buscar a descrição do Cod_CID_10 no CSV
        descricao = df.loc[df['Cod_CID_10'] == cod_cid_10, 'DESCRICAO'].values
        if len(descricao) == 0:
            return jsonify({"error": "Descrição não encontrada para o código CID-10 fornecido."})

        # Preparar entrada para predição
        entrada = {
            'faixa_etaria': faixa_etaria,
            'sexo': sexo,
            'Cod_CID_10': cod_cid_10
        }

        # Transformação One-Hot Encoding no DataFrame de entrada
        entrada_df = pd.DataFrame([entrada])
        entrada_df = pd.get_dummies(entrada_df, columns=['faixa_etaria', 'sexo', 'Cod_CID_10'], drop_first=True)

        # Garantir que as colunas do modelo treinado estão presentes na entrada
        for coluna in modelo.feature_names_in_:
            if coluna not in entrada_df.columns:
                entrada_df[coluna] = 0  # Adiciona colunas faltantes com valor 0

        # Fazer a predição
        previsao = modelo.predict(entrada_df)[0]
        previsao_limitada = limitar_previsao(previsao)

        # Retornar o resultado, incluindo a descrição
        return jsonify({
            'faixa_etaria': faixa_etaria,
            'sexo': sexo,
            'Cod_CID_10': cod_cid_10,
            'previsao_mortalidade': previsao_limitada,  # Previsão limitada entre 0 e 1
            'descricao': descricao[0]
        })

# Função para treinar o modelo ARIMA e fazer a previsão
def treinar_e_prever_mortes_ano(caminho_csv, cod_cid_10, sexo, faixa_etaria, anos_a_prever):
    try:
        # Carregar os dados filtrados por Cod_CID_10, sexo e faixa etária e a descrição da causa
        df_agrupado, descricao = carregar_dados_filtrados(caminho_csv, cod_cid_10, sexo, faixa_etaria)
        if "error" in df_agrupado:
            return df_agrupado  # Retornar o erro se ocorrer

        # Treinar o modelo ARIMA com os dados históricos
        modelo = ARIMA(df_agrupado['total_obitos'], order=(5,1,0))
        modelo_fit = modelo.fit()

        # Fazer a previsão para os próximos anos
        previsao = modelo_fit.forecast(steps=anos_a_prever)

        # Organizar os resultados
        anos_futuros = list(range(int(df_agrupado['ano'].max()) + 1, int(df_agrupado['ano'].max()) + anos_a_prever + 1))
        previsao_dict = dict(zip(anos_futuros, previsao))

        return previsao_dict, descricao
    except Exception as e:
        return {"error": str(e)}, None

# Classe que processa a previsão de mortes por ano com Cod_CID_10, sexo, faixa etária e anos via POST
class PrevisaoMortesAnuaisFiltradaResource(Resource):
    def post(self):
        # Caminho para o arquivo CSV de mortalidade
        caminho_arquivo = r'app\data\mortalidade_com_descricao.csv'

        try:
            # Pegar os parâmetros enviados na requisição
            dados = request.json
            cod_cid_10 = dados.get('Cod_CID_10')
            sexo = dados.get('sexo')
            faixa_etaria = dados.get('faixa_etaria')
            anos_a_prever = int(dados.get('anos_a_prever', 5))  # Prever 5 anos por padrão

            # Verificar se os parâmetros foram fornecidos
            if not cod_cid_10 or not sexo or not faixa_etaria:
                return jsonify({"error": "Faltam parâmetros Cod_CID_10, sexo e/ou faixa etária."})

            # Carregar e treinar o modelo para prever mortes por ano
            previsao, descricao = treinar_e_prever_mortes_ano(caminho_arquivo, cod_cid_10, sexo, faixa_etaria, anos_a_prever)

            if "error" in previsao:
                return jsonify({"error": previsao["error"]})

            # Retornar o resultado da previsão e a descrição da causa de óbito
            return jsonify({
                'previsao_mortes_proximos_anos': previsao,
                'descricao_causa_obito': descricao
            })

        except Exception as e:
            return jsonify({"error": str(e)})

# Função para prever as principais causas de morte em Inhumas com filtros
def prever_causas_morte_inhumas(caminho_csv, faixa_etaria=None, sexo=None, anos_a_prever=5, top_n_causas=5):
    try:
        # Carregar os dados filtrados para Inhumas com os filtros de faixa etária e sexo
        df_inhumas = carregar_dados_inhumas_prever_causas(caminho_csv, faixa_etaria, sexo)
        if "error" in df_inhumas:
            return df_inhumas  # Retornar o erro se ocorrer

        # Agrupar os dados por causa de morte (Cod_CID_10), faixa etária, sexo e ano, somando o total de óbitos
        df_agrupado = df_inhumas.groupby(['Cod_CID_10', 'DESCRICAO', 'faixa_etaria', 'sexo', 'ano'])['total_obitos'].sum().reset_index()

        # Dicionário para armazenar previsões
        previsoes = {}

        # Para cada causa de morte, treinar um modelo e prever os óbitos futuros
        for cid in df_agrupado['Cod_CID_10'].unique():
            # Filtrar os dados para a causa de morte atual
            df_cid = df_agrupado[df_agrupado['Cod_CID_10'] == cid]

            if len(df_cid) < 3:  # Verificar se há observações suficientes para o ARIMA
                continue  # Ignorar causas com menos de 3 observações

            # Treinar o modelo ARIMA
            modelo = ARIMA(df_cid['total_obitos'], order=(1, 1, 0))
            modelo_fit = modelo.fit()

            # Prever o número de mortes para os próximos anos
            previsao = modelo_fit.forecast(steps=anos_a_prever)

            # Adicionar a previsão ao dicionário
            previsoes[cid] = {
                "previsao": previsao.sum(),  # Somar as previsões ao longo dos anos
                "descricao": df_cid['DESCRICAO'].values[0],
                "faixa_etaria": df_cid['faixa_etaria'].values[0],
                "sexo": df_cid['sexo'].values[0]
            }

        # Ordenar as causas de morte com base nas previsões de maior impacto
        previsoes_ordenadas = sorted(previsoes.items(), key=lambda x: x[1]["previsao"], reverse=True)

        # Retornar as top N causas de morte que mais afetarão Inhumas
        top_causas = previsoes_ordenadas[:top_n_causas]

        return top_causas
    except Exception as e:
        return {"error": str(e)}

# Classe que processa a rota para as principais causas de morte em Inhumas
class PreverCausasMorteInhumasResource(Resource):
    def post(self):
        # Caminho para o arquivo CSV de mortalidade
        caminho_arquivo = r'app\data\mortalidade_com_descricao.csv'

        try:
            # Pegar os parâmetros enviados na requisição
            dados = request.json
            anos_a_prever = int(dados.get('anos_a_prever', 5))  # Prever 5 anos por padrão
            top_n_causas = int(dados.get('top_n_causas', 5))  # Retornar as top 5 causas por padrão
            faixa_etaria = dados.get('faixa_etaria')
            sexo = dados.get('sexo')

            # Prever as principais causas de morte em Inhumas
            top_causas = prever_causas_morte_inhumas(caminho_arquivo, faixa_etaria, sexo, anos_a_prever, top_n_causas)

            if "error" in top_causas:
                return jsonify({"error": top_causas["error"]})

            # Retornar o resultado das previsões
            return jsonify({
                'top_causas': top_causas
            })

        except Exception as e:
            return jsonify({"error": str(e)})

# Função para treinar e salvar o modelo de regressão linear múltipla
def treinar_regressao_linear_e_salvar(X, y, caminho_modelo=r'C:\Users\flavi\OneDrive\Documentos\POO II\API PIPS\app\modelo_arvore.joblib'):
    try:
        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinar o modelo de regressão linear
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        # Avaliar o modelo
        y_pred = modelo.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Salvar o modelo treinado
        joblib.dump(modelo, caminho_modelo)

        return modelo, mse
    except Exception as e:
        return {"error": str(e)}

# Função para carregar o modelo salvo
def carregar_modelo(caminho_modelo=r'C:\Users\flavi\OneDrive\Documentos\POO II\API PIPS\app\modelo_arvore.joblib'):
    try:
        modelo = joblib.load(caminho_modelo)
        return modelo
    except Exception as e:
        return {"error": str(e)}

# Função para treinar e salvar o modelo de árvore de decisão
def treinar_arvore_decisao_e_salvar(X, y, caminho_modelo=r'C:\Users\flavi\OneDrive\Documentos\POO II\API PIPS\app\modelo_arvore.joblib'):
    try:
        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinar o modelo de árvore de decisão
        modelo = DecisionTreeRegressor()
        modelo.fit(X_train, y_train)

        # Avaliar o modelo
        y_pred = modelo.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Salvar o modelo treinado
        joblib.dump(modelo, caminho_modelo)

        return modelo, mse
    except Exception as e:
        return {"error": str(e)}

# Modificar a classe para treinar o modelo escolhido (regressão linear ou árvore de decisão)
class AnaliseExpectativaVidaResource(Resource):
    def post(self):
        # Caminho para o arquivo CSV com os dados de saúde
        caminho_arquivo = r'app\data\expectativa_de_vida_inhumas.csv'

        try:
            # Carregar os dados
            dados_carregados = carregar_dados_analise_expectativa_vida(caminho_arquivo)

            # Verifique se dados_carregados é um dicionário de erro
            if isinstance(dados_carregados, dict) and "error" in dados_carregados:
                return jsonify({"error": dados_carregados["error"]})

            X, y = dados_carregados  # X são as features, y é o target

            # Pegar o tipo de modelo solicitado pelo usuário
            dados = request.json
            modelo_selecionado = dados.get('modelo', 'linear')  # Padrão é regressão linear

            # Treinar o modelo com base na escolha do usuário
            if modelo_selecionado == 'linear':
                modelo, mse = treinar_regressao_linear_e_salvar(X, y, caminho_modelo=r'C:\Users\flavi\OneDrive\Documentos\POO II\API PIPS\app\modelo_arvore.joblib')
            elif modelo_selecionado == 'arvore':
                modelo, mse = treinar_arvore_decisao_e_salvar(X, y, caminho_modelo=r'C:\Users\flavi\OneDrive\Documentos\POO II\API PIPS\app\modelo_arvore.joblib')
            else:
                return jsonify({"error": "Modelo não suportado."})

            # Retornar o erro quadrático médio e sucesso
            return jsonify({
                'modelo': modelo_selecionado,
                'mse': mse,
                'mensagem': f'Modelo {modelo_selecionado} treinado e salvo com sucesso.'
            })
        except Exception as e:
            return jsonify({"error": str(e)})

# Função para carregar o modelo salvo
def carregar_modelo(caminho_modelo=r'C:\Users\flavi\OneDrive\Documentos\POO II\API PIPS\app\modelo_arvore.joblib'):
    try:
        modelo = joblib.load(caminho_modelo)
        return modelo
    except Exception as e:
        return {"error": str(e)}

# Ajustar o endpoint de previsão para lidar com o modelo apropriado
class PrevisaoExpectativaVidaResource(Resource):
    def post(self):
        try:
            # Carregar os dados de entrada enviados pelo frontend
            dados = request.json
            
            # Dados fornecidos pelo frontend para a previsão (total_obitos, idade_media, ano, faixa_etaria, sexo)
            total_obitos = dados.get('total_obitos')
            idade_media = dados.get('idade_media')
            ano = dados.get('ano')
            faixa_etaria = dados.get('faixa_etaria')
            sexo = dados.get('sexo')
            modelo_selecionado = dados.get('modelo', 'linear')  # Padrão é regressão linear

            # Validar se todos os dados necessários foram fornecidos
            if not all([total_obitos, idade_media, ano, faixa_etaria, sexo]):
                return jsonify({"error": "Todos os parâmetros (total_obitos, idade_media, ano, faixa_etaria, sexo) são necessários."})

            # Escolher o caminho do modelo com base no tipo selecionado
            caminho_modelo = r'C:\Users\flavi\OneDrive\Documentos\POO II\API PIPS\app\modelo_arvore.joblib' if modelo_selecionado == 'linear' else 'modelo_arvore.joblib'

            # Carregar o modelo salvo
            modelo = carregar_modelo(caminho_modelo)
            if isinstance(modelo, dict) and "error" in modelo:
                return jsonify({"error": modelo["error"]})

            # Carregar o CSV para obter as variáveis dummy
            caminho_arquivo = r'app\data\expectativa_de_vida_inhumas.csv'
            dados_carregados = carregar_dados_analise_expectativa_vida(caminho_arquivo)
            X, _ = dados_carregados

            # Reconstruir as variáveis dummy para faixa_etaria e sexo
            df_input = pd.DataFrame({
                'total_obitos': [total_obitos],
                'idade_media': [idade_media],
                'ano': [ano],
                **{f'faixa_etaria_{faixa_etaria}': [1] if f'faixa_etaria_{faixa_etaria}' in X.columns else 0 for f in X.columns if f.startswith('faixa_etaria_')},
                **{f'sexo_{sexo}': [1] if f'sexo_{sexo}' in X.columns else 0 for f in X.columns if f.startswith('sexo_')}
            })

            # Garantir que todas as colunas necessárias estão presentes (mesmo as dummies não usadas)
            for col in X.columns:
                if col not in df_input.columns:
                    df_input[col] = 0  # Adiciona colunas faltantes com valor 0

            # Fazer a previsão com o modelo carregado
            previsao = modelo.predict(df_input)[0]

            # Retornar o resultado da previsão
            return jsonify({
                'previsao_expectativa_vida': previsao,
                'mensagem': 'Previsão realizada com sucesso.'
            })
        except Exception as e:
            return jsonify({"error": str(e)})

# Classe para retornar a análise
class AnaliseMortalidade(Resource):
    def get(self):
        try:
            mortalidade_df, especialidades_df = carregar_dados_mortalide_especialidades()

            # Verificar o conteúdo dos dados
            logging.debug(mortalidade_df.head())
            logging.debug(especialidades_df.head())

            # Agrupar dados de mortalidade por código CID-10
            mortalidade_agrupada = mortalidade_df.groupby('Cod_CID_10').agg({
                'total_obitos': 'sum',
                'DESCRICAO': 'first'  # Retorna a descrição da CID-10
            }).reset_index()

            # Preparar um dicionário para armazenar a análise
            analise = []

            # Loop para correlacionar mortalidade com especialidades
            for index, row in mortalidade_agrupada.iterrows():
                cid_10 = row['Cod_CID_10']
                descricao_cid = row['DESCRICAO']
                total_obitos = row['total_obitos']

                especialidades_list = especialidades_df['Descrição'].tolist()

                analise._append({
                    'CID_10': cid_10,
                    'Descricao_CID': descricao_cid,
                    'Total_Obitos': total_obitos,
                    'Especialidades_Oferecidas': especialidades_list
                })

            return jsonify(analise)
        except Exception as e:
            logging.error(f"Erro ao processar os dados: {e}")
            return jsonify({"error": "Erro ao processar os dados"}), 500


# Função para carregar os dados em CSV
def carregar_dados_prever_aumento_atendimento():
    # Carregar CSV de mortalidade e especialidades
    mortalidade_df = pd.read_csv(r'app\data\mortalidade_com_descricao.csv')
    especialidades_df = pd.read_csv(r'app\data\especialidades.csv')
    return mortalidade_df, especialidades_df

# Função para prever a necessidade futura de atendimentos com aumento percentual
def prever_aumento_atendimento_percentual(cid_10, anos=10, crescimento_anual=0.03):
    # Carregar os dados
    mortalidade_df, especialidades_df = carregar_dados_prever_aumento_atendimento()
    
    # Filtrar os dados de mortalidade para o CID-10 específico
    filtro_mortalidade = mortalidade_df[mortalidade_df['Cod_CID_10'] == cid_10]
    
    # Agrupar os dados por ano e calcular o total de óbitos
    mortalidade_ano = filtro_mortalidade.groupby('ano')['total_obitos'].sum().reset_index()

    # Verificar se há dados de mortalidade suficientes
    if mortalidade_ano.empty:
        return {"error": f"Nenhum dado de mortalidade encontrado para o CID-10: {cid_10}"}
    
    # Calcular a taxa de crescimento anual para os próximos 10 anos
    mortalidade_ano['predicao_obitos'] = mortalidade_ano['total_obitos']
    ultimo_ano = mortalidade_ano['ano'].max()
    ultimo_total_obitos = mortalidade_ano.loc[mortalidade_ano['ano'] == ultimo_ano, 'total_obitos'].values[0]

    predicoes = []
    
    for i in range(1, anos + 1):
        ano_futuro = ultimo_ano + i
        novo_total_obitos = ultimo_total_obitos * (1 + crescimento_anual)  # Crescimento anual de 3%
        
        # Calcular aumento percentual em relação ao último total de óbitos
        aumento_percentual = ((novo_total_obitos - ultimo_total_obitos) / ultimo_total_obitos) * 100
        
        # Atualizar o último total para a próxima iteração
        ultimo_total_obitos = novo_total_obitos
        
        # Adicionar a predição para o ano futuro
        predicoes.append({
            'ano': ano_futuro,
            'predicao_obitos': novo_total_obitos,
            'aumento_percentual': aumento_percentual
        })

    # Calcular a necessidade futura de atendimentos com base nas especialidades disponíveis
    especialidades_list = especialidades_df['Descrição'].tolist()
    necessidade_aumento = {
        "CID_10": cid_10,
        "descricao": filtro_mortalidade['DESCRICAO'].iloc[0],
        "especialidades_necessarias": especialidades_list,
        "predicoes": predicoes
    }

    return necessidade_aumento

# Classe do Flask para retornar a previsão
class PredicaoAtendimento(Resource):
    def get(self, cid_10):
        resultado = prever_aumento_atendimento_percentual(cid_10, anos=10, crescimento_anual=0.03)
        return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)