import os
import joblib
import pandas as pd

def carregar_dados(caminho):
    df = pd.read_csv(caminho)
    df['data_sintomas'] = pd.to_datetime(df['data_sintomas'], format='%Y%m%d')
    df['data_notificacao'] = pd.to_datetime(df['data_notificacao'], format='%Y%m%d')
    return df

def carregar_dados_srag(caminho):
    df = pd.read_csv(caminho)
    df['data_sintomas'] = pd.to_datetime(df['data_sintomas'], format='%Y%m%d', errors='coerce')
    df['data_notificacao'] = pd.to_datetime(df['data_notificacao'], format='%Y%m%d', errors='coerce')
    return df

def carregar_dados_nascimentos(caminho):
    df = pd.read_csv(caminho)
    df['data_nascimento'] = pd.to_datetime(df['data_nascimento'], format='%Y-%m-%d', errors='coerce')
    return df

def carregar_dados_mortalidade(caminho):
    df = pd.read_csv(caminho)
    return df

def carregar_dados_mortalidade(caminho):
    df_mortalidade = pd.read_csv(caminho)
    return df_mortalidade

def carregar_dados_expectativa_vida(caminho_csv):
    df = pd.read_csv(caminho_csv)

    # Calcular a média da expectativa de vida por município
    expectativa_por_municipio = df.groupby('Municipio_residencia')['expectativa_vida'].mean().reset_index()

    return expectativa_por_municipio

def carregar_evolucao_atendimentos(caminho_csv):
    df = pd.read_csv(caminho_csv)

    # Filtrar as linhas com 'TOTAL' na coluna 'Codigo'
    df_total = df[df['Codigo'] == 'TOTAL']

    # Extrair o ano a partir da competência
    df_total['Ano'] = df_total['Competencia'].astype(str).str[:4]

    # Agrupar por ano e somar os totais de atendimentos
    evolucao_atendimentos = df_total.groupby('Ano')['Total'].sum().reset_index()

    return evolucao_atendimentos

def carregar_atendimentos_por_ano(caminho_csv, ano=None):
    df = pd.read_csv(caminho_csv)

    # Filtrar as linhas com 'TOTAL' na coluna 'Codigo'
    df_total = df[df['Codigo'] == 'TOTAL']

    # Extrair o ano a partir da competência
    df_total['Ano'] = df_total['Competencia'].astype(str).str[:4]

    if ano:
        # Filtrar pelo ano solicitado
        df_total = df_total[df_total['Ano'] == str(ano)]

    # Agrupar por competência e somar os totais de atendimentos
    evolucao_atendimentos = df_total.groupby('Competencia')['Total'].sum().reset_index()

    return evolucao_atendimentos

def carregar_dados_atendimento_clientela(caminho_csv):
    try:
        df = pd.read_csv(caminho_csv)
        
        # Agrupar os dados por Tipo de Atendimento e Fluxo de Clientela
        atendimentos_clientela = df.groupby(['Tipo de Atendimento', 'Fluxo de Clientela']).size().reset_index(name='quantidade')
        
        print(atendimentos_clientela)  # Log para verificar os dados
        return atendimentos_clientela
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return []
    
def carregar_dados_capacidade_consultorios_leitos(caminho_csv):
    df = pd.read_csv(caminho_csv)

    # Selecionar as colunas relevantes para a capacidade de consultórios e leitos
    colunas_capacidade = [
        'Nome Fantasia', 
        'Qtde./Consultório 1', 'Leitos/Equipamentos 1', 
        'Qtde./Consultório 2', 'Leitos/Equipamentos 2', 
        'Qtde./Consultório 3', 'Leitos/Equipamentos 3',
        'Qtde./Consultório 4', 'Leitos/Equipamentos 4',
        'Qtde./Consultório 5', 'Leitos/Equipamentos 5'
    ]

    # Filtrar o DataFrame para as colunas relevantes
    df_capacidade = df[colunas_capacidade].copy()

    # Calcular a capacidade total de consultórios e leitos por estabelecimento
    df_capacidade['Total Consultórios'] = df_capacidade[
        ['Qtde./Consultório 1', 'Qtde./Consultório 2', 'Qtde./Consultório 3', 
         'Qtde./Consultório 4', 'Qtde./Consultório 5']
    ].sum(axis=1)

    df_capacidade['Total Leitos'] = df_capacidade[
        ['Leitos/Equipamentos 1', 'Leitos/Equipamentos 2', 'Leitos/Equipamentos 3', 
         'Leitos/Equipamentos 4', 'Leitos/Equipamentos 5']
    ].sum(axis=1)

    df_resultado = df_capacidade[['Nome Fantasia', 'Total Consultórios', 'Total Leitos']]

    return df_resultado

def carregar_dados_unidades_saude(caminho_csv):
    df = pd.read_csv(caminho_csv)

    # Remover a linha "Total" se houver
    df = df[df['Descricao'] != 'Total']

    return df

def carregar_dados_servicos_saude(caminho_csv):
    df = pd.read_csv(caminho_csv)

    # Remover a linha "TOTAL" se houver
    df = df[df['Descrição'] != 'TOTAL']

    return df

def carregar_dados_infraestrutura(caminho_csv):
    try:
        df = pd.read_csv(caminho_csv)

        # Convertendo os valores para garantir que são inteiros (se necessário)
        df['Qtde./Consultório 1'] = pd.to_numeric(df['Qtde./Consultório 1'], errors='coerce').fillna(0).astype(int)
        df['Qtde./Consultório 2'] = pd.to_numeric(df['Qtde./Consultório 2'], errors='coerce').fillna(0).astype(int)
        df['Leitos/Equipamentos 1'] = pd.to_numeric(df['Leitos/Equipamentos 1'], errors='coerce').fillna(0).astype(int)
        df['Leitos/Equipamentos 7'] = pd.to_numeric(df['Leitos/Equipamentos 7'], errors='coerce').fillna(0).astype(int)

        # Extraindo informações de consultórios e leitos
        infraestrutura = {
            'Consultórios Médicos': df['Qtde./Consultório 1'].sum(),
            'Consultórios Não Médicos': df['Qtde./Consultório 2'].sum(),
            'Leitos': df['Leitos/Equipamentos 1'].sum(),
            'Salas de Curativo': df['Qtde./Consultório 5'].sum() if 'Qtde./Consultório 5' in df else 0,
            'Salas de Observação': df['Leitos/Equipamentos 7'].sum(),
            'Salas de Imunização': df['Qtde./Consultório 6'].sum() if 'Qtde./Consultório 6' in df else 0
        }

        return infraestrutura
    except Exception as e:
        return {"error": str(e)}
    
def carregar_dados_prever_mortalidade(caminho_csv):
    try:
        # Lendo o arquivo CSV
        df = pd.read_csv(caminho_csv)

        # Garantindo que as colunas estão no formato correto
        df['total_obitos'] = pd.to_numeric(df['total_obitos'], errors='coerce').fillna(0).astype(int)
        df['faixa_etaria'] = df['faixa_etaria'].astype(str)
        df['sexo'] = df['sexo'].astype(str)
        df['Cod_CID_10'] = df['Cod_CID_10'].astype(str)

        # Variáveis preditoras e alvo
        X = df[['faixa_etaria', 'sexo', 'Cod_CID_10']]
        y = df['total_obitos']

        # Convertendo variáveis categóricas em numéricas
        X = pd.get_dummies(X, columns=['faixa_etaria', 'sexo', 'Cod_CID_10'], drop_first=True)

        return X, y
    except Exception as e:
        return {"error": str(e)}
    
def carregar_dados_filtrados(caminho_csv, cod_cid_10, sexo, faixa_etaria):
    try:
        # Lendo o arquivo CSV
        df = pd.read_csv(caminho_csv)

        # Filtrar os dados pelo Cod_CID_10, sexo e faixa etária fornecidos
        df_filtrado = df[(df['Cod_CID_10'] == cod_cid_10) & (df['sexo'] == sexo) & (df['faixa_etaria'] == faixa_etaria)]

        # Verificar se há dados suficientes após o filtro
        if df_filtrado.empty:
            return {"error": "Nenhum dado encontrado para o Cod_CID_10, sexo e faixa etária fornecidos."}

        # Agrupar os dados por ano e somar o total de óbitos
        df_agrupado = df_filtrado.groupby('ano')['total_obitos'].sum().reset_index()

        # Pegar a descrição da causa de óbito (primeira ocorrência para o Cod_CID_10)
        descricao = df_filtrado['DESCRICAO'].values[0] if 'DESCRICAO' in df_filtrado.columns else "Descrição não disponível"

        return df_agrupado, descricao
    except Exception as e:
        return {"error": str(e)}, None
    
def carregar_dados_inhumas_prever_causas(caminho_csv, faixa_etaria=None, sexo=None):
    try:
        # Lendo o arquivo CSV
        df = pd.read_csv(caminho_csv)

        # Filtrar os dados apenas para a cidade de Inhumas
        df_inhumas = df[df['Municipio_residencia'] == 'Inhumas']

        # Filtrar por faixa etária se fornecida
        if faixa_etaria:
            df_inhumas = df_inhumas[df_inhumas['faixa_etaria'] == faixa_etaria]

        # Filtrar por sexo se fornecido
        if sexo:
            df_inhumas = df_inhumas[df_inhumas['sexo'] == sexo]

        return df_inhumas
    except Exception as e:
        return {"error": str(e)}
    
def carregar_dados_analise_expectativa_vida(caminho_csv):
    try:
        # Carregar o CSV com os dados
        df = pd.read_csv(caminho_csv)

        # Verificar se todas as colunas necessárias estão no CSV
        colunas_necessarias = ['total_obitos', 'idade_media', 'faixa_etaria', 'sexo', 'ano', 'expectativa_vida']
        for coluna in colunas_necessarias:
            if coluna not in df.columns:
                return {"error": f"A coluna '{coluna}' está faltando no arquivo CSV."}

        # Converter faixa etária e sexo em variáveis categóricas (dummy variables)
        df = pd.get_dummies(df, columns=['faixa_etaria', 'sexo'], drop_first=True)

        # Variáveis independentes (features)
        X = df[['total_obitos', 'idade_media', 'ano'] + [col for col in df.columns if col.startswith('faixa_etaria_') or col.startswith('sexo_')]]

        # Variável dependente (target) - expectativa de vida
        y = df['expectativa_vida']

        return X, y
    except Exception as e:
        return {"error": str(e)}
    
def carregar_modelo(caminho_modelo=r'C:\Users\flavi\OneDrive\Documentos\POO II\API PIPS\app\modelo_arvore.joblib'):
    try:
        modelo = joblib.load(caminho_modelo)
        return modelo
    except Exception as e:
        return {"error": str(e)}
    
def carregar_dados_mortalide_especialidades():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mortalidade_path = os.path.join(base_dir, 'file/mortalidade_com_descricao.csv')
    especialidades_path = os.path.join(base_dir, 'file/especialidades.csv')

    # Carregar CSV de mortalidade e especialidades
    mortalidade_df = pd.read_csv(mortalidade_path)
    especialidades_df = pd.read_csv(especialidades_path)
    
    return mortalidade_df, especialidades_df

def carregar_dados_prever_aumento_atendimento():
    # Carregar CSV de mortalidade e especialidades
    mortalidade_df = pd.read_csv(r'C:\Users\flavi\OneDrive\Documentos\POO II\API PIPS\app\file\mortalidade_com_descricao.csv')
    especialidades_df = pd.read_csv(r'C:\Users\flavi\OneDrive\Documentos\POO II\API PIPS\app\file\especialidades.csv')
    return mortalidade_df, especialidades_df