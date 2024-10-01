from app.resources import (
    PredicaoAtendimento, AnaliseMortalidade, PrevisaoExpectativaVidaResource,
    AnaliseExpectativaVidaResource, PreverCausasMorteInhumasResource, 
    PrevisaoMortesAnuaisFiltradaResource, PredicaoMortalidadeResource,
    GraficoCapacidadeInfraestruturaResource, GraficoServicosSaudeResource,
    GraficoUnidadesSaudeResource, GraficoCapacidadeConsultoriosLeitosResource,
    GraficoAtendimentoClientelaResource, EvolucaoAtendimentosAnoResource,
    EvolucaoAtendimentosResource, GraficoExpectativaVidaResource,
    GraficoTaxasMortalidadeResource, GraficoEvolucaoObitosResource,
    GraficoCausasMorteCID10Resource, GraficoDistribuicaoObitosFaixaEtariaSexoResource,
    GraficoEvolucaoNascimentosResource, GraficoEvolucaoSRAGResource,
    GraficoEvolucaoDengueResource, UserLoginResource
)

def register_routes(api):
    api.add_resource(PredicaoAtendimento, '/predicao/<string:cid_10>')
    api.add_resource(AnaliseMortalidade, '/grafico/analise-mortalidade')
    api.add_resource(PrevisaoExpectativaVidaResource, '/previsao/expectativa_vida')
    api.add_resource(AnaliseExpectativaVidaResource, '/analise/expectativa_vida')
    api.add_resource(PreverCausasMorteInhumasResource, '/previsao/causas_morte_inhumas')
    api.add_resource(PrevisaoMortesAnuaisFiltradaResource, '/previsao/mortes_filtrada')
    api.add_resource(PredicaoMortalidadeResource, '/predicoes/mortalidade')
    api.add_resource(GraficoCapacidadeInfraestruturaResource, '/grafico/capacidade-infraestrutura')
    api.add_resource(GraficoServicosSaudeResource, '/grafico/servicos-saude')
    api.add_resource(GraficoUnidadesSaudeResource, '/grafico/unidades-saude')
    api.add_resource(GraficoCapacidadeConsultoriosLeitosResource, '/grafico/capacidade-consultorios-leitos')
    api.add_resource(GraficoAtendimentoClientelaResource, '/grafico/atendimento-clientela')
    api.add_resource(EvolucaoAtendimentosAnoResource, '/grafico/evolucao-atendimentos-ano')
    api.add_resource(EvolucaoAtendimentosResource, '/grafico/evolucao-atendimentos')
    api.add_resource(GraficoExpectativaVidaResource, '/grafico/expectativa-vida')
    api.add_resource(GraficoTaxasMortalidadeResource, '/grafico/taxas-mortalidade')
    api.add_resource(GraficoEvolucaoObitosResource, '/grafico/evolucao-obitos')
    api.add_resource(GraficoCausasMorteCID10Resource, '/grafico/causas-morte-cid10')
    api.add_resource(GraficoDistribuicaoObitosFaixaEtariaSexoResource, '/grafico/distribuicao-obitos-faixa-etaria-sexo')
    api.add_resource(GraficoEvolucaoNascimentosResource, '/grafico/evolucao-nascimentos-inhumas')
    api.add_resource(GraficoEvolucaoSRAGResource, '/grafico/evolucao-srag-inhumas')
    api.add_resource(GraficoEvolucaoDengueResource, '/grafico/evolucao-dengue-inhumas')
    api.add_resource(UserLoginResource, '/login')
