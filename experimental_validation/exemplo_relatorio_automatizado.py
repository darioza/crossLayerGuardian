"""
Exemplo de uso do Sistema de Relatórios Automatizados
Demonstra como gerar relatórios completos com dados experimentais
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from automated_reporting import (
    GeradorRelatorioAutomatizado, 
    ConfiguracaoRelatorio, 
    DadosRelatorio
)
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simular estruturas de dados do sistema (normalmente viriam dos módulos reais)
@dataclass
class MetricasComparativasSimulada:
    """Simula dados de MetricasComparativas para demonstração"""
    acuracia: float
    precisao: float
    recall: float
    f1_score: float
    roc_auc: float
    throughput_ops_por_segundo: float
    latencia_media_ms: float
    taxa_deteccao: float
    taxa_falso_positivo: float
    matriz_confusao: np.ndarray

def criar_dados_demonstracao() -> DadosRelatorio:
    """Cria dados de demonstração para teste do sistema de relatórios"""
    
    # Dados do CrossLayerGuardian (sistema proposto)
    crosslayer_metrics = MetricasComparativasSimulada(
        acuracia=0.9847,
        precisao=0.9823,
        recall=0.9871,
        f1_score=0.9847,
        roc_auc=0.9934,
        throughput_ops_por_segundo=8500.0,
        latencia_media_ms=12.3,
        taxa_deteccao=0.9871,
        taxa_falso_positivo=0.0177,
        matriz_confusao=np.array([[8820, 156], [103, 7921]])
    )
    
    # Dados de sistemas baseline para comparação
    snort_metrics = MetricasComparativasSimulada(
        acuracia=0.8934,
        precisao=0.8756,
        recall=0.9123,
        f1_score=0.8936,
        roc_auc=0.9234,
        throughput_ops_por_segundo=12000.0,
        latencia_media_ms=8.7,
        taxa_deteccao=0.9123,
        taxa_falso_positivo=0.1244,
        matriz_confusao=np.array([[7834, 1142], [701, 7323]])
    )
    
    suricata_metrics = MetricasComparativasSimulada(
        acuracia=0.9123,
        precisao=0.9034,
        recall=0.9234,
        f1_score=0.9133,
        roc_auc=0.9456,
        throughput_ops_por_segundo=9800.0,
        latencia_media_ms=15.2,
        taxa_deteccao=0.9234,
        taxa_falso_positivo=0.0966,
        matriz_confusao=np.array([[8112, 864], [612, 7412]])
    )
    
    zeek_metrics = MetricasComparativasSimulada(
        acuracia=0.8756,
        precisao=0.8634,
        recall=0.8923,
        f1_score=0.8776,
        roc_auc=0.9123,
        throughput_ops_por_segundo=7200.0,
        latencia_media_ms=22.1,
        taxa_deteccao=0.8923,
        taxa_falso_positivo=0.1366,
        matriz_confusao=np.array([[7756, 1220], [856, 7168]])
    )
    
    # Criar dados de relatório
    dados = DadosRelatorio()
    
    # Análise comparativa
    dados.analise_comparativa = {
        'CrossLayerGuardian': crosslayer_metrics,
        'Snort': snort_metrics,
        'Suricata': suricata_metrics,
        'Zeek/Bro': zeek_metrics
    }
    
    # Simular dados de performance temporal
    from collections import namedtuple
    PerformanceMetrics = namedtuple('PerformanceMetrics', ['detailed_metrics'])
    
    # Gerar dados temporais simulados para 5 minutos
    base_time = datetime.now().timestamp()
    temporal_data = []
    
    for i in range(300):  # 5 minutos, 1 amostra por segundo
        timestamp = base_time + i
        cpu_percent = 25.0 + np.random.normal(0, 3.0)  # CPU base 25% ± 3%
        memory_mb = 512.0 + np.random.normal(0, 50.0)  # RAM base 512MB ± 50MB
        
        # Simular picos de processamento ocasionais
        if i % 60 == 0:  # A cada minuto
            cpu_percent += np.random.uniform(10, 20)
            memory_mb += np.random.uniform(100, 200)
        
        temporal_data.append({
            'timestamp': timestamp,
            'cpu_percent': max(0, min(100, cpu_percent)),
            'memory_mb': max(100, memory_mb)
        })
    
    dados.metricas_performance = [
        PerformanceMetrics(detailed_metrics={'samples': temporal_data})
    ]
    
    # Simular resultados de sistema (latências end-to-end)
    from collections import namedtuple
    SystemTestResult = namedtuple('SystemTestResult', ['end_to_end_latency_ms'])
    
    # Gerar distribuição de latências realística
    latencias = []
    for i in range(1000):
        # Distribuição log-normal típica de latências de rede
        latencia = np.random.lognormal(mean=np.log(12.3), sigma=0.3)
        latencias.append(max(1.0, min(100.0, latencia)))  # Entre 1ms e 100ms
    
    dados.resultados_sistema = [
        SystemTestResult(end_to_end_latency_ms=lat) for lat in latencias
    ]
    
    # Metadados do experimento
    dados.metadados_experimento = {
        'duracao_experimento_horas': 24,
        'total_eventos_processados': 1500000,
        'datasets_utilizados': ['NSL-KDD', 'ADFA-LD', 'CICIDS2017'],
        'ambiente_teste': 'Ubuntu 20.04, Intel i7-9700K, 32GB RAM',
        'versao_crosslayer': '1.0.0',
        'data_coleta': datetime.now().isoformat()
    }
    
    return dados

def exemplo_relatorio_academico():
    """Gera relatório no estilo acadêmico para dissertação"""
    
    logger.info("🎓 Gerando relatório acadêmico para dissertação...")
    
    # Configuração acadêmica
    config = ConfiguracaoRelatorio(
        titulo="Validação Experimental do CrossLayerGuardian",
        subtitulo="Sistema de Detecção de Intrusão Cross-Layer Baseado em Machine Learning e eBPF",
        autor="Daniel Arioza",
        instituicao="Programa de Pós-Graduação em Ciência da Computação - UFRGS",
        data_experimento=datetime.now() - timedelta(days=7),
        incluir_graficos_interativos=True,
        incluir_analise_estatistica=True,
        incluir_codigo_fonte=False,
        formato_saida=['html', 'pdf'],
        tema_cores='academic',
        idioma='pt-BR',
        nivel_detalhamento='dissertacao'
    )
    
    # Criar gerador
    gerador = GeradorRelatorioAutomatizado(config)
    
    # Carregar dados de demonstração
    dados = criar_dados_demonstracao()
    gerador.carregar_dados_experimentais(dados)
    
    # Gerar relatório completo
    arquivos_gerados = gerador.gerar_relatorio_completo()
    
    logger.info(f"✅ Relatório acadêmico gerado com sucesso!")
    for formato, arquivo in arquivos_gerados.items():
        logger.info(f"   📄 {formato.upper()}: {arquivo}")
    
    return arquivos_gerados

def exemplo_relatorio_corporativo():
    """Gera relatório no estilo corporativo"""
    
    logger.info("🏢 Gerando relatório corporativo...")
    
    # Configuração corporativa
    config = ConfiguracaoRelatorio(
        titulo="CrossLayerGuardian - Análise de Performance",
        subtitulo="Avaliação Técnica e Comparativa de Sistema IDS Inovador",
        autor="Equipe de Pesquisa e Desenvolvimento",
        instituicao="Centro de Pesquisa em Cybersecurity",
        data_experimento=datetime.now() - timedelta(days=3),
        incluir_graficos_interativos=True,
        incluir_analise_estatistica=True,
        formato_saida=['html'],
        tema_cores='corporate',
        nivel_detalhamento='completo'
    )
    
    # Criar gerador
    gerador = GeradorRelatorioAutomatizado(config)
    
    # Carregar dados
    dados = criar_dados_demonstracao()
    gerador.carregar_dados_experimentais(dados)
    
    # Gerar relatório
    arquivos_gerados = gerador.gerar_relatorio_completo()
    
    logger.info(f"✅ Relatório corporativo gerado!")
    for formato, arquivo in arquivos_gerados.items():
        logger.info(f"   📄 {formato.upper()}: {arquivo}")
    
    return arquivos_gerados

def exemplo_relatorio_moderno():
    """Gera relatório com tema moderno/colorido"""
    
    logger.info("🎨 Gerando relatório com tema moderno...")
    
    # Configuração moderna
    config = ConfiguracaoRelatorio(
        titulo="CrossLayerGuardian Analytics",
        subtitulo="Next-Gen Intrusion Detection Performance Report",
        autor="Research Team",
        instituicao="Advanced Cybersecurity Lab",
        data_experimento=datetime.now() - timedelta(days=1),
        formato_saida=['html'],
        tema_cores='modern',
        nivel_detalhamento='resumido'
    )
    
    # Criar gerador
    gerador = GeradorRelatorioAutomatizado(config)
    
    # Carregar dados
    dados = criar_dados_demonstracao()
    gerador.carregar_dados_experimentais(dados)
    
    # Gerar relatório
    arquivos_gerados = gerador.gerar_relatorio_completo()
    
    logger.info(f"✅ Relatório moderno gerado!")
    for formato, arquivo in arquivos_gerados.items():
        logger.info(f"   📄 {formato.upper()}: {arquivo}")
    
    return arquivos_gerados

def demonstrar_capacidades():
    """Demonstra todas as capacidades do sistema de relatórios"""
    
    print("🚀 Sistema de Relatórios Automatizados - CrossLayerGuardian")
    print("=" * 60)
    print()
    
    print("📋 Capacidades do Sistema:")
    print("  ✅ Geração automática de relatórios HTML/PDF")
    print("  ✅ Visualizações interativas com Plotly")
    print("  ✅ Análise estatística rigorosa")
    print("  ✅ Comparação multi-sistema")
    print("  ✅ Gráficos de performance temporal")
    print("  ✅ Distribuição de latências")
    print("  ✅ Matrizes de confusão avançadas")
    print("  ✅ Radar charts comparativos")
    print("  ✅ Temas visuais configuráveis")
    print("  ✅ Exportação de dados brutos")
    print()
    
    print("🎓 Exemplo 1: Relatório Acadêmico (Dissertação)")
    print("-" * 50)
    try:
        arquivos1 = exemplo_relatorio_academico()
        print(f"   ✅ Gerado com sucesso: {len(arquivos1)} formato(s)")
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        logger.error(f"Erro no relatório acadêmico: {e}")
    
    print()
    print("🏢 Exemplo 2: Relatório Corporativo")
    print("-" * 40)
    try:
        arquivos2 = exemplo_relatorio_corporativo()
        print(f"   ✅ Gerado com sucesso: {len(arquivos2)} formato(s)")
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        logger.error(f"Erro no relatório corporativo: {e}")
    
    print()
    print("🎨 Exemplo 3: Relatório Moderno")
    print("-" * 35)
    try:
        arquivos3 = exemplo_relatorio_moderno()
        print(f"   ✅ Gerado com sucesso: {len(arquivos3)} formato(s)")
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        logger.error(f"Erro no relatório moderno: {e}")
    
    print()
    print("📊 Recursos Técnicos Implementados:")
    print("  🔬 Testes estatísticos: t-test, Mann-Whitney U, Wilcoxon")
    print("  📏 Effect size: Cohen's d com interpretação")
    print("  📈 Visualizações: 6 tipos diferentes de gráficos")
    print("  🎨 Temas: Academic, Corporate, Modern")
    print("  📄 Formatos: HTML (interativo), PDF (impressão)")
    print("  💾 Exportação: JSON, CSV para análise externa")
    print()
    
    print("🔄 Status: Sistema pronto para integração com framework experimental")
    print("🎯 Próximo: Integração com validation_framework.py")

if __name__ == "__main__":
    demonstrar_capacidades()