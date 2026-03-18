"""
Versão Simplificada do Sistema de Relatórios Automatizados
Funciona com dependências básicas, sem geração de PDF
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from jinja2 import Template
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para usar backend sem GUI
plt.switch_backend('Agg')

logger = logging.getLogger(__name__)

@dataclass
class ConfiguracaoRelatorio:
    """Configuração para geração de relatórios"""
    titulo: str
    subtitulo: str
    autor: str
    instituicao: str
    data_experimento: datetime
    incluir_graficos_interativos: bool = True
    incluir_analise_estatistica: bool = True
    formato_saida: List[str] = field(default_factory=lambda: ['html'])
    tema_cores: str = 'academic'
    idioma: str = 'pt-BR'
    nivel_detalhamento: str = 'completo'

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

class GeradorRelatorioSimplificado:
    """Gerador simplificado de relatórios com funcionalidades essenciais"""
    
    def __init__(self, config: ConfiguracaoRelatorio):
        self.config = config
        self.dados_comparativos = {}
        
        # Cores por tema
        self.cores = {
            'academic': {
                'primary': '#2E8B57',
                'secondary': '#4682B4',
                'accent': '#DC143C',
                'palette': ['#2E8B57', '#4682B4', '#DC143C', '#FF8C00', '#9370DB', '#20B2AA']
            },
            'corporate': {
                'primary': '#1F4E79',
                'secondary': '#2E75B6',
                'accent': '#D32F2F',
                'palette': ['#1F4E79', '#2E75B6', '#D32F2F', '#F57C00', '#7B1FA2', '#00796B']
            },
            'modern': {
                'primary': '#6C5CE7',
                'secondary': '#A29BFE',
                'accent': '#FD79A8',
                'palette': ['#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E', '#6C5CE7', '#74B9FF']
            }
        }[config.tema_cores]
        
        # Criar diretório de saída
        self.dir_saida = Path(f"relatorios_automatizados_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.dir_saida.mkdir(exist_ok=True)
    
    def carregar_dados_comparativos(self, dados: Dict[str, MetricasComparativasSimulada]):
        """Carrega dados comparativos entre sistemas"""
        self.dados_comparativos = dados
        logger.info(f"Dados carregados para {len(dados)} sistemas")
    
    def criar_grafico_comparacao_barras(self) -> str:
        """Cria gráfico de barras comparativo"""
        
        if not self.dados_comparativos:
            return "<p>Dados não disponíveis</p>"
        
        sistemas = list(self.dados_comparativos.keys())
        metricas = ['acuracia', 'precisao', 'recall', 'f1_score', 'roc_auc']
        
        fig = make_subplots(
            rows=1, cols=len(metricas),
            subplot_titles=[m.replace('_', ' ').title() for m in metricas]
        )
        
        for i, metrica in enumerate(metricas):
            valores = [getattr(self.dados_comparativos[s], metrica) for s in sistemas]
            cores = [self.cores['primary'] if 'CrossLayer' in s else self.cores['secondary'] for s in sistemas]
            
            fig.add_trace(
                go.Bar(
                    x=sistemas,
                    y=valores,
                    name=metrica.title(),
                    marker_color=cores,
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title_text="Comparação de Performance - Sistemas IDS",
            title_x=0.5,
            height=500,
            font=dict(family="Arial", size=12)
        )
        
        # Configurar eixos Y entre 0 e 1
        for i in range(1, len(metricas) + 1):
            fig.update_yaxes(range=[0, 1], row=1, col=i)
        
        fig.update_xaxes(tickangle=45)
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def criar_radar_chart(self) -> str:
        """Cria radar chart comparativo"""
        
        if not self.dados_comparativos:
            return "<p>Dados não disponíveis</p>"
        
        fig = go.Figure()
        
        metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'ROC AUC']
        
        for i, (sistema, dados) in enumerate(self.dados_comparativos.items()):
            valores = [
                dados.acuracia,
                dados.precisao,
                dados.recall,
                dados.f1_score,
                dados.roc_auc
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=valores,
                theta=metricas,
                fill='toself',
                name=sistema,
                line_color=self.cores['palette'][i % len(self.cores['palette'])]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Comparação Multi-dimensional - Radar Chart",
            showlegend=True,
            height=600
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def criar_matriz_confusao(self, sistema: str, dados: MetricasComparativasSimulada) -> str:
        """Cria matriz de confusão para um sistema"""
        
        matriz = dados.matriz_confusao
        nomes_classes = ['Normal', 'Ataque']
        
        # Calcular percentuais
        matriz_percentual = matriz.astype('float') / matriz.sum(axis=1)[:, np.newaxis] * 100
        
        fig = go.Figure(data=go.Heatmap(
            z=matriz,
            x=nomes_classes,
            y=nomes_classes,
            colorscale='Blues',
            text=[[f'{matriz[i][j]}<br>({matriz_percentual[i][j]:.1f}%)' 
                   for j in range(len(nomes_classes))] 
                  for i in range(len(nomes_classes))],
            texttemplate="%{text}",
            textfont={"size": 14},
            colorbar=dict(title="Contagem")
        ))
        
        fig.update_layout(
            title=f"Matriz de Confusão - {sistema}",
            xaxis_title="Predição",
            yaxis_title="Valor Real",
            font=dict(size=12),
            width=400,
            height=350
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def criar_analise_estatistica(self) -> Dict[str, Any]:
        """Cria análise estatística básica"""
        
        if not self.dados_comparativos:
            return {}
        
        # Encontrar CrossLayerGuardian
        crosslayer_data = None
        baseline_data = []
        
        for nome, dados in self.dados_comparativos.items():
            if 'CrossLayer' in nome:
                crosslayer_data = dados
            else:
                baseline_data.append((nome, dados))
        
        if not crosslayer_data:
            return {}
        
        # Análise comparativa
        analise = {
            'crosslayer_superior': {},
            'melhorias_percentuais': {},
            'resumo': {}
        }
        
        metricas = ['acuracia', 'precisao', 'recall', 'f1_score', 'roc_auc']
        
        for metrica in metricas:
            valor_cl = getattr(crosslayer_data, metrica)
            
            superior_count = 0
            melhorias = []
            
            for nome_bl, dados_bl in baseline_data:
                valor_bl = getattr(dados_bl, metrica)
                
                if valor_cl > valor_bl:
                    superior_count += 1
                    melhoria = ((valor_cl - valor_bl) / valor_bl) * 100
                    melhorias.append(melhoria)
            
            analise['crosslayer_superior'][metrica] = superior_count
            analise['melhorias_percentuais'][metrica] = {
                'media': np.mean(melhorias) if melhorias else 0,
                'max': np.max(melhorias) if melhorias else 0,
                'min': np.min(melhorias) if melhorias else 0
            }
        
        # Resumo geral
        total_comparacoes = len(baseline_data) * len(metricas)
        total_superior = sum(analise['crosslayer_superior'].values())
        
        analise['resumo'] = {
            'percentual_superior': (total_superior / total_comparacoes) * 100,
            'total_comparacoes': total_comparacoes,
            'total_superior': total_superior
        }
        
        return analise
    
    def gerar_tabela_comparativa(self) -> str:
        """Gera tabela HTML comparativa"""
        
        if not self.dados_comparativos:
            return "<p>Dados não disponíveis</p>"
        
        html = """
        <table class="table-comparativa">
            <thead>
                <tr>
                    <th>Sistema</th>
                    <th>Acurácia</th>
                    <th>Precisão</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>ROC AUC</th>
                    <th>Throughput<br>(ops/sec)</th>
                    <th>Latência<br>(ms)</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for sistema, dados in self.dados_comparativos.items():
            classe_linha = 'crosslayer-row' if 'CrossLayer' in sistema else 'baseline-row'
            
            html += f"""
                <tr class="{classe_linha}">
                    <td><strong>{sistema}</strong></td>
                    <td>{dados.acuracia:.4f}</td>
                    <td>{dados.precisao:.4f}</td>
                    <td>{dados.recall:.4f}</td>
                    <td>{dados.f1_score:.4f}</td>
                    <td>{dados.roc_auc:.4f}</td>
                    <td>{dados.throughput_ops_por_segundo:.0f}</td>
                    <td>{dados.latencia_media_ms:.1f}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def gerar_css_styles(self) -> str:
        """Gera estilos CSS"""
        
        return f"""
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #2F4F4F;
            background-color: #FFFFFF;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, {self.cores['primary']}, {self.cores['secondary']});
            color: white;
            padding: 40px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header h2 {{
            font-size: 1.5em;
            font-weight: 300;
            margin-bottom: 20px;
        }}
        
        section {{
            margin: 40px 0;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        section h2 {{
            color: {self.cores['primary']};
            border-bottom: 3px solid {self.cores['primary']};
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        
        .table-comparativa {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        .table-comparativa th,
        .table-comparativa td {{
            border: 1px solid #dee2e6;
            padding: 12px;
            text-align: center;
        }}
        
        .table-comparativa th {{
            background-color: {self.cores['primary']};
            color: white;
            font-weight: 600;
        }}
        
        .crosslayer-row {{
            background-color: #e8f5e8 !important;
            font-weight: bold;
        }}
        
        .baseline-row:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        .visualization {{
            margin: 30px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 8px;
        }}
        
        .stats-summary {{
            background: #e3f2fd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        .highlight-box {{
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid {self.cores['accent']};
            margin: 15px 0;
        }}
        
        .matrices-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        """
    
    def gerar_relatorio_html(self) -> str:
        """Gera relatório HTML completo"""
        
        # Gerar visualizações
        grafico_comparacao = self.criar_grafico_comparacao_barras()
        radar_chart = self.criar_radar_chart()
        tabela_comparativa = self.gerar_tabela_comparativa()
        
        # Matrizes de confusão
        matrizes_html = ""
        for sistema, dados in self.dados_comparativos.items():
            matriz_html = self.criar_matriz_confusao(sistema, dados)
            matrizes_html += f'<div class="matrix-container">{matriz_html}</div>'
        
        # Análise estatística
        analise_stats = self.criar_analise_estatistica()
        
        # Template HTML
        template_html = f"""
<!DOCTYPE html>
<html lang="{self.config.idioma}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.titulo}</title>
    <style>
        {self.gerar_css_styles()}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        <!-- Cabeçalho -->
        <header class="header">
            <h1>{self.config.titulo}</h1>
            <h2>{self.config.subtitulo}</h2>
            <div class="meta-info">
                <p><strong>Autor:</strong> {self.config.autor}</p>
                <p><strong>Instituição:</strong> {self.config.instituicao}</p>
                <p><strong>Data do Experimento:</strong> {self.config.data_experimento.strftime('%d/%m/%Y')}</p>
                <p><strong>Relatório Gerado:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            </div>
        </header>

        <!-- Sumário Executivo -->
        <section class="executive-summary">
            <h2>📊 Sumário Executivo</h2>
            {self._gerar_sumario_executivo(analise_stats)}
        </section>

        <!-- Tabela Comparativa -->
        <section class="comparative-table">
            <h2>📋 Comparação Detalhada de Sistemas</h2>
            {tabela_comparativa}
        </section>

        <!-- Gráficos Comparativos -->
        <section class="comparative-charts">
            <h2>📈 Análise Visual Comparativa</h2>
            <div class="visualization">
                <h3>Comparação por Métricas</h3>
                {grafico_comparacao}
            </div>
            
            <div class="visualization">
                <h3>Radar Chart Multi-dimensional</h3>
                {radar_chart}
            </div>
        </section>

        <!-- Matrizes de Confusão -->
        <section class="confusion-matrices">
            <h2>🎯 Matrizes de Confusão</h2>
            <div class="matrices-grid">
                {matrizes_html}
            </div>
        </section>

        <!-- Análise Estatística -->
        <section class="statistical-analysis">
            <h2>🔬 Análise Estatística</h2>
            {self._gerar_analise_estatistica_html(analise_stats)}
        </section>

        <!-- Conclusões -->
        <section class="conclusions">
            <h2>🎓 Conclusões</h2>
            {self._gerar_conclusoes_html()}
        </section>
    </div>
</body>
</html>
        """
        
        return template_html
    
    def _gerar_sumario_executivo(self, analise_stats: Dict[str, Any]) -> str:
        """Gera sumário executivo"""
        
        if not analise_stats:
            return "<p>Análise estatística não disponível</p>"
        
        resumo = analise_stats.get('resumo', {})
        percentual_superior = resumo.get('percentual_superior', 0)
        
        crosslayer = self.dados_comparativos.get('CrossLayerGuardian')
        if not crosslayer:
            return "<p>Dados do CrossLayerGuardian não disponíveis</p>"
        
        return f"""
        <div class="highlight-box">
            <h3>🏆 Performance Superior</h3>
            <p>O CrossLayerGuardian demonstrou superioridade em <strong>{percentual_superior:.1f}%</strong> 
            das métricas avaliadas, com F1-Score de <strong>{crosslayer.f1_score:.4f}</strong> 
            e ROC AUC de <strong>{crosslayer.roc_auc:.4f}</strong>.</p>
        </div>
        
        <div class="highlight-box">
            <h3>⚡ Eficiência Operacional</h3>
            <p>Mantém throughput competitivo de <strong>{crosslayer.throughput_ops_por_segundo:.0f} ops/sec</strong> 
            com latência de apenas <strong>{crosslayer.latencia_media_ms:.1f} ms</strong>.</p>
        </div>
        
        <div class="highlight-box">
            <h3>🎯 Precisão Elevada</h3>
            <p>Taxa de detecção de <strong>{crosslayer.taxa_deteccao:.1%}</strong> 
            com baixa taxa de falsos positivos ({crosslayer.taxa_falso_positivo:.1%}).</p>
        </div>
        """
    
    def _gerar_analise_estatistica_html(self, analise_stats: Dict[str, Any]) -> str:
        """Gera seção HTML da análise estatística"""
        
        if not analise_stats:
            return "<p>Análise estatística não disponível</p>"
        
        html = '<div class="stats-summary">'
        html += '<h3>Resumo Estatístico</h3>'
        
        resumo = analise_stats.get('resumo', {})
        html += f"""
        <ul>
            <li><strong>Comparações realizadas:</strong> {resumo.get('total_comparacoes', 0)}</li>
            <li><strong>CrossLayerGuardian superior em:</strong> {resumo.get('total_superior', 0)} casos</li>
            <li><strong>Percentual de superioridade:</strong> {resumo.get('percentual_superior', 0):.1f}%</li>
        </ul>
        """
        
        html += '<h3>Melhorias por Métrica</h3>'
        melhorias = analise_stats.get('melhorias_percentuais', {})
        
        html += '<ul>'
        for metrica, valores in melhorias.items():
            if valores['max'] > 0:
                html += f"""
                <li><strong>{metrica.replace('_', ' ').title()}:</strong> 
                    Melhoria média de {valores['media']:.1f}% 
                    (máx: {valores['max']:.1f}%)</li>
                """
        html += '</ul>'
        html += '</div>'
        
        return html
    
    def _gerar_conclusoes_html(self) -> str:
        """Gera seção de conclusões"""
        
        return """
        <p>O CrossLayerGuardian demonstrou capacidade superior de detecção de intrusão através da 
        correlação inovadora entre eventos de rede e sistema de arquivos. Os resultados indicam:</p>
        
        <ul>
            <li><strong>Superior performance de detecção:</strong> Supera sistemas IDS tradicionais em múltiplas métricas</li>
            <li><strong>Eficiência operacional:</strong> Mantém throughput competitivo com baixa latência</li>
            <li><strong>Redução de falsos positivos:</strong> Correlação cross-layer melhora precisão</li>
            <li><strong>Inovação tecnológica:</strong> Primeira implementação prática de correlação eBPF + ML em tempo real</li>
        </ul>
        
        <p>Os resultados validam a hipótese de que a correlação entre camadas de rede e sistema de arquivos 
        pode melhorar significativamente a capacidade de detecção de ataques complexos.</p>
        """
    
    def salvar_relatorio(self) -> str:
        """Salva o relatório HTML"""
        
        conteudo_html = self.gerar_relatorio_html()
        arquivo_html = self.dir_saida / f"relatorio_simplificado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with open(arquivo_html, 'w', encoding='utf-8') as f:
            f.write(conteudo_html)
        
        logger.info(f"Relatório salvo: {arquivo_html}")
        return str(arquivo_html)

def criar_dados_demonstracao() -> Dict[str, MetricasComparativasSimulada]:
    """Cria dados de demonstração"""
    
    # CrossLayerGuardian - sistema proposto
    crosslayer = MetricasComparativasSimulada(
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
    
    # Sistemas baseline
    snort = MetricasComparativasSimulada(
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
    
    suricata = MetricasComparativasSimulada(
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
    
    zeek = MetricasComparativasSimulada(
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
    
    return {
        'CrossLayerGuardian': crosslayer,
        'Snort': snort,
        'Suricata': suricata,
        'Zeek/Bro': zeek
    }

def demonstrar_relatorio_simplificado():
    """Demonstra o sistema de relatórios simplificado"""
    
    print("🚀 Sistema de Relatórios Automatizados - Versão Simplificada")
    print("=" * 65)
    print()
    
    # Configuração
    config = ConfiguracaoRelatorio(
        titulo="CrossLayerGuardian - Validação Experimental",  
        subtitulo="Sistema de Detecção de Intrusão Cross-Layer com Machine Learning",
        autor="Daniel Arioza",
        instituicao="UFRGS - Programa de Pós-Graduação em Ciência da Computação",
        data_experimento=datetime.now() - timedelta(days=7),
        tema_cores='academic'
    )
    
    print(f"📋 Configuração do Relatório:")
    print(f"   📝 Título: {config.titulo}")
    print(f"   🎨 Tema: {config.tema_cores}")
    print(f"   📊 Formato: {config.formato_saida}")
    print()
    
    # Criar gerador
    gerador = GeradorRelatorioSimplificado(config)
    
    # Carregar dados de demonstração
    dados = criar_dados_demonstracao()
    gerador.carregar_dados_comparativos(dados)
    
    print(f"💾 Dados Carregados:")
    for sistema, metricas in dados.items():
        destaque = "🏆" if "CrossLayer" in sistema else "📊"
        print(f"   {destaque} {sistema}: F1={metricas.f1_score:.4f}, ROC AUC={metricas.roc_auc:.4f}")
    print()
    
    # Gerar relatório
    try:
        print("⚙️  Gerando relatório...")
        arquivo_gerado = gerador.salvar_relatorio()
        
        print("✅ Relatório gerado com sucesso!")
        print(f"   📄 Arquivo: {arquivo_gerado}")
        print()
        
        print("🎨 Componentes Incluídos:")
        print("   ✅ Cabeçalho com metadados")
        print("   ✅ Sumário executivo")
        print("   ✅ Tabela comparativa detalhada")
        print("   ✅ Gráficos de barras comparativos")
        print("   ✅ Radar chart multi-dimensional")
        print("   ✅ Matrizes de confusão individuais")
        print("   ✅ Análise estatística básica")
        print("   ✅ Conclusões e implicações")
        print("   ✅ Estilos CSS responsivos")
        print()
        
        print("🔧 Tecnologias Utilizadas:")
        print("   📊 Plotly.js para gráficos interativos")
        print("   🎨 CSS Grid e Flexbox para layout")
        print("   📱 Design responsivo")
        print("   🌐 HTML5 semântico")
        print()
        
        print("🎯 Status: Pronto para integração com framework experimental")
        
        return arquivo_gerado
        
    except Exception as e:
        print(f"❌ Erro na geração: {e}")
        logger.error(f"Erro: {e}")
        return None

if __name__ == "__main__":
    demonstrar_relatorio_simplificado()