"""
Exemplo de Uso do Framework de Análise Comparativa
Demonstra como executar comparações entre CrossLayerGuardian e sistemas IDS baseline
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from comparative_analysis import AnalisadorComparativo
from validation_framework import SyntheticDataGenerator
import numpy as np
import time
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def executar_exemplo_analise_comparativa():
    """Exemplo completo de análise comparativa"""
    
    print("🚀 Iniciando Exemplo de Análise Comparativa")
    print("=" * 60)
    
    # 1. Gerar dados sintéticos para teste
    print("\n📊 Gerando dados sintéticos para teste...")
    gerador = SyntheticDataGenerator(seed=42)
    
    # Gerar dataset com ataques variados
    dataset_sintetico = gerador.generate_dataset(
        n_samples=1000,
        attack_ratio=0.3  # 30% ataques, 70% tráfego normal
    )
    
    # Converter para formato adequado
    dados_teste = []
    rotulos_verdadeiros = []
    
    for grupo_evento, rotulo in dataset_sintetico:
        # Pegar primeiro evento do grupo para simplificar
        if grupo_evento.events:
            evento = grupo_evento.events[0]
            dados_teste.append(evento)
            rotulos_verdadeiros.append(rotulo)
    
    print(f"✅ Dataset gerado: {len(dados_teste)} eventos")
    print(f"   - Normal: {rotulos_verdadeiros.count(0)} eventos")
    print(f"   - Ataques: {rotulos_verdadeiros.count(1)} eventos")
    
    # 2. Inicializar analisador comparativo
    print("\n🔧 Inicializando Framework de Análise Comparativa...")
    analisador = AnalisadorComparativo(diretorio_saida="exemplo_analise_comparativa")
    
    # 3. Executar análise completa
    print("\n🏃‍♂️ Executando análise comparativa completa...")
    print("   Isso pode levar alguns minutos...")
    
    try:
        resultados = analisador.executar_analise_completa(
            dados_teste=dados_teste,
            rotulos_verdadeiros=rotulos_verdadeiros
        )
        
        # 4. Apresentar resultados
        print("\n📈 RESULTADOS DA ANÁLISE COMPARATIVA")
        print("=" * 50)
        
        for nome_sistema, metricas in resultados.items():
            print(f"\n🛡️  {nome_sistema}")
            print(f"   Acurácia:     {metricas.acuracia:.4f}")
            print(f"   Precisão:     {metricas.precisao:.4f}")
            print(f"   Recall:       {metricas.recall:.4f}")
            print(f"   F1-Score:     {metricas.f1_score:.4f}")
            print(f"   ROC AUC:      {metricas.roc_auc:.4f}")
            print(f"   Throughput:   {metricas.throughput_ops_por_segundo:.0f} ops/sec")
            print(f"   Latência:     {metricas.latencia_media_ms:.2f} ms")
        
        # 5. Identificar melhor sistema
        melhor_sistema = max(resultados.keys(), key=lambda k: resultados[k].f1_score)
        melhor_f1 = resultados[melhor_sistema].f1_score
        
        print(f"\n🏆 MELHOR SISTEMA: {melhor_sistema}")
        print(f"   F1-Score: {melhor_f1:.4f}")
        
        # 6. Análise específica do CrossLayerGuardian
        if 'CrossLayerGuardian' in resultados:
            clg_metricas = resultados['CrossLayerGuardian']
            print(f"\n🎯 ANÁLISE CROSSLAYERGUARDIAN")
            print(f"   Posição no ranking F1: ", end="")
            ranking = sorted(resultados.items(), key=lambda x: x[1].f1_score, reverse=True)
            posicao = next(i for i, (nome, _) in enumerate(ranking) if nome == 'CrossLayerGuardian') + 1
            print(f"{posicao}º lugar de {len(resultados)} sistemas")
            
            # Comparar com baselines
            print(f"\n   Comparação com Baselines:")
            for nome, metricas in resultados.items():
                if nome != 'CrossLayerGuardian':
                    melhoria_f1 = ((clg_metricas.f1_score - metricas.f1_score) / metricas.f1_score * 100) if metricas.f1_score > 0 else 0
                    melhoria_throughput = ((clg_metricas.throughput_ops_por_segundo - metricas.throughput_ops_por_segundo) / metricas.throughput_ops_por_segundo * 100) if metricas.throughput_ops_por_segundo > 0 else 0
                    
                    print(f"   vs {nome}:")
                    print(f"     F1-Score: {melhoria_f1:+.1f}%")
                    print(f"     Throughput: {melhoria_throughput:+.1f}%")
        
        # 7. Informações sobre arquivos gerados
        print(f"\n📁 ARQUIVOS GERADOS:")
        print(f"   📊 Relatório HTML: exemplo_analise_comparativa/relatorio_comparativo.html")
        print(f"   📈 Gráficos: exemplo_analise_comparativa/comparacao_sistemas_ids.png")
        print(f"   📋 Dados JSON: exemplo_analise_comparativa/resultados_comparativos.json")
        print(f"   🔬 Análise Estatística: exemplo_analise_comparativa/analise_estatistica.json")
        
        return resultados
        
    except Exception as e:
        logger.error(f"Erro durante análise comparativa: {e}")
        return None

def demonstrar_configuracao_baselines():
    """Demonstra configuração dos sistemas baseline"""
    
    print("\n🔧 CONFIGURAÇÃO DOS SISTEMAS BASELINE")
    print("=" * 45)
    
    print("\n1️⃣  Snort IDS")
    print("   Tipo: Network-based IDS")
    print("   Detecção: Baseada em assinatura")
    print("   Características:")
    print("   - Regras estáticas pré-definidas")
    print("   - Alta velocidade de processamento")
    print("   - Baixa taxa de falsos negativos para ataques conhecidos")
    print("   - Dificuldade com ataques zero-day")
    
    print("\n2️⃣  Suricata IDS")
    print("   Tipo: Network-based IDS/IPS")
    print("   Detecção: Assinatura + Anomalia")
    print("   Características:")
    print("   - Multi-threading nativo")
    print("   - Análise de protocolo mais avançada")
    print("   - Detecção de fluxo de rede")
    print("   - Metadata enrichment")
    
    print("\n3️⃣  OSSEC HIDS")
    print("   Tipo: Host-based IDS")
    print("   Detecção: Log analysis + File integrity")
    print("   Características:")
    print("   - Monitoramento de arquivos de sistema")
    print("   - Análise de logs centralizada")
    print("   - Detecção de rootkits")
    print("   - Resposta ativa a incidentes")
    
    print("\n🛡️  CrossLayerGuardian")
    print("   Tipo: Cross-layer ML-based IDS")
    print("   Detecção: Machine Learning Ensemble")
    print("   Características:")
    print("   - Correlação entre eventos de rede e host")
    print("   - Ensemble XGBoost + MLP")
    print("   - 127 features cross-layer")
    print("   - Aprendizado adaptativo")
    print("   - Detecção de ataques multi-estágio")

def exibir_metricas_esperadas():
    """Exibe métricas esperadas para cada sistema"""
    
    print("\n📊 MÉTRICAS ESPERADAS")
    print("=" * 25)
    
    sistemas_esperados = {
        "CrossLayerGuardian": {
            "acuracia": "0.92-0.96",
            "f1_score": "0.88-0.94",
            "throughput": "800-1200 ops/sec",
            "latencia": "8-15 ms",
            "vantagens": ["Detecção multi-estágio", "Baixos falsos positivos", "Adaptativo"]
        },
        "Snort": {
            "acuracia": "0.85-0.92",
            "f1_score": "0.80-0.88",
            "throughput": "1500-2500 ops/sec",
            "latencia": "2-5 ms",
            "vantagens": ["Velocidade", "Regras bem estabelecidas", "Baixo overhead"]
        },
        "Suricata": {
            "acuracia": "0.87-0.93",
            "f1_score": "0.82-0.90",
            "throughput": "1200-2000 ops/sec",
            "latencia": "3-8 ms",
            "vantagens": ["Multi-threading", "Análise avançada", "Metadata"]
        },
        "OSSEC": {
            "acuracia": "0.80-0.88",
            "f1_score": "0.75-0.85",
            "throughput": "500-1000 ops/sec",
            "latencia": "10-20 ms",
            "vantagens": ["Integridade de arquivos", "Análise de logs", "Host-based"]
        }
    }
    
    for sistema, metricas in sistemas_esperados.items():
        print(f"\n{sistema}:")
        print(f"  Acurácia esperada: {metricas['acuracia']}")
        print(f"  F1-Score esperado: {metricas['f1_score']}")
        print(f"  Throughput esperado: {metricas['throughput']}")
        print(f"  Latência esperada: {metricas['latencia']}")
        print(f"  Vantagens: {', '.join(metricas['vantagens'])}")

if __name__ == "__main__":
    print("🔬 FRAMEWORK DE ANÁLISE COMPARATIVA - CROSSLAYERGUARDIAN")
    print("========================================================")
    
    # Demonstrar configurações
    demonstrar_configuracao_baselines()
    
    # Mostrar métricas esperadas
    exibir_metricas_esperadas()
    
    # Executar exemplo (comentado para não executar automaticamente)
    print("\n" + "="*60)
    print("Para executar a análise comparativa completa:")
    print("python exemplo_analise_comparativa.py --executar")
    print("="*60)
    
    # Verificar se deve executar
    if len(sys.argv) > 1 and sys.argv[1] == "--executar":
        resultados = executar_exemplo_analise_comparativa()
        
        if resultados:
            print("\n✅ Análise comparativa concluída com sucesso!")
            print("📊 Verifique os arquivos gerados para análise detalhada.")
        else:
            print("\n❌ Erro durante execução da análise comparativa.")
    else:
        print("\n💡 RESUMO DO FRAMEWORK:")
        print("   ✓ Comparação com 3 sistemas IDS baseline")
        print("   ✓ Métricas padronizadas (Acurácia, F1, ROC AUC, Throughput)")
        print("   ✓ Análise estatística de significância")
        print("   ✓ Relatórios HTML detalhados")
        print("   ✓ Gráficos comparativos")
        print("   ✓ Análise por tipo de ataque")
        print("   ✓ Suporte para dados CICIDS2018 e sintéticos")