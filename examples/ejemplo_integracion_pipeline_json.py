"""
Ejemplo de Integración Mejorada: Logging JSON en Pipeline Principal
==================================================================

Este ejemplo demuestra la integración completa del sistema de logging JSON
avanzado en el PipelineOrchestrator, proporcionando trazabilidad end-to-end
con formato estructurado para sistemas de monitoreo.
"""

import pandas as pd
import time
import json
from datetime import datetime
from typing import Dict, Any

# Importar el PipelineOrchestrator mejorado
from orchestrator.pipeline_orchestrator import PipelineOrchestrator


def crear_datos_ejemplo() -> pd.DataFrame:
    """Crea datos de ejemplo para las pruebas del pipeline"""
    return pd.DataFrame({
        'edad': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'genero': ['Masculino', 'Femenino', 'Masculino', 'Femenino', 'Masculino', 
                  'Femenino', 'Masculino', 'Femenino', 'Masculino', 'Femenino'],
        'nivel_educacion': ['Primaria', 'Secundaria', 'Universitaria', 'Postgrado', 'Doctorado',
                           'Primaria', 'Secundaria', 'Universitaria', 'Postgrado', 'Doctorado'],
        'ingresos': [500000, 800000, 1200000, 1500000, 2000000, 
                    600000, 900000, 1300000, 1600000, 2200000],
        'region': ['Metropolitana', 'Valparaíso', 'Antofagasta', 'Tarapacá', 'Atacama',
                  'Metropolitana', 'Valparaíso', 'Antofagasta', 'Tarapacá', 'Atacama'],
        'pregunta_1_likert': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'pregunta_2_likert': [2, 3, 4, 5, 1, 2, 3, 4, 5, 1]
    })


def ejemplo_pipeline_completo_json():
    """
    Ejemplo de ejecución completa del pipeline con logging JSON integrado.
    """
    print("🚀 EJEMPLO: PIPELINE COMPLETO CON LOGGING JSON INTEGRADO")
    print("="*80)
    
    # Crear datos de ejemplo y guardarlos
    df = crear_datos_ejemplo()
    file_path = "datos_ejemplo_pipeline.csv"
    df.to_csv(file_path, index=False)
    
    print(f"📊 Datos de ejemplo creados: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"💾 Archivo guardado: {file_path}")
    
    # Crear PipelineOrchestrator con logging JSON
    print(f"\n🔧 Inicializando PipelineOrchestrator...")
    orchestrator = PipelineOrchestrator()
    
    print(f"📝 Session ID generado: {orchestrator.session_id}")
    print(f"🔍 JSON Logger configurado: {orchestrator.json_logger.session_id}")
    
    # Ejecutar pipeline completo
    print(f"\n🔄 Ejecutando pipeline completo...")
    
    start_time = time.time()
    
    success = orchestrator.run_full_pipeline(
        file_path=file_path,
        schema=None,  # Usar detección automática
        filters=None  # Sin filtros
    )
    
    execution_time = time.time() - start_time
    
    print(f"\n✅ Pipeline {'completado exitosamente' if success else 'falló'}")
    print(f"⏱️ Tiempo total de ejecución: {execution_time:.3f} segundos")
    
    # Obtener resumen completo
    summary = orchestrator.get_pipeline_summary()
    
    print(f"\n📈 RESUMEN DEL PIPELINE:")
    print(f"   - Session ID: {summary['session_id']}")
    print(f"   - Datos cargados: {summary['data_loaded']}")
    print(f"   - Forma de datos: {summary['data_shape']}")
    print(f"   - Reportes generados: {summary['reports_generated']}")
    
    # Resumen de logs JSON
    json_summary = summary['json_logs_summary']
    print(f"\n📋 RESUMEN DE LOGS JSON:")
    print(f"   - Total de logs: {json_summary['total_logs']}")
    print(f"   - Distribución por nivel: {json_summary['level_distribution']}")
    print(f"   - Distribución por categoría: {json_summary['category_distribution']}")
    print(f"   - Errores: {json_summary['error_count']}")
    print(f"   - Tiempo total de logs: {json_summary['total_execution_time']:.3f}s")
    
    # Exportar resultados
    export_path = f"resultados_pipeline_{orchestrator.session_id}.json"
    orchestrator.export_results(export_path)
    
    print(f"\n💾 Resultados exportados a: {export_path}")
    
    return orchestrator, success


def ejemplo_pipeline_paso_a_paso():
    """
    Ejemplo de ejecución paso a paso del pipeline con logging detallado.
    """
    print("\n" + "="*80)
    print("EJEMPLO: PIPELINE PASO A PASO CON LOGGING DETALLADO")
    print("="*80)
    
    # Crear datos de ejemplo
    df = crear_datos_ejemplo()
    file_path = "datos_ejemplo_paso_a_paso.csv"
    df.to_csv(file_path, index=False)
    
    # Crear PipelineOrchestrator
    orchestrator = PipelineOrchestrator()
    print(f"📝 Pipeline inicializado con session_id: {orchestrator.session_id}")
    
    # Ejecutar pasos individualmente
    steps = [
        ("Carga de Datos", lambda: orchestrator.load_data(file_path)),
        ("Validación de Esquema", lambda: orchestrator.validate_schema()),
        ("Clasificación Semántica", lambda: orchestrator.classify_semantics()),
        ("Validación de Reglas de Negocio", lambda: orchestrator.validate_business_rules_step()),
        ("Filtrado de Datos", lambda: orchestrator.apply_filters()),
        ("Ingeniería de Características", lambda: orchestrator.run_feature_engineering()),
        ("Análisis Estadístico", lambda: orchestrator.run_statistical_analysis()),
        ("Generación de Visualizaciones", lambda: orchestrator.generate_visualizations()),
        ("Generación de Reportes", lambda: orchestrator.generate_reports()),
    ]
    
    print(f"\n🔄 Ejecutando pasos individualmente:")
    
    for i, (step_name, step_function) in enumerate(steps, 1):
        print(f"\n   {i}. {step_name}...")
        
        try:
            start_time = time.time()
            success = step_function()
            execution_time = time.time() - start_time
            
            status = "✅" if success else "❌"
            print(f"      {status} Completado en {execution_time:.3f}s")
            
            if not success:
                print(f"      ⚠️ Paso falló, deteniendo pipeline")
                break
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
            break
    
    # Obtener logs específicos por categoría
    print(f"\n📊 ANÁLISIS DE LOGS POR CATEGORÍA:")
    
    categories = ["data_load", "validation", "processing", "analysis", "visualization", "export"]
    
    for category in categories:
        logs = orchestrator.json_logger.get_logs_by_category(category)
        if logs:
            print(f"   - {category}: {len(logs)} logs")
            
            # Mostrar ejemplo de log
            example_log = logs[0]
            print(f"     Ejemplo: {example_log.message}")
    
    # Obtener logs de errores
    error_logs = orchestrator.json_logger.get_error_logs()
    if error_logs:
        print(f"\n❌ LOGS DE ERRORES ({len(error_logs)}):")
        for log in error_logs:
            print(f"   - {log.timestamp}: {log.message}")
            if log.error_details:
                print(f"     Detalles: {log.error_details}")
    
    return orchestrator


def ejemplo_multiple_pipelines():
    """
    Ejemplo de múltiples pipelines ejecutándose con logging independiente.
    """
    print("\n" + "="*80)
    print("EJEMPLO: MÚLTIPLES PIPELINES CON LOGGING INDEPENDIENTE")
    print("="*80)
    
    from processing.json_logging import get_json_logger_manager
    
    # Crear gestor de logging
    manager = get_json_logger_manager()
    
    # Crear datos de ejemplo
    df = crear_datos_ejemplo()
    
    # Crear múltiples pipelines
    pipelines = []
    for i in range(3):
        # Crear archivo único para cada pipeline
        file_path = f"datos_pipeline_{i+1}.csv"
        df.to_csv(file_path, index=False)
        
        # Crear pipeline con session_id único
        session_id = f"pipeline_{i+1}_{datetime.now().strftime('%H%M%S')}"
        orchestrator = PipelineOrchestrator()
        orchestrator.session_id = session_id
        orchestrator.json_logger = manager.create_session(session_id)
        
        pipelines.append((orchestrator, file_path, f"Pipeline {i+1}"))
    
    print(f"📝 Pipelines creados: {len(pipelines)}")
    
    # Ejecutar pipelines en paralelo (simulado)
    results = []
    for orchestrator, file_path, pipeline_name in pipelines:
        print(f"\n🔄 Ejecutando {pipeline_name}...")
        
        try:
            # Ejecutar solo algunos pasos para demostración
            success1 = orchestrator.load_data(file_path)
            success2 = orchestrator.validate_schema()
            success3 = orchestrator.validate_business_rules_step()
            
            success = all([success1, success2, success3])
            results.append((pipeline_name, success))
            
            print(f"   ✅ {pipeline_name}: {'Exitoso' if success else 'Falló'}")
            
        except Exception as e:
            print(f"   ❌ {pipeline_name}: Error - {e}")
            results.append((pipeline_name, False))
    
    # Obtener resumen de todas las sesiones
    all_summaries = manager.get_all_sessions_summary()
    
    print(f"\n📊 RESUMEN DE TODAS LAS SESIONES:")
    print(f"   - Total de sesiones: {all_summaries['total_sessions']}")
    print(f"   - Sesiones activas: {len(all_summaries['active_sessions'])}")
    
    for session_id, summary in all_summaries['session_summaries'].items():
        print(f"\n   📈 Sesión {session_id}:")
        print(f"      - Logs: {summary['total_logs']}")
        print(f"      - Errores: {summary['error_count']}")
        print(f"      - Tiempo total: {summary['total_execution_time']:.3f}s")
    
    # Exportar todas las sesiones
    export_path = manager.export_all_sessions()
    print(f"\n💾 Todas las sesiones exportadas a: {export_path}")
    
    return manager, results


def ejemplo_formato_json_generado():
    """
    Ejemplo que muestra el formato JSON generado por el pipeline.
    """
    print("\n" + "="*80)
    print("EJEMPLO: FORMATO JSON GENERADO POR EL PIPELINE")
    print("="*80)
    
    # Crear pipeline
    orchestrator = PipelineOrchestrator()
    
    # Crear datos de ejemplo
    df = crear_datos_ejemplo()
    file_path = "datos_ejemplo_formato.csv"
    df.to_csv(file_path, index=False)
    
    # Ejecutar solo el primer paso para obtener logs
    orchestrator.load_data(file_path)
    
    # Obtener logs de la sesión
    logs = orchestrator.json_logger.get_session_logs()
    
    print(f"📄 FORMATO JSON GENERADO:")
    print(f"   - Total de logs: {len(logs)}")
    
    if logs:
        # Mostrar el primer log como ejemplo
        example_log = logs[0]
        print(f"\n🔍 EJEMPLO DE LOG JSON:")
        print(json.dumps(example_log.to_dict(), indent=2, ensure_ascii=False, default=to_serializable))
        
        # Mostrar campos específicos
        print(f"\n📋 CAMPOS CLAVE:")
        print(f"   - Timestamp: {example_log.timestamp}")
        print(f"   - Session ID: {example_log.session_id}")
        print(f"   - Level: {example_log.level}")
        print(f"   - Category: {example_log.category}")
        print(f"   - Module: {example_log.module}")
        print(f"   - Function: {example_log.function}")
        print(f"   - Step: {example_log.step}")
        print(f"   - Execution Time: {example_log.execution_time}s")
        print(f"   - Tags: {example_log.tags}")
        
        # Mostrar métricas
        print(f"\n📊 MÉTRICAS:")
        print(f"   - Before: {example_log.before_metrics}")
        print(f"   - After: {example_log.after_metrics}")
        
        # Mostrar información del sistema
        if example_log.system_info:
            print(f"\n💻 INFORMACIÓN DEL SISTEMA:")
            print(f"   - Platform: {example_log.system_info.platform}")
            print(f"   - Python: {example_log.system_info.python_version}")
            print(f"   - CPU Count: {example_log.system_info.cpu_count}")
            print(f"   - Memory: {example_log.system_info.memory_total / (1024**3):.1f} GB")
            print(f"   - Process ID: {example_log.system_info.process_id}")
    
    return orchestrator


def ejemplo_integracion_con_sistemas_monitoreo():
    """
    Ejemplo de configuración para integración con sistemas de monitoreo.
    """
    print("\n" + "="*80)
    print("EJEMPLO: INTEGRACIÓN CON SISTEMAS DE MONITOREO")
    print("="*80)
    
    # Configuraciones para diferentes sistemas
    configs = {
        "elk_stack": {
            "json_logging": {
                "enabled": True,
                "json_file": "logs/elk_pipeline.json",
                "console_json": True,
                "monitoring": {
                    "elk": {
                        "enabled": True,
                        "host": "localhost",
                        "port": 9200,
                        "index_prefix": "pipeline-logs"
                    }
                }
            }
        },
        "datadog": {
            "json_logging": {
                "enabled": True,
                "json_file": "logs/datadog_pipeline.json",
                "console_json": True,
                "monitoring": {
                    "datadog": {
                        "enabled": True,
                        "api_key": "your_api_key_here",
                        "host": "localhost",
                        "port": 8126
                    }
                }
            }
        },
        "prometheus": {
            "json_logging": {
                "enabled": True,
                "json_file": "logs/prometheus_pipeline.json",
                "console_json": True,
                "monitoring": {
                    "prometheus": {
                        "enabled": True,
                        "port": 9090,
                        "metrics_path": "/metrics"
                    }
                }
            }
        }
    }
    
    print("🔧 CONFIGURACIONES PARA SISTEMAS DE MONITOREO:")
    
    for system, config in configs.items():
        print(f"\n📊 {system.upper()}:")
        print(f"   - Archivo JSON: {config['json_logging']['json_file']}")
        print(f"   - Console JSON: {config['json_logging']['console_json']}")
        
        monitoring_config = config['json_logging']['monitoring']
        for monitor, monitor_config in monitoring_config.items():
            if monitor_config.get('enabled', False):
                print(f"   - {monitor}: Habilitado")
                if 'host' in monitor_config:
                    print(f"     Host: {monitor_config['host']}:{monitor_config.get('port', 'default')}")
    
    print(f"\n💡 VENTAJAS DE LA INTEGRACIÓN:")
    print(f"   ✅ Formato JSON estandarizado para fácil ingestión")
    print(f"   ✅ Session ID único para trazabilidad end-to-end")
    print(f"   ✅ Métricas detalladas de rendimiento")
    print(f"   ✅ Compatibilidad con ELK, Datadog, Prometheus")
    print(f"   ✅ Análisis de errores y debugging")
    print(f"   ✅ Auditoría completa del pipeline")
    
    print(f"\n🔧 PRÓXIMOS PASOS PARA INTEGRACIÓN:")
    print(f"   1. Configurar config.yml con la configuración deseada")
    print(f"   2. Instalar y configurar Filebeat para ELK")
    print(f"   3. Configurar Datadog Agent para recolección")
    print(f"   4. Configurar Prometheus para métricas")
    print(f"   5. Crear dashboards y alertas")


def main():
    """
    Función principal que ejecuta todos los ejemplos de integración.
    """
    print("🚀 EJEMPLO DE INTEGRACIÓN MEJORADA: LOGGING JSON EN PIPELINE")
    print("="*80)
    
    try:
        # Ejemplo 1: Pipeline completo
        orchestrator1, success1 = ejemplo_pipeline_completo_json()
        
        # Ejemplo 2: Pipeline paso a paso
        orchestrator2 = ejemplo_pipeline_paso_a_paso()
        
        # Ejemplo 3: Múltiples pipelines
        manager, results = ejemplo_multiple_pipelines()
        
        # Ejemplo 4: Formato JSON
        orchestrator3 = ejemplo_formato_json_generado()
        
        # Ejemplo 5: Integración con sistemas de monitoreo
        ejemplo_integracion_con_sistemas_monitoreo()
        
        print("\n" + "="*80)
        print("✅ TODOS LOS EJEMPLOS COMPLETADOS EXITOSAMENTE")
        print("="*80)
        
        print(f"\n🎯 BENEFICIOS LOGRADOS:")
        print(f"   ✅ Trazabilidad completa con session_id único")
        print(f"   ✅ Logs estructurados en formato JSON")
        print(f"   ✅ Métricas detalladas de cada paso")
        print(f"   ✅ Integración con sistemas de monitoreo")
        print(f"   ✅ Análisis de errores y debugging")
        print(f"   ✅ Auditoría completa del pipeline")
        
        print(f"\n📚 ARCHIVOS IMPORTANTES:")
        print(f"   - orchestrator/pipeline_orchestrator.py - Pipeline con JSON logging")
        print(f"   - processing/json_logging.py - Sistema JSON avanzado")
        print(f"   - config/config.yml - Configuración de logging")
        print(f"   - logs/pipeline.json - Logs JSON generados")
        
        print(f"\n🔧 INTEGRACIÓN COMPLETADA:")
        print(f"   El PipelineOrchestrator ahora incluye logging JSON avanzado")
        print(f"   Cada paso del pipeline genera logs estructurados")
        print(f"   Session ID único para trazabilidad end-to-end")
        print(f"   Compatible con sistemas de monitoreo (ELK, Datadog, Prometheus)")
        
    except Exception as e:
        print(f"\n❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 