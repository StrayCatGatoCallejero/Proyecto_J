"""
Ejemplo de Logging JSON Avanzado
===============================

Este ejemplo demuestra el uso del sistema de logging JSON estructurado
para facilitar la ingestión en sistemas de monitoreo (ELK, Datadog, etc.).
"""

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, Any

# Importar el sistema de logging JSON
from processing.json_logging import (
    JSONLogger,
    JSONLoggingManager,
    LogLevel,
    LogCategory,
    create_json_logger,
    get_json_logger_manager
)
from processing.business_rules import validate_business_rules
from processing.json_utils import to_serializable  # Ajusta el import si la función está en otro módulo


def crear_datos_ejemplo() -> pd.DataFrame:
    """Crea datos de ejemplo para las pruebas"""
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


def ejemplo_logging_basico():
    """
    Ejemplo básico de uso del logger JSON.
    """
    print("\n" + "="*60)
    print("EJEMPLO 1: LOGGING JSON BÁSICO")
    print("="*60)
    
    # Crear logger JSON
    logger = create_json_logger()
    
    print(f"📝 Logger creado con session_id: {logger.session_id}")
    
    # Simular diferentes tipos de eventos
    print("\n📊 Registrando eventos de ejemplo...")
    
    # Evento de carga de datos
    start_time = time.time()
    time.sleep(0.1)  # Simular procesamiento
    execution_time = time.time() - start_time
    
    logger.log_data_load(
        function="load_csv_file",
        file_path="datos_ejemplo.csv",
        file_size=1024 * 50,  # 50KB
        rows=1000,
        columns=10,
        execution_time=execution_time,
        success=True
    )
    
    # Evento de validación
    start_time = time.time()
    time.sleep(0.05)  # Simular validación
    execution_time = time.time() - start_time
    
    logger.log_validation(
        function="validate_schema",
        validation_type="schema_validation",
        total_checks=15,
        passed_checks=14,
        failed_checks=1,
        execution_time=execution_time,
        details={"failed_column": "edad", "error_type": "out_of_range"}
    )
    
    # Evento de análisis
    start_time = time.time()
    time.sleep(0.2)  # Simular análisis
    execution_time = time.time() - start_time
    
    logger.log_analysis(
        function="correlation_analysis",
        analysis_type="pearson_correlation",
        input_size=1000,
        output_size=45,  # 10x10 correlation matrix - diagonal
        execution_time=execution_time,
        success=True,
        results={
            "max_correlation": 0.85,
            "min_correlation": -0.12,
            "significant_correlations": 8
        }
    )
    
    # Evento de error
    try:
        # Simular un error
        raise ValueError("Error de ejemplo para demostración")
    except Exception as e:
        logger.log_error(
            function="process_data",
            error=e,
            context="data_processing",
            execution_time=0.5,
            additional_data={"step": "data_cleaning", "attempt": 1}
        )
    
    # Obtener resumen de la sesión
    summary = logger.get_session_summary()
    print(f"\n📈 Resumen de la sesión:")
    print(f"   - Total de logs: {summary['total_logs']}")
    print(f"   - Distribución por nivel: {summary['level_distribution']}")
    print(f"   - Distribución por categoría: {summary['category_distribution']}")
    print(f"   - Tiempo total de ejecución: {summary['total_execution_time']:.3f}s")
    print(f"   - Errores: {summary['error_count']}")
    
    # Exportar logs
    export_path = logger.export_session_logs()
    print(f"\n💾 Logs exportados a: {export_path}")
    
    return logger


def ejemplo_logging_avanzado():
    """
    Ejemplo avanzado con múltiples sesiones y eventos detallados.
    """
    print("\n" + "="*60)
    print("EJEMPLO 2: LOGGING JSON AVANZADO")
    print("="*60)
    
    # Crear gestor de logging
    manager = get_json_logger_manager()
    
    # Crear múltiples sesiones
    session1 = manager.create_session("session_analisis_1")
    session2 = manager.create_session("session_analisis_2")
    
    print(f"📝 Sesiones creadas:")
    print(f"   - Session 1: {session1.session_id}")
    print(f"   - Session 2: {session2.session_id}")
    
    # Simular pipeline completo en sesión 1
    print(f"\n🔄 Simulando pipeline completo en sesión 1...")
    
    # Paso 1: Carga de datos
    start_time = time.time()
    time.sleep(0.1)
    execution_time = time.time() - start_time
    
    session1.log_data_load(
        function="load_demographic_data",
        file_path="encuesta_demografica.csv",
        file_size=1024 * 200,  # 200KB
        rows=5000,
        columns=15,
        execution_time=execution_time,
        success=True
    )
    
    # Paso 2: Validación de reglas de negocio
    start_time = time.time()
    time.sleep(0.15)
    execution_time = time.time() - start_time
    
    session1.log_business_rules(
        function="validate_demographics",
        rules_executed=8,
        rules_failed=2,
        rules_warnings=1,
        execution_time=execution_time,
        details={
            "failed_rules": ["age_range", "income_validation"],
            "warnings": ["gender_distribution"]
        }
    )
    
    # Paso 3: Análisis estadístico
    start_time = time.time()
    time.sleep(0.3)
    execution_time = time.time() - start_time
    
    session1.log_analysis(
        function="descriptive_statistics",
        analysis_type="descriptive_analysis",
        input_size=5000,
        output_size=120,  # 15 variables * 8 estadísticas
        execution_time=execution_time,
        success=True,
        results={
            "mean_age": 42.5,
            "gender_distribution": {"Masculino": 0.48, "Femenino": 0.52},
            "education_levels": {"Primaria": 0.15, "Secundaria": 0.35, "Universitaria": 0.50}
        }
    )
    
    # Simular análisis paralelo en sesión 2
    print(f"\n🔄 Simulando análisis paralelo en sesión 2...")
    
    start_time = time.time()
    time.sleep(0.2)
    execution_time = time.time() - start_time
    
    session2.log_analysis(
        function="correlation_analysis",
        analysis_type="pearson_correlation",
        input_size=5000,
        output_size=105,  # 15x15 - diagonal
        execution_time=execution_time,
        success=True,
        results={
            "strong_correlations": 5,
            "moderate_correlations": 12,
            "weak_correlations": 88
        }
    )
    
    # Obtener resumen de todas las sesiones
    all_summaries = manager.get_all_sessions_summary()
    print(f"\n📊 Resumen de todas las sesiones:")
    print(f"   - Total de sesiones: {all_summaries['total_sessions']}")
    print(f"   - Sesiones activas: {all_summaries['active_sessions']}")
    
    for session_id, summary in all_summaries['session_summaries'].items():
        print(f"\n   📈 Sesión {session_id}:")
        print(f"      - Logs: {summary['total_logs']}")
        print(f"      - Errores: {summary['error_count']}")
        print(f"      - Tiempo total: {summary['total_execution_time']:.3f}s")
    
    # Exportar todas las sesiones
    export_path = manager.export_all_sessions()
    print(f"\n💾 Todas las sesiones exportadas a: {export_path}")
    
    return manager


def ejemplo_integracion_con_business_rules():
    """
    Ejemplo de integración con el sistema de reglas de negocio.
    """
    print("\n" + "="*60)
    print("EJEMPLO 3: INTEGRACIÓN CON BUSINESS RULES")
    print("="*60)
    
    # Crear logger JSON
    logger = create_json_logger()
    
    # Crear datos de ejemplo
    df = crear_datos_ejemplo()
    print(f"📊 Datos creados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Simular validación de reglas de negocio con logging detallado
    print(f"\n🔍 Ejecutando validación de reglas de negocio...")
    
    start_time = time.time()
    
    try:
        # Ejecutar validaciones
        metadata = {'dataset_type': 'social_sciences'}
        validation_results = validate_business_rules(df, metadata)
        
        execution_time = time.time() - start_time
        
        # Contar resultados
        total_rules = len(validation_results)
        failed_rules = sum(1 for r in validation_results if not r.is_valid)
        warning_rules = len([r for r in validation_results if r.details.get('alertas_generadas', 0) > 0])
        
        # Log del evento de validación
        logger.log_business_rules(
            function="validate_business_rules",
            rules_executed=total_rules,
            rules_failed=failed_rules,
            rules_warnings=warning_rules,
            execution_time=execution_time,
            details={
                "validation_results": [r.rule_name for r in validation_results],
                "failed_rules": [r.rule_name for r in validation_results if not r.is_valid],
                "warning_rules": [r.rule_name for r in validation_results if r.details.get('alertas_generadas', 0) > 0]
            }
        )
        
        # Log detallado de cada regla
        for result in validation_results:
            if not result.is_valid:
                logger.log_event(
                    level=LogLevel.ERROR,
                    message=f"Regla de negocio fallida: {result.rule_name}",
                    module="business_rules",
                    function=result.rule_name,
                    step="business_rules_validation",
                    category=LogCategory.BUSINESS_RULES,
                    parameters={},
                    before_metrics={"rule_name": result.rule_name},
                    after_metrics={"is_valid": False, "message": result.message},
                    execution_time=0.0,
                    error_details=result.message,
                    metadata=result.details,
                    tags=["business_rule", "validation_failed"]
                )
            elif result.details.get('alertas_generadas', 0) > 0:
                logger.log_event(
                    level=LogLevel.WARNING,
                    message=f"Advertencia en regla de negocio: {result.rule_name}",
                    module="business_rules",
                    function=result.rule_name,
                    step="business_rules_validation",
                    category=LogCategory.BUSINESS_RULES,
                    parameters={},
                    before_metrics={"rule_name": result.rule_name},
                    after_metrics={"is_valid": True, "warnings": result.details.get('alertas_generadas', 0)},
                    execution_time=0.0,
                    metadata=result.details,
                    tags=["business_rule", "warning"]
                )
        
        print(f"✅ Validación completada:")
        print(f"   - Reglas ejecutadas: {total_rules}")
        print(f"   - Reglas fallidas: {failed_rules}")
        print(f"   - Advertencias: {warning_rules}")
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        logger.log_error(
            function="validate_business_rules",
            error=e,
            context="business_rules_validation",
            execution_time=execution_time,
            additional_data={"dataset_shape": df.shape}
        )
        
        print(f"❌ Error en validación: {e}")
    
    # Obtener logs de errores
    error_logs = logger.get_error_logs()
    print(f"\n📋 Logs de errores generados: {len(error_logs)}")
    
    for log in error_logs:
        print(f"   - {log.timestamp}: {log.message}")
    
    return logger


def ejemplo_formato_json():
    """
    Ejemplo que muestra el formato JSON generado.
    """
    print("\n" + "="*60)
    print("EJEMPLO 4: FORMATO JSON GENERADO")
    print("="*60)
    
    # Crear logger JSON
    logger = create_json_logger()
    
    # Crear un evento de ejemplo
    log_entry = logger.log_event(
        level=LogLevel.INFO,
        message="Ejemplo de log estructurado",
        module="ejemplo",
        function="ejemplo_formato_json",
        step="demonstration",
        category=LogCategory.SYSTEM,
        parameters={"param1": "valor1", "param2": 42},
        before_metrics={"rows": 100, "columns": 10},
        after_metrics={"processed_rows": 95, "errors": 5},
        execution_time=1.5,
        user_id="usuario_ejemplo",
        request_id="req_12345",
        correlation_id="corr_67890",
        tags=["ejemplo", "demo"],
        metadata={"version": "1.0", "environment": "development"}
    )
    
    # Mostrar el formato JSON
    print("📄 Formato JSON generado:")
    print(json.dumps(log_entry.to_dict(), indent=2, ensure_ascii=False, default=to_serializable))
    
    # Mostrar campos específicos
    print(f"\n🔍 Campos específicos:")
    print(f"   - Timestamp: {log_entry.timestamp}")
    print(f"   - Session ID: {log_entry.session_id}")
    print(f"   - Level: {log_entry.level}")
    print(f"   - Category: {log_entry.category}")
    print(f"   - Execution Time: {log_entry.execution_time}s")
    print(f"   - Tags: {log_entry.tags}")
    
    return logger


def ejemplo_monitoreo_sistemas():
    """
    Ejemplo de configuración para sistemas de monitoreo.
    """
    print("\n" + "="*60)
    print("EJEMPLO 5: CONFIGURACIÓN PARA MONITOREO")
    print("="*60)
    
    # Configuración para diferentes sistemas de monitoreo
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
    
    print("🔧 Configuraciones para sistemas de monitoreo:")
    
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
    
    print(f"\n💡 Para usar con sistemas de monitoreo:")
    print(f"   1. Configurar el archivo config.yml con la configuración deseada")
    print(f"   2. Los logs JSON se generarán automáticamente")
    print(f"   3. Usar herramientas como Filebeat para enviar logs a ELK")
    print(f"   4. Configurar Datadog Agent para recolección automática")
    print(f"   5. Usar Prometheus para métricas de rendimiento")


def main():
    """
    Función principal que ejecuta todos los ejemplos.
    """
    print("🚀 EJEMPLO DE LOGGING JSON AVANZADO")
    print("="*80)
    
    # Ejecutar ejemplos
    logger1 = ejemplo_logging_basico()
    manager = ejemplo_logging_avanzado()
    logger2 = ejemplo_integracion_con_business_rules()
    logger3 = ejemplo_formato_json()
    ejemplo_monitoreo_sistemas()
    
    print("\n" + "="*80)
    print("✅ TODOS LOS EJEMPLOS COMPLETADOS")
    print("="*80)
    
    print("\n💡 PRÓXIMOS PASOS:")
    print("1. Integrar JSONLogger en el pipeline existente")
    print("2. Configurar sistemas de monitoreo (ELK, Datadog)")
    print("3. Implementar métricas de rendimiento")
    print("4. Configurar alertas basadas en logs")
    print("5. Crear dashboards de monitoreo")
    
    print("\n📚 ARCHIVOS IMPORTANTES:")
    print("- processing/json_logging.py - Sistema principal")
    print("- config/config.yml - Configuración de logging")
    print("- logs/pipeline.json - Logs JSON generados")
    
    print("\n🔧 INTEGRACIÓN CON PIPELINE:")
    print("Para integrar con el pipeline existente, reemplazar:")
    print("from processing.logging import get_logger")
    print("por:")
    print("from processing.json_logging import create_json_logger")


if __name__ == "__main__":
    main() 