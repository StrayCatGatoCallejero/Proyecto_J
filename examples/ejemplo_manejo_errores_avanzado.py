"""
Ejemplo Avanzado de Manejo de Errores de Negocio
===============================================

Este ejemplo demuestra:
1. Cómo integrar el sistema de manejo de errores con el pipeline existente
2. Visualización avanzada de errores en Streamlit
3. Extracción y análisis de errores desde logs
4. Generación de reportes detallados
5. Recomendaciones automáticas basadas en errores
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List

# Importar módulos del sistema
from processing.business_error_handler import (
    BusinessErrorHandler, 
    get_business_errors, 
    display_business_errors_in_streamlit,
    BusinessError
)
from processing.business_rules import (
    validate_business_rules, 
    ValidationResult,
    BusinessRuleError
)
from processing.logging import UnifiedLogger, get_logger
from orchestrator.pipeline_orchestrator import PipelineOrchestrator, SessionData


def crear_datos_con_errores() -> Dict[str, pd.DataFrame]:
    """
    Crea datasets con diferentes tipos de errores para demostración.
    
    Returns:
        Diccionario con datasets problemáticos
    """
    
    # Dataset 1: Errores demográficos
    datos_demograficos = pd.DataFrame({
        'edad': [25, 30, -5, 150, 45, 18, 200, 35],  # Edades inválidas
        'genero': ['Masculino', 'Femenino', 'X', 'M', 'F', 'Otro', 'Hombre', 'Mujer'],  # Género inválido
        'nivel_educacion': ['Primaria', 'Secundaria', 'Universitaria', 'PhD', 'Básica', 'Media', 'Superior', 'Doctorado'],
        'ingresos': [500000, 800000, -10000, 5000000, 1200000, 300000, 10000000, 900000]  # Ingresos negativos y outliers
    })
    
    # Dataset 2: Errores geográficos
    datos_geograficos = pd.DataFrame({
        'region': ['Metropolitana', 'Valparaíso', 'Región Inexistente', 'Antofagasta', 'Tarapacá'],
        'comuna': ['Santiago', 'Valparaíso', 'Comuna Inexistente', 'Antofagasta', 'Iquique'],
        'edad': [30, 25, 40, 35, 28],
        'genero': ['Masculino', 'Femenino', 'Masculino', 'Femenino', 'Masculino']
    })
    
    # Dataset 3: Errores en escalas Likert
    datos_likert = pd.DataFrame({
        'pregunta_1_likert': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Valores fuera de escala
        'pregunta_2_likert': [1, 2, 3, 4, 5, 'Muy de acuerdo', 'En desacuerdo', 3, 4, 5],  # Mezcla de tipos
        'pregunta_3_likert': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],  # Datos válidos
        'edad': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'genero': ['Masculino', 'Femenino', 'Masculino', 'Femenino', 'Masculino', 
                  'Femenino', 'Masculino', 'Femenino', 'Masculino', 'Femenino']
    })
    
    # Dataset 4: Errores de consistencia cruzada
    datos_consistencia = pd.DataFrame({
        'edad': [15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        'nivel_educacion': ['Universitaria', 'Postgrado', 'Doctorado', 'Secundaria', 'Primaria',
                           'Universitaria', 'Postgrado', 'Doctorado', 'Secundaria', 'Primaria'],
        'empleo': ['Estudiante', 'Empleado', 'Empleado', 'Desempleado', 'Jubilado',
                  'Empleado', 'Empleado', 'Empleado', 'Desempleado', 'Jubilado'],
        'ingresos': [0, 500000, 800000, 0, 0, 1200000, 1500000, 1800000, 0, 0]
    })
    
    return {
        'demograficos': datos_demograficos,
        'geograficos': datos_geograficos,
        'likert': datos_likert,
        'consistencia': datos_consistencia
    }


def simular_pipeline_con_errores(dataset: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Simula la ejecución del pipeline con detección de errores.
    
    Args:
        dataset: DataFrame a procesar
        dataset_name: Nombre del dataset para logging
        
    Returns:
        Diccionario con resultados del pipeline
    """
    # Inicializar logger
    logger = get_logger()
    
    # Crear metadata básica
    metadata = {
        'dataset_type': 'social_sciences',
        'dataset_name': dataset_name,
        'columns': list(dataset.columns)
    }
    
    # Simular paso 1: Carga de datos
    logger.log_action(
        function="load_data",
        step="data_load",
        parameters={"dataset_name": dataset_name},
        before_metrics={"n_rows": 0},
        after_metrics={"n_rows": len(dataset), "n_columns": len(dataset.columns)},
        status="success",
        message=f"Datos cargados: {len(dataset)} filas, {len(dataset.columns)} columnas"
    )
    
    # Simular paso 2: Validación de reglas de negocio
    try:
        validation_results = validate_business_rules(dataset, metadata)
        
        # Contar errores y advertencias
        total_errors = sum(1 for r in validation_results if not r.is_valid)
        total_warnings = len([r for r in validation_results if r.details.get('alertas_generadas', 0) > 0])
        
        # Logging de resultados
        logger.log_action(
            function="validate_business_rules",
            step="business_rules",
            parameters={"dataset_type": metadata['dataset_type']},
            before_metrics={"n_rows": len(dataset)},
            after_metrics={
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "validation_results": [r.rule_name for r in validation_results]
            },
            status="error" if total_errors > 0 else "warning" if total_warnings > 0 else "success",
            message=f"Validación completada: {total_errors} errores, {total_warnings} advertencias"
        )
        
        return {
            'success': True,
            'validation_results': validation_results,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'logs': logger.get_session_history()
        }
        
    except Exception as e:
        logger.log_action(
            function="validate_business_rules",
            step="business_rules",
            parameters={"dataset_type": metadata['dataset_type']},
            before_metrics={"n_rows": len(dataset)},
            after_metrics={"error": str(e)},
            status="error",
            message=f"Error en validación: {str(e)}",
            error_details=str(e)
        )
        
        return {
            'success': False,
            'error': str(e),
            'logs': logger.get_session_history()
        }


def ejemplo_extraccion_errores_basica():
    """
    Ejemplo básico de extracción de errores desde logs.
    """
    print("\n" + "="*60)
    print("EJEMPLO 1: EXTRACCIÓN BÁSICA DE ERRORES")
    print("="*60)
    
    # Crear datos con errores
    datasets = crear_datos_con_errores()
    dataset = datasets['demograficos']
    
    # Simular pipeline
    result = simular_pipeline_con_errores(dataset, 'datos_demograficos')
    
    if result['success']:
        # Extraer errores usando la función básica
        errors = get_business_errors(result['logs'])
        
        print(f"📊 Errores extraídos: {len(errors)}")
        for i, error in enumerate(errors, 1):
            print(f"   Error {i}:")
            print(f"      Regla: {error['regla']}")
            print(f"      Detalle: {error['detalle']}")
            print(f"      Timestamp: {error['timestamp']}")
            print(f"      Severidad: {error['severidad']}")
            print()


def ejemplo_manejo_avanzado_errores():
    """
    Ejemplo avanzado usando BusinessErrorHandler.
    """
    print("\n" + "="*60)
    print("EJEMPLO 2: MANEJO AVANZADO DE ERRORES")
    print("="*60)
    
    # Crear handler
    handler = BusinessErrorHandler()
    
    # Procesar múltiples datasets
    datasets = crear_datos_con_errores()
    
    for dataset_name, dataset in datasets.items():
        print(f"\n📊 Procesando dataset: {dataset_name}")
        
        # Simular pipeline
        result = simular_pipeline_con_errores(dataset, dataset_name)
        
        if result['success']:
            # Extraer errores desde logs
            handler.extract_business_errors_from_logs(result['logs'])
            
            # Extraer errores desde resultados de validación
            handler.extract_errors_from_validation_results(result['validation_results'])
            
            # Obtener resumen
            summary = handler.get_error_summary()
            
            print(f"   Total errores: {summary['total_errors']}")
            print(f"   Reglas afectadas: {len(summary['errors_by_rule'])}")
            print(f"   Errores críticos: {summary['errors_by_severity'].get('error', 0)}")
            
            # Mostrar recomendaciones
            recommendations = handler.get_recommendations()
            if recommendations:
                print("   Recomendaciones:")
                for rec in recommendations:
                    print(f"      - {rec}")
        else:
            print(f"   ❌ Error en procesamiento: {result['error']}")


def ejemplo_visualizacion_streamlit():
    """
    Ejemplo de visualización en Streamlit (simulado).
    """
    print("\n" + "="*60)
    print("EJEMPLO 3: VISUALIZACIÓN EN STREAMLIT")
    print("="*60)
    
    # Crear datos con errores
    datasets = crear_datos_con_errores()
    dataset = datasets['likert']  # Usar dataset con errores Likert
    
    # Simular pipeline
    result = simular_pipeline_con_errores(dataset, 'datos_likert')
    
    if result['success']:
        print("📊 Simulando visualización en Streamlit...")
        
        # Extraer errores
        errors = get_business_errors(result['logs'])
        
        print(f"   Métricas mostradas:")
        print(f"      - Total de errores: {len(errors)}")
        print(f"      - Reglas afectadas: {len(set(e['regla'] for e in errors))}")
        print(f"      - Columnas afectadas: {len(set(e['columna'] for e in errors if e['columna']))}")
        
        # Simular gráficos
        print(f"   Gráficos generados:")
        print(f"      - Gráfico de barras: Errores por regla")
        print(f"      - Gráfico de pastel: Errores por severidad")
        print(f"      - Gráfico de dispersión: Timeline de errores")
        
        # Simular tabla
        print(f"   Tabla de errores:")
        for i, error in enumerate(errors[:3], 1):  # Mostrar solo los primeros 3
            print(f"      {i}. {error['regla']} - {error['detalle']}")
        
        # Simular botones de acción
        print(f"   Botones de acción:")
        print(f"      - 📋 Copiar Reporte")
        print(f"      - 📊 Exportar CSV")
        print(f"      - 📄 Generar Reporte")
        
        # Simular recomendaciones
        handler = BusinessErrorHandler()
        handler.extract_business_errors_from_logs(result['logs'])
        recommendations = handler.get_recommendations()
        
        if recommendations:
            print(f"   Recomendaciones mostradas:")
            for rec in recommendations:
                print(f"      - {rec}")


def ejemplo_integracion_pipeline_completo():
    """
    Ejemplo de integración completa con el pipeline existente.
    """
    print("\n" + "="*60)
    print("EJEMPLO 4: INTEGRACIÓN CON PIPELINE COMPLETO")
    print("="*60)
    
    # Crear datos
    datasets = crear_datos_con_errores()
    dataset = datasets['consistencia']
    
    # Crear SessionData
    session_data = SessionData(
        df=dataset,
        metadata={
            'dataset_type': 'social_sciences',
            'dataset_name': 'datos_consistencia',
            'columns': list(dataset.columns)
        }
    )
    
    # Simular pipeline con validación de reglas de negocio
    try:
        print("🔍 Ejecutando validación de reglas de negocio...")
        
        # Validar reglas de negocio
        validation_results = validate_business_rules(dataset, session_data.metadata)
        
        # Contar resultados
        failed_validations = [r for r in validation_results if not r.is_valid]
        warnings = [r for r in validation_results if r.details.get('alertas_generadas', 0) > 0]
        
        print(f"   Resultados de validación:")
        print(f"      - Total reglas ejecutadas: {len(validation_results)}")
        print(f"      - Reglas fallidas: {len(failed_validations)}")
        print(f"      - Advertencias: {len(warnings)}")
        
        # Mostrar detalles de errores
        for result in validation_results:
            if not result.is_valid:
                print(f"      ❌ {result.rule_name}: {result.message}")
            elif result.details.get('alertas_generadas', 0) > 0:
                print(f"      ⚠️ {result.rule_name}: {result.message}")
            else:
                print(f"      ✅ {result.rule_name}: {result.message}")
        
        # Simular manejo de errores en Streamlit
        if failed_validations:
            print(f"\n🚨 Se detectaron errores críticos. Simulando manejo en Streamlit...")
            
            # Crear handler y extraer errores
            handler = BusinessErrorHandler()
            handler.extract_errors_from_validation_results(validation_results)
            
            # Obtener resumen
            summary = handler.get_error_summary()
            
            print(f"   Resumen de errores:")
            print(f"      - Total errores: {summary['total_errors']}")
            print(f"      - Errores críticos: {summary['errors_by_severity'].get('error', 0)}")
            print(f"      - Advertencias: {summary['errors_by_severity'].get('warning', 0)}")
            
            # Mostrar recomendaciones
            recommendations = handler.get_recommendations()
            if recommendations:
                print(f"   Recomendaciones generadas:")
                for rec in recommendations:
                    print(f"      - {rec}")
            
            print(f"\n💡 En Streamlit, se mostraría:")
            print(f"   - Mensaje de error principal")
            print(f"   - Métricas de errores")
            print(f"   - Gráficos de análisis")
            print(f"   - Tabla detallada de errores")
            print(f"   - Botones de acción (copiar, exportar, reporte)")
            print(f"   - Recomendaciones específicas")
        
        else:
            print(f"✅ Todas las validaciones pasaron exitosamente")
            
    except Exception as e:
        print(f"❌ Error en el pipeline: {e}")


def ejemplo_exportacion_reportes():
    """
    Ejemplo de exportación y generación de reportes.
    """
    print("\n" + "="*60)
    print("EJEMPLO 5: EXPORTACIÓN Y REPORTES")
    print("="*60)
    
    # Crear datos con errores
    datasets = crear_datos_con_errores()
    dataset = datasets['geograficos']
    
    # Simular pipeline
    result = simular_pipeline_con_errores(dataset, 'datos_geograficos')
    
    if result['success']:
        # Crear handler
        handler = BusinessErrorHandler()
        handler.extract_business_errors_from_logs(result['logs'])
        handler.extract_errors_from_validation_results(result['validation_results'])
        
        # Obtener resumen
        summary = handler.get_error_summary()
        
        print("📊 Generando reportes...")
        
        # Simular reporte de texto
        print(f"   Reporte de texto generado:")
        print(f"      - Total de errores: {summary['total_errors']}")
        print(f"      - Reglas afectadas: {len(summary['errors_by_rule'])}")
        print(f"      - Columnas afectadas: {len(summary['errors_by_column'])}")
        
        # Simular exportación CSV
        print(f"   CSV generado:")
        print(f"      - Filas de errores: {len(handler.errors)}")
        print(f"      - Columnas: rule_name, error_type, severity, message, column_name, invalid_value, timestamp")
        
        # Simular reporte detallado
        print(f"   Reporte detallado generado:")
        print(f"      - Resumen ejecutivo")
        print(f"      - Análisis por categoría")
        print(f"      - Recomendaciones")
        print(f"      - Detalle completo de errores")
        
        # Mostrar estadísticas específicas
        print(f"\n📈 Estadísticas específicas:")
        for rule, count in summary['errors_by_rule'].items():
            print(f"   - {rule}: {count} errores")
        
        for severity, count in summary['errors_by_severity'].items():
            print(f"   - {severity}: {count} errores")


def main():
    """
    Función principal que ejecuta todos los ejemplos.
    """
    print("🚀 EJEMPLO AVANZADO DE MANEJO DE ERRORES DE NEGOCIO")
    print("="*80)
    
    # Ejecutar ejemplos
    ejemplo_extraccion_errores_basica()
    ejemplo_manejo_avanzado_errores()
    ejemplo_visualizacion_streamlit()
    ejemplo_integracion_pipeline_completo()
    ejemplo_exportacion_reportes()
    
    print("\n" + "="*80)
    print("✅ TODOS LOS EJEMPLOS COMPLETADOS")
    print("="*80)
    
    print("\n💡 PRÓXIMOS PASOS:")
    print("1. Integrar BusinessErrorHandler en tu app_front.py")
    print("2. Usar display_business_errors_in_streamlit() después de validaciones")
    print("3. Configurar exportación automática de reportes")
    print("4. Personalizar recomendaciones según tu dominio")
    print("5. Agregar notificaciones externas si es necesario")


if __name__ == "__main__":
    main() 