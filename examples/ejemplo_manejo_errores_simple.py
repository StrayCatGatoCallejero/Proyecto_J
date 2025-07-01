"""
Ejemplo Simplificado de Manejo de Errores de Negocio
===================================================

Este ejemplo demuestra el uso del sistema de manejo de errores
sin depender del pipeline_orchestrator completo.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# Importar módulos del sistema de manejo de errores
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
from processing.logging import get_logger


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
    
    return {
        'demograficos': datos_demograficos,
        'geograficos': datos_geograficos,
        'likert': datos_likert
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
    
    # Crear metadata
    metadata = {
        'dataset_type': 'social_sciences',
        'dataset_name': 'datos_demograficos',
        'columns': list(dataset.columns)
    }
    
    # Ejecutar validaciones
    print("🔍 Ejecutando validaciones de reglas de negocio...")
    validation_results = validate_business_rules(dataset, metadata)
    
    # Contar errores
    total_errors = sum(1 for r in validation_results if not r.is_valid)
    total_warnings = len([r for r in validation_results if r.details.get('alertas_generadas', 0) > 0])
    
    print(f"📊 Resultados de validación:")
    print(f"   - Total reglas ejecutadas: {len(validation_results)}")
    print(f"   - Reglas fallidas: {total_errors}")
    print(f"   - Advertencias: {total_warnings}")
    
    # Mostrar detalles de errores
    for result in validation_results:
        if not result.is_valid:
            print(f"   ❌ {result.rule_name}: {result.message}")
        elif result.details.get('alertas_generadas', 0) > 0:
            print(f"   ⚠️ {result.rule_name}: {result.message}")
        else:
            print(f"   ✅ {result.rule_name}: {result.message}")


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
        
        # Crear metadata
        metadata = {
            'dataset_type': 'social_sciences',
            'dataset_name': dataset_name,
            'columns': list(dataset.columns)
        }
        
        # Ejecutar validaciones
        validation_results = validate_business_rules(dataset, metadata)
        
        # Extraer errores desde resultados de validación
        handler.extract_errors_from_validation_results(validation_results)
        
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
    
    # Crear metadata
    metadata = {
        'dataset_type': 'social_sciences',
        'dataset_name': 'datos_likert',
        'columns': list(dataset.columns)
    }
    
    # Ejecutar validaciones
    validation_results = validate_business_rules(dataset, metadata)
    
    print("📊 Simulando visualización en Streamlit...")
    
    # Contar errores
    total_errors = sum(1 for r in validation_results if not r.is_valid)
    total_warnings = len([r for r in validation_results if r.details.get('alertas_generadas', 0) > 0])
    
    print(f"   Métricas mostradas:")
    print(f"      - Total de errores: {total_errors}")
    print(f"      - Total de advertencias: {total_warnings}")
    print(f"      - Reglas ejecutadas: {len(validation_results)}")
    
    # Simular gráficos
    print(f"   Gráficos generados:")
    print(f"      - Gráfico de barras: Errores por regla")
    print(f"      - Gráfico de pastel: Errores por severidad")
    print(f"      - Gráfico de dispersión: Timeline de errores")
    
    # Simular tabla
    print(f"   Tabla de errores:")
    for i, result in enumerate(validation_results, 1):
        if not result.is_valid:
            print(f"      {i}. {result.rule_name} - {result.message}")
    
    # Simular botones de acción
    print(f"   Botones de acción:")
    print(f"      - 📋 Copiar Reporte")
    print(f"      - 📊 Exportar CSV")
    print(f"      - 📄 Generar Reporte")
    
    # Simular recomendaciones
    handler = BusinessErrorHandler()
    handler.extract_errors_from_validation_results(validation_results)
    recommendations = handler.get_recommendations()
    
    if recommendations:
        print(f"   Recomendaciones mostradas:")
        for rec in recommendations:
            print(f"      - {rec}")


def ejemplo_exportacion_reportes():
    """
    Ejemplo de exportación y generación de reportes.
    """
    print("\n" + "="*60)
    print("EJEMPLO 4: EXPORTACIÓN Y REPORTES")
    print("="*60)
    
    # Crear datos con errores
    datasets = crear_datos_con_errores()
    dataset = datasets['geograficos']
    
    # Crear metadata
    metadata = {
        'dataset_type': 'social_sciences',
        'dataset_name': 'datos_geograficos',
        'columns': list(dataset.columns)
    }
    
    # Ejecutar validaciones
    validation_results = validate_business_rules(dataset, metadata)
    
    # Crear handler
    handler = BusinessErrorHandler()
    handler.extract_errors_from_validation_results(validation_results)
    
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


def ejemplo_integracion_practica():
    """
    Ejemplo de integración práctica con una aplicación real.
    """
    print("\n" + "="*60)
    print("EJEMPLO 5: INTEGRACIÓN PRÁCTICA")
    print("="*60)
    
    # Simular carga de datos desde archivo
    print("📁 Simulando carga de datos...")
    dataset = crear_datos_con_errores()['demograficos']
    print(f"   ✅ Datos cargados: {dataset.shape[0]} filas, {dataset.shape[1]} columnas")
    
    # Simular validación automática
    print("\n🔍 Ejecutando validación automática...")
    metadata = {'dataset_type': 'social_sciences'}
    validation_results = validate_business_rules(dataset, metadata)
    
    # Contar resultados
    failed_validations = [r for r in validation_results if not r.is_valid]
    warnings = [r for r in validation_results if r.details.get('alertas_generadas', 0) > 0]
    
    print(f"   Resultados:")
    print(f"      - Reglas ejecutadas: {len(validation_results)}")
    print(f"      - Reglas fallidas: {len(failed_validations)}")
    print(f"      - Advertencias: {len(warnings)}")
    
    # Simular manejo de errores en aplicación
    if failed_validations:
        print(f"\n🚨 Se detectaron errores críticos. Simulando manejo en aplicación...")
        
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
        
        print(f"\n💡 En una aplicación real, se mostraría:")
        print(f"   - Mensaje de error principal")
        print(f"   - Métricas de errores")
        print(f"   - Gráficos de análisis")
        print(f"   - Tabla detallada de errores")
        print(f"   - Botones de acción (copiar, exportar, reporte)")
        print(f"   - Recomendaciones específicas")
        
        # Simular opciones del usuario
        print(f"\n🤔 Opciones para el usuario:")
        print(f"   1. 🔄 Corregir datos automáticamente")
        print(f"   2. ⚠️ Continuar con advertencias")
        print(f"   3. ❌ Cancelar operación")
        print(f"   4. 📋 Generar reporte de errores")
        
    else:
        print(f"✅ Todas las validaciones pasaron exitosamente")


def main():
    """
    Función principal que ejecuta todos los ejemplos.
    """
    print("🚀 EJEMPLO SIMPLIFICADO DE MANEJO DE ERRORES DE NEGOCIO")
    print("="*80)
    
    # Ejecutar ejemplos
    ejemplo_extraccion_errores_basica()
    ejemplo_manejo_avanzado_errores()
    ejemplo_visualizacion_streamlit()
    ejemplo_exportacion_reportes()
    ejemplo_integracion_practica()
    
    print("\n" + "="*80)
    print("✅ TODOS LOS EJEMPLOS COMPLETADOS")
    print("="*80)
    
    print("\n💡 PRÓXIMOS PASOS:")
    print("1. Integrar BusinessErrorHandler en tu app_front.py")
    print("2. Usar display_business_errors_in_streamlit() después de validaciones")
    print("3. Configurar exportación automática de reportes")
    print("4. Personalizar recomendaciones según tu dominio")
    print("5. Agregar notificaciones externas si es necesario")
    
    print("\n📚 ARCHIVOS IMPORTANTES:")
    print("- processing/business_error_handler.py - Sistema principal")
    print("- processing/business_rules.py - Reglas de validación")
    print("- README_MANEJO_ERRORES.md - Documentación completa")


if __name__ == "__main__":
    main() 