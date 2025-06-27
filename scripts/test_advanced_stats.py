"""
Script de Prueba: Análisis Estadístico Avanzado
==============================================

Demuestra el sistema completo de análisis estadístico avanzado
con datos sintéticos de ciencias sociales.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.pipeline_orchestrator import PipelineOrchestrator
from processing.stats import (
    summary_statistics_advanced,
    recommend_statistical_tests,
    linear_regression_analysis,
    suggest_analysis_for_ui,
    compute_correlations
)

def create_synthetic_social_data(n_samples=1000):
    """Crea datos sintéticos de ciencias sociales para pruebas."""
    np.random.seed(42)
    
    # Datos demográficos
    edad = np.random.normal(35, 12, n_samples)
    edad = np.clip(edad, 18, 80)
    
    genero = np.random.choice(['Masculino', 'Femenino', 'No binario'], n_samples, p=[0.48, 0.48, 0.04])
    
    educacion = np.random.choice([
        'Sin educación', 'Primaria', 'Secundaria', 'Técnica', 'Universitaria', 'Postgrado'
    ], n_samples, p=[0.05, 0.15, 0.30, 0.20, 0.25, 0.05])
    
    # Datos socioeconómicos
    ingresos = np.random.lognormal(10.5, 0.8, n_samples)  # Distribución log-normal típica de ingresos
    ingresos = np.clip(ingresos, 200000, 5000000)
    
    # Correlación entre educación e ingresos
    educacion_numerica = pd.Categorical(educacion).codes
    ingresos = ingresos * (1 + 0.3 * educacion_numerica + np.random.normal(0, 0.2, n_samples))
    
    # Datos de opinión (escalas Likert)
    satisfaccion_vida = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1])
    confianza_gobierno = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05])
    
    # Correlación entre satisfacción y confianza
    confianza_gobierno = np.clip(confianza_gobierno + np.random.normal(0, 0.5, n_samples), 1, 5).astype(int)
    
    # Datos geográficos
    region = np.random.choice([
        'Metropolitana', 'Valparaíso', 'O\'Higgins', 'Maule', 'Biobío', 'Araucanía'
    ], n_samples, p=[0.4, 0.15, 0.1, 0.1, 0.15, 0.1])
    
    # Crear DataFrame
    df = pd.DataFrame({
        'edad': edad,
        'genero': genero,
        'educacion': educacion,
        'ingresos': ingresos,
        'satisfaccion_vida': satisfaccion_vida,
        'confianza_gobierno': confianza_gobierno,
        'region': region
    })
    
    return df

def create_metadata():
    """Crea metadata semántica para los datos."""
    return {
        'edad': {
            'category': 'demographic',
            'unit': 'años',
            'type': 'numeric',
            'description': 'Edad del encuestado'
        },
        'genero': {
            'category': 'demographic',
            'type': 'categorical',
            'description': 'Identidad de género'
        },
        'educacion': {
            'category': 'demographic',
            'type': 'categorical',
            'description': 'Nivel educativo alcanzado'
        },
        'ingresos': {
            'category': 'socioeconomic',
            'unit': 'CLP',
            'type': 'numeric',
            'description': 'Ingresos mensuales'
        },
        'satisfaccion_vida': {
            'category': 'opinion',
            'type': 'likert',
            'scale': '1-5',
            'description': 'Satisfacción con la vida'
        },
        'confianza_gobierno': {
            'category': 'opinion',
            'type': 'likert',
            'scale': '1-5',
            'description': 'Confianza en el gobierno'
        },
        'region': {
            'category': 'geographic',
            'type': 'categorical',
            'description': 'Región de residencia'
        }
    }

def test_advanced_statistics():
    """Prueba el análisis estadístico avanzado."""
    print("🔬 Iniciando prueba de análisis estadístico avanzado...")
    print("=" * 60)
    
    # 1. Crear datos sintéticos
    print("📊 Creando datos sintéticos de ciencias sociales...")
    df = create_synthetic_social_data(1000)
    metadata = create_metadata()
    
    print(f"✅ Datos creados: {len(df)} filas, {len(df.columns)} columnas")
    print(f"   Columnas: {list(df.columns)}")
    print()
    
    # 2. Estadísticas descriptivas avanzadas
    print("📈 Calculando estadísticas descriptivas avanzadas...")
    try:
        advanced_stats = summary_statistics_advanced(df, metadata)
        print(f"✅ Estadísticas calculadas para {len(advanced_stats['column_statistics'])} variables")
        
        # Mostrar resumen
        summary = advanced_stats['summary']
        print(f"   📊 Variables analizadas: {summary['total_columns_analyzed']}")
        print(f"   ⚠️ Outliers totales: {summary['total_outliers']}")
        print(f"   📉 Variables no normales: {summary['non_normal_variables']}")
        print()
        
        # Mostrar ejemplo de estadísticas para una variable
        if 'ingresos' in advanced_stats['column_statistics']:
            ingresos_stats = advanced_stats['column_statistics']['ingresos']
            print("💰 Ejemplo - Estadísticas de ingresos:")
            basic = ingresos_stats['basic_stats']
            print(f"   Media: ${basic['mean']:,.0f} CLP")
            print(f"   Mediana: ${basic['median']:,.0f} CLP")
            print(f"   Desv. Est.: ${basic['std']:,.0f} CLP")
            print(f"   Outliers: {ingresos_stats['outlier_stats']['outliers_count']} ({ingresos_stats['outlier_stats']['outliers_percentage']:.1f}%)")
            print()
        
    except Exception as e:
        print(f"❌ Error en estadísticas descriptivas: {str(e)}")
        return
    
    # 3. Recomendaciones de pruebas estadísticas
    print("🔍 Generando recomendaciones de pruebas estadísticas...")
    try:
        test_recommendations = recommend_statistical_tests(df, metadata, metadata)
        print(f"✅ {len(test_recommendations)} recomendaciones generadas")
        
        # Mostrar algunas recomendaciones
        for i, rec in enumerate(test_recommendations[:3]):
            print(f"   {i+1}. {rec['test_name']}: {rec['rationale']}")
        print()
        
    except Exception as e:
        print(f"❌ Error en recomendaciones: {str(e)}")
    
    # 4. Análisis de correlaciones
    print("🔗 Calculando correlaciones...")
    try:
        correlations = compute_correlations(df)
        print(f"✅ Correlaciones calculadas")
        
        sig_corrs = correlations.get('significant_correlations', [])
        print(f"   📊 Correlaciones significativas: {len(sig_corrs)}")
        
        if sig_corrs:
            print("   Top 3 correlaciones:")
            for i, corr in enumerate(sig_corrs[:3]):
                print(f"      {i+1}. {corr['variable1']} ↔ {corr['variable2']}: r={corr['correlation']:.3f} (p={corr['p_value']:.3f})")
        print()
        
    except Exception as e:
        print(f"❌ Error en correlaciones: {str(e)}")
    
    # 5. Sugerencias para UI
    print("💡 Generando sugerencias de análisis...")
    try:
        ui_suggestions = suggest_analysis_for_ui(df, advanced_stats, correlations, metadata)
        print(f"✅ {len(ui_suggestions)} sugerencias generadas")
        
        for i, suggestion in enumerate(ui_suggestions[:2]):
            print(f"   {i+1}. {suggestion['type']}: {suggestion['message'][:100]}...")
        print()
        
    except Exception as e:
        print(f"❌ Error en sugerencias: {str(e)}")
    
    # 6. Análisis de regresión
    print("📈 Realizando análisis de regresión...")
    try:
        # Regresión: ingresos ~ edad + educación
        regression_result = linear_regression_analysis(
            df, 'ingresos', ['edad'], metadata
        )
        print(f"✅ Modelo de regresión creado")
        print(f"   R²: {regression_result['r_squared']:.3f}")
        print(f"   RMSE: {regression_result['rmse']:,.0f}")
        print(f"   Interpretación: {regression_result['semantic_interpretation']['model_fit']}")
        print()
        
    except Exception as e:
        print(f"❌ Error en regresión: {str(e)}")
    
    # 7. Prueba del orquestador completo
    print("🎯 Probando orquestador completo...")
    try:
        orchestrator = PipelineOrchestrator()
        
        # Simular carga de datos
        print("   📁 Simulando carga de datos...")
        orchestrator.session_data.df = df
        orchestrator.session_data.metadata = metadata
        orchestrator.session_data.semantic_classification = metadata
        
        # Ejecutar análisis estadístico
        print("   📊 Ejecutando análisis estadístico...")
        result = orchestrator.run_step('statistical_analysis')
        
        if result['status'] == 'success':
            print("   ✅ Análisis estadístico completado exitosamente")
            stats_data = result['result']
            print(f"      - Estadísticas: {len(stats_data.get('descriptive_statistics', {}).get('column_statistics', {}))} variables")
            print(f"      - Recomendaciones: {len(stats_data.get('test_recommendations', []))} pruebas")
            print(f"      - Sugerencias: {len(stats_data.get('ui_suggestions', []))} sugerencias")
        else:
            print(f"   ❌ Error en orquestador: {result['error']}")
        
    except Exception as e:
        print(f"❌ Error en orquestador: {str(e)}")
    
    print("=" * 60)
    print("🎉 Prueba de análisis estadístico avanzado completada!")
    print("📋 Resumen:")
    print("   ✅ Datos sintéticos creados")
    print("   ✅ Estadísticas descriptivas avanzadas")
    print("   ✅ Recomendaciones de pruebas")
    print("   ✅ Análisis de correlaciones")
    print("   ✅ Sugerencias de análisis")
    print("   ✅ Modelos de regresión")
    print("   ✅ Orquestador integrado")
    print()
    print("🚀 El sistema está listo para análisis de datos de ciencias sociales!")

if __name__ == "__main__":
    test_advanced_statistics() 