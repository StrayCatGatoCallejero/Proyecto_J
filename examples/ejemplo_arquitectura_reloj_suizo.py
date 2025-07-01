"""
Ejemplo de Uso - Arquitectura "Reloj Suizo"
==========================================

Este archivo demuestra cómo usar la nueva arquitectura modular refactorizada
siguiendo el patrón de "reloj suizo" con separación clara de responsabilidades.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Importar módulos refactorizados
from processing.io import cargar_archivo, validar_dataframe, obtener_info_archivo
from processing.stats import (
    summary_statistics, compute_correlations, contingency_analysis,
    normality_test, t_test_independent, linear_regression,
    frequency_table, outlier_detection
)
from processing.visualization import (
    plot_histogram, plot_boxplot, plot_scatter, plot_heatmap, plot_bar_chart,
    plotly_histogram, plotly_heatmap, plotly_scatter
)
from processing.filters import (
    filter_by_condition, filter_by_range, filter_by_values,
    remove_outliers, handle_missing_values, sample_data,
    select_columns, drop_duplicates
)
from processing.features import (
    create_numeric_features, encode_categorical, scale_features,
    create_interaction_features, select_features, create_time_features,
    create_binning_features
)

def ejemplo_analisis_completo():
    """
    Ejemplo completo de análisis usando la nueva arquitectura.
    """
    print("🔧 Iniciando análisis con arquitectura 'Reloj Suizo'")
    print("=" * 60)
    
    # 1. CARGAR DATOS
    print("\n📁 1. CARGANDO DATOS")
    print("-" * 30)
    
    # Crear datos de ejemplo
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'edad': np.random.normal(35, 10, n_samples),
        'ingresos': np.random.lognormal(10, 0.5, n_samples),
        'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universitaria'], n_samples),
        'genero': np.random.choice(['M', 'F'], n_samples),
        'region': np.random.choice(['Norte', 'Centro', 'Sur'], n_samples),
        'satisfaccion': np.random.randint(1, 11, n_samples),
        'fecha_encuesta': pd.date_range('2023-01-01', periods=n_samples, freq='D')
    }
    
    df = pd.DataFrame(data)
    
    # Simular algunos valores faltantes y outliers
    df.loc[np.random.choice(df.index, 50), 'edad'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'ingresos'] = df['ingresos'].max() * 3
    
    print(f"✅ Datos creados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # 2. VALIDAR DATOS
    print("\n🔍 2. VALIDANDO DATOS")
    print("-" * 30)
    
    validation = validar_dataframe(df, {})
    print(f"✅ Validación completada: {validation['is_valid']}")
    if validation['warnings']:
        print(f"⚠️ Advertencias: {validation['warnings']}")
    
    # 3. LIMPIEZA DE DATOS
    print("\n🧹 3. LIMPIEZA DE DATOS")
    print("-" * 30)
    
    # Manejar valores faltantes
    df_clean = handle_missing_values(df, method='fill_median')
    print(f"✅ Valores faltantes manejados")
    
    # Eliminar outliers de ingresos
    df_clean = remove_outliers(df_clean, 'ingresos', method='iqr')
    print(f"✅ Outliers eliminados de ingresos")
    
    # Eliminar duplicados
    df_clean = drop_duplicates(df_clean)
    print(f"✅ Duplicados eliminados")
    
    print(f"📊 Datos finales: {df_clean.shape[0]} filas, {df_clean.shape[1]} columnas")
    
    # 4. ANÁLISIS ESTADÍSTICO
    print("\n📊 4. ANÁLISIS ESTADÍSTICO")
    print("-" * 30)
    
    # Estadísticas descriptivas
    numeric_cols = ['edad', 'ingresos', 'satisfaccion']
    stats = summary_statistics(df_clean, numeric_cols)
    print("📈 Estadísticas descriptivas:")
    print(stats[['variable', 'mean', 'std', 'skew']].to_string(index=False))
    
    # Correlaciones
    corr_matrix = compute_correlations(df_clean, numeric_cols, method='pearson')
    print("\n🔗 Matriz de correlaciones:")
    print(corr_matrix.round(3))
    
    # Análisis de contingencia
    contingency_table, contingency_stats = contingency_analysis(df_clean, 'educacion', 'genero')
    print(f"\n📋 Análisis de contingencia (Educación vs Género):")
    print(f"Chi² = {contingency_stats['chi2']:.3f}, p = {contingency_stats['p_value']:.3f}")
    
    # Pruebas de normalidad
    normality_results = normality_test(df_clean, 'edad')
    print(f"\n📏 Prueba de normalidad (edad):")
    for test, result in normality_results.items():
        if 'error' not in result:
            print(f"  {test}: p = {result['p_value']:.3f} ({result['interpretation']})")
    
    # 5. CREACIÓN DE FEATURES
    print("\n⚙️ 5. CREACIÓN DE FEATURES")
    print("-" * 30)
    
    # Features numéricas derivadas
    df_features = create_numeric_features(df_clean, ['edad', 'ingresos'])
    print(f"✅ Features numéricas creadas: {len(df_features.columns)} columnas total")
    
    # Codificación categórica
    df_features = encode_categorical(df_features, ['educacion', 'genero'], method='label')
    print(f"✅ Variables categóricas codificadas")
    
    # Features de interacción
    df_features = create_interaction_features(df_features, ['edad', 'ingresos'], max_interactions=3)
    print(f"✅ Features de interacción creadas")
    
    # Features temporales
    df_features = create_time_features(df_features, 'fecha_encuesta')
    print(f"✅ Features temporales creadas")
    
    # 6. VISUALIZACIONES
    print("\n📊 6. VISUALIZACIONES")
    print("-" * 30)
    
    # Histograma
    fig1 = plot_histogram(df_clean, 'edad', bins=20)
    fig1.savefig('histograma_edad.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("✅ Histograma de edad guardado")
    
    # Boxplot por educación
    fig2 = plot_boxplot(df_clean, 'ingresos', 'educacion')
    fig2.savefig('boxplot_ingresos_educacion.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("✅ Boxplot de ingresos por educación guardado")
    
    # Gráfico de dispersión
    fig3 = plot_scatter(df_clean, 'edad', 'ingresos', color_column='genero')
    fig3.savefig('scatter_edad_ingresos.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("✅ Gráfico de dispersión guardado")
    
    # Mapa de calor de correlaciones
    fig4 = plot_heatmap(corr_matrix)
    fig4.savefig('heatmap_correlaciones.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("✅ Mapa de calor de correlaciones guardado")
    
    # 7. ANÁLISIS AVANZADO
    print("\n🔬 7. ANÁLISIS AVANZADO")
    print("-" * 30)
    
    # T-test independiente
    t_test_result = t_test_independent(df_clean, 'ingresos', 'genero')
    print(f"📊 T-test (Ingresos por Género):")
    print(f"  t = {t_test_result['t_statistic']:.3f}, p = {t_test_result['p_value']:.3f}")
    print(f"  Significativo: {t_test_result['significant']}")
    
    # Regresión lineal
    regression_result = linear_regression(df_clean, 'satisfaccion', ['edad', 'ingresos'])
    print(f"\n📈 Regresión lineal (Satisfacción ~ Edad + Ingresos):")
    print(f"  R² = {regression_result['r2']:.3f}")
    print(f"  RMSE = {regression_result['rmse']:.3f}")
    
    # Selección de features
    selected_features = select_features(df_features, 'satisfaccion', method='correlation', n_features=5)
    print(f"\n🎯 Features seleccionadas para satisfacción:")
    print(f"  {selected_features}")
    
    # 8. FILTRADO Y MUESTREO
    print("\n🔍 8. FILTRADO Y MUESTREO")
    print("-" * 30)
    
    # Filtrar por condición
    df_filtered = filter_by_condition(df_clean, "edad > 30 and ingresos > 1000")
    print(f"✅ Filtro aplicado: {len(df_filtered)} filas")
    
    # Filtrar por rango
    df_filtered = filter_by_range(df_filtered, 'satisfaccion', min_val=5, max_val=10)
    print(f"✅ Filtro por rango aplicado: {len(df_filtered)} filas")
    
    # Muestreo
    df_sample = sample_data(df_filtered, fraction=0.5, random_state=42)
    print(f"✅ Muestreo aplicado: {len(df_sample)} filas")
    
    # 9. RESUMEN FINAL
    print("\n📋 9. RESUMEN FINAL")
    print("-" * 30)
    
    print(f"📊 Datos originales: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"🧹 Datos limpios: {df_clean.shape[0]} filas, {df_clean.shape[1]} columnas")
    print(f"⚙️ Con features: {df_features.shape[0]} filas, {df_features.shape[1]} columnas")
    print(f"🔍 Muestra final: {df_sample.shape[0]} filas, {df_sample.shape[1]} columnas")
    
    print(f"\n📈 Variables numéricas analizadas: {len(numeric_cols)}")
    print(f"🔗 Correlaciones calculadas: {len(numeric_cols)} variables")
    print(f"📊 Pruebas estadísticas realizadas: 3")
    print(f"📊 Visualizaciones generadas: 4")
    
    print(f"\n✅ Análisis completado exitosamente!")
    print(f"📁 Archivos generados:")
    print(f"  - histograma_edad.png")
    print(f"  - boxplot_ingresos_educacion.png")
    print(f"  - scatter_edad_ingresos.png")
    print(f"  - heatmap_correlaciones.png")

def ejemplo_funciones_individuales():
    """
    Ejemplo de uso de funciones individuales.
    """
    print("\n🔧 EJEMPLO DE FUNCIONES INDIVIDUALES")
    print("=" * 60)
    
    # Crear datos de ejemplo
    df = pd.DataFrame({
        'x': np.random.normal(0, 1, 100),
        'y': np.random.normal(0, 1, 100),
        'categoria': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Ejemplo de cada módulo
    print("\n📁 I/O:")
    info = obtener_info_archivo("datos_ejemplo_features.csv")
    print(f"  Información de archivo: {info['name']} ({info['size']} bytes)")
    
    print("\n📊 Estadísticas:")
    stats = summary_statistics(df, ['x', 'y'])
    print(f"  Estadísticas calculadas para {len(stats)} variables")
    
    print("\n📊 Visualización:")
    fig = plot_histogram(df, 'x', bins=15)
    plt.close(fig)
    print("  Histograma generado")
    
    print("\n🔍 Filtros:")
    df_filtrado = filter_by_range(df, 'x', min_val=-1, max_val=1)
    print(f"  Datos filtrados: {len(df_filtrado)} filas")
    
    print("\n⚙️ Features:")
    df_features = create_numeric_features(df, ['x', 'y'])
    print(f"  Features creadas: {len(df_features.columns)} columnas")

if __name__ == "__main__":
    # Ejecutar ejemplos
    ejemplo_analisis_completo()
    ejemplo_funciones_individuales()
    
    print("\n🎉 ¡Todos los ejemplos completados exitosamente!")
    print("📚 La arquitectura 'Reloj Suizo' está funcionando correctamente.") 