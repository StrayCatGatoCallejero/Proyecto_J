"""
Test de la Arquitectura "Reloj Suizo"
====================================

Script para verificar que todos los módulos refactorizados funcionan correctamente.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

def test_imports():
    """Prueba que todos los módulos se pueden importar correctamente."""
    print("🔍 Probando imports...")
    
    try:
        # I/O
        from processing.io import cargar_archivo, validar_dataframe, obtener_info_archivo
        print("✅ processing.io - OK")
        
        # Stats
        from processing.stats import (
            summary_statistics, compute_correlations, contingency_analysis,
            normality_test, t_test_independent, linear_regression,
            frequency_table, outlier_detection
        )
        print("✅ processing.stats - OK")
        
        # Visualization
        from processing.visualization import (
            plot_histogram, plot_boxplot, plot_scatter, plot_heatmap, plot_bar_chart,
            plotly_histogram, plotly_heatmap, plotly_scatter
        )
        print("✅ processing.visualization - OK")
        
        # Filters
        from processing.filters import (
            filter_by_condition, filter_by_range, filter_by_values,
            remove_outliers, handle_missing_values, sample_data,
            select_columns, drop_duplicates
        )
        print("✅ processing.filters - OK")
        
        # Features
        from processing.features import (
            create_numeric_features, encode_categorical, scale_features,
            create_interaction_features, select_features, create_time_features,
            create_binning_features
        )
        print("✅ processing.features - OK")
        
        # Logging
        from processing.logging import log_action, setup_logging
        print("✅ processing.logging - OK")
        
        # Utils
        from processing.utils import (
            obtener_info_archivo, validar_dataframe, summary_statistics, compute_correlations, contingency_analysis, normality_test, t_test_independent, linear_regression, plot_histogram, plot_boxplot, plot_scatter, plot_heatmap, plot_bar_chart, filter_by_condition, filter_by_range, filter_by_values, remove_outliers, handle_missing_values, sample_data, select_columns, drop_duplicates, create_numeric_features, encode_categorical, scale_features, create_interaction_features, select_features, create_time_features, create_binning_features
        )
        print("✅ processing.utils - OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de import: {e}")
        return False

def create_test_data():
    """Crea datos de prueba para las funciones."""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'edad': np.random.normal(35, 10, n_samples),
        'ingresos': np.random.lognormal(10, 0.5, n_samples),
        'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universitaria'], n_samples),
        'genero': np.random.choice(['M', 'F'], n_samples),
        'region': np.random.choice(['Norte', 'Centro', 'Sur'], n_samples),
        'satisfaccion': np.random.randint(1, 11, n_samples),
        'fecha': pd.date_range('2023-01-01', periods=n_samples, freq='D')
    }
    
    df = pd.DataFrame(data)
    
    # Simular algunos valores faltantes y outliers
    df.loc[np.random.choice(df.index, 20), 'edad'] = np.nan
    df.loc[np.random.choice(df.index, 10), 'ingresos'] = df['ingresos'].max() * 2
    
    return df

def test_io_functions(df):
    """Prueba las funciones de I/O."""
    print("\n📁 Probando funciones de I/O...")
    
    try:
        # Guardar datos de prueba
        df.to_csv('test_data.csv', index=False)
        
        # Probar obtener_info_archivo
        info = obtener_info_archivo('test_data.csv')
        assert 'test_data.csv' in info['name']
        print("✅ obtener_info_archivo - OK")
        
        # Probar validar_dataframe
        validation = validar_dataframe(df, {})
        assert isinstance(validation, dict)
        assert 'is_valid' in validation
        print("✅ validar_dataframe - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en I/O: {e}")
        return False

def test_stats_functions(df):
    """Prueba las funciones de estadísticas."""
    print("\n📊 Probando funciones de estadísticas...")
    
    try:
        # Estadísticas descriptivas
        stats = summary_statistics(df, ['edad', 'ingresos'])
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) > 0
        print("✅ summary_statistics - OK")
        
        # Correlaciones
        corr = compute_correlations(df, ['edad', 'ingresos'])
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (2, 2)
        print("✅ compute_correlations - OK")
        
        # Análisis de contingencia
        table, stats = contingency_analysis(df, 'educacion', 'genero')
        assert isinstance(table, pd.DataFrame)
        assert isinstance(stats, dict)
        print("✅ contingency_analysis - OK")
        
        # Pruebas de normalidad
        normality = normality_test(df, 'edad')
        assert isinstance(normality, dict)
        print("✅ normality_test - OK")
        
        # T-test
        t_result = t_test_independent(df, 'ingresos', 'genero')
        assert isinstance(t_result, dict)
        assert 't_statistic' in t_result
        print("✅ t_test_independent - OK")
        
        # Regresión lineal
        regression = linear_regression(df, 'satisfaccion', ['edad', 'ingresos'])
        assert isinstance(regression, dict)
        assert 'r2' in regression
        print("✅ linear_regression - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en estadísticas: {e}")
        return False

def test_visualization_functions(df):
    """Prueba las funciones de visualización."""
    print("\n📊 Probando funciones de visualización...")
    
    try:
        # Histograma
        fig1 = plot_histogram(df, 'edad', bins=15)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        print("✅ plot_histogram - OK")
        
        # Boxplot
        fig2 = plot_boxplot(df, 'ingresos', 'educacion')
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
        print("✅ plot_boxplot - OK")
        
        # Scatter plot
        fig3 = plot_scatter(df, 'edad', 'ingresos')
        assert isinstance(fig3, plt.Figure)
        plt.close(fig3)
        print("✅ plot_scatter - OK")
        
        # Correlaciones para heatmap
        corr = compute_correlations(df, ['edad', 'ingresos'])
        fig4 = plot_heatmap(corr)
        assert isinstance(fig4, plt.Figure)
        plt.close(fig4)
        print("✅ plot_heatmap - OK")
        
        # Bar chart
        fig5 = plot_bar_chart(df, 'educacion')
        assert isinstance(fig5, plt.Figure)
        plt.close(fig5)
        print("✅ plot_bar_chart - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en visualización: {e}")
        return False

def test_filters_functions(df):
    """Prueba las funciones de filtros."""
    print("\n🔍 Probando funciones de filtros...")
    
    try:
        # Filtrar por condición
        df_filtered = filter_by_condition(df, "edad > 30")
        assert len(df_filtered) < len(df)
        print("✅ filter_by_condition - OK")
        
        # Filtrar por rango
        df_filtered = filter_by_range(df, 'ingresos', min_val=1000, max_val=5000)
        assert len(df_filtered) < len(df)
        print("✅ filter_by_range - OK")
        
        # Filtrar por valores
        df_filtered = filter_by_values(df, 'region', ['Norte', 'Centro'])
        assert len(df_filtered) < len(df)
        print("✅ filter_by_values - OK")
        
        # Eliminar outliers
        df_clean = remove_outliers(df, 'ingresos', method='iqr')
        assert len(df_clean) <= len(df)
        print("✅ remove_outliers - OK")
        
        # Manejar valores faltantes
        df_clean = handle_missing_values(df, method='fill_median')
        assert df_clean.isnull().sum().sum() == 0
        print("✅ handle_missing_values - OK")
        
        # Muestreo
        df_sample = sample_data(df, fraction=0.5, random_state=42)
        assert len(df_sample) == len(df) // 2
        print("✅ sample_data - OK")
        
        # Seleccionar columnas
        df_selected = select_columns(df, ['edad', 'ingresos'])
        assert len(df_selected.columns) == 2
        print("✅ select_columns - OK")
        
        # Eliminar duplicados
        df_dedup = drop_duplicates(df)
        assert len(df_dedup) <= len(df)
        print("✅ drop_duplicates - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en filtros: {e}")
        return False

def test_features_functions(df):
    """Prueba las funciones de features."""
    print("\n⚙️ Probando funciones de features...")
    
    try:
        # Features numéricas
        df_features = create_numeric_features(df, ['edad', 'ingresos'])
        assert len(df_features.columns) > len(df.columns)
        print("✅ create_numeric_features - OK")
        
        # Codificación categórica
        df_encoded = encode_categorical(df, ['educacion'], method='label')
        assert len(df_encoded.columns) > len(df.columns)
        print("✅ encode_categorical - OK")
        
        # Escalado
        df_scaled = scale_features(df, ['edad', 'ingresos'], method='standard')
        assert len(df_scaled.columns) > len(df.columns)
        print("✅ scale_features - OK")
        
        # Features de interacción
        df_interactions = create_interaction_features(df, ['edad', 'ingresos'], max_interactions=2)
        assert len(df_interactions.columns) > len(df.columns)
        print("✅ create_interaction_features - OK")
        
        # Selección de features
        selected = select_features(df, 'satisfaccion', method='correlation', n_features=3)
        assert isinstance(selected, list)
        assert len(selected) <= 3
        print("✅ select_features - OK")
        
        # Features temporales
        df_time = create_time_features(df, 'fecha')
        assert len(df_time.columns) > len(df.columns)
        print("✅ create_time_features - OK")
        
        # Binning
        df_binned = create_binning_features(df, ['edad'], n_bins=5)
        assert len(df_binned.columns) > len(df.columns)
        print("✅ create_binning_features - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en features: {e}")
        return False

def test_logging():
    """Prueba el sistema de logging."""
    print("\n📝 Probando sistema de logging...")
    
    try:
        from processing.logging import log_action
        
        # Probar logging básico
        log_action(
            function="test_function",
            step="test",
            parameters={"test": True},
            before_metrics={"n": 1},
            after_metrics={"n": 1},
            status="success",
            message="Test de logging"
        )
        print("✅ log_action - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en logging: {e}")
        return False

def run_all_tests():
    """Ejecuta todas las pruebas."""
    print("🧪 INICIANDO PRUEBAS DE LA ARQUITECTURA 'RELOJ SUIZO'")
    print("=" * 60)
    
    # Contador de pruebas
    total_tests = 0
    passed_tests = 0
    
    # 1. Probar imports
    total_tests += 1
    if test_imports():
        passed_tests += 1
    
    # 2. Crear datos de prueba
    df = create_test_data()
    print(f"\n📊 Datos de prueba creados: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # 3. Probar funciones de I/O
    total_tests += 1
    if test_io_functions(df):
        passed_tests += 1
    
    # 4. Probar funciones de estadísticas
    total_tests += 1
    if test_stats_functions(df):
        passed_tests += 1
    
    # 5. Probar funciones de visualización
    total_tests += 1
    if test_visualization_functions(df):
        passed_tests += 1
    
    # 6. Probar funciones de filtros
    total_tests += 1
    if test_filters_functions(df):
        passed_tests += 1
    
    # 7. Probar funciones de features
    total_tests += 1
    if test_features_functions(df):
        passed_tests += 1
    
    # 8. Probar sistema de logging
    total_tests += 1
    if test_logging():
        passed_tests += 1
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 60)
    print(f"✅ Pruebas pasadas: {passed_tests}/{total_tests}")
    print(f"📊 Porcentaje de éxito: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON!")
        print("🚀 La arquitectura 'Reloj Suizo' está funcionando correctamente.")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} pruebas fallaron.")
        print("🔧 Revisar los errores anteriores.")
    
    # Limpiar archivos de prueba
    try:
        import os
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
    except:
        pass
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 