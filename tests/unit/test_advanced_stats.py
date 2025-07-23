"""
Tests para estadísticas avanzadas
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing.stats import (
    summary_statistics,
    compute_correlations,
    contingency_analysis,
    normality_test,
    t_test_independent,
    linear_regression,
    frequency_table,
    outlier_detection
)

@pytest.fixture
def sample_data():
    """Datos de prueba para tests avanzados"""
    np.random.seed(42)
    n = 200
    
    # Datos con grupos para t-test
    grupo_a = np.random.normal(25, 5, n//2)
    grupo_b = np.random.normal(30, 5, n//2)
    
    data = {
        'edad': np.concatenate([grupo_a, grupo_b]),
        'ingresos': np.random.normal(50000, 15000, n),
        'satisfaccion': np.random.randint(1, 6, n),
        'genero': np.random.choice(['M', 'F'], n),
        'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universitaria'], n),
        'grupo': ['A'] * (n//2) + ['B'] * (n//2),
        'horas_trabajo': np.random.normal(40, 8, n),
        'productividad': np.random.normal(7.5, 1.5, n)
    }
    return pd.DataFrame(data)

def test_advanced_summary_statistics(sample_data):
    """Test para estadísticas descriptivas avanzadas"""
    result = summary_statistics(sample_data, ['edad', 'ingresos', 'horas_trabajo'])
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert all(col in result.columns for col in ['mean', 'median', 'std', 'skew', 'kurtosis', 'iqr'])

def test_correlation_analysis(sample_data):
    """Test para análisis de correlaciones"""
    result = compute_correlations(sample_data, ['edad', 'ingresos', 'horas_trabajo'], method='pearson')
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    assert all(col in result.index for col in ['edad', 'ingresos', 'horas_trabajo'])

def test_contingency_analysis_advanced(sample_data):
    """Test para análisis de contingencia avanzado"""
    table, stats = contingency_analysis(sample_data, 'genero', 'educacion')
    
    assert isinstance(table, pd.DataFrame)
    assert isinstance(stats, dict)
    assert 'chi2' in stats
    assert 'p_value' in stats
    assert 'cramer_v' in stats
    assert 'significant' in stats

def test_normality_test_advanced(sample_data):
    """Test para pruebas de normalidad avanzadas"""
    result = normality_test(sample_data, 'edad')
    
    assert isinstance(result, dict)
    assert 'shapiro_wilk' in result
    assert 'kolmogorov_smirnov' in result
    # Verificar estructura anidada
    assert 'statistic' in result['shapiro_wilk']
    assert 'p_value' in result['shapiro_wilk']
    assert 'is_normal' in result['shapiro_wilk']

def test_t_test_independent(sample_data):
    """Test para t-test de muestras independientes"""
    result = t_test_independent(sample_data, 'edad', 'grupo')
    
    assert isinstance(result, dict)
    assert 't_statistic' in result
    assert 'p_value' in result
    assert 'significant' in result
    assert 'cohens_d' in result  # Cambio de 'effect_size' a 'cohens_d'

def test_linear_regression(sample_data):
    """Test para regresión lineal"""
    result = linear_regression(sample_data, 'productividad', ['horas_trabajo', 'edad'])
    
    assert isinstance(result, dict)
    assert 'r2' in result
    assert 'mse' in result
    assert 'coefficients' in result
    assert 'model_fitted' in result  # Cambio de 'intercept' a 'model_fitted'

def test_frequency_table_advanced(sample_data):
    """Test para tabla de frecuencias avanzada"""
    result = frequency_table(sample_data, 'educacion')
    
    assert isinstance(result, pd.DataFrame)
    assert 'category' in result.columns
    assert 'frequency' in result.columns
    assert 'relative_frequency' in result.columns
    assert 'cumulative_frequency' in result.columns
    assert len(result) == 3  # Tres niveles de educación

def test_outlier_detection(sample_data):
    """Test para detección de outliers"""
    result = outlier_detection(sample_data, 'ingresos', method='iqr')
    
    assert isinstance(result, dict)
    assert 'outliers_count' in result
    assert 'outliers_values' in result
    assert 'method' in result
    assert 'outliers_percentage' in result
    assert 'total_observations' in result

def test_outlier_detection_zscore(sample_data):
    """Test para detección de outliers con z-score"""
    result = outlier_detection(sample_data, 'edad', method='zscore')
    
    assert isinstance(result, dict)
    assert 'outliers_count' in result
    assert 'outliers_values' in result
    assert 'method' in result

def test_statistical_robustness(sample_data):
    """Test para verificar robustez estadística"""
    # Test con datos que contienen valores nulos
    sample_data_with_nulls = sample_data.copy()
    sample_data_with_nulls.loc[0, 'edad'] = np.nan
    sample_data_with_nulls.loc[1, 'ingresos'] = np.nan
    
    result = summary_statistics(sample_data_with_nulls, ['edad', 'ingresos'])
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    # Debería manejar nulos correctamente
    assert result.loc[result['variable'] == 'edad', 'count'].iloc[0] < len(sample_data_with_nulls)

def test_error_handling():
    """Test para manejo de errores"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    
    # Test con columnas inexistentes
    with pytest.raises(ValueError):
        summary_statistics(df, ['inexistente'])
    
    # Test con DataFrame vacío
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        summary_statistics(empty_df, ['a']) 