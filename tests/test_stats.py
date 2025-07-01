"""
Tests para el módulo de estadísticas
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from processing.stats import (
    summary_statistics,
    compute_correlations,
    contingency_analysis,
    normality_test,
    frequency_table
)

@pytest.fixture
def sample_data():
    """Datos de prueba para los tests"""
    np.random.seed(42)
    data = {
        'edad': np.random.normal(35, 10, 100),
        'ingresos': np.random.normal(50000, 15000, 100),
        'genero': np.random.choice(['M', 'F'], 100),
        'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universitaria'], 100),
        'satisfaccion': np.random.randint(1, 6, 100)
    }
    return pd.DataFrame(data)

def test_summary_statistics(sample_data):
    """Test para estadísticas descriptivas"""
    result = summary_statistics(sample_data, ['edad', 'ingresos'])
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert 'edad' in result['variable'].values
    assert 'ingresos' in result['variable'].values
    assert 'mean' in result.columns
    assert 'std' in result.columns

def test_compute_correlations(sample_data):
    """Test para correlaciones"""
    result = compute_correlations(sample_data, ['edad', 'ingresos'])
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert 'edad' in result.index
    assert 'ingresos' in result.index

def test_contingency_analysis(sample_data):
    """Test para análisis de contingencia"""
    table, stats = contingency_analysis(sample_data, 'genero', 'educacion')
    
    assert isinstance(table, pd.DataFrame)
    assert isinstance(stats, dict)
    assert 'chi2' in stats
    assert 'p_value' in stats
    assert 'cramer_v' in stats

def test_normality_test(sample_data):
    """Test para pruebas de normalidad"""
    result = normality_test(sample_data, 'edad')
    
    assert isinstance(result, dict)
    assert 'shapiro_wilk' in result
    assert 'kolmogorov_smirnov' in result
    # Verificar estructura anidada
    assert 'statistic' in result['shapiro_wilk']
    assert 'p_value' in result['shapiro_wilk']
    assert 'is_normal' in result['shapiro_wilk']

def test_frequency_table(sample_data):
    """Test para tabla de frecuencias"""
    result = frequency_table(sample_data, 'genero')
    
    assert isinstance(result, pd.DataFrame)
    assert 'category' in result.columns
    assert 'frequency' in result.columns
    assert 'relative_frequency' in result.columns
    assert len(result) == 2  # M y F

def test_summary_statistics_empty_columns():
    """Test para manejo de columnas vacías"""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    
    with pytest.raises(ValueError, match="La lista de columnas no puede estar vacía"):
        summary_statistics(df, [])

def test_compute_correlations_invalid_method():
    """Test para método de correlación inválido"""
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    
    with pytest.raises(ValueError, match="Método debe ser uno de"):
        compute_correlations(df, ['a', 'b'], method='invalid')

def test_contingency_analysis_invalid_columns():
    """Test para columnas inexistentes"""
    df = pd.DataFrame({'a': [1, 2, 3]})
    
    with pytest.raises(ValueError, match="Las columnas especificadas no existen"):
        contingency_analysis(df, 'a', 'inexistente') 