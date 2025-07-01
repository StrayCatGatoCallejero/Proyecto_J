"""
Tests de integración para EDA (Exploratory Data Analysis)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from processing.io import cargar_archivo
from processing.stats import (
    summary_statistics,
    compute_correlations,
    contingency_analysis,
    normality_test
)

@pytest.fixture
def sample_csv_data(tmp_path):
    """Crea un archivo CSV de prueba"""
    np.random.seed(42)
    data = {
        'edad': np.random.normal(35, 10, 100),
        'ingresos': np.random.normal(50000, 15000, 100),
        'genero': np.random.choice(['M', 'F'], 100),
        'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universitaria'], 100),
        'satisfaccion': np.random.randint(1, 6, 100)
    }
    df = pd.DataFrame(data)
    
    # Guardar como CSV
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

def test_integration_data_loading_and_analysis(sample_csv_data):
    """Test de integración: carga de datos + análisis estadístico"""
    # 1. Cargar datos
    df, metadata = cargar_archivo(sample_csv_data)
    
    assert isinstance(df, pd.DataFrame)
    assert isinstance(metadata, dict)
    assert len(df) == 100
    assert 'edad' in df.columns
    
    # 2. Estadísticas descriptivas
    stats_result = summary_statistics(df, ['edad', 'ingresos'])
    
    assert isinstance(stats_result, pd.DataFrame)
    assert len(stats_result) == 2
    assert 'edad' in stats_result['variable'].values
    
    # 3. Correlaciones
    corr_result = compute_correlations(df, ['edad', 'ingresos'])
    
    assert isinstance(corr_result, pd.DataFrame)
    assert corr_result.shape == (2, 2)
    
    # 4. Análisis de contingencia
    table, stats = contingency_analysis(df, 'genero', 'educacion')
    
    assert isinstance(table, pd.DataFrame)
    assert isinstance(stats, dict)
    assert 'chi2' in stats
    
    # 5. Prueba de normalidad
    normality_result = normality_test(df, 'edad')
    
    assert isinstance(normality_result, dict)
    assert 'shapiro_wilk' in normality_result
    assert 'kolmogorov_smirnov' in normality_result
    assert 'is_normal' in normality_result['shapiro_wilk']

def test_integration_error_handling():
    """Test de integración: manejo de errores"""
    # Test con archivo inexistente
    with pytest.raises(FileNotFoundError):
        cargar_archivo("archivo_inexistente.csv")
    
    # Test con DataFrame vacío
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        summary_statistics(empty_df, ['a'])

def test_integration_data_validation():
    """Test de integración: validación de datos"""
    # Crear datos con problemas
    problematic_data = {
        'edad': [25, 30, np.nan, 35],
        'ingresos': [50000, 60000, 55000, np.nan],
        'genero': ['M', 'F', 'M', 'F']
    }
    df = pd.DataFrame(problematic_data)
    
    # Debería manejar nulos correctamente
    stats_result = summary_statistics(df, ['edad', 'ingresos'])
    
    assert isinstance(stats_result, pd.DataFrame)
    # Debería tener menos filas que el original debido a nulos
    assert len(stats_result) <= 2

def test_integration_metadata_consistency(sample_csv_data):
    """Test de integración: consistencia de metadatos"""
    df, metadata = cargar_archivo(sample_csv_data)
    
    # Verificar que los metadatos sean consistentes
    assert metadata['n_rows'] == len(df)
    assert metadata['n_cols'] == len(df.columns)
    assert metadata['format'] == 'csv'
    assert 'columns' in metadata 