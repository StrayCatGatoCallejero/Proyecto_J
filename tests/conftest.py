"""
Configuración pytest para Proyecto J
Fixtures y configuración común para todos los tests
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Agregar el directorio raíz al path para importaciones
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def sample_data():
    """Datos de ejemplo para tests"""
    return pd.DataFrame({
        'edad': [25, 30, 35, 40, 45, 50, 55, 60],
        'ingresos': [30000, 45000, 55000, 65000, 75000, 85000, 95000, 105000],
        'genero': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'region': ['RM', 'Valparaiso', 'Bio Bio', 'RM', 'Valparaiso', 'Bio Bio', 'RM', 'Valparaiso'],
        'educacion': ['Basica', 'Media', 'Superior', 'Superior', 'Media', 'Superior', 'Basica', 'Media'],
        'satisfaccion': [7, 8, 6, 9, 7, 8, 6, 9]
    })


@pytest.fixture(scope="session")
def sample_data_large():
    """Datos de ejemplo más grandes para tests de rendimiento"""
    np.random.seed(42)
    n = 1000
    
    return pd.DataFrame({
        'id': range(1, n + 1),
        'edad': np.random.randint(18, 80, n),
        'ingresos': np.random.randint(20000, 150000, n),
        'genero': np.random.choice(['M', 'F'], n),
        'region': np.random.choice(['RM', 'Valparaiso', 'Bio Bio', 'Maule', 'Araucania'], n),
        'educacion': np.random.choice(['Basica', 'Media', 'Superior', 'Postgrado'], n),
        'satisfaccion': np.random.randint(1, 11, n),
        'fecha': pd.date_range('2020-01-01', periods=n, freq='D')
    })


@pytest.fixture(scope="session")
def sample_data_missing():
    """Datos con valores faltantes para tests de limpieza"""
    df = pd.DataFrame({
        'edad': [25, 30, None, 40, 45, None, 55, 60],
        'ingresos': [30000, None, 55000, 65000, None, 85000, 95000, None],
        'genero': ['M', 'F', 'M', None, 'M', 'F', None, 'F'],
        'region': ['RM', None, 'Bio Bio', 'RM', 'Valparaiso', None, 'RM', 'Valparaiso'],
        'educacion': ['Basica', 'Media', None, 'Superior', None, 'Superior', 'Basica', 'Media'],
        'satisfaccion': [7, None, 6, 9, 7, None, 6, 9]
    })
    return df


@pytest.fixture(scope="session")
def temp_dir():
    """Directorio temporal para tests"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="session")
def test_config():
    """Configuración de prueba"""
    return {
        'input': {
            'path': 'tests/fixtures/sample_data.csv',
            'format': 'csv',
            'encoding': 'utf-8'
        },
        'processing': {
            'clean_data': True,
            'validate_types': True,
            'handle_missing': True
        },
        'output': {
            'format': 'pdf',
            'path': 'tests/fixtures/expected_outputs/',
            'include_charts': True
        },
        'logging': {
            'level': 'INFO',
            'format': 'json',
            'file': 'tests/fixtures/test.log'
        }
    }


@pytest.fixture(scope="function")
def mock_streamlit():
    """Mock de Streamlit para tests"""
    class MockStreamlit:
        def __init__(self):
            self.session_state = {}
            self.calls = []
        
        def title(self, text):
            self.calls.append(('title', text))
        
        def header(self, text):
            self.calls.append(('header', text))
        
        def subheader(self, text):
            self.calls.append(('subheader', text))
        
        def write(self, text):
            self.calls.append(('write', text))
        
        def dataframe(self, data):
            self.calls.append(('dataframe', data))
        
        def plotly_chart(self, fig):
            self.calls.append(('plotly_chart', fig))
        
        def sidebar(self):
            return self
        
        def selectbox(self, label, options, **kwargs):
            self.calls.append(('selectbox', (label, options)))
            return options[0] if options else None
        
        def file_uploader(self, label, **kwargs):
            self.calls.append(('file_uploader', label))
            return None
        
        def button(self, label, **kwargs):
            self.calls.append(('button', label))
            return False
        
        def success(self, text):
            self.calls.append(('success', text))
        
        def error(self, text):
            self.calls.append(('error', text))
        
        def warning(self, text):
            self.calls.append(('warning', text))
        
        def info(self, text):
            self.calls.append(('info', text))
    
    return MockStreamlit()


@pytest.fixture(scope="session")
def chile_geographic_data():
    """Datos geográficos de Chile para tests de validación"""
    return {
        'regiones': {
            'RM': 'Región Metropolitana de Santiago',
            'Valparaiso': 'Región de Valparaíso',
            'Bio Bio': 'Región del Biobío',
            'Maule': 'Región del Maule',
            'Araucania': 'Región de La Araucanía'
        },
        'comunas': {
            'Santiago': 'RM',
            'Providencia': 'RM',
            'Valparaiso': 'Valparaiso',
            'Concepcion': 'Bio Bio',
            'Temuco': 'Araucania'
        }
    }


@pytest.fixture(scope="function")
def sample_csv_file(temp_dir):
    """Archivo CSV de ejemplo para tests"""
    df = pd.DataFrame({
        'edad': [25, 30, 35, 40],
        'ingresos': [30000, 45000, 55000, 65000],
        'genero': ['M', 'F', 'M', 'F'],
        'region': ['RM', 'Valparaiso', 'Bio Bio', 'RM']
    })
    
    csv_path = os.path.join(temp_dir, 'sample_data.csv')
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="function")
def sample_excel_file(temp_dir):
    """Archivo Excel de ejemplo para tests"""
    df = pd.DataFrame({
        'edad': [25, 30, 35, 40],
        'ingresos': [30000, 45000, 55000, 65000],
        'genero': ['M', 'F', 'M', 'F'],
        'region': ['RM', 'Valparaiso', 'Bio Bio', 'RM']
    })
    
    excel_path = os.path.join(temp_dir, 'sample_data.xlsx')
    df.to_excel(excel_path, index=False)
    return excel_path


@pytest.fixture(scope="session")
def expected_outputs():
    """Salidas esperadas para tests de validación"""
    return {
        'estadisticas_basicas': {
            'count': 8,
            'mean_edad': 42.5,
            'mean_ingresos': 67500.0,
            'unique_genero': 2,
            'unique_region': 3
        },
        'validacion_chile': {
            'regiones_validas': True,
            'comunas_validas': True,
            'codigos_validos': True
        }
    }


# Configuración de pytest
def pytest_configure(config):
    """Configuración adicional de pytest"""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modificar items de colección para agregar marcadores automáticos"""
    for item in items:
        # Marcar tests por ubicación
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Marcar tests lentos por nombre
        if "performance" in item.name or "large" in item.name:
            item.add_marker(pytest.mark.slow) 