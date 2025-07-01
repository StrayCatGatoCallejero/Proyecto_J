"""
Tests para el sistema de configuración centralizada
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import tempfile
import yaml
from processing.config_manager import (
    ConfigManager,
    SystemConfig,
    ValidationConfig,
    MethodsConfig,
    UIConfig,
    get_config,
    get_validation_config,
    get_methods_config
)

@pytest.fixture
def temp_config_file():
    """Crea un archivo de configuración temporal para testing"""
    config_data = {
        'validation': {
            'schema': {
                'strict': True,
                'allow_extra_columns': False
            },
            'integrity': {
                'check_duplicates': True,
                'outlier_threshold': 2.5
            }
        },
        'methods': {
            'correlation': {
                'default_method': 'spearman',
                'min_correlation': 0.2
            },
            'chi_square': {
                'min_expected_frequency': 10
            }
        },
        'ui': {
            'thresholds': {
                'max_rows_display': 500,
                'chunk_size': 5000
            },
            'colors': {
                'primary': '#ff0000',
                'secondary': '#00ff00'
            }
        },
        'logging': {
            'level': 'DEBUG',
            'file': 'test.log'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        yaml.dump(config_data, f)
        yield f.name
    
    # Limpiar archivo temporal
    os.unlink(f.name)

def test_config_manager_singleton():
    """Test que verifica que ConfigManager implementa el patrón Singleton"""
    manager1 = ConfigManager()
    manager2 = ConfigManager()
    
    assert manager1 is manager2

def test_load_config_from_file(temp_config_file):
    """Test que verifica la carga de configuración desde archivo"""
    manager = ConfigManager()
    config = manager.load_config(temp_config_file)
    
    assert isinstance(config, SystemConfig)
    assert config.validation.strict is True
    assert config.validation.allow_extra_columns is False
    assert config.methods.correlation_default == 'spearman'
    assert config.methods.min_correlation == 0.2
    assert config.ui.max_rows_display == 500

def test_config_defaults():
    """Test que verifica que se usan valores por defecto cuando no están en el archivo"""
    manager = ConfigManager()
    # Forzar recarga para obtener configuración por defecto
    config = manager.reload_config("default")
    
    # Verificar valores por defecto
    assert config.validation.strict is False
    assert config.methods.correlation_default == 'pearson'
    assert config.ui.max_rows_display == 1000

def test_get_config_function():
    """Test que verifica la función helper get_config()"""
    config = get_config()
    
    assert isinstance(config, SystemConfig)
    assert hasattr(config, 'validation')
    assert hasattr(config, 'methods')
    assert hasattr(config, 'ui')

def test_get_validation_config():
    """Test que verifica la función helper get_validation_config()"""
    validation_config = get_validation_config()
    
    assert isinstance(validation_config, ValidationConfig)
    assert hasattr(validation_config, 'strict')
    assert hasattr(validation_config, 'outlier_threshold')

def test_get_methods_config():
    """Test que verifica la función helper get_methods_config()"""
    methods_config = get_methods_config()
    
    assert isinstance(methods_config, MethodsConfig)
    assert hasattr(methods_config, 'correlation_default')
    assert hasattr(methods_config, 'significance_level')

def test_config_reload(temp_config_file):
    """Test que verifica la recarga de configuración"""
    manager = ConfigManager()
    
    # Cargar configuración por defecto
    config1 = manager.reload_config("default")
    
    # Recargar con archivo temporal
    config2 = manager.reload_config(temp_config_file)
    
    # Verificar que son diferentes
    assert config1.validation.strict != config2.validation.strict
    assert config1.methods.correlation_default != config2.methods.correlation_default

def test_config_metadata():
    """Test que verifica los metadatos de configuración"""
    manager = ConfigManager()
    config = manager.load_config()
    
    assert hasattr(config, 'config_version')
    assert hasattr(config, 'loaded_at')
    assert hasattr(config, 'config_file')
    assert config.config_version == "1.0.0"

def test_config_structure():
    """Test que verifica la estructura completa de configuración"""
    config = get_config()
    
    # Verificar que todas las secciones existen
    assert hasattr(config, 'validation')
    assert hasattr(config, 'methods')
    assert hasattr(config, 'ui')
    assert hasattr(config, 'logging')
    assert hasattr(config, 'visualization')
    assert hasattr(config, 'export')
    assert hasattr(config, 'semantic')
    
    # Verificar tipos de datos
    assert isinstance(config.validation, ValidationConfig)
    assert isinstance(config.methods, MethodsConfig)
    assert isinstance(config.ui, UIConfig)

def test_config_file_not_found():
    """Test que verifica el manejo de archivo de configuración no encontrado"""
    manager = ConfigManager()
    
    # Este test ya no debería fallar porque usamos configuración por defecto
    config = manager.load_config("archivo_inexistente.yml")
    assert isinstance(config, SystemConfig)

def test_invalid_yaml_config():
    """Test que verifica el manejo de YAML inválido"""
    # Crear archivo YAML realmente inválido
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write("invalid: yaml: content: [\n  unclosed: bracket")
        temp_file = f.name
    
    try:
        # Crear una nueva instancia para este test
        manager = ConfigManager()
        manager._config = None  # Forzar recarga
        
        with pytest.raises(ValueError):
            manager.load_config(temp_file)
    finally:
        os.unlink(temp_file) 