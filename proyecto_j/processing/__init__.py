"""
Proyecto J - Módulo de Procesamiento de Datos

Este paquete proporciona funcionalidades avanzadas para:
- Validación de esquemas y tipos de datos
- Filtros con validación y logging
- Verificación de integridad de datos
- Limpieza y transformación de datos
"""

from .schema_validator import validate_schema, SchemaValidationError
from .filter_manager import apply_filters, revert_last, FilterManager
from .integrity_checker import validate_integrity, IntegrityWarning
from .data_cleaner import clean_data, DataCleaningReport
from .logging_utils import log_action, get_action_log

__version__ = "1.0.0"
__author__ = "Proyecto J Team"

__all__ = [
    # Schema validation
    'validate_schema',
    'SchemaValidationError',
    
    # Filter management
    'apply_filters', 
    'revert_last',
    'FilterManager',
    
    # Integrity checking
    'validate_integrity',
    'IntegrityWarning',
    
    # Data cleaning
    'clean_data',
    'DataCleaningReport',
    
    # Logging
    'log_action',
    'get_action_log'
] 