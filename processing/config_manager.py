"""
Gestor de Configuración Centralizada - Patrón "Reloj Suizo"
=========================================================

Responsabilidades:
- Carga y validación de configuración desde YAML
- Dependency Injection para todos los módulos
- Gestión de valores por defecto y validaciones
- Contexto de ejecución centralizado
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Importar esquemas de validación y sistema de errores
from .config_schema import SystemConfigSchema
from .error_reporter import report_config_error, set_notifications_enabled

@dataclass
class ValidationConfig:
    """Configuración de validación de datos"""
    strict: bool = False
    allow_extra_columns: bool = True
    allow_missing_columns: bool = False
    check_duplicates: bool = True
    check_outliers: bool = True
    outlier_threshold: float = 3.0

@dataclass
class MethodsConfig:
    """Configuración de métodos estadísticos"""
    correlation_default: str = "pearson"
    correlation_alternatives: list = field(default_factory=lambda: ["spearman", "kendall"])
    min_correlation: float = 0.1
    significance_level: float = 0.05
    chi_square_min_expected: int = 5
    normalization_similarity: float = 0.8
    normalization_confidence: float = 0.7

@dataclass
class UIConfig:
    """Configuración de interfaz de usuario"""
    max_rows_display: int = 1000
    max_columns_display: int = 20
    chunk_size: int = 10000
    timeout_seconds: int = 30
    colors: Dict[str, str] = field(default_factory=dict)
    icons: Dict[str, str] = field(default_factory=dict)

@dataclass
class LoggingConfig:
    """Configuración de logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/pipeline.log"
    max_file_size: str = "10MB"
    backup_count: int = 5

@dataclass
class VisualizationConfig:
    """Configuración de visualización"""
    chart_types: Dict[str, list] = field(default_factory=dict)
    default_figure_size: tuple = (10, 6)
    dpi: int = 100
    style: str = "default"

@dataclass
class ExportConfig:
    """Configuración de exportación"""
    formats: list = field(default_factory=lambda: ["csv", "excel", "json", "html"])
    compression: bool = True
    include_metadata: bool = True
    include_logs: bool = True

@dataclass
class SemanticConfig:
    """Configuración de análisis semántico"""
    keywords: Dict[str, list] = field(default_factory=dict)
    similarity_threshold: float = 0.7
    confidence_threshold: float = 0.6

@dataclass
class SystemConfig:
    """Configuración completa del sistema"""
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    methods: MethodsConfig = field(default_factory=MethodsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    
    # Metadatos de configuración
    config_version: str = "1.0.0"
    loaded_at: datetime = field(default_factory=datetime.now)
    config_file: str = ""

class ConfigManager:
    """
    Gestor centralizado de configuración con Dependency Injection y validación Pydantic.
    
    Implementa el patrón Singleton para asegurar una única instancia
    de configuración en toda la aplicación, con validación estricta
    de la configuración usando Pydantic.
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Optional[SystemConfig] = None
    
    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._config: Optional[SystemConfig] = None
            self._logger: Optional[logging.Logger] = None
    
    def load_config(self, config_path: Optional[str] = None) -> SystemConfig:
        """
        Carga y valida la configuración desde archivo YAML usando Pydantic.
        
        Args:
            config_path: Ruta al archivo de configuración (opcional)
            
        Returns:
            SystemConfig: Configuración cargada y validada
            
        Raises:
            ValidationError: Si la configuración es inválida
        """
        if self._config is not None:
            return self._config
        
        # Determinar ruta de configuración
        if config_path is None:
            config_path = self._find_config_file()
        
        try:
            # Cargar configuración YAML
            config_data = self._load_yaml(config_path)
            
            # Validar configuración usando Pydantic
            validated_config = self._validate_config_with_pydantic(config_data, config_path)
            
            # Crear instancia de configuración del sistema
            self._config = self._create_config_instance(validated_config)
            self._config.config_file = config_path
            
            # Configurar logging
            self._setup_logging()
            
            # Configurar notificaciones de errores
            if hasattr(validated_config, 'notifications'):
                set_notifications_enabled(validated_config.notifications.enabled)
            
            if self._logger is not None:
                self._logger.info(f"Configuración validada y cargada desde: {config_path}")
            
            return self._config
            
        except Exception as e:
            # Reportar error de configuración
            report_config_error(
                message=f"Error al cargar configuración: {str(e)}",
                context=f"load_config({config_path})",
                details={
                    "config_path": config_path,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            # El error_reporter detendrá la aplicación
    
    def _validate_config_with_pydantic(self, config_data: Dict[str, Any], config_path: str) -> SystemConfigSchema:
        """
        Valida la configuración usando Pydantic.
        
        Args:
            config_data: Datos de configuración cargados del YAML
            config_path: Ruta del archivo de configuración
            
        Returns:
            SystemConfigSchema: Configuración validada
            
        Raises:
            ValidationError: Si la configuración es inválida
        """
        try:
            # Validar con Pydantic
            validated_config = SystemConfigSchema(**config_data)
            return validated_config
            
        except Exception as e:
            # Reportar error de validación específico
            error_details = {
                "config_path": config_path,
                "validation_error": str(e),
                "config_data_keys": list(config_data.keys()) if config_data else []
            }
            
            if hasattr(e, 'errors'):
                error_details["pydantic_errors"] = e.errors()
            
            report_config_error(
                message=f"Configuración inválida en {config_path}",
                context="validate_config_with_pydantic",
                details=error_details
            )
            # El error_reporter detendrá la aplicación
    
    def _load_yaml(self, config_path: str) -> Dict[str, Any]:
        """Carga archivo YAML con manejo de errores mejorado"""
        if config_path == "default":
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
                if not isinstance(config_data, dict):
                    raise ValueError("El archivo YAML debe contener un diccionario en la raíz")
                return config_data
                
        except FileNotFoundError:
            report_config_error(
                message=f"Archivo de configuración no encontrado: {config_path}",
                context="load_yaml",
                details={"config_path": config_path, "error_type": "FileNotFoundError"}
            )
        except yaml.YAMLError as e:
            report_config_error(
                message=f"Error al parsear YAML: {str(e)}",
                context="load_yaml",
                details={"config_path": config_path, "error_type": "YAMLError", "yaml_error": str(e)}
            )
        except Exception as e:
            report_config_error(
                message=f"Error inesperado al cargar configuración: {str(e)}",
                context="load_yaml",
                details={"config_path": config_path, "error_type": type(e).__name__, "error": str(e)}
            )
    
    def _create_config_instance(self, validated_config: SystemConfigSchema) -> SystemConfig:
        """Crea instancia de SystemConfig desde configuración validada"""
        
        # Configuración de validación
        validation_config = ValidationConfig(
            strict=validated_config.validation.schema.get('strict', False),
            allow_extra_columns=validated_config.validation.schema.get('allow_extra_columns', True),
            allow_missing_columns=validated_config.validation.schema.get('allow_missing_columns', False),
            check_duplicates=validated_config.validation.integrity.get('check_duplicates', True),
            check_outliers=validated_config.validation.integrity.get('check_outliers', True),
            outlier_threshold=validated_config.validation.integrity.get('outlier_threshold', 3.0)
        )
        
        # Configuración de métodos
        methods_config = MethodsConfig(
            correlation_default=validated_config.methods.correlation.get('default_method', 'pearson'),
            correlation_alternatives=validated_config.methods.correlation.get('alternatives', ['spearman', 'kendall']),
            min_correlation=validated_config.methods.correlation.get('min_correlation', 0.1),
            significance_level=validated_config.methods.correlation.get('significance_level', 0.05),
            chi_square_min_expected=validated_config.methods.chi_square.get('min_expected_frequency', 5),
            normalization_similarity=validated_config.methods.normalization.get('similarity_threshold', 0.8),
            normalization_confidence=validated_config.methods.normalization.get('confidence_threshold', 0.7)
        )
        
        # Configuración de UI
        ui_config = UIConfig(
            max_rows_display=validated_config.ui.thresholds.get('max_rows_display', 1000),
            max_columns_display=validated_config.ui.thresholds.get('max_columns_display', 20),
            chunk_size=validated_config.ui.thresholds.get('chunk_size', 10000),
            timeout_seconds=validated_config.ui.thresholds.get('timeout_seconds', 30),
            colors=validated_config.ui.colors,
            icons=validated_config.ui.icons
        )
        
        # Configuración de logging
        logging_config = LoggingConfig(
            level=validated_config.logging.level,
            format=validated_config.logging.format,
            file=validated_config.logging.file,
            max_file_size=validated_config.logging.max_file_size,
            backup_count=validated_config.logging.backup_count
        )
        
        # Configuración de visualización
        visualization_config = VisualizationConfig(
            chart_types=validated_config.visualization.chart_types,
            default_figure_size=(10, 6),
            dpi=100,
            style='default'
        )
        
        # Configuración de exportación
        export_config = ExportConfig(
            formats=validated_config.export.formats,
            compression=validated_config.export.compression,
            include_metadata=validated_config.export.include_metadata,
            include_logs=validated_config.export.include_logs
        )
        
        # Configuración semántica
        semantic_config = SemanticConfig(
            keywords=validated_config.semantic.keywords,
            similarity_threshold=validated_config.semantic.classification.get('similarity_threshold', 0.7),
            confidence_threshold=validated_config.semantic.classification.get('confidence_threshold', 0.6)
        )
        
        return SystemConfig(
            validation=validation_config,
            methods=methods_config,
            ui=ui_config,
            logging=logging_config,
            visualization=visualization_config,
            export=export_config,
            semantic=semantic_config
        )
    
    def get_config(self) -> SystemConfig:
        """Obtiene la configuración cargada"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración por clave.
        
        Args:
            key: Clave de configuración (ej: 'app_name', 'validation.strict')
            default: Valor por defecto si no se encuentra la clave
            
        Returns:
            Valor de configuración o default
        """
        if self._config is None:
            self.load_config()
        
        # Buscar en la configuración usando notación de punto
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except (AttributeError, KeyError, TypeError):
            return default
    
    def reload_config(self, config_path: Optional[str] = None) -> SystemConfig:
        """Recarga la configuración desde archivo"""
        # Forzar recarga limpiando la instancia
        self._config = None
        return self.load_config(config_path)
    
    def _find_config_file(self) -> str:
        """Encuentra el archivo de configuración por defecto"""
        possible_paths = [
            "config/config.yml",
            "config.yml",
            "../config/config.yml",
            "../../config/config.yml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Si no se encuentra ningún archivo, usar configuración por defecto
        return "default"
    
    def _setup_logging(self) -> None:
        """Configura el sistema de logging"""
        if self._config is None:
            return
            
        config = self._config.logging
        
        # Crear directorio de logs si no existe
        log_dir = os.path.dirname(config.file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configurar logging
        logging.basicConfig(
            level=getattr(logging, config.level.upper()),
            format=config.format,
            handlers=[
                logging.FileHandler(config.file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self._logger = logging.getLogger(__name__)

# Instancia global para Dependency Injection
config_manager = ConfigManager()

def get_config() -> SystemConfig:
    """Función helper para obtener configuración desde cualquier módulo"""
    return config_manager.get_config()

def get_validation_config() -> ValidationConfig:
    """Obtiene configuración de validación"""
    return get_config().validation

def get_methods_config() -> MethodsConfig:
    """Obtiene configuración de métodos"""
    return get_config().methods

def get_ui_config() -> UIConfig:
    """Obtiene configuración de UI"""
    return get_config().ui

def get_logging_config() -> LoggingConfig:
    """Obtiene configuración de logging"""
    return get_config().logging

def get_visualization_config() -> VisualizationConfig:
    """Obtiene configuración de visualización"""
    return get_config().visualization

def get_export_config() -> ExportConfig:
    """Obtiene configuración de exportación"""
    return get_config().export

def get_semantic_config() -> SemanticConfig:
    """Obtiene configuración semántica"""
    return get_config().semantic 