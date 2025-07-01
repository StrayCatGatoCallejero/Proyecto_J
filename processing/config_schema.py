"""
Modelos Pydantic para validación de configuración del sistema.

Este módulo define esquemas estrictos para validar la estructura y valores
del archivo config.yml, asegurando que toda la aplicación arranque con
una configuración válida y coherente.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator, model_validator
import re
from pathlib import Path


class DataTypesConfig(BaseModel):
    """Configuración de tipos de datos soportados"""
    numeric: List[str] = Field(default_factory=lambda: ["int64", "float64", "int32", "float32"])
    categorical: List[str] = Field(default_factory=lambda: ["object", "category", "string"])
    temporal: List[str] = Field(default_factory=lambda: ["datetime64[ns]", "datetime64[us]", "datetime64[ms]"])
    boolean: List[str] = Field(default_factory=lambda: ["bool", "boolean"])


class ValidationRange(BaseModel):
    """Rango de validación para un tipo de dato"""
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    unit: str = Field(..., min_length=1, max_length=20)
    
    @validator('min', 'max')
    def validate_range_values(cls, v):
        if v is not None and not isinstance(v, (int, float)):
            raise ValueError("Los valores min/max deben ser numéricos")
        return v
    
    @model_validator(mode="after")
    def validate_range_logic(cls, values):
        min_val = values.get('min')
        max_val = values.get('max')
        if min_val is not None and max_val is not None and min_val >= max_val:
            raise ValueError("El valor mínimo debe ser menor que el máximo")
        return values


class ValidationPatterns(BaseModel):
    """Patrones de validación para formatos específicos"""
    email: str = Field(..., pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    phone: str = Field(..., pattern=r'^[+]?[0-9]{8,15}$')
    dni: str = Field(..., pattern=r'^[0-9]{7,8}[A-Z]$')


class SchemaConfig(BaseModel):
    """Configuración del esquema de datos"""
    data_types: 'DataTypesConfig' = Field(default_factory=lambda: DataTypesConfig())
    validation_ranges: Dict[str, 'ValidationRange'] = Field(default_factory=dict)
    patterns: Dict[str, str] = Field(default_factory=dict)


class ImputationConfig(BaseModel):
    """Configuración de métodos de imputación"""
    numeric: str = Field(..., pattern=r'^(median|mean|mode|forward_fill|backward_fill)$')
    categorical: str = Field(..., pattern=r'^(mode|most_frequent|constant)$')
    temporal: str = Field(..., pattern=r'^(forward_fill|backward_fill|interpolate)$')
    boolean: str = Field(..., pattern=r'^(mode|most_frequent|constant)$')


class CorrelationConfig(BaseModel):
    """Configuración de análisis de correlación"""
    default_method: str = Field(..., pattern=r'^(pearson|spearman|kendall)$')
    alternatives: List[str] = Field(..., min_items=1, max_items=5, default_factory=list)
    min_correlation: float = Field(..., ge=0.0, le=1.0)
    significance_level: float = Field(..., gt=0.0, lt=1.0)
    
    @validator('alternatives')
    def validate_alternatives(cls, v):
        valid_methods = ['pearson', 'spearman', 'kendall']
        for method in v:
            if method not in valid_methods:
                raise ValueError(f"Método de correlación inválido: {method}")
        return v


class ChiSquareConfig(BaseModel):
    """Configuración de prueba chi-cuadrado"""
    min_expected_frequency: int = Field(..., ge=1, le=100)
    significance_level: float = Field(..., gt=0.0, lt=1.0)


class NormalizationConfig(BaseModel):
    """Configuración de normalización"""
    similarity_threshold: float = Field(..., ge=0.0, le=1.0)
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)


class MethodsConfig(BaseModel):
    """Configuración de métodos estadísticos"""
    imputation: Dict[str, str] = Field(default_factory=dict)
    correlation: Dict[str, Any] = Field(default_factory=dict)
    chi_square: Dict[str, Any] = Field(default_factory=dict)
    normalization: Dict[str, float] = Field(default_factory=dict)
    
    @validator('correlation')
    def validate_correlation(cls, v):
        if 'default_method' in v and v['default_method'] not in ['pearson', 'spearman', 'kendall']:
            raise ValueError("Método de correlación inválido")
        if 'min_correlation' in v and not (0.0 <= v['min_correlation'] <= 1.0):
            raise ValueError("min_correlation debe estar entre 0.0 y 1.0")
        return v


class ColorsConfig(BaseModel):
    """Configuración de colores de la UI"""
    primary: str = Field(..., pattern=r'^#[0-9A-Fa-f]{6}$')
    secondary: str = Field(..., pattern=r'^#[0-9A-Fa-f]{6}$')
    success: str = Field(..., pattern=r'^#[0-9A-Fa-f]{6}$')
    warning: str = Field(..., pattern=r'^#[0-9A-Fa-f]{6}$')
    info: str = Field(..., pattern=r'^#[0-9A-Fa-f]{6}$')


class IconsConfig(BaseModel):
    """Configuración de iconos de la UI"""
    data_load: str = Field(..., min_length=1, max_length=10)
    validation: str = Field(..., min_length=1, max_length=10)
    analysis: str = Field(..., min_length=1, max_length=10)
    visualization: str = Field(..., min_length=1, max_length=10)
    export: str = Field(..., min_length=1, max_length=10)
    error: str = Field(..., min_length=1, max_length=10)
    warning: str = Field(..., min_length=1, max_length=10)
    info: str = Field(..., min_length=1, max_length=10)


class ThresholdsConfig(BaseModel):
    """Umbrales de la interfaz de usuario"""
    max_rows_display: int = Field(..., ge=100, le=100000)
    max_columns_display: int = Field(..., ge=5, le=100)
    chunk_size: int = Field(..., ge=1000, le=1000000)
    timeout_seconds: int = Field(..., ge=5, le=300)


class UIConfig(BaseModel):
    """Configuración de la interfaz de usuario"""
    colors: Dict[str, str] = Field(default_factory=dict)
    icons: Dict[str, str] = Field(default_factory=dict)
    thresholds: Dict[str, int] = Field(default_factory=dict)
    
    @validator('thresholds')
    def validate_thresholds(cls, v):
        if 'max_rows_display' in v and not (100 <= v['max_rows_display'] <= 100000):
            raise ValueError("max_rows_display debe estar entre 100 y 100000")
        if 'timeout_seconds' in v and not (5 <= v['timeout_seconds'] <= 300):
            raise ValueError("timeout_seconds debe estar entre 5 y 300")
        return v


class LoggingConfig(BaseModel):
    """Configuración de logging"""
    level: str = Field(..., pattern=r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$')
    format: str = Field(..., min_length=10, max_length=200)
    file: str = Field(..., min_length=1, max_length=200)
    max_file_size: str = Field(..., pattern=r'^\d+[KMGT]?B$')
    backup_count: int = Field(..., ge=0, le=100)
    
    @validator('file')
    def validate_file_path(cls, v):
        # Validar que la ruta sea válida
        try:
            Path(v).parent
        except Exception:
            raise ValueError("Ruta de archivo de log inválida")
        return v


class ExportConfig(BaseModel):
    """Configuración de exportación"""
    formats: List[str] = Field(..., min_items=1, max_items=10, default_factory=list)
    compression: bool = True
    include_metadata: bool = True
    include_logs: bool = True
    
    @validator('formats')
    def validate_formats(cls, v):
        valid_formats = ['csv', 'excel', 'json', 'html', 'pdf', 'png', 'jpg']
        for fmt in v:
            if fmt not in valid_formats:
                raise ValueError(f"Formato de exportación no soportado: {fmt}")
        return v


class SchemaValidationConfig(BaseModel):
    """Configuración de validación de esquema"""
    strict: bool = False
    allow_extra_columns: bool = True
    allow_missing_columns: bool = False


class IntegrityConfig(BaseModel):
    """Configuración de validación de integridad"""
    check_duplicates: bool = True
    check_outliers: bool = True
    outlier_threshold: float = Field(..., ge=1.0, le=10.0)


class ConsistencyConfig(BaseModel):
    """Configuración de validación de consistencia"""
    age_marital_check: bool = True
    age_education_check: bool = True
    income_education_check: bool = True


class ValidationConfig(BaseModel):
    """Configuración de validación"""
    schema: Dict[str, Any] = Field(default_factory=dict)
    integrity: Dict[str, Any] = Field(default_factory=dict)
    consistency: Dict[str, bool] = Field(default_factory=dict)
    
    @validator('integrity')
    def validate_integrity(cls, v):
        if 'outlier_threshold' in v and not (1.0 <= v['outlier_threshold'] <= 10.0):
            raise ValueError("outlier_threshold debe estar entre 1.0 y 10.0")
        return v


class KeywordsConfig(BaseModel):
    """Configuración de palabras clave semánticas"""
    demographic: List[str] = Field(..., min_items=1, max_items=50, default_factory=list)
    socioeconomic: List[str] = Field(..., min_items=1, max_items=50, default_factory=list)
    opinion: List[str] = Field(..., min_items=1, max_items=50, default_factory=list)
    likert: List[str] = Field(..., min_items=1, max_items=50, default_factory=list)


class ClassificationConfig(BaseModel):
    """Configuración de clasificación semántica"""
    similarity_threshold: float = Field(..., ge=0.0, le=1.0)
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)


class SemanticConfig(BaseModel):
    """Configuración de análisis semántico"""
    keywords: Dict[str, List[str]] = Field(default_factory=dict)
    classification: Dict[str, float] = Field(default_factory=dict)
    
    @validator('classification')
    def validate_classification(cls, v):
        if 'similarity_threshold' in v and not (0.0 <= v['similarity_threshold'] <= 1.0):
            raise ValueError("similarity_threshold debe estar entre 0.0 y 1.0")
        if 'confidence_threshold' in v and not (0.0 <= v['confidence_threshold'] <= 1.0):
            raise ValueError("confidence_threshold debe estar entre 0.0 y 1.0")
        return v


class ChartTypesConfig(BaseModel):
    """Configuración de tipos de gráficos"""
    demographic: List[str] = Field(..., min_items=1, max_items=20)
    numeric: List[str] = Field(..., min_items=1, max_items=20)
    temporal: List[str] = Field(..., min_items=1, max_items=20)
    likert: List[str] = Field(..., min_items=1, max_items=20)
    
    @validator('demographic', 'numeric', 'temporal', 'likert')
    def validate_chart_types(cls, v):
        valid_charts = [
            'bar_chart', 'pie_chart', 'horizontal_bar', 'histogram', 
            'box_plot', 'scatter_plot', 'line_chart', 'area_chart', 
            'heatmap', 'stacked_bar', 'diverging_bar'
        ]
        for chart in v:
            if chart not in valid_charts:
                raise ValueError(f"Tipo de gráfico no soportado: {chart}")
        return v


class VisualizationConfig(BaseModel):
    """Configuración de visualización"""
    chart_types: Dict[str, List[str]] = Field(default_factory=dict)


class NotificationsConfig(BaseModel):
    """Configuración de notificaciones"""
    enabled: bool = False
    slack_webhook: Optional[str] = None
    email_recipients: List[str] = Field(default_factory=list)
    
    @validator('slack_webhook')
    def validate_slack_webhook(cls, v):
        if v is not None and not v.startswith('https://hooks.slack.com/'):
            raise ValueError("URL de Slack webhook inválida")
        return v
    
    @validator('email_recipients')
    def validate_email_recipients(cls, v):
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        for email in v:
            if not email_pattern.match(email):
                raise ValueError(f"Email inválido: {email}")
        return v


class SystemConfigSchema(BaseModel):
    """Esquema completo de configuración del sistema"""
    schema: SchemaConfig = Field(default_factory=SchemaConfig)
    methods: MethodsConfig = Field(default_factory=MethodsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    
    class Config:
        extra = "forbid"  # No permitir campos adicionales no definidos 