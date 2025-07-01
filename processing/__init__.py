"""
Módulo de Procesamiento - Patrón "Reloj Suizo"
=============================================

Este módulo contiene todas las funcionalidades de procesamiento de datos
siguiendo el patrón arquitectónico "Reloj Suizo" con responsabilidades
bien definidas y separadas.
"""

from .io import cargar_archivo, validar_dataframe, obtener_info_archivo
from .stats import (
    summary_statistics,
    compute_correlations,
    contingency_analysis,
    normality_test,
    t_test_independent,
    linear_regression,
    frequency_table,
    outlier_detection
)
from .features import (
    create_numeric_features,
    encode_categorical,
    scale_features,
    create_interaction_features,
    select_features,
    create_time_features,
    create_binning_features
)
from .filters import (
    filter_by_condition,
    filter_by_range,
    filter_by_values,
    remove_outliers,
    handle_missing_values,
    sample_data,
    select_columns,
    drop_duplicates,
    apply_custom_filter
)
from .visualization import (
    plot_histogram,
    plot_boxplot,
    plot_scatter,
    plot_heatmap,
    plot_bar_chart,
    plotly_histogram,
    plotly_heatmap,
    plotly_scatter
)
from .logging import log_action
from .config_manager import (
    get_config,
    get_validation_config,
    get_methods_config,
    get_ui_config,
    get_logging_config,
    get_visualization_config,
    get_export_config,
    get_semantic_config,
    SystemConfig,
    ValidationConfig,
    MethodsConfig,
    UIConfig,
    LoggingConfig,
    VisualizationConfig,
    ExportConfig,
    SemanticConfig
)

__all__ = [
    # IO
    'cargar_archivo',
    'validar_dataframe', 
    'obtener_info_archivo',
    
    # Stats
    'summary_statistics',
    'compute_correlations',
    'contingency_analysis',
    'normality_test',
    't_test_independent',
    'linear_regression',
    'frequency_table',
    'outlier_detection',
    
    # Features
    'create_numeric_features',
    'encode_categorical',
    'scale_features',
    'create_interaction_features',
    'select_features',
    'create_time_features',
    'create_binning_features',
    
    # Filters
    'filter_by_condition',
    'filter_by_range',
    'filter_by_values',
    'remove_outliers',
    'handle_missing_values',
    'sample_data',
    'select_columns',
    'drop_duplicates',
    'apply_custom_filter',
    
    # Visualization
    'plot_histogram',
    'plot_boxplot',
    'plot_scatter',
    'plot_heatmap',
    'plot_bar_chart',
    'plotly_histogram',
    'plotly_heatmap',
    'plotly_scatter',
    
    # Logging
    'log_action',
    
    # Configuration
    'get_config',
    'get_validation_config',
    'get_methods_config',
    'get_ui_config',
    'get_logging_config',
    'get_visualization_config',
    'get_export_config',
    'get_semantic_config',
    'SystemConfig',
    'ValidationConfig',
    'MethodsConfig',
    'UIConfig',
    'LoggingConfig',
    'VisualizationConfig',
    'ExportConfig',
    'SemanticConfig'
]
