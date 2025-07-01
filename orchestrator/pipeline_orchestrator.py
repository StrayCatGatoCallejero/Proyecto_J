"""
Pipeline Orchestrator Mejorado con Logging JSON Avanzado
=======================================================

Sistema completo de orquestación de pipelines con logging JSON estructurado,
métricas detalladas, manejo robusto de errores y trazabilidad end-to-end.
"""

import uuid
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import traceback
import sys
import os

# Importar sistema de logging JSON
from processing.json_logging import create_json_logger, LogLevel, LogCategory, serialize_for_json
from processing.io import DataLoader
from processing.types import SchemaValidator
from processing.filters import DataFilter
from processing.visualization import VisualizationGenerator
from processing.business_rules import validate_business_rules


class PipelineStatus(Enum):
    """Estados del pipeline"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Estados de los pasos"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepMetrics:
    """Métricas detalladas de un paso"""
    start_time: float = 0.0
    end_time: float = 0.0
    execution_time: float = 0.0
    memory_before: float = 0.0
    memory_after: float = 0.0
    memory_delta: float = 0.0
    rows_before: int = 0
    rows_after: int = 0
    columns_before: int = 0
    columns_after: int = 0
    data_size_before: float = 0.0
    data_size_after: float = 0.0
    errors_count: int = 0
    warnings_count: int = 0
    success_rate: float = 1.0


@dataclass
class PipelineMetrics:
    """Métricas completas del pipeline"""
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    total_execution_time: float = 0.0
    total_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    data_processing_efficiency: float = 0.0
    error_rate: float = 0.0
    step_metrics: Dict[str, StepMetrics] = field(default_factory=dict)


@dataclass
class SessionData:
    """Datos de la sesión del pipeline"""
    df: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    reports: Dict[str, Any] = field(default_factory=dict)
    visualizations: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)


def log_pipeline_step(step_name: str, capture_metrics: bool = True):
    """
    Decorador avanzado para logging de pasos del pipeline.
    
    Args:
        step_name: Nombre del paso
        capture_metrics: Si capturar métricas detalladas
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            step_id = f"{step_name}_{int(time.time())}"
            start_time = time.time()
            
            # Métricas antes
            before_metrics = {}
            if capture_metrics and hasattr(self, 'session') and self.session.df is not None:
                before_metrics = self._compute_detailed_metrics(self.session.df)
            
            # Log de inicio del paso
            self.json_logger.log_event(
                level=LogLevel.INFO,
                message=f"Iniciando paso: {step_name}",
                module=self.__class__.__module__,
                function=func.__name__,
                step=step_name,
                category=LogCategory.PROCESSING.value,
                parameters={"args": str(args), "kwargs": str(kwargs), "step_id": step_id},
                before_metrics=before_metrics,
                after_metrics=before_metrics,  # Mismo valor al inicio
                execution_time=0.0,
                tags=["pipeline_step", step_name.lower().replace(" ", "_"), "start"],
                metadata={"step_id": step_id}
            )
            
            try:
                # Actualizar estado del paso
                self._update_step_status(step_name, StepStatus.RUNNING)
                
                # Ejecutar función
                result = func(self, *args, **kwargs)
                
                # Métricas después
                after_metrics = {}
                if capture_metrics and hasattr(self, 'session') and self.session.df is not None:
                    after_metrics = self._compute_detailed_metrics(self.session.df)
                
                execution_time = time.time() - start_time
                
                # Actualizar métricas del pipeline
                self._update_pipeline_metrics(step_name, execution_time, before_metrics, after_metrics)
                
                # Log de éxito
                self.json_logger.log_event(
                    level=LogLevel.INFO,
                    message=f"Paso completado exitosamente: {step_name}",
                    module=self.__class__.__module__,
                    function=func.__name__,
                    step=step_name,
                    category=LogCategory.PROCESSING.value,
                    parameters={"args": str(args), "kwargs": str(kwargs), "step_id": step_id},
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    execution_time=execution_time,
                    tags=["pipeline_step", step_name.lower().replace(" ", "_"), "success"],
                    metadata={
                        "step_id": step_id,
                        "success": True,
                        "result_type": type(result).__name__
                    }
                )
                
                # Actualizar estado del paso
                self._update_step_status(step_name, StepStatus.COMPLETED)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Capturar error detallado
                error_details = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "stack_trace": traceback.format_exc(),
                    "step_id": step_id
                }
                
                # Log de error
                self.json_logger.log_error(
                    function=func.__name__,
                    error=e,
                    context=f"pipeline_step_{step_name.lower().replace(' ', '_')}",
                    execution_time=execution_time,
                    additional_data={
                        "step_name": step_name,
                        "step_id": step_id,
                        "args": str(args),
                        "kwargs": str(kwargs),
                        "before_metrics": before_metrics
                    }
                )
                
                # Registrar error en la sesión
                self.session.errors.append(error_details)
                
                # Actualizar estado del paso
                self._update_step_status(step_name, StepStatus.FAILED)
                
                # Actualizar métricas del pipeline
                self.pipeline_metrics.failed_steps += 1
                self.pipeline_metrics.error_rate = self.pipeline_metrics.failed_steps / self.pipeline_metrics.total_steps
                
                raise
        
        return wrapper
    
    return decorator


class PipelineOrchestrator:
    """
    Orquestador de pipeline mejorado con logging JSON avanzado.
    
    Características:
    - Logging JSON estructurado con métricas detalladas
    - Trazabilidad completa con session_id
    - Manejo robusto de errores y recuperación
    - Métricas de rendimiento y memoria
    - Estados de pipeline y pasos
    - Configuración flexible
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el orquestador del pipeline.
        
        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.session_id = str(uuid.uuid4())
        self.session = SessionData()
        self.pipeline_metrics = PipelineMetrics()
        self.step_statuses = {}
        self.pipeline_status = PipelineStatus.INITIALIZED
        
        # Inicializar componentes
        self.data_loader = DataLoader()
        self.schema_validator = SchemaValidator()
        self.data_filter = DataFilter()
        self.viz_generator = VisualizationGenerator()
        
        # Inicializar JSON Logger
        log_conf = config.get("logging", {}).get("json_logging", {})
        if log_conf.get("enabled", True):
            self.json_logger = create_json_logger(config, self.session_id)
        else:
            # Logger dummy si está deshabilitado
            self.json_logger = DummyJsonLogger(self.session_id)
        
        # Log de inicialización
        self.json_logger.log_system_event(
            level=LogLevel.INFO,
            message="Pipeline Orchestrator initialized",
            metadata={
                "session_id": self.session_id,
                "config_path": config.get("config_path", "default"),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform
            }
        )
        
        print(f"🚀 Pipeline Orchestrator inicializado - Session ID: {self.session_id}")
    
    def _compute_detailed_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcula métricas detalladas de un DataFrame.
        
        Args:
            df: DataFrame a analizar
            
        Returns:
            Diccionario con métricas detalladas
        """
        try:
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            
            metrics = {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": memory_usage,
                "data_types": df.dtypes.value_counts().to_dict(),
                "missing_values": df.isnull().sum().sum(),
                "missing_ratio": df.isnull().sum().sum() / (len(df) * len(df.columns)),
                "duplicate_rows": df.duplicated().sum(),
                "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
                "datetime_columns": len(df.select_dtypes(include=['datetime']).columns)
            }
            
            # Métricas adicionales para columnas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                metrics.update({
                    "numeric_stats": {
                        "mean": df[numeric_cols].mean().to_dict(),
                        "std": df[numeric_cols].std().to_dict(),
                        "min": df[numeric_cols].min().to_dict(),
                        "max": df[numeric_cols].max().to_dict()
                    }
                })
            
            return metrics
            
        except Exception as e:
            return {
                "error": f"Error computing metrics: {str(e)}",
                "rows": len(df) if df is not None else 0,
                "columns": len(df.columns) if df is not None else 0
            }
    
    def _update_step_status(self, step_name: str, status: StepStatus):
        """Actualiza el estado de un paso."""
        self.step_statuses[step_name] = {
            "status": status.value,
            "timestamp": datetime.now().isoformat(),
            "step_name": step_name
        }
    
    def _update_pipeline_metrics(self, step_name: str, execution_time: float, 
                                before_metrics: Dict, after_metrics: Dict):
        """Actualiza las métricas del pipeline."""
        self.pipeline_metrics.total_execution_time += execution_time
        self.pipeline_metrics.completed_steps += 1
        
        # Crear métricas del paso
        step_metrics = StepMetrics(
            execution_time=execution_time,
            memory_before=before_metrics.get("memory_usage_mb", 0),
            memory_after=after_metrics.get("memory_usage_mb", 0),
            rows_before=before_metrics.get("rows", 0),
            rows_after=after_metrics.get("rows", 0),
            columns_before=before_metrics.get("columns", 0),
            columns_after=after_metrics.get("columns", 0)
        )
        step_metrics.memory_delta = step_metrics.memory_after - step_metrics.memory_before
        
        self.pipeline_metrics.step_metrics[step_name] = step_metrics
        
        # Actualizar métricas globales
        self.pipeline_metrics.peak_memory_usage = max(
            self.pipeline_metrics.peak_memory_usage, 
            step_metrics.memory_after
        )
    
    @log_pipeline_step("Data Loading", capture_metrics=True)
    def load_data(self, path: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo con logging detallado.
        
        Args:
            path: Ruta al archivo de datos
            
        Returns:
            DataFrame cargado
        """
        # Log específico de carga de datos
        file_size = os.path.getsize(path) if os.path.exists(path) else 0
        
        self.json_logger.log_event(
            level=LogLevel.INFO,
            message=f"Cargando datos desde: {path}",
            module=self.__class__.__module__,
            function="load_data",
            step="data_loading",
            category=LogCategory.DATA_LOAD.value,
            parameters={"file_path": path, "file_size_bytes": file_size},
            before_metrics={"file_size_mb": file_size / (1024 * 1024)},
            after_metrics={"file_size_mb": file_size / (1024 * 1024)},
            execution_time=0.0,
            tags=["data_load", "file_operation"],
            metadata={"file_path": path}
        )
        
        # Cargar datos
        df = self.data_loader.load_file(path)
        
        if df is not None:
            self.session.df = df
            self.session.metadata["source_file"] = path
            self.session.metadata["file_format"] = self.data_loader.detected_format
            self.session.metadata["original_shape"] = df.shape
            
            # Log de carga exitosa
            self.json_logger.log_event(
                level=LogLevel.INFO,
                message=f"Datos cargados exitosamente: {df.shape}",
                module=self.__class__.__module__,
                function="load_data",
                step="data_loading",
                category=LogCategory.DATA_LOAD.value,
                parameters={"file_path": path},
                before_metrics={"file_size_mb": file_size / (1024 * 1024)},
                after_metrics={
                    "rows": len(df),
                    "columns": len(df.columns),
                    "file_format": self.data_loader.detected_format,
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
                },
                execution_time=0.0,
                tags=["data_load", "success"],
                metadata={"file_path": path}
            )
            
            return df
        else:
            raise ValueError(f"No se pudieron cargar los datos desde: {path}")
    
    @log_pipeline_step("Schema Validation", capture_metrics=True)
    def validate_schema(self, schema: Optional[Dict] = None) -> bool:
        """
        Valida el esquema de datos con logging detallado.
        
        Args:
            schema: Esquema opcional para validación
            
        Returns:
            True si la validación es exitosa
        """
        if self.session.df is None:
            raise ValueError("No hay datos cargados para validar")
        
        # Log de inicio de validación
        self.json_logger.log_validation(
            function="validate_schema",
            validation_type="schema_validation",
            total_checks=len(self.session.df.columns),
            passed_checks=0,  # Se calculará después
            failed_checks=0,  # Se calculará después
            execution_time=0.0,
            details={"schema_provided": schema is not None}
        )
        
        # Ejecutar validación
        validation_result = self.schema_validator.validate(self.session.df, schema)
        
        # Contar resultados
        total_checks = len(validation_result)
        passed_checks = sum(1 for result in validation_result.values() if result.get('is_valid', False))
        failed_checks = total_checks - passed_checks
        
        # Log de resultados
        self.json_logger.log_event(
            level=LogLevel.INFO if failed_checks == 0 else LogLevel.WARNING,
            message=f"Validación de esquema completada: {passed_checks}/{total_checks} pasaron",
            module=self.__class__.__module__,
            function="validate_schema",
            step="schema_validation",
            category=LogCategory.VALIDATION.value,
            parameters={"schema_provided": schema is not None},
            before_metrics={"total_columns": len(self.session.df.columns)},
            after_metrics={
                "validation_passed": passed_checks,
                "validation_failed": failed_checks,
                "success_rate": passed_checks / total_checks if total_checks > 0 else 0
            },
            execution_time=0.0,
            tags=["validation", "schema"],
            metadata=validation_result
        )
        
        return failed_checks == 0
    
    @log_pipeline_step("Business Rules Validation", capture_metrics=True)
    def validate_business_rules(self) -> bool:
        """
        Valida reglas de negocio con logging detallado.
        
        Returns:
            True si la validación es exitosa
        """
        if self.session.df is None:
            raise ValueError("No hay datos cargados para validar reglas de negocio")
        
        # Log de inicio de validación de reglas de negocio
        self.json_logger.log_business_rules(
            function="validate_business_rules",
            rules_executed=0,  # Se calculará después
            rules_failed=0,  # Se calculará después
            rules_warnings=0,  # Se calculará después
            execution_time=0.0,
            details={"dataset_shape": self.session.df.shape}
        )
        
        # Ejecutar validaciones
        metadata = {'dataset_type': 'social_sciences'}
        validation_results = validate_business_rules(self.session.df, metadata)
        
        # Contar resultados
        total_rules = len(validation_results)
        failed_rules = sum(1 for r in validation_results if not r.is_valid)
        warning_rules = len([r for r in validation_results if r.details.get('alertas_generadas', 0) > 0])
        
        # Log de resultados
        self.json_logger.log_event(
            level=LogLevel.INFO if failed_rules == 0 else LogLevel.ERROR,
            message=f"Validación de reglas de negocio: {total_rules - failed_rules}/{total_rules} pasaron",
            module=self.__class__.__module__,
            function="validate_business_rules",
            step="business_rules_validation",
            category=LogCategory.BUSINESS_RULES.value,
            parameters={"total_rules": total_rules},
            before_metrics={"total_rules": total_rules},
            after_metrics={
                "rules_failed": failed_rules,
                "rules_warnings": warning_rules,
                "success_rate": (total_rules - failed_rules) / total_rules if total_rules > 0 else 0
            },
            execution_time=0.0,
            tags=["business_rules", "validation"],
            metadata={
                "validation_results": [r.rule_name for r in validation_results],
                "failed_rules": [r.rule_name for r in validation_results if not r.is_valid],
                "warning_rules": [r.rule_name for r in validation_results if r.details.get('alertas_generadas', 0) > 0]
            }
        )
        
        return failed_rules == 0
    
    @log_pipeline_step("Data Filtering", capture_metrics=True)
    def apply_filters(self, filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Aplica filtros a los datos con logging detallado.
        
        Args:
            filters: Filtros a aplicar
            
        Returns:
            DataFrame filtrado
        """
        if self.session.df is None:
            raise ValueError("No hay datos cargados para filtrar")
        
        initial_rows = len(self.session.df)
        
        # Log de inicio de filtrado
        self.json_logger.log_event(
            level=LogLevel.INFO,
            message=f"Aplicando filtros: {len(filters) if filters else 0} filtros",
            module=self.__class__.__module__,
            function="apply_filters",
            step="data_filtering",
            category=LogCategory.PROCESSING.value,
            parameters={"filters_count": len(filters) if filters else 0},
            before_metrics={"initial_rows": initial_rows},
            after_metrics={"initial_rows": initial_rows},
            execution_time=0.0,
            tags=["filtering", "data_processing"],
            metadata={"filters": filters}
        )
        
        # Aplicar filtros
        if filters:
            self.session.df = self.data_filter.apply_filters(self.session.df, filters)
        
        final_rows = len(self.session.df)
        rows_removed = initial_rows - final_rows
        
        # Log de filtrado completado
        self.json_logger.log_event(
            level=LogLevel.INFO,
            message=f"Filtrado completado: {rows_removed} filas removidas",
            module=self.__class__.__module__,
            function="apply_filters",
            step="data_filtering",
            category=LogCategory.PROCESSING.value,
            parameters={"filters_applied": len(filters) if filters else 0},
            before_metrics={"initial_rows": initial_rows},
            after_metrics={
                "final_rows": final_rows,
                "rows_removed": rows_removed,
                "removal_percentage": (rows_removed / initial_rows * 100) if initial_rows > 0 else 0
            },
            execution_time=0.0,
            tags=["filtering", "data_processing", "success"],
            metadata={"filters": filters}
        )
        
        return self.session.df
    
    @log_pipeline_step("Statistical Analysis", capture_metrics=True)
    def run_statistical_analysis(self) -> Dict[str, Any]:
        """
        Ejecuta análisis estadístico con logging detallado.
        
        Returns:
            Resultados del análisis estadístico
        """
        if self.session.df is None:
            raise ValueError("No hay datos cargados para análisis estadístico")
        
        # Log de inicio de análisis
        self.json_logger.log_analysis(
            function="run_statistical_analysis",
            analysis_type="statistical_analysis",
            input_size=len(self.session.df),
            output_size=0,  # Se calculará después
            execution_time=0.0,
            success=True,
            results={"dataset_size": len(self.session.df)}
        )
        
        # Ejecutar análisis estadístico
        analysis_results = {
            "summary_stats": self._compute_summary_statistics(),
            "correlations": self._compute_correlations(),
            "distributions": self._compute_distributions()
        }
        
        # Log de análisis completado
        self.json_logger.log_event(
            level=LogLevel.INFO,
            message="Análisis estadístico completado exitosamente",
            module=self.__class__.__module__,
            function="run_statistical_analysis",
            step="statistical_analysis",
            category=LogCategory.ANALYSIS.value,
            parameters={},
            before_metrics={"dataset_size": len(self.session.df)},
            after_metrics={"analysis_types": len(analysis_results)},
            execution_time=0.0,
            tags=["analysis", "statistical"],
            metadata=analysis_results
        )
        
        self.session.reports["statistical_analysis"] = analysis_results
        return analysis_results
    
    def _compute_summary_statistics(self) -> Dict[str, Any]:
        """Calcula estadísticas resumen."""
        if self.session.df is None:
            return {}
        
        numeric_cols = self.session.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {"message": "No hay columnas numéricas para análisis"}
        
        return {
            "descriptive_stats": self.session.df[numeric_cols].describe().to_dict(),
            "missing_values": self.session.df[numeric_cols].isnull().sum().to_dict(),
            "skewness": self.session.df[numeric_cols].skew().to_dict(),
            "kurtosis": self.session.df[numeric_cols].kurtosis().to_dict()
        }
    
    def _compute_correlations(self) -> Dict[str, Any]:
        """Calcula correlaciones."""
        if self.session.df is None:
            return {}
        
        numeric_cols = self.session.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {"message": "Se necesitan al menos 2 columnas numéricas para correlación"}
        
        corr_matrix = self.session.df[numeric_cols].corr()
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": self._find_strong_correlations(corr_matrix),
            "correlation_summary": {
                "mean_correlation": corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
                "max_correlation": corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max(),
                "min_correlation": corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()
            }
        }
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Encuentra correlaciones fuertes."""
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_corr.append({
                        "variable1": corr_matrix.columns[i],
                        "variable2": corr_matrix.columns[j],
                        "correlation": corr_value
                    })
        return strong_corr
    
    def _compute_distributions(self) -> Dict[str, Any]:
        """Calcula distribuciones."""
        if self.session.df is None:
            return {}
        
        numeric_cols = self.session.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {"message": "No hay columnas numéricas para análisis de distribución"}
        
        distributions = {}
        for col in numeric_cols:
            distributions[col] = {
                "histogram": np.histogram(self.session.df[col].dropna(), bins=10),
                "percentiles": self.session.df[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
            }
        
        return distributions
    
    @log_pipeline_step("Visualization Generation", capture_metrics=True)
    def generate_visualizations(self) -> Dict[str, Any]:
        """
        Genera visualizaciones con logging detallado.
        
        Returns:
            Visualizaciones generadas
        """
        if self.session.df is None:
            raise ValueError("No hay datos cargados para generar visualizaciones")
        
        # Log de inicio de generación de visualizaciones
        self.json_logger.log_analysis(
            function="generate_visualizations",
            analysis_type="visualization_generation",
            input_size=len(self.session.df.columns),
            output_size=0,  # Se calculará después
            execution_time=0.0,
            success=True,
            results={"variables_to_visualize": len(self.session.df.columns)}
        )
        
        # Generar visualizaciones
        visualizations = self.viz_generator.generate_visualizations(self.session.df, self.config.get("visualization", {}))
        
        # Log de generación completada
        self.json_logger.log_event(
            level=LogLevel.INFO,
            message=f"Generación de visualizaciones completada: {len(visualizations)} tipos generados",
            module=self.__class__.__module__,
            function="generate_visualizations",
            step="visualization_generation",
            category=LogCategory.VISUALIZATION.value,
            parameters={},
            before_metrics={"variables": len(self.session.df.columns)},
            after_metrics={"visualization_types": len(visualizations)},
            execution_time=0.0,
            tags=["visualization", "generation"],
            metadata=visualizations
        )
        
        self.session.visualizations = visualizations
        return visualizations
    
    def run_pipeline(self, path: str, filters: Optional[Dict] = None, 
                    schema: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo con logging detallado.
        
        Args:
            path: Ruta al archivo de datos
            filters: Filtros opcionales
            schema: Esquema opcional
            
        Returns:
            Resultados del pipeline
        """
        pipeline_start_time = time.time()
        self.pipeline_status = PipelineStatus.RUNNING
        
        # Log de inicio del pipeline
        self.json_logger.log_system_event(
            level=LogLevel.INFO,
            message="Pipeline execution started",
            metadata={
                "file_path": path,
                "filters_provided": filters is not None,
                "schema_provided": schema is not None,
                "session_id": self.session_id
            }
        )
        
        try:
            # Definir pasos del pipeline
            steps = [
                ("Data Loading", lambda: self.load_data(path)),
                ("Schema Validation", lambda: self.validate_schema(schema)),
                ("Business Rules Validation", lambda: self.validate_business_rules()),
                ("Data Filtering", lambda: self.apply_filters(filters)),
                ("Statistical Analysis", lambda: self.run_statistical_analysis()),
                ("Visualization Generation", lambda: self.generate_visualizations()),
            ]
            
            self.pipeline_metrics.total_steps = len(steps)
            
            # Ejecutar pasos
            for step_name, step_function in steps:
                try:
                    step_function()
                    self.pipeline_metrics.completed_steps += 1
                except Exception as e:
                    self.pipeline_metrics.failed_steps += 1
                    self.pipeline_status = PipelineStatus.FAILED
                    raise
            
            # Pipeline completado exitosamente
            pipeline_execution_time = time.time() - pipeline_start_time
            self.pipeline_status = PipelineStatus.COMPLETED
            
            # Log de finalización
            self.json_logger.log_system_event(
                level=LogLevel.INFO,
                message="Pipeline execution completed successfully",
                metadata={
                    "total_steps": self.pipeline_metrics.total_steps,
                    "completed_steps": self.pipeline_metrics.completed_steps,
                    "failed_steps": self.pipeline_metrics.failed_steps,
                    "pipeline_execution_time": pipeline_execution_time,
                    "session_id": self.session_id
                }
            )
            
            return self.get_pipeline_results()
            
        except Exception as e:
            pipeline_execution_time = time.time() - pipeline_start_time
            self.pipeline_status = PipelineStatus.FAILED
            
            # Log de error en el pipeline
            self.json_logger.log_error(
                function="run_pipeline",
                error=e,
                context="full_pipeline_execution",
                execution_time=pipeline_execution_time,
                additional_data={
                    "file_path": path,
                    "filters_provided": filters is not None,
                    "schema_provided": schema is not None
                }
            )
            
            raise
    
    def get_pipeline_results(self) -> Dict[str, Any]:
        """
        Obtiene los resultados completos del pipeline.
        
        Returns:
            Diccionario con todos los resultados
        """
        # Preparar resultados
        results = {
            "session_id": self.session_id,
            "pipeline_status": self.pipeline_status.value,
            "pipeline_metrics": {
                "total_steps": self.pipeline_metrics.total_steps,
                "completed_steps": self.pipeline_metrics.completed_steps,
                "failed_steps": self.pipeline_metrics.failed_steps,
                "total_execution_time": self.pipeline_metrics.total_execution_time,
                "error_rate": self.pipeline_metrics.error_rate
            },
            "data_info": {
                "shape": self.session.df.shape if self.session.df is not None else None,
                "memory_usage_mb": self.session.df.memory_usage(deep=True).sum() / (1024 * 1024) if self.session.df is not None else 0
            },
            "reports": self.session.reports,
            "visualizations": list(self.session.visualizations.keys()) if self.session.visualizations else [],
            "errors": self.session.errors,
            "warnings": self.session.warnings,
            "step_statuses": self.step_statuses
        }
        
        # Limpiar todos los metadatos para serialización JSON
        cleaned_results = serialize_for_json(results)
        
        return cleaned_results
    
    def export_results(self, output_path: str) -> bool:
        """
        Exporta los resultados del pipeline.
        
        Args:
            output_path: Ruta de exportación
            
        Returns:
            True si la exportación fue exitosa
        """
        try:
            # Log de inicio de exportación
            self.json_logger.log_event(
                level=LogLevel.INFO,
                message=f"Exporting results to: {output_path}",
                module=self.__class__.__module__,
                function="export_results",
                step="export",
                category=LogCategory.EXPORT.value,
                parameters={"output_path": output_path},
                before_metrics={"reports_count": len(self.session.reports)},
                after_metrics={"reports_count": len(self.session.reports)},
                execution_time=0.0,
                tags=["export", "results"]
            )
            
            # Aquí iría la lógica de exportación
            # Por ahora, solo exportamos los logs JSON
            logs_path = f"logs/pipeline_{self.session_id}.json"
            
            # Log de exportación completada
            self.json_logger.log_event(
                level=LogLevel.INFO,
                message="Results export completed",
                module=self.__class__.__module__,
                function="export_results",
                step="export",
                category=LogCategory.EXPORT.value,
                parameters={"output_path": output_path},
                before_metrics={"reports_count": len(self.session.reports)},
                after_metrics={"logs_exported": True},
                execution_time=0.0,
                tags=["export", "results", "success"]
            )
            
            return True
            
        except Exception as e:
            self.json_logger.log_error(
                function="export_results",
                error=e,
                context="results_export",
                execution_time=0.0,
                additional_data={"output_path": output_path}
            )
            return False


class DummyJsonLogger:
    """Logger dummy para cuando el logging JSON está deshabilitado"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
