"""
Sistema de Logging JSON Mejorado
===============================

Sistema de logging JSON robusto, independiente y fácil de mantener.
Proporciona trazabilidad completa con formato estructurado para sistemas de monitoreo.
"""

import logging
import uuid
import os
import json
import time
import sys
import platform
import psutil
import threading
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import numpy as np
import pandas as pd


def serialize_for_json(obj: Any) -> Any:
    """
    Convierte cualquier objeto a tipos serializables por JSON.
    
    Args:
        obj: Objeto a serializar
        
    Returns:
        Objeto serializable por JSON
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.datetime64,)):
        return str(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        # Convertir Series a dict con claves como strings
        return {str(k): serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Index):
        return obj.tolist()
    elif hasattr(obj, "to_dict"):
        try:
            return serialize_for_json(obj.to_dict())
        except:
            return str(obj)
    elif hasattr(obj, "__dict__"):
        try:
            return serialize_for_json(obj.__dict__)
        except:
            return str(obj)
    else:
        return str(obj)


class LogLevel:
    """Niveles de log estandarizados"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Categorías de eventos para clasificación"""
    SYSTEM = "system"
    DATA_LOAD = "data_load"
    VALIDATION = "validation"
    PROCESSING = "processing"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    EXPORT = "export"
    BUSINESS_RULES = "business_rules"
    ERROR_HANDLING = "error_handling"


@dataclass
class SystemInfo:
    """Información del sistema para contexto"""
    platform: str = platform.platform()
    python_version: str = sys.version
    cpu_count: int = psutil.cpu_count()
    memory_total: int = psutil.virtual_memory().total
    process_id: int = os.getpid()


class JSONFormatter(logging.Formatter):
    """Formateador personalizado para logs JSON"""
    
    def format(self, record):
        """Formatea el record como JSON"""
        try:
            # Si el mensaje ya es JSON, parsearlo
            if isinstance(record.msg, str) and record.msg.strip().startswith('{'):
                try:
                    return record.msg
                except:
                    pass
            
            # Crear estructura JSON básica
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "logger_name": record.name
            }
            
            # Agregar información adicional si está disponible
            if hasattr(record, 'session_id'):
                log_entry['session_id'] = record.session_id
            
            if hasattr(record, 'event_type'):
                log_entry['event_type'] = record.event_type
            
            if hasattr(record, 'metadata'):
                log_entry['metadata'] = record.metadata
            
            return json.dumps(log_entry, ensure_ascii=False, default=serialize_for_json)
            
        except Exception as e:
            # Fallback a formato simple
            return f'{{"timestamp": "{datetime.now(timezone.utc).isoformat()}", "level": "ERROR", "message": "Error formatting log: {str(e)}"}}'


def to_serializable(obj):
    """
    Convierte objetos no serializables (numpy, pandas, plotly, etc.) a tipos serializables por JSON.
    """
    # Numpy types
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.datetime64,)):
        return str(obj)
    # Pandas types
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Index):
        return obj.tolist()
    # Plotly Figure - manejar referencias circulares
    try:
        import plotly.graph_objs as go
        if isinstance(obj, go.Figure):
            # Extraer solo la información esencial para evitar referencias circulares
            return {
                "type": "plotly_figure",
                "layout_title": obj.layout.title.text if hasattr(obj.layout, 'title') and obj.layout.title else None,
                "data_count": len(obj.data),
                "data_types": [trace.type for trace in obj.data] if obj.data else []
            }
    except ImportError:
        pass
    # Otros objetos con método to_dict
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    # Otros objetos con método __dict__
    if hasattr(obj, "__dict__"):
        try:
            # Filtrar atributos que podrían causar referencias circulares
            safe_dict = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_') and not callable(value):
                    try:
                        # Intentar serializar cada valor
                        json.dumps(value, default=lambda x: str(x))
                        safe_dict[key] = value
                    except:
                        safe_dict[key] = str(value)
            return safe_dict
        except Exception:
            pass
    # Fallback: string
    return str(obj)


class JsonLogger:
    """
    Logger JSON robusto y configurable.
    
    Características:
    - Rotación automática de archivos
    - Formato JSON estructurado
    - Métodos especializados para diferentes tipos de eventos
    - Configuración flexible
    - Compatibilidad con sistemas de monitoreo
    """
    
    def __init__(self, file_path: str, level: str, session_id: str, 
                 rotation: Optional[Dict[str, Any]] = None,
                 console_output: bool = False):
        """
        Inicializa el logger JSON.
        
        Args:
            file_path: Ruta al archivo de logs
            level: Nivel de logging
            session_id: ID único de la sesión
            rotation: Configuración de rotación
            console_output: Mostrar logs en consola
        """
        self.session_id = session_id
        self.system_info = SystemInfo()
        self.lock = threading.Lock()
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Configurar logger
        self.logger = logging.getLogger(f"pipeline_{session_id}")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Limpiar handlers existentes
        self.logger.handlers.clear()
        
        # Handler para archivo con rotación
        if rotation and rotation.get('enabled', False):
            if rotation.get('max_bytes'):
                # Rotación por tamaño
                max_bytes = self._parse_size(rotation['max_bytes'])
                handler = RotatingFileHandler(
                    filename=file_path,
                    maxBytes=max_bytes,
                    backupCount=rotation.get('backup_count', 7),
                    encoding="utf-8"
                )
            else:
                # Rotación por tiempo
                handler = TimedRotatingFileHandler(
                    filename=file_path,
                    when=rotation.get('when', 'midnight'),
                    interval=rotation.get('interval', 1),
                    backupCount=rotation.get('backup_count', 7),
                    encoding="utf-8"
                )
        else:
            # Handler simple sin rotación
            handler = logging.FileHandler(file_path, encoding="utf-8")
        
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
        
        # Handler para consola si está habilitado
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(console_handler)
    
    def _parse_size(self, size_str: str) -> int:
        """Convierte string de tamaño a bytes"""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def _write_log(self, log_event):
        """Escribe el log usando el logger estándar de Python"""
        try:
            with self.lock:
                # Limpiar todos los campos del log_event para asegurar serialización JSON
                cleaned_log_event = serialize_for_json(log_event)
                
                # Agregar campos estándar
                cleaned_log_event.update({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": self.session_id,
                    "system_info": serialize_for_json(asdict(self.system_info))
                })
                
                # Serializar y escribir (ahora sin default porque ya está limpio)
                log_line = json.dumps(cleaned_log_event, ensure_ascii=False)
                
                # Usar el logger estándar
                level = getattr(logging, cleaned_log_event.get('level', 'INFO').upper())
                self.logger.log(level, log_line)
                
        except Exception as e:
            print(f"[JsonLogger] Error serializando log: {e}")
            import traceback
            traceback.print_exc()
    
    def log_system_event(self, level: str, message: str, metadata: Optional[Dict] = None):
        """
        Registra un evento del sistema.
        
        Args:
            level: Nivel del log
            message: Mensaje del evento
            metadata: Metadatos adicionales
        """
        self._write_log({
            "level": level,
            "event": "system",
            "message": message,
            "metadata": metadata or {}
        })
    
    def log_event(self,
                  level: str,
                  message: str,
                  module: str,
                  function: str,
                  step: str,
                  category: str,
                  parameters: Optional[Dict] = None,
                  before_metrics: Optional[Dict] = None,
                  after_metrics: Optional[Dict] = None,
                  execution_time: Optional[float] = None,
                  tags: Optional[List[str]] = None,
                  metadata: Optional[Dict] = None):
        """
        Registra un evento estructurado.
        
        Args:
            level: Nivel del log
            message: Mensaje del evento
            module: Módulo donde ocurrió
            function: Función que generó el evento
            step: Paso del pipeline
            category: Categoría del evento
            parameters: Parámetros de entrada
            before_metrics: Métricas antes de la ejecución
            after_metrics: Métricas después de la ejecución
            execution_time: Tiempo de ejecución
            tags: Tags para clasificación
            metadata: Metadatos adicionales
        """
        self._write_log({
            "level": level,
            "event": "step",
            "message": message,
            "module": module,
            "function": function,
            "step": step,
            "category": category,
            "parameters": parameters or {},
            "before_metrics": before_metrics or {},
            "after_metrics": after_metrics or {},
            "execution_time": execution_time,
            "tags": tags or [],
            "metadata": metadata or {}
        })
    
    def log_data_load(self,
                     function: str,
                     file_path: str,
                     file_size: int,
                     rows: int,
                     columns: int,
                     execution_time: float,
                     success: bool,
                     error_details: Optional[str] = None):
        """
        Registra un evento de carga de datos.
        
        Args:
            function: Función de carga
            file_path: Ruta del archivo
            file_size: Tamaño del archivo
            rows: Número de filas
            columns: Número de columnas
            execution_time: Tiempo de ejecución
            success: Si la carga fue exitosa
            error_details: Detalles del error si aplica
        """
        level = LogLevel.INFO if success else LogLevel.ERROR
        
        self.log_event(
            level=level,
            message=f"Data load {'completed' if success else 'failed'}: {file_path}",
            module="data_loader",
            function=function,
            step="data_load",
            category=LogCategory.DATA_LOAD.value,
            parameters={"file_path": file_path, "file_size": file_size},
            before_metrics={"file_size": file_size},
            after_metrics={"rows": rows, "columns": columns},
            execution_time=execution_time,
            tags=["data_load", "file_operation"],
            metadata={"error_details": error_details} if error_details else {}
        )
    
    def log_validation(self,
                      function: str,
                      validation_type: str,
                      total_checks: int,
                      passed_checks: int,
                      failed_checks: int,
                      execution_time: float,
                      details: Dict[str, Any]):
        """
        Registra un evento de validación.
        
        Args:
            function: Función de validación
            validation_type: Tipo de validación
            total_checks: Total de verificaciones
            passed_checks: Verificaciones exitosas
            failed_checks: Verificaciones fallidas
            execution_time: Tiempo de ejecución
            details: Detalles de la validación
        """
        success = failed_checks == 0
        level = LogLevel.INFO if success else LogLevel.WARNING
        
        self.log_event(
            level=level,
            message=f"Validation {validation_type}: {passed_checks}/{total_checks} passed",
            module="validation",
            function=function,
            step="validation",
            category=LogCategory.VALIDATION.value,
            parameters={"validation_type": validation_type},
            before_metrics={"total_checks": total_checks},
            after_metrics={
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "success_rate": passed_checks / total_checks if total_checks > 0 else 0
            },
            execution_time=execution_time,
            tags=["validation", validation_type],
            metadata=details
        )
    
    def log_business_rules(self,
                          function: str,
                          rules_executed: int,
                          rules_failed: int,
                          rules_warnings: int,
                          execution_time: float,
                          details: Dict[str, Any]):
        """
        Registra un evento de validación de reglas de negocio.
        
        Args:
            function: Función de validación
            rules_executed: Reglas ejecutadas
            rules_failed: Reglas fallidas
            rules_warnings: Reglas con advertencias
            execution_time: Tiempo de ejecución
            details: Detalles de la validación
        """
        level = LogLevel.INFO if rules_failed == 0 else LogLevel.ERROR
        
        self.log_event(
            level=level,
            message=f"Business rules validation: {rules_executed} executed, {rules_failed} failed",
            module="business_rules",
            function=function,
            step="business_rules",
            category=LogCategory.BUSINESS_RULES.value,
            parameters={"rules_executed": rules_executed},
            before_metrics={"rules_executed": rules_executed},
            after_metrics={
                "rules_failed": rules_failed,
                "rules_warnings": rules_warnings,
                "success_rate": (rules_executed - rules_failed) / rules_executed if rules_executed > 0 else 0
            },
            execution_time=execution_time,
            tags=["business_rules", "validation"],
            metadata=details
        )
    
    def log_analysis(self,
                    function: str,
                    analysis_type: str,
                    input_size: int,
                    output_size: int,
                    execution_time: float,
                    success: bool,
                    results: Dict[str, Any],
                    error_details: Optional[str] = None):
        """
        Registra un evento de análisis.
        
        Args:
            function: Función de análisis
            analysis_type: Tipo de análisis
            input_size: Tamaño de entrada
            output_size: Tamaño de salida
            execution_time: Tiempo de ejecución
            success: Si el análisis fue exitoso
            results: Resultados del análisis
            error_details: Detalles del error si aplica
        """
        level = LogLevel.INFO if success else LogLevel.ERROR
        
        self.log_event(
            level=level,
            message=f"Analysis {analysis_type} {'completed' if success else 'failed'}",
            module="analysis",
            function=function,
            step="analysis",
            category=LogCategory.ANALYSIS.value,
            parameters={"analysis_type": analysis_type},
            before_metrics={"input_size": input_size},
            after_metrics={"output_size": output_size},
            execution_time=execution_time,
            tags=["analysis", analysis_type],
            metadata={
                "results": results,
                "error_details": error_details
            } if error_details else {"results": results}
        )
    
    def log_error(self,
                  function: str,
                  error: Exception,
                  context: str,
                  execution_time: Optional[float] = None,
                  additional_data: Optional[Dict] = None):
        """
        Registra un error detallado.
        
        Args:
            function: Función donde ocurrió el error
            error: Excepción capturada
            context: Contexto del error
            execution_time: Tiempo de ejecución
            additional_data: Datos adicionales
        """
        self._write_log({
            "level": LogLevel.ERROR,
            "event": "error",
            "message": str(error),
            "function": function,
            "context": context,
            "execution_time": execution_time,
            "error_type": type(error).__name__,
            "stack_trace": traceback.format_exc(),
            "additional_data": additional_data or {},
            "tags": ["error", context]
        })


def create_json_logger(config: Dict[str, Any], session_id: str) -> JsonLogger:
    """
    Crea un logger JSON configurado.
    
    Args:
        config: Configuración del sistema
        session_id: ID de la sesión
        
    Returns:
        JsonLogger configurado
    """
    log_conf = config.get("logging", {}).get("json_logging", {})
    
    if not log_conf.get("enabled", False):
        # Retornar logger dummy si está deshabilitado
        return DummyJsonLogger(session_id)
    
    # Configurar rotación correctamente
    rotation_config = {}
    if log_conf.get("rotation"):
        if isinstance(log_conf["rotation"], dict):
            rotation_config = log_conf["rotation"]
        elif isinstance(log_conf["rotation"], str):
            # Si es string, configurar rotación básica
            rotation_config = {
                "enabled": True,
                "when": "midnight",
                "interval": 1,
                "backup_count": log_conf.get("backup_count", 7)
            }
    
    return JsonLogger(
        file_path=log_conf.get("log_file", "logs/pipeline.json"),
        level=log_conf.get("log_level", "INFO"),
        session_id=session_id,
        rotation=rotation_config,
        console_output=log_conf.get("console_output", False)
    )


class DummyJsonLogger:
    """Logger dummy para cuando el logging JSON está deshabilitado"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
    
    def __getattr__(self, name):
        # Todos los métodos retornan None o no hacen nada
        return lambda *args, **kwargs: None


# Decorador para logging de pasos del pipeline
def log_pipeline_step(step_name: str):
    """
    Decorador para logging automático de pasos del pipeline.
    
    Args:
        step_name: Nombre del paso
        
    Returns:
        Decorador configurado
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            
            # Métricas antes
            before_metrics = {}
            if hasattr(self, "session") and hasattr(self.session, "df") and self.session.df is not None:
                before_metrics = {
                    "rows": len(self.session.df),
                    "columns": len(self.session.df.columns),
                    "memory_usage_mb": self.session.df.memory_usage(deep=True).sum() / (1024 * 1024),
                }
            
            try:
                # Ejecutar función
                result = func(self, *args, **kwargs)
                
                # Métricas después
                after_metrics = {}
                if hasattr(self, "session") and hasattr(self.session, "df") and self.session.df is not None:
                    after_metrics = {
                        "rows": len(self.session.df),
                        "columns": len(self.session.df.columns),
                        "memory_usage_mb": self.session.df.memory_usage(deep=True).sum() / (1024 * 1024),
                    }
                
                execution_time = time.time() - start_time
                
                # Log de éxito
                if hasattr(self, "json_logger"):
                    self.json_logger.log_event(
                        level=LogLevel.INFO,
                        message=f"Paso completado exitosamente: {step_name}",
                        module=func.__module__,
                        function=func.__name__,
                        step=step_name,
                        category=LogCategory.PROCESSING.value,
                        parameters={"args": str(args), "kwargs": str(kwargs)},
                        before_metrics=before_metrics,
                        after_metrics=after_metrics,
                        execution_time=execution_time,
                        tags=["pipeline_step", step_name.lower().replace(" ", "_"), "success"]
                    )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log de error
                if hasattr(self, "json_logger"):
                    self.json_logger.log_error(
                        function=func.__name__,
                        error=e,
                        context=f"pipeline_step_{step_name.lower().replace(' ', '_')}",
                        execution_time=execution_time,
                        additional_data={
                            "step_name": step_name,
                            "args": str(args),
                            "kwargs": str(kwargs),
                            "before_metrics": before_metrics
                        }
                    )
                
                raise
        
        return wrapper
    
    return decorator 