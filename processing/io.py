"""
Módulo de I/O - Patrón "Reloj Suizo"
====================================

Responsabilidades:
- Carga de archivos de datos (.sav, .dta, .csv, .xlsx)
- Validación de entrada y manejo de errores
- Logging sistemático de operaciones
- Metadatos consistentes y estructurados
- Validación automática de entrada usando decoradores
"""

import os
import pandas as pd
import numpy as np
import chardet
from typing import Tuple, Dict, Any, Optional
import warnings
from datetime import datetime
import time

warnings.filterwarnings("ignore")

# Importar logging y validación
from .logging import log_action
from .validation_decorators import validate_io, create_dataframe_schema
from .json_logging import LogLevel, LogCategory, serialize_for_json


def log_io_operation(func):
    """
    Decorador para logging JSON de operaciones de I/O.
    
    Args:
        func: Función a decorar
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        operation_name = func.__name__
        
        # Extraer información del archivo si está disponible
        file_path = None
        if args and isinstance(args[0], str):
            file_path = args[0]
        
        try:
            # Log de inicio de operación
            if hasattr(wrapper, 'json_logger'):
                wrapper.json_logger.log_event(
                    level=LogLevel.INFO,
                    message=f"Iniciando operación de I/O: {operation_name}",
                    module=func.__module__,
                    function=func.__name__,
                    step="io_operation",
                    category=LogCategory.DATA_LOAD.value,
                    parameters={"file_path": file_path, "args_count": len(args), "kwargs_count": len(kwargs)},
                    before_metrics={"file_path": file_path},
                    after_metrics={"file_path": file_path},
                    execution_time=0.0,
                    tags=["io_operation", operation_name, "start"],
                    metadata={"operation": operation_name}
                )
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Calcular métricas del resultado
            after_metrics = {}
            if isinstance(result, pd.DataFrame):
                after_metrics = {
                    "rows": len(result),
                    "columns": len(result.columns),
                    "memory_usage_mb": result.memory_usage(deep=True).sum() / (1024 * 1024),
                    "data_types": result.dtypes.value_counts().to_dict(),
                    "missing_values": result.isnull().sum().sum()
                }
            elif isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], pd.DataFrame):
                df = result[0]
                after_metrics = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                    "data_types": df.dtypes.value_counts().to_dict(),
                    "missing_values": df.isnull().sum().sum(),
                    "metadata_included": len(result) > 1
                }
            
            # Log de éxito
            if hasattr(wrapper, 'json_logger'):
                wrapper.json_logger.log_event(
                    level=LogLevel.INFO,
                    message=f"Operación de I/O completada: {operation_name}",
                    module=func.__module__,
                    function=func.__name__,
                    step="io_operation",
                    category=LogCategory.DATA_LOAD.value,
                    parameters={"file_path": file_path},
                    before_metrics={"file_path": file_path},
                    after_metrics=after_metrics,
                    execution_time=execution_time,
                    tags=["io_operation", operation_name, "success"],
                    metadata={
                        "operation": operation_name,
                        "result_type": type(result).__name__,
                        "success": True
                    }
                )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log de error
            if hasattr(wrapper, 'json_logger'):
                wrapper.json_logger.log_error(
                    function=func.__name__,
                    error=e,
                    context=f"io_operation_{operation_name}",
                    execution_time=execution_time,
                    additional_data={
                        "file_path": file_path,
                        "operation": operation_name,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    }
                )
            
            raise
    
    return wrapper


class DataLoader:
    """
    Clase para carga de datos con detección automática de formato.
    """
    
    def __init__(self, json_logger=None):
        """
        Inicializa el DataLoader.
        
        Args:
            json_logger: Logger JSON opcional para logging estructurado
        """
        self.detected_format = None
        self.supported_formats = ['.sav', '.dta', '.csv', '.xlsx', '.xls']
        self.json_logger = json_logger
    
    @log_io_operation
    def load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Carga un archivo de datos con detección automática de formato.
        
        Args:
            file_path: Ruta al archivo de datos
            
        Returns:
            DataFrame cargado o None si hay error
        """
        try:
            # Obtener información del archivo
            file_info = self._get_file_info(file_path)
            
            # Log de información del archivo
            if self.json_logger:
                self.json_logger.log_event(
                    level=LogLevel.INFO,
                    message=f"Información del archivo obtenida: {file_path}",
                    module=self.__class__.__module__,
                    function="load_file",
                    step="file_info",
                    category=LogCategory.DATA_LOAD.value,
                    parameters={"file_path": file_path},
                    before_metrics={"file_path": file_path},
                    after_metrics=file_info,
                    execution_time=0.0,
                    tags=["file_info", "metadata"],
                    metadata=file_info
                )
            
            # Detectar formato
            ext = os.path.splitext(file_path)[1].lower()
            self.detected_format = ext
            
            if ext not in self.supported_formats:
                raise ValueError(f"Formato no soportado: {ext}")
            
            # Cargar archivo según formato
            if ext == '.csv':
                df = self._load_csv(file_path)
            elif ext in ['.xlsx', '.xls']:
                df = self._load_excel(file_path)
            elif ext == '.sav':
                df = self._load_sav(file_path)
            elif ext == '.dta':
                df = self._load_dta(file_path)
            else:
                raise ValueError(f"Formato no soportado: {ext}")
            
            # Log de carga exitosa
            if self.json_logger and df is not None:
                self.json_logger.log_event(
                    level=LogLevel.INFO,
                    message=f"Archivo cargado exitosamente: {file_path}",
                    module=self.__class__.__module__,
                    function="load_file",
                    step="file_loaded",
                    category=LogCategory.DATA_LOAD.value,
                    parameters={"file_path": file_path, "format": ext},
                    before_metrics=file_info,
                    after_metrics={
                        "rows": len(df),
                        "columns": len(df.columns),
                        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                        "format": ext
                    },
                    execution_time=0.0,
                    tags=["file_loaded", "success"],
                    metadata={"format": ext, "file_info": file_info}
                )
            
            return df
            
        except Exception as e:
            # Log de error de carga
            if self.json_logger:
                self.json_logger.log_error(
                    function="load_file",
                    error=e,
                    context="file_loading",
                    execution_time=0.0,
                    additional_data={
                        "file_path": file_path,
                        "detected_format": self.detected_format
                    }
                )
            raise
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Obtiene información detallada del archivo."""
        try:
            stat = os.stat(file_path)
            return {
                "file_size_bytes": stat.st_size,
                "file_size_mb": stat.st_size / (1024 * 1024),
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": os.path.splitext(file_path)[1].lower(),
                "filename": os.path.basename(file_path),
                "directory": os.path.dirname(file_path)
            }
        except Exception as e:
            return {"error": f"Error obteniendo información del archivo: {str(e)}"}
    
    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """Carga archivo CSV con detección de encoding."""
        try:
            # Detectar encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding']
            
            # Log de encoding detectado
            if self.json_logger:
                self.json_logger.log_event(
                    level=LogLevel.DEBUG,
                    message=f"Encoding detectado para CSV: {encoding}",
                    module=self.__class__.__module__,
                    function="_load_csv",
                    step="encoding_detection",
                    category=LogCategory.DATA_LOAD.value,
                    parameters={"file_path": file_path},
                    before_metrics={"encoding": encoding},
                    after_metrics={"encoding": encoding},
                    execution_time=0.0,
                    tags=["encoding_detection", "csv"],
                    metadata={"detected_encoding": encoding, "confidence": detected.get('confidence', 0)}
                )
            
            return pd.read_csv(file_path, encoding=encoding)
            
        except Exception as e:
            # Fallback a encoding por defecto
            if self.json_logger:
                self.json_logger.log_event(
                    level=LogLevel.WARNING,
                    message=f"Fallback a encoding por defecto para CSV: {file_path}",
                    module=self.__class__.__module__,
                    function="_load_csv",
                    step="encoding_fallback",
                    category=LogCategory.DATA_LOAD.value,
                    parameters={"file_path": file_path},
                    before_metrics={"encoding": "default"},
                    after_metrics={"encoding": "default"},
                    execution_time=0.0,
                    tags=["encoding_fallback", "csv"],
                    metadata={"error": str(e)}
                )
            
            return pd.read_csv(file_path)
    
    def _load_excel(self, file_path: str) -> pd.DataFrame:
        """Carga archivo Excel."""
        return pd.read_excel(file_path)
    
    def _load_sav(self, file_path: str) -> pd.DataFrame:
        """Carga archivo SPSS (.sav)."""
        try:
            # Intentar usar pyreadstat si está disponible
            import pyreadstat
            df, meta = pyreadstat.read_sav(file_path)
            
            # Log de metadatos SPSS
            if self.json_logger:
                self.json_logger.log_event(
                    level=LogLevel.INFO,
                    message=f"Metadatos SPSS cargados: {file_path}",
                    module=self.__class__.__module__,
                    function="_load_sav",
                    step="spss_metadata",
                    category=LogCategory.DATA_LOAD.value,
                    parameters={"file_path": file_path},
                    before_metrics={"format": "spss"},
                    after_metrics={
                        "format": "spss",
                        "variable_labels": len(meta.get('variable_labels', {})),
                        "value_labels": len(meta.get('value_labels', {}))
                    },
                    execution_time=0.0,
                    tags=["spss", "metadata"],
                    metadata={"spss_metadata": meta}
                )
            
            return df
            
        except ImportError:
            # Fallback a CSV si pyreadstat no está disponible
            if self.json_logger:
                self.json_logger.log_event(
                    level=LogLevel.WARNING,
                    message=f"pyreadstat no disponible, fallback a CSV: {file_path}",
                    module=self.__class__.__module__,
                    function="_load_sav",
                    step="spss_fallback",
                    category=LogCategory.DATA_LOAD.value,
                    parameters={"file_path": file_path},
                    before_metrics={"format": "spss"},
                    after_metrics={"format": "csv_fallback"},
                    execution_time=0.0,
                    tags=["spss_fallback", "csv"],
                    metadata={"fallback_reason": "pyreadstat_not_available"}
                )
            
            return pd.read_csv(file_path)
    
    def _load_dta(self, file_path: str) -> pd.DataFrame:
        """Carga archivo Stata (.dta)."""
        try:
            return pd.read_stata(file_path)
        except Exception as e:
            # Fallback a CSV
            if self.json_logger:
                self.json_logger.log_event(
                    level=LogLevel.WARNING,
                    message=f"Error cargando Stata, fallback a CSV: {file_path}",
                    module=self.__class__.__module__,
                    function="_load_dta",
                    step="stata_fallback",
                    category=LogCategory.DATA_LOAD.value,
                    parameters={"file_path": file_path},
                    before_metrics={"format": "stata"},
                    after_metrics={"format": "csv_fallback"},
                    execution_time=0.0,
                    tags=["stata_fallback", "csv"],
                    metadata={"error": str(e)}
                )
            
            return pd.read_csv(file_path)


# Esquema específico para validación de rutas de archivos
class FilePathSchema:
    """Esquema para validación de rutas de archivos"""
    def __init__(self, required_extensions: Optional[list] = None):
        self.required_extensions = required_extensions or ['.sav', '.dta', '.csv', '.xlsx', '.xls']
    
    def validate_path(self, path: str, context: str) -> Dict[str, Any]:
        """Valida una ruta de archivo"""
        from .data_validators import ValidationResult
        
        # Verificar que la ruta existe
        if not os.path.exists(path):
            return {
                'is_valid': False,
                'errors': [f"El archivo no existe: {path}"],
                'warnings': [],
                'details': {'path': path, 'exists': False}
            }
        
        # Verificar que es un archivo (no directorio)
        if not os.path.isfile(path):
            return {
                'is_valid': False,
                'errors': [f"La ruta no es un archivo: {path}"],
                'warnings': [],
                'details': {'path': path, 'is_file': False}
            }
        
        # Verificar extensión
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.required_extensions:
            return {
                'is_valid': False,
                'errors': [f"Extensión no soportada: {ext}. Extensiones válidas: {self.required_extensions}"],
                'warnings': [],
                'details': {'extension': ext, 'supported_extensions': self.required_extensions}
            }
        
        # Verificar que el archivo no esté vacío
        size = os.path.getsize(path)
        if size == 0:
            return {
                'is_valid': False,
                'errors': ["El archivo está vacío"],
                'warnings': [],
                'details': {'file_size': size}
            }
        
        return {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'details': {'path': path, 'extension': ext, 'file_size': size}
        }


# Esquema específico para DataFrames cargados
class LoadedDataFrameSchema(create_dataframe_schema(min_rows=1)):
    """Esquema para DataFrames cargados desde archivos"""
    def validate_dataframe(self, df: pd.DataFrame, context: str):
        """Validación específica para DataFrames cargados"""
        from .data_validators import ValidationResult
        
        # Verificar que el DataFrame no esté completamente vacío
        if df.empty:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame cargado está completamente vacío"],
                warnings=[],
                details={"shape": df.shape, "columns": list(df.columns)}
            )
        
        # Verificar que hay al menos una columna
        if len(df.columns) == 0:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame cargado no tiene columnas"],
                warnings=[],
                details={"shape": df.shape, "columns": list(df.columns)}
            )
        
        # Verificar que las columnas no son todas nulas
        non_null_columns = df.columns[df.notna().any()].tolist()
        if len(non_null_columns) == 0:
            return ValidationResult(
                is_valid=False,
                errors=["Todas las columnas del DataFrame están completamente vacías"],
                warnings=[],
                details={"shape": df.shape, "columns": list(df.columns), "non_null_columns": non_null_columns}
            )
        
        return super().validate_dataframe(df, context)


def validate_file_path(func):
    """Decorador específico para validación de rutas de archivos"""
    def wrapper(path: str, *args, **kwargs):
        # Validar ruta antes de procesar
        schema = FilePathSchema()
        validation = schema.validate_path(path, f"{func.__module__}.{func.__name__}")
        
        if not validation['is_valid']:
            from .error_reporter import report_dataframe_error
            report_dataframe_error(
                message=f"Ruta de archivo inválida en {func.__name__}: {validation['errors'][0]}",
                context=f"{func.__module__}.{func.__name__}",
                details={
                    "validation_errors": validation['errors'],
                    "validation_warnings": validation['warnings'],
                    "validation_details": validation['details'],
                    "function_name": func.__name__,
                    "module": func.__module__
                }
            )
        
        return func(path, *args, **kwargs)
    
    return wrapper


@validate_file_path
@validate_io(df_schema=LoadedDataFrameSchema)
def cargar_archivo(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Carga un archivo .sav/.dta/.csv/.xlsx y devuelve (df, metadata).
    Metadata incluye: format, n_rows, n_cols, variable_labels (si aplica).
    
    Esta función está protegida por validación automática que:
    1. Verifica que la ruta del archivo existe y es válida
    2. Valida que el archivo no esté vacío y tenga formato soportado
    3. Verifica que el DataFrame cargado tenga datos válidos
    4. Reporta errores detallados si la validación falla
    
    Args:
        path: Ruta del archivo a cargar
        
    Returns:
        Tuple: (DataFrame, metadata)
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo está vacío o formato no soportado
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    # Detectar formato
    ext = path.lower().rsplit('.', 1)[-1]
    size = os.path.getsize(path)
    metadata = {'format': ext, 'file_size': size}
    
    try:
        if ext == 'sav':
            df, meta = _cargar_sav(path)
            metadata.update({
                'variable_labels': meta.variable_labels if hasattr(meta, 'variable_labels') else {},
                'value_labels': meta.value_labels if hasattr(meta, 'value_labels') else {}
            })
        elif ext == 'dta':
            df, meta = _cargar_dta(path)
            metadata.update({
                'variable_labels': meta.variable_labels if hasattr(meta, 'variable_labels') else {}
            })
        elif ext == 'csv':
            df = _cargar_csv(path)
        elif ext in ('xls', 'xlsx'):
            df = _cargar_excel(path)
        else:
            raise ValueError(f"Formato no soportado: {ext}")
        
        # Post-proceso genérico
        df = df.rename(columns=str.strip).copy()
        metadata.update({
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'load_time': (datetime.now() - start_time).total_seconds()
        })
        
        # Limpiar metadatos para serialización JSON
        cleaned_metadata = serialize_for_json(metadata)
        
        return df, cleaned_metadata
        
    except Exception as e:
        # Log del error
        log_action(
            function="cargar_archivo",
            step="file_loading",
            parameters={"path": path, "format": ext},
            before_metrics={"file_size": size},
            after_metrics={},
            status="error",
            message=f"Error cargando archivo: {str(e)}",
            execution_time=(datetime.now() - start_time).total_seconds(),
            error_details=str(e)
        )
        raise


@validate_file_path
def _cargar_sav(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Carga archivo .sav (SPSS)"""
    try:
        import pyreadstat
        df, meta = pyreadstat.read_sav(path)
        return df, meta
    except ImportError:
        raise ImportError("pyreadstat no está instalado. Instalar con: pip install pyreadstat")
    except Exception as e:
        raise ValueError(f"Error cargando archivo .sav: {e}")


@validate_file_path
def _cargar_dta(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Carga archivo .dta (Stata)"""
    try:
        df = pd.read_stata(path)
        meta = {'variable_labels': {}}  # Placeholder
        return df, meta
    except Exception as e:
        raise ValueError(f"Error cargando archivo .dta: {e}")


@validate_file_path
def _cargar_csv(path: str) -> pd.DataFrame:
    """Carga archivo .csv con detección automática de encoding"""
    try:
        # Detectar encoding
        with open(path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        # Intentar cargar con encoding detectado
        try:
            df = pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            # Fallback a utf-8
            df = pd.read_csv(path, encoding='utf-8')
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error cargando archivo .csv: {e}")


@validate_file_path
def _cargar_excel(path: str) -> pd.DataFrame:
    """Carga archivo .xlsx/.xls"""
    try:
        df = pd.read_excel(path)
        return df
    except Exception as e:
        raise ValueError(f"Error cargando archivo Excel: {e}")


@validate_io(df_schema=LoadedDataFrameSchema)
def validar_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida un DataFrame cargado y retorna reporte de validación.
    
    Args:
        df: DataFrame a validar
        metadata: Metadatos del archivo
        
    Returns:
        Dict con resultados de validación
    """
    start_time = datetime.now()
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'details': {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum()
        }
    }
    
    # Validaciones específicas
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame está vacío")
    
    if len(df.columns) == 0:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame no tiene columnas")
    
    # Verificar columnas completamente vacías
    empty_columns = df.columns[df.isnull().all()].tolist()
    if empty_columns:
        validation_results['warnings'].append(f"Columnas completamente vacías: {empty_columns}")
    
    # Verificar filas duplicadas
    if df.duplicated().sum() > 0:
        validation_results['warnings'].append(f"Hay {df.duplicated().sum()} filas duplicadas")
    
    # Log de validación
    log_action(
        function="validar_dataframe",
        step="dataframe_validation",
        parameters={"metadata": metadata},
        before_metrics={"shape": df.shape},
        after_metrics={
            "is_valid": validation_results['is_valid'],
            "errors_count": len(validation_results['errors']),
            "warnings_count": len(validation_results['warnings'])
        },
        status="success" if validation_results['is_valid'] else "error",
        message=f"Validación DataFrame: {'Válido' if validation_results['is_valid'] else 'Inválido'}",
        execution_time=(datetime.now() - start_time).total_seconds(),
        error_details="; ".join(validation_results['errors']) if validation_results['errors'] else None
    )
    
    return validation_results


def obtener_info_archivo(path: str) -> Dict[str, Any]:
    """
    Obtiene información básica de un archivo sin cargarlo completamente.
    
    Args:
        path: Ruta al archivo
        
    Returns:
        Dict con información del archivo
    """
    try:
        if not os.path.exists(path):
            return {
                'exists': False,
                'error': 'Archivo no encontrado'
            }
        
        info = {
            'exists': True,
            'path': path,
            'size_bytes': os.path.getsize(path),
            'extension': os.path.splitext(path)[1].lower(),
            'modified_time': datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
        }
        
        # Información específica por formato
        ext = info['extension']
        if ext == '.csv':
            # Leer primeras líneas para detectar separador y encoding
            with open(path, 'rb') as f:
                raw_data = f.read(1024)
                result = chardet.detect(raw_data)
                info['encoding'] = result['encoding']
                
            # Detectar separador
            with open(path, 'r', encoding=info['encoding']) as f:
                first_line = f.readline()
                separators = [',', ';', '\t', '|']
                detected_sep = None
                for sep in separators:
                    if sep in first_line:
                        detected_sep = sep
                        break
                info['separator'] = detected_sep
        
        elif ext in ['.xlsx', '.xls']:
            # Información básica de Excel
            info['excel_sheets'] = pd.ExcelFile(path).sheet_names
        
        return info
        
    except Exception as e:
        return {
            'exists': False,
            'error': str(e)
        }
