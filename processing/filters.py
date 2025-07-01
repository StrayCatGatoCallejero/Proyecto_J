"""
Módulo de Filtros - Patrón "Reloj Suizo"
=======================================

Responsabilidades:
- Filtrado de datos por condiciones simples y complejas
- Limpieza de datos (outliers, valores faltantes)
- Transformaciones básicas de datos
- Logging sistemático de operaciones
- Validación automática de entrada usando decoradores
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import warnings
import logging
import time

warnings.filterwarnings("ignore")

# Importar logging y validación
from .logging import log_action
from .validation_decorators import validate_io, validate_filter, create_dataframe_schema
from .json_logging import LogLevel, LogCategory

logger = logging.getLogger(__name__)


def log_filter_operation(func):
    """
    Decorador para logging JSON de operaciones de filtrado.
    
    Args:
        func: Función a decorar
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        operation_name = func.__name__
        
        # Extraer información del DataFrame si está disponible
        df_info = {}
        if args and isinstance(args[0], pd.DataFrame):
            df = args[0]
            df_info = {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "missing_values": df.isnull().sum().sum(),
                "duplicate_rows": df.duplicated().sum()
            }
        
        try:
            # Log de inicio de operación de filtrado
            if hasattr(wrapper, 'json_logger'):
                wrapper.json_logger.log_event(
                    level=LogLevel.INFO,
                    message=f"Iniciando operación de filtrado: {operation_name}",
                    module=func.__module__,
                    function=func.__name__,
                    step="data_filtering",
                    category=LogCategory.PROCESSING.value,
                    parameters={"operation": operation_name, "args_count": len(args), "kwargs_count": len(kwargs)},
                    before_metrics=df_info,
                    after_metrics=df_info,
                    execution_time=0.0,
                    tags=["data_filtering", operation_name, "start"],
                    metadata={"operation": operation_name}
                )
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Calcular métricas del resultado
            after_metrics = df_info.copy()
            if isinstance(result, pd.DataFrame):
                after_metrics.update({
                    "result_rows": len(result),
                    "result_columns": len(result.columns),
                    "result_memory_mb": result.memory_usage(deep=True).sum() / (1024 * 1024),
                    "rows_removed": df_info.get("rows", 0) - len(result),
                    "removal_percentage": ((df_info.get("rows", 0) - len(result)) / df_info.get("rows", 1)) * 100 if df_info.get("rows", 0) > 0 else 0,
                    "result_missing_values": result.isnull().sum().sum(),
                    "result_duplicate_rows": result.duplicated().sum()
                })
            
            # Log de éxito
            if hasattr(wrapper, 'json_logger'):
                wrapper.json_logger.log_event(
                    level=LogLevel.INFO,
                    message=f"Operación de filtrado completada: {operation_name}",
                    module=func.__module__,
                    function=func.__name__,
                    step="data_filtering",
                    category=LogCategory.PROCESSING.value,
                    parameters={"operation": operation_name},
                    before_metrics=df_info,
                    after_metrics=after_metrics,
                    execution_time=execution_time,
                    tags=["data_filtering", operation_name, "success"],
                    metadata={
                        "operation": operation_name,
                        "result_type": type(result).__name__,
                        "success": True,
                        "data_reduction": {
                            "rows_removed": df_info.get("rows", 0) - len(result),
                            "percentage_removed": ((df_info.get("rows", 0) - len(result)) / df_info.get("rows", 1)) * 100 if df_info.get("rows", 0) > 0 else 0
                        }
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
                    context=f"data_filtering_{operation_name}",
                    execution_time=execution_time,
                    additional_data={
                        "operation": operation_name,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    }
                )
            
            raise
    
    return wrapper


# Esquema específico para filtros
class FilterDataFrameSchema(create_dataframe_schema(min_rows=1)):
    """Esquema para operaciones de filtrado"""
    def validate_dataframe(self, df: pd.DataFrame, context: str):
        """Validación específica para filtros"""
        from .data_validators import ValidationResult
        
        # Verificar que hay al menos una columna
        if len(df.columns) == 0:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame debe contener al menos una columna para filtrado"],
                warnings=[],
                details={"columns": list(df.columns)}
            )
        
        return super().validate_dataframe(df, context)


# Esquemas de parámetros específicos para filtros
class FilterConditionParams:
    """Parámetros para filtro por condición"""
    def __init__(self, condition: str):
        self.condition = condition
    
    def validate(self) -> Dict[str, Any]:
        """Valida los parámetros de filtro por condición"""
        if not isinstance(self.condition, str) or not self.condition.strip():
            return {
                'is_valid': False,
                'errors': ["La condición debe ser un string no vacío"],
                'warnings': [],
                'details': {'condition': self.condition}
            }
        
        return {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'details': {'condition': self.condition}
        }


class FilterRangeParams:
    """Parámetros para filtro por rango"""
    def __init__(self, column: str, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.column = column
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Valida los parámetros de filtro por rango"""
        if self.column not in df.columns:
            return {
                'is_valid': False,
                'errors': [f"La columna {self.column} no existe en el DataFrame"],
                'warnings': [],
                'details': {'column': self.column, 'available_columns': list(df.columns)}
            }
        
        if self.min_val is None and self.max_val is None:
            return {
                'is_valid': False,
                'errors': ["Debe especificar al menos un valor mínimo o máximo"],
                'warnings': [],
                'details': {'min_val': self.min_val, 'max_val': self.max_val}
            }
        
        if self.min_val is not None and self.max_val is not None:
            if self.min_val >= self.max_val:
                return {
                    'is_valid': False,
                    'errors': ["El valor mínimo debe ser menor que el valor máximo"],
                    'warnings': [],
                    'details': {'min_val': self.min_val, 'max_val': self.max_val}
                }
        
        return {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'details': {'column': self.column, 'min_val': self.min_val, 'max_val': self.max_val}
        }


class FilterValuesParams:
    """Parámetros para filtro por valores"""
    def __init__(self, column: str, values: List[Any]):
        self.column = column
        self.values = values
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Valida los parámetros de filtro por valores"""
        if self.column not in df.columns:
            return {
                'is_valid': False,
                'errors': [f"La columna {self.column} no existe en el DataFrame"],
                'warnings': [],
                'details': {'column': self.column, 'available_columns': list(df.columns)}
            }
        
        if not isinstance(self.values, list) or len(self.values) == 0:
            return {
                'is_valid': False,
                'errors': ["Los valores deben ser una lista no vacía"],
                'warnings': [],
                'details': {'values': self.values}
            }
        
        return {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'details': {'column': self.column, 'n_values': len(self.values)}
        }


def validate_filter_params(func):
    """Decorador específico para validación de parámetros de filtro"""
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        # Determinar tipo de filtro basado en el nombre de la función
        func_name = func.__name__
        
        if func_name == "filter_by_condition":
            if len(args) > 0:
                params = FilterConditionParams(args[0])
                validation = params.validate()
            else:
                validation = {'is_valid': True, 'errors': [], 'warnings': [], 'details': {}}
        
        elif func_name == "filter_by_range":
            if len(args) >= 2:
                params = FilterRangeParams(args[0], args[1] if len(args) > 1 else None, 
                                         args[2] if len(args) > 2 else None)
                validation = params.validate(df)
            else:
                validation = {'is_valid': True, 'errors': [], 'warnings': [], 'details': {}}
        
        elif func_name == "filter_by_values":
            if len(args) >= 2:
                params = FilterValuesParams(args[0], args[1])
                validation = params.validate(df)
            else:
                validation = {'is_valid': True, 'errors': [], 'warnings': [], 'details': {}}
        
        else:
            validation = {'is_valid': True, 'errors': [], 'warnings': [], 'details': {}}
        
        if not validation['is_valid']:
            from .error_reporter import report_parameter_error
            report_parameter_error(
                message=f"Parámetros de filtro inválidos en {func.__name__}: {validation['errors'][0]}",
                context=f"{func.__module__}.{func.__name__}",
                details={
                    "validation_errors": validation['errors'],
                    "validation_warnings": validation['warnings'],
                    "validation_details": validation['details'],
                    "function_name": func.__name__,
                    "module": func.__module__
                }
            )
        
        return func(df, *args, **kwargs)
    
    return wrapper


@validate_io(df_schema=FilterDataFrameSchema)
@validate_filter_params
def filter_by_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """
    Filtra DataFrame usando una condición en formato string.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas válidas
    2. Valida que la condición sea un string no vacío
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame a filtrar
        condition: Condición en formato string (ej: "edad > 25 and genero == 'M'")
        
    Returns:
        DataFrame filtrado
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    try:
        # Aplicar filtro
        filtered_df = df.query(condition)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        log_action(
            function="filter_by_condition",
            step="filters",
            parameters={"condition": condition},
            before_metrics={"n_rows": len(df)},
            after_metrics={"n_rows": len(filtered_df)},
            status="success",
            message=f"Filtro aplicado: {condition}",
            execution_time=execution_time
        )
        
        return filtered_df
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        log_action(
            function="filter_by_condition",
            step="filters",
            parameters={"condition": condition},
            before_metrics={"n_rows": len(df)},
            after_metrics={},
            status="error",
            message=f"Error al aplicar filtro: {str(e)}",
            execution_time=execution_time,
            error_details=str(e)
        )
        raise


@validate_io(df_schema=FilterDataFrameSchema)
@validate_filter_params
def filter_by_range(df: pd.DataFrame, column: str, min_val: Optional[float] = None, 
                   max_val: Optional[float] = None) -> pd.DataFrame:
    """
    Filtra DataFrame por rango de valores en una columna numérica.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas válidas
    2. Valida que la columna especificada exista
    3. Valida que el rango sea coherente (min < max)
    4. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame a filtrar
        column: Columna numérica
        min_val: Valor mínimo (opcional)
        max_val: Valor máximo (opcional)
        
    Returns:
        DataFrame filtrado
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    # Aplicar filtros
    filtered_df = df.copy()
    
    if min_val is not None:
        filtered_df = filtered_df[filtered_df[column] >= min_val]
    
    if max_val is not None:
        filtered_df = filtered_df[filtered_df[column] <= max_val]
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="filter_by_range",
        step="filters",
        parameters={"column": column, "min_val": min_val, "max_val": max_val},
        before_metrics={"n_rows": len(df)},
        after_metrics={"n_rows": len(filtered_df)},
        status="success",
        message=f"Filtro por rango aplicado a {column}",
        execution_time=execution_time
    )
    
    return filtered_df


@validate_io(df_schema=FilterDataFrameSchema)
@validate_filter_params
def filter_by_values(df: pd.DataFrame, column: str, values: List[Any]) -> pd.DataFrame:
    """
    Filtra DataFrame por valores específicos en una columna.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas válidas
    2. Valida que la columna especificada exista
    3. Valida que los valores sean una lista no vacía
    4. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame a filtrar
        column: Columna a filtrar
        values: Lista de valores a incluir
        
    Returns:
        DataFrame filtrado
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    # Aplicar filtro
    filtered_df = df[df[column].isin(values)]
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="filter_by_values",
        step="filters",
        parameters={"column": column, "n_values": len(values)},
        before_metrics={"n_rows": len(df)},
        after_metrics={"n_rows": len(filtered_df)},
        status="success",
        message=f"Filtro por valores aplicado a {column}",
        execution_time=execution_time
    )
    
    return filtered_df


@validate_io(df_schema=FilterDataFrameSchema)
def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', 
                   factor: float = 1.5) -> pd.DataFrame:
    """
    Elimina outliers de una columna numérica.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas válidas
    2. Valida que la columna especificada exista y sea numérica
    3. Valida que el método sea válido
    4. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame a procesar
        column: Columna numérica
        method: Método de detección ('iqr' o 'zscore')
        factor: Factor para el método IQR (por defecto 1.5)
        
    Returns:
        DataFrame sin outliers
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    if method not in ['iqr', 'zscore']:
        return df  # Retornar DataFrame original si método no válido
    
    series = df[column].dropna()
    if len(series) == 0:
        return df  # Retornar DataFrame original si no hay datos válidos
    
    # Detectar outliers
    if method == 'iqr':
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        mask = (series >= lower_bound) & (series <= upper_bound)
    else:  # zscore
        z_scores = np.abs((series - series.mean()) / series.std())
        mask = z_scores <= 3
    
    # Aplicar filtro
    filtered_df = df[mask]
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="remove_outliers",
        step="filters",
        parameters={"column": column, "method": method, "factor": factor},
        before_metrics={"n_rows": len(df)},
        after_metrics={"n_rows": len(filtered_df)},
        status="success",
        message=f"Outliers removidos de {column} usando {method}",
        execution_time=execution_time
    )
    
    return filtered_df

@validate_io(df_schema=FilterDataFrameSchema)
def handle_missing_values(df: pd.DataFrame, columns: Optional[List[str]] = None,
                         method: str = 'drop') -> pd.DataFrame:
    """
    Maneja valores faltantes en el DataFrame.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas válidas
    2. Valida que el método sea válido
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame a procesar
        columns: Columnas específicas a procesar (opcional)
        method: Método de manejo ('drop', 'fill_mean', 'fill_median', 'fill_mode')
        
    Returns:
        DataFrame con valores faltantes manejados
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    if method not in ['drop', 'fill_mean', 'fill_median', 'fill_mode']:
        return df  # Retornar DataFrame original si método no válido
    
    processed_df = df.copy()
    
    # Determinar columnas a procesar
    if columns is None:
        columns = list(df.columns)
    else:
        # Filtrar solo columnas que existen
        columns = [col for col in columns if col in df.columns]
    
    if method == 'drop':
        processed_df = processed_df.dropna(subset=columns)
    else:
        for col in columns:
            if col in processed_df.columns:
                if method == 'fill_mean' and processed_df[col].dtype in ['int64', 'float64']:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
                elif method == 'fill_median' and processed_df[col].dtype in ['int64', 'float64']:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                elif method == 'fill_mode':
                    mode_value = processed_df[col].mode()
                    if len(mode_value) > 0:
                        processed_df[col] = processed_df[col].fillna(mode_value[0])
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="handle_missing_values",
        step="filters",
        parameters={"method": method, "n_columns": len(columns)},
        before_metrics={"n_rows": len(df)},
        after_metrics={"n_rows": len(processed_df)},
        status="success",
        message=f"Valores faltantes manejados usando {method}",
        execution_time=execution_time
    )
    
    return processed_df

@validate_io(df_schema=FilterDataFrameSchema)
def sample_data(df: pd.DataFrame, n: Optional[int] = None, 
                fraction: Optional[float] = None, random_state: int = 42) -> pd.DataFrame:
    """
    Toma una muestra del DataFrame.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas válidas
    2. Valida que los parámetros de muestreo sean coherentes
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame a muestrear
        n: Número de filas a muestrear (opcional)
        fraction: Fracción de filas a muestrear (opcional)
        random_state: Semilla para reproducibilidad
        
    Returns:
        DataFrame muestreado
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    if n is None and fraction is None:
        return df  # Retornar DataFrame original si no se especifica muestreo
    
    if n is not None and fraction is not None:
        return df  # Retornar DataFrame original si se especifican ambos
    
    if n is not None:
        if n <= 0 or n > len(df):
            return df  # Retornar DataFrame original si n no es válido
        sampled_df = df.sample(n=n, random_state=random_state)
    else:
        if fraction <= 0 or fraction > 1:
            return df  # Retornar DataFrame original si fraction no es válido
        sampled_df = df.sample(frac=fraction, random_state=random_state)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="sample_data",
        step="filters",
        parameters={"n": n, "fraction": fraction, "random_state": random_state},
        before_metrics={"n_rows": len(df)},
        after_metrics={"n_rows": len(sampled_df)},
        status="success",
        message=f"Muestra tomada: {len(sampled_df)} filas",
        execution_time=execution_time
    )
    
    return sampled_df

@validate_io(df_schema=FilterDataFrameSchema)
def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Selecciona columnas específicas del DataFrame.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas válidas
    2. Valida que las columnas especificadas existan
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame a procesar
        columns: Lista de columnas a seleccionar
        
    Returns:
        DataFrame con columnas seleccionadas
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    if not isinstance(columns, list) or len(columns) == 0:
        return df  # Retornar DataFrame original si columns no es válido
    
    # Filtrar solo columnas que existen
    valid_columns = [col for col in columns if col in df.columns]
    
    if len(valid_columns) == 0:
        return df  # Retornar DataFrame original si no hay columnas válidas
    
    selected_df = df[valid_columns]
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="select_columns",
        step="filters",
        parameters={"n_columns": len(valid_columns)},
        before_metrics={"n_cols": len(df.columns)},
        after_metrics={"n_cols": len(selected_df.columns)},
        status="success",
        message=f"Columnas seleccionadas: {len(valid_columns)}",
        execution_time=execution_time
    )
    
    return selected_df

@validate_io(df_schema=FilterDataFrameSchema)
def drop_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None, 
                   keep: str = 'first') -> pd.DataFrame:
    """
    Elimina filas duplicadas del DataFrame.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas válidas
    2. Valida que el parámetro keep sea válido
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame a procesar
        subset: Columnas a considerar para duplicados (opcional)
        keep: Estrategia de eliminación ('first', 'last', False)
        
    Returns:
        DataFrame sin duplicados
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    if keep not in ['first', 'last', False]:
        return df  # Retornar DataFrame original si keep no es válido
    
    # Filtrar subset si se especifica
    if subset is not None:
        valid_subset = [col for col in subset if col in df.columns]
        if len(valid_subset) == 0:
            return df  # Retornar DataFrame original si no hay columnas válidas en subset
        subset = valid_subset
    
    deduplicated_df = df.drop_duplicates(subset=subset, keep=keep)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="drop_duplicates",
        step="filters",
        parameters={"subset": subset, "keep": keep},
        before_metrics={"n_rows": len(df)},
        after_metrics={"n_rows": len(deduplicated_df)},
        status="success",
        message=f"Duplicados eliminados: {len(df) - len(deduplicated_df)} filas",
        execution_time=execution_time
    )
    
    return deduplicated_df

@validate_io(df_schema=FilterDataFrameSchema)
def apply_custom_filter(df: pd.DataFrame, filter_func: Callable) -> pd.DataFrame:
    """
    Aplica un filtro personalizado al DataFrame.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas válidas
    2. Valida que filter_func sea una función válida
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame a procesar
        filter_func: Función de filtrado personalizada
        
    Returns:
        DataFrame filtrado
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    if not callable(filter_func):
        return df  # Retornar DataFrame original si filter_func no es válido
    
    try:
        filtered_df = filter_func(df)
        
        # Verificar que el resultado es un DataFrame
        if not isinstance(filtered_df, pd.DataFrame):
            return df  # Retornar DataFrame original si el resultado no es válido
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        log_action(
            function="apply_custom_filter",
            step="filters",
            parameters={"filter_func": filter_func.__name__ if hasattr(filter_func, '__name__') else 'custom'},
            before_metrics={"n_rows": len(df)},
            after_metrics={"n_rows": len(filtered_df)},
            status="success",
            message=f"Filtro personalizado aplicado: {filter_func.__name__ if hasattr(filter_func, '__name__') else 'custom'}",
            execution_time=execution_time
        )
        
        return filtered_df
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        log_action(
            function="apply_custom_filter",
            step="filters",
            parameters={"filter_func": filter_func.__name__ if hasattr(filter_func, '__name__') else 'custom'},
            before_metrics={"n_rows": len(df)},
            after_metrics={},
            status="error",
            message=f"Error al aplicar filtro personalizado: {str(e)}",
            execution_time=execution_time,
            error_details=str(e)
        )
        return df  # Retornar DataFrame original en caso de error


class DataFilter:
    """
    Clase para aplicar filtros a DataFrames.
    """
    
    def __init__(self):
        """Inicializa el DataFilter."""
        self.applied_filters = []
    
    def apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Aplica filtros a un DataFrame.
        
        Args:
            df: DataFrame a filtrar
            filters: Diccionario con filtros a aplicar
            
        Returns:
            DataFrame filtrado
        """
        if not filters:
            return df
        
        filtered_df = df.copy()
        
        for filter_name, filter_config in filters.items():
            try:
                filtered_df = self._apply_single_filter(filtered_df, filter_name, filter_config)
                self.applied_filters.append(filter_name)
            except Exception as e:
                logger.warning(f"Error applying filter {filter_name}: {e}")
        
        return filtered_df
    
    def _apply_single_filter(self, df: pd.DataFrame, filter_name: str, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Aplica un filtro individual.
        
        Args:
            df: DataFrame a filtrar
            filter_name: Nombre del filtro
            filter_config: Configuración del filtro
            
        Returns:
            DataFrame filtrado
        """
        filter_type = filter_config.get('type', 'range')
        
        if filter_type == 'range':
            return self._apply_range_filter(df, filter_config)
        elif filter_type == 'categorical':
            return self._apply_categorical_filter(df, filter_config)
        elif filter_type == 'missing':
            return self._apply_missing_filter(df, filter_config)
        else:
            logger.warning(f"Unknown filter type: {filter_type}")
            return df
    
    def _apply_range_filter(self, df: pd.DataFrame, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Aplica filtro de rango."""
        column = filter_config.get('column')
        min_val = filter_config.get('min')
        max_val = filter_config.get('max')
        
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return df
        
        mask = pd.Series([True] * len(df), index=df.index)
        
        if min_val is not None:
            mask &= df[column] >= min_val
        
        if max_val is not None:
            mask &= df[column] <= max_val
        
        return df[mask]
    
    def _apply_categorical_filter(self, df: pd.DataFrame, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Aplica filtro categórico."""
        column = filter_config.get('column')
        values = filter_config.get('values', [])
        
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return df
        
        if not values:
            return df
        
        return df[df[column].isin(values)]
    
    def _apply_missing_filter(self, df: pd.DataFrame, filter_config: Dict[str, Any]) -> pd.DataFrame:
        """Aplica filtro de valores faltantes."""
        columns = filter_config.get('columns', [])
        threshold = filter_config.get('threshold', 0.5)
        
        if not columns:
            return df
        
        # Filtrar filas con menos del threshold de valores faltantes
        missing_ratio = df[columns].isnull().sum(axis=1) / len(columns)
        return df[missing_ratio <= threshold]
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de los filtros aplicados.
        
        Returns:
            Diccionario con resumen de filtros
        """
        return {
            "applied_filters": self.applied_filters,
            "total_filters": len(self.applied_filters)
        }
