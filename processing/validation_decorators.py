"""
Decoradores de validación universal para el pipeline de procesamiento.

Este módulo implementa decoradores que validan automáticamente:
1. DataFrames y sus columnas requeridas
2. Parámetros de funciones críticas
3. Reglas de negocio específicas
4. Integración con el sistema híbrido de reporte de errores
"""

import functools
import inspect
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints
import pandas as pd
from pydantic import BaseModel, ValidationError, Field

from .error_reporter import report_dataframe_error, report_parameter_error
from .data_validators import DataFrameSchema, ValidationResult, validate_dataframe


class DataFrameModel(BaseModel):
    """Modelo base para validación de DataFrames"""
    required_columns: List[str] = Field(default_factory=list)
    optional_columns: List[str] = Field(default_factory=list)
    min_rows: int = 1
    max_rows: Optional[int] = None
    column_types: Optional[Dict[str, str]] = None
    
    def validate_dataframe(self, df: pd.DataFrame, context: str) -> ValidationResult:
        """Valida un DataFrame contra este esquema"""
        schema = DataFrameSchema(
            required_columns=self.required_columns,
            optional_columns=self.optional_columns,
            min_rows=self.min_rows,
            max_rows=self.max_rows,
            column_types=self.column_types
        )
        return validate_dataframe(df, schema, context)


class ParameterModel(BaseModel):
    """Modelo base para validación de parámetros"""
    pass


def validate_io(
    df_schema: Optional[Type[DataFrameModel]] = None,
    param_schema: Optional[Type[ParameterModel]] = None,
    context: Optional[str] = None
):
    """
    Decorador universal para validación de entrada/salida.
    
    Args:
        df_schema: Esquema Pydantic para validar DataFrames
        param_schema: Esquema Pydantic para validar parámetros
        context: Contexto de la validación (opcional, se infiere del nombre de la función)
    
    Returns:
        Decorador que valida automáticamente DataFrames y parámetros
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determinar contexto
            func_context = context or f"{func.__module__}.{func.__name__}"
            
            # Validar DataFrames en argumentos
            if df_schema:
                df_schema_instance = df_schema()
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        validation_result = df_schema_instance.validate_dataframe(arg, func_context)
                        if not validation_result.is_valid:
                            report_dataframe_error(
                                message=f"DataFrame inválido en {func.__name__}: {validation_result.errors[0]}",
                                context=func_context,
                                details={
                                    "validation_errors": validation_result.errors,
                                    "validation_warnings": validation_result.warnings,
                                    "validation_details": validation_result.details,
                                    "function_name": func.__name__,
                                    "module": func.__module__
                                }
                            )
                        # Si hay warnings, registrarlos pero continuar
                        if validation_result.warnings:
                            print(f"⚠️ Warnings en {func_context}: {validation_result.warnings}")
            
            # Validar parámetros usando Pydantic
            if param_schema:
                try:
                    # Extraer parámetros de la función
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    
                    # Validar parámetros con Pydantic
                    validated_params = param_schema(**bound_args.arguments)
                    
                    # Reemplazar argumentos con valores validados
                    for name, value in validated_params.dict().items():
                        if name in bound_args.arguments:
                            bound_args.arguments[name] = value
                    
                    # Llamar función con parámetros validados
                    return func(*bound_args.args, **bound_args.kwargs)
                    
                except ValidationError as e:
                    report_parameter_error(
                        message=f"Parámetros inválidos en {func.__name__}: {str(e)}",
                        context=func_context,
                        details={
                            "validation_errors": e.errors(),
                            "function_name": func.__name__,
                            "module": func.__module__,
                            "parameters": bound_args.arguments if 'bound_args' in locals() else {}
                        }
                    )
                except Exception as e:
                    report_parameter_error(
                        message=f"Error inesperado en validación de parámetros: {str(e)}",
                        context=func_context,
                        details={
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "function_name": func.__name__,
                            "module": func.__module__
                        }
                    )
            
            # Si no hay esquemas de validación, ejecutar función directamente
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Esquemas específicos para diferentes tipos de operaciones
class SummaryStatsSchema(DataFrameModel):
    """Esquema para estadísticas descriptivas"""
    required_columns: List[str] = Field(default_factory=lambda: ["any_numeric"])
    optional_columns: List[str] = []
    min_rows: int = 1
    
    def validate_dataframe(self, df: pd.DataFrame, context: str) -> ValidationResult:
        """Validación específica para estadísticas descriptivas"""
        # Verificar que hay al menos una columna numérica
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) == 0:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame debe contener al menos una columna numérica para estadísticas"],
                warnings=[],
                details={"numeric_columns": numeric_cols}
            )
        
        # Crear un esquema dinámico con las columnas numéricas encontradas
        dynamic_schema = DataFrameSchema(
            required_columns=numeric_cols[:1],  # Al menos una columna numérica
            optional_columns=numeric_cols[1:],  # El resto como opcionales
            min_rows=self.min_rows,
            max_rows=self.max_rows,
            column_types=self.column_types
        )
        
        return validate_dataframe(df, dynamic_schema, context)


class CorrelationSchema(DataFrameModel):
    """Esquema para análisis de correlación"""
    required_columns: List[str] = Field(default_factory=lambda: ["any_numeric"])
    optional_columns: List[str] = []
    min_rows: int = 3  # Mínimo para correlación
    
    def validate_dataframe(self, df: pd.DataFrame, context: str) -> ValidationResult:
        """Validación específica para correlación"""
        # Verificar que hay al menos 2 columnas numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame debe contener al menos 2 columnas numéricas para correlación"],
                warnings=[],
                details={"numeric_columns": numeric_cols}
            )
        
        # Crear un esquema dinámico con las columnas numéricas encontradas
        dynamic_schema = DataFrameSchema(
            required_columns=numeric_cols[:2],  # Al menos 2 columnas numéricas
            optional_columns=numeric_cols[2:],  # El resto como opcionales
            min_rows=self.min_rows,
            max_rows=self.max_rows,
            column_types=self.column_types
        )
        
        return validate_dataframe(df, dynamic_schema, context)


class FilterSchema(DataFrameModel):
    """Esquema para operaciones de filtrado"""
    required_columns: List[str] = Field(default_factory=lambda: ["any_column"])
    optional_columns: List[str] = []
    min_rows: int = 1
    
    def validate_dataframe(self, df: pd.DataFrame, context: str) -> ValidationResult:
        """Validación específica para filtros"""
        # Verificar que las columnas requeridas existen
        missing_cols = [col for col in self.required_columns if col not in df.columns and col != "any_column"]
        if missing_cols:
            return ValidationResult(
                is_valid=False,
                errors=[f"Columnas requeridas para filtrado no encontradas: {missing_cols}"],
                warnings=[],
                details={"missing_columns": missing_cols, "available_columns": list(df.columns)}
            )
        
        # Crear un esquema dinámico con al menos una columna
        if len(df.columns) > 0:
            dynamic_schema = DataFrameSchema(
                required_columns=df.columns[:1],  # Al menos una columna
                optional_columns=df.columns[1:],  # El resto como opcionales
                min_rows=self.min_rows,
                max_rows=self.max_rows,
                column_types=self.column_types
            )
            return validate_dataframe(df, dynamic_schema, context)
        else:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame debe contener al menos una columna"],
                warnings=[],
                details={"available_columns": list(df.columns)}
            )


# Esquemas de parámetros específicos
class SummaryStatsParams(ParameterModel):
    """Parámetros para estadísticas descriptivas"""
    columns: Optional[List[str]] = None
    include_all: bool = True
    
    def validate_columns_exist(self, df: pd.DataFrame) -> bool:
        """Valida que las columnas especificadas existen en el DataFrame"""
        if self.columns:
            missing = [col for col in self.columns if col not in df.columns]
            if missing:
                raise ValueError(f"Columnas no encontradas: {missing}")
        return True


class CorrelationParams(ParameterModel):
    """Parámetros para análisis de correlación"""
    method: str = "pearson"
    columns: Optional[List[str]] = None
    min_correlation: float = 0.0
    
    def validate_method(self) -> bool:
        """Valida el método de correlación"""
        valid_methods = ["pearson", "spearman", "kendall"]
        if self.method not in valid_methods:
            raise ValueError(f"Método debe ser uno de: {valid_methods}")
        return True
    
    def validate_min_correlation(self) -> bool:
        """Valida el umbral de correlación mínima"""
        if not (0.0 <= self.min_correlation <= 1.0):
            raise ValueError("min_correlation debe estar entre 0.0 y 1.0")
        return True


class FilterParams(ParameterModel):
    """Parámetros para filtros"""
    column: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[str]] = None
    
    def validate_range(self) -> bool:
        """Valida que el rango sea coherente"""
        if self.min_value is not None and self.max_value is not None:
            if self.min_value >= self.max_value:
                raise ValueError("min_value debe ser menor que max_value")
        return True


# Decoradores específicos para diferentes tipos de operaciones
def validate_summary_stats(func):
    """Decorador específico para estadísticas descriptivas"""
    return validate_io(df_schema=SummaryStatsSchema, param_schema=SummaryStatsParams)(func)


def validate_correlation(func):
    """Decorador específico para análisis de correlación"""
    return validate_io(df_schema=CorrelationSchema, param_schema=CorrelationParams)(func)


def validate_filter(func):
    """Decorador específico para operaciones de filtrado"""
    return validate_io(df_schema=FilterSchema, param_schema=FilterParams)(func)


# Función helper para crear esquemas dinámicos
def create_dataframe_schema(
    required_columns: List[str] = None,
    optional_columns: List[str] = None,
    min_rows: int = 1,
    max_rows: Optional[int] = None,
    column_types: Optional[Dict[str, str]] = None
) -> Type[DataFrameModel]:
    """
    Crea un esquema de DataFrame dinámicamente.
    
    Args:
        required_columns: Columnas requeridas
        optional_columns: Columnas opcionales
        min_rows: Número mínimo de filas
        max_rows: Número máximo de filas
        column_types: Tipos esperados de columnas
        
    Returns:
        Clase de esquema de DataFrame
    """
    class DynamicDataFrameSchema(DataFrameModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.required_columns = required_columns or ["any_column"]
            self.optional_columns = optional_columns or []
            self.min_rows = min_rows
            self.max_rows = max_rows
            self.column_types = column_types
        
        def validate_dataframe(self, df: pd.DataFrame, context: str) -> ValidationResult:
            """Validación dinámica que se adapta al contenido del DataFrame"""
            # Si no hay columnas requeridas específicas, usar las del DataFrame
            if not self.required_columns or self.required_columns == ["any_column"]:
                if len(df.columns) > 0:
                    dynamic_schema = DataFrameSchema(
                        required_columns=df.columns[:1],  # Al menos una columna
                        optional_columns=df.columns[1:],  # El resto como opcionales
                        min_rows=self.min_rows,
                        max_rows=self.max_rows,
                        column_types=self.column_types
                    )
                    return validate_dataframe(df, dynamic_schema, context)
                else:
                    return ValidationResult(
                        is_valid=False,
                        errors=["El DataFrame debe contener al menos una columna"],
                        warnings=[],
                        details={"available_columns": list(df.columns)}
                    )
            else:
                # Usar validación estándar si hay columnas requeridas específicas
                return super().validate_dataframe(df, context)
    
    return DynamicDataFrameSchema 