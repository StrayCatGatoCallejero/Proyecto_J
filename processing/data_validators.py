"""
Validadores Pydantic para datos de entrada al procesamiento.

Este módulo define esquemas de validación para:
1. DataFrames y sus columnas requeridas
2. Parámetros de funciones de análisis
3. Configuraciones de métodos estadísticos
4. Validación de tipos de datos y rangos
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
import numpy as np


class DataFrameSchema(BaseModel):
    """Esquema para validar DataFrames"""
    required_columns: List[str] = Field(..., min_items=1)
    optional_columns: List[str] = Field(default_factory=list)
    min_rows: int = Field(default=1, ge=1)
    max_rows: Optional[int] = Field(default=None, ge=1)
    column_types: Optional[Dict[str, str]] = Field(default=None)
    
    @validator('required_columns', 'optional_columns')
    def validate_column_names(cls, v):
        for col in v:
            if not isinstance(col, str) or len(col.strip()) == 0:
                raise ValueError("Los nombres de columnas deben ser strings no vacíos")
        return v
    
    @validator('column_types')
    def validate_column_types(cls, v):
        if v is not None:
            valid_types = ['int64', 'float64', 'object', 'bool', 'datetime64[ns]']
            for col_type in v.values():
                if col_type not in valid_types:
                    raise ValueError(f"Tipo de columna no válido: {col_type}")
        return v


class FeatureSelectionParams(BaseModel):
    """Parámetros para selección de features"""
    target_column: str = Field(..., min_length=1)
    method: str = Field(..., pattern=r'^(correlation|mutual_info|f_regression|f_classif)$')
    n_features: int = Field(..., ge=1, le=1000)
    feature_columns: Optional[List[str]] = Field(default=None)
    
    @validator('target_column')
    def validate_target_column(cls, v):
        if not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError("La columna objetivo debe ser un string no vacío")
        return v.strip()
    
    @validator('feature_columns')
    def validate_feature_columns(cls, v):
        if v is not None:
            for col in v:
                if not isinstance(col, str) or len(col.strip()) == 0:
                    raise ValueError("Los nombres de columnas de features deben ser strings no vacíos")
        return v


class CorrelationParams(BaseModel):
    """Parámetros para análisis de correlación"""
    method: str = Field(..., pattern=r'^(pearson|spearman|kendall)$')
    min_correlation: float = Field(..., ge=0.0, le=1.0)
    significance_level: float = Field(..., gt=0.0, lt=1.0)
    columns: Optional[List[str]] = Field(default=None)
    
    @validator('columns')
    def validate_columns(cls, v):
        if v is not None:
            for col in v:
                if not isinstance(col, str) or len(col.strip()) == 0:
                    raise ValueError("Los nombres de columnas deben ser strings no vacíos")
        return v


class StatisticalTestParams(BaseModel):
    """Parámetros para pruebas estadísticas"""
    test_type: str = Field(..., pattern=r'^(t_test|chi_square|anova|mann_whitney|wilcoxon)$')
    group_column: Optional[str] = Field(default=None)
    value_column: Optional[str] = Field(default=None)
    significance_level: float = Field(default=0.05, gt=0.0, lt=1.0)
    
    @validator('group_column', 'value_column')
    def validate_column_names(cls, v):
        if v is not None and (not isinstance(v, str) or len(v.strip()) == 0):
            raise ValueError("Los nombres de columnas deben ser strings no vacíos")
        return v.strip() if v else v


class DataCleaningParams(BaseModel):
    """Parámetros para limpieza de datos"""
    remove_duplicates: bool = Field(default=True)
    handle_missing: str = Field(default="drop", pattern=r'^(drop|fill|interpolate)$')
    fill_method: Optional[str] = Field(default=None, pattern=r'^(mean|median|mode|forward_fill|backward_fill)$')
    outlier_method: Optional[str] = Field(default=None, pattern=r'^(iqr|zscore|isolation_forest)$')
    outlier_threshold: float = Field(default=3.0, ge=1.0, le=10.0)


class ValidationResult(BaseModel):
    """Resultado de validación de datos"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


def validate_dataframe(
    df: pd.DataFrame, 
    schema: DataFrameSchema,
    context: str = "validate_dataframe"
) -> ValidationResult:
    """
    Valida un DataFrame contra un esquema específico.
    
    Args:
        df: DataFrame a validar
        schema: Esquema de validación
        context: Contexto de la validación
        
    Returns:
        ValidationResult: Resultado de la validación
    """
    errors = []
    warnings = []
    details = {
        "dataframe_shape": df.shape,
        "dataframe_columns": list(df.columns),
        "schema_required": schema.required_columns,
        "schema_optional": schema.optional_columns
    }
    
    # Validar número de filas
    if len(df) < schema.min_rows:
        errors.append(f"El DataFrame debe tener al menos {schema.min_rows} filas, tiene {len(df)}")
    
    if schema.max_rows is not None and len(df) > schema.max_rows:
        errors.append(f"El DataFrame debe tener máximo {schema.max_rows} filas, tiene {len(df)}")
    
    # Validar columnas requeridas
    missing_columns = [col for col in schema.required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Columnas requeridas faltantes: {missing_columns}")
    
    # Validar columnas opcionales
    optional_missing = [col for col in schema.optional_columns if col not in df.columns]
    if optional_missing:
        warnings.append(f"Columnas opcionales no encontradas: {optional_missing}")
    
    # Validar tipos de columnas si se especifican
    if schema.column_types:
        type_errors = []
        for col, expected_type in schema.column_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    type_errors.append(f"Columna '{col}': esperado {expected_type}, encontrado {actual_type}")
        
        if type_errors:
            warnings.extend(type_errors)
            details["type_mismatches"] = type_errors
    
    # Validar que no haya columnas completamente vacías
    empty_columns = []
    for col in df.columns:
        if df[col].isna().all():
            empty_columns.append(col)
    
    if empty_columns:
        warnings.append(f"Columnas completamente vacías: {empty_columns}")
        details["empty_columns"] = empty_columns
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        details=details
    )


def validate_feature_selection_params(
    params: FeatureSelectionParams,
    df: pd.DataFrame,
    context: str = "validate_feature_selection"
) -> ValidationResult:
    """
    Valida parámetros de selección de features.
    
    Args:
        params: Parámetros de selección
        df: DataFrame de referencia
        context: Contexto de la validación
        
    Returns:
        ValidationResult: Resultado de la validación
    """
    errors = []
    warnings = []
    details = {
        "target_column": params.target_column,
        "method": params.method,
        "n_features": params.n_features,
        "available_columns": list(df.columns)
    }
    
    # Validar que la columna objetivo existe
    if params.target_column not in df.columns:
        errors.append(f"La columna objetivo '{params.target_column}' no existe en el DataFrame")
    
    # Validar que la columna objetivo no está completamente vacía
    elif df[params.target_column].isna().all():
        errors.append(f"La columna objetivo '{params.target_column}' está completamente vacía")
    
    # Validar columnas de features si se especifican
    if params.feature_columns:
        missing_features = [col for col in params.feature_columns if col not in df.columns]
        if missing_features:
            errors.append(f"Columnas de features faltantes: {missing_features}")
        
        # Validar que no se seleccione la columna objetivo como feature
        if params.target_column in params.feature_columns:
            errors.append(f"La columna objetivo '{params.target_column}' no puede ser una feature")
        
        # Validar número de features
        available_features = [col for col in params.feature_columns if col in df.columns]
        if params.n_features > len(available_features):
            warnings.append(f"Se solicitan {params.n_features} features pero solo hay {len(available_features)} disponibles")
    
    # Validar método de selección
    if params.method in ['f_regression', 'f_classif']:
        if params.target_column in df.columns:
            target_dtype = str(df[params.target_column].dtype)
            if params.method == 'f_regression' and target_dtype not in ['int64', 'float64']:
                errors.append(f"El método '{params.method}' requiere una columna objetivo numérica")
            elif params.method == 'f_classif' and target_dtype not in ['object', 'category']:
                warnings.append(f"El método '{params.method}' funciona mejor con columnas categóricas")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        details=details
    )


def validate_correlation_params(
    params: CorrelationParams,
    df: pd.DataFrame,
    context: str = "validate_correlation"
) -> ValidationResult:
    """
    Valida parámetros de análisis de correlación.
    
    Args:
        params: Parámetros de correlación
        df: DataFrame de referencia
        context: Contexto de la validación
        
    Returns:
        ValidationResult: Resultado de la validación
    """
    errors = []
    warnings = []
    details = {
        "method": params.method,
        "min_correlation": params.min_correlation,
        "significance_level": params.significance_level,
        "available_columns": list(df.columns)
    }
    
    # Validar columnas si se especifican
    if params.columns:
        missing_columns = [col for col in params.columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Columnas faltantes: {missing_columns}")
        
        # Validar que las columnas sean numéricas
        non_numeric = []
        for col in params.columns:
            if col in df.columns and str(df[col].dtype) not in ['int64', 'float64']:
                non_numeric.append(col)
        
        if non_numeric:
            errors.append(f"Columnas no numéricas: {non_numeric}")
    
    # Validar que hay suficientes datos para correlación
    if len(df) < 3:
        errors.append("Se necesitan al menos 3 filas para calcular correlaciones")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        details=details
    ) 