"""
Módulo de Features - Patrón "Reloj Suizo"
========================================

Responsabilidades:
- Creación de features básicas y derivadas
- Transformaciones de variables
- Codificación de variables categóricas
- Escalado y normalización
- Logging sistemático de operaciones
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import warnings

warnings.filterwarnings("ignore")

# Importar logging
from .logging import log_action

# Importar sistema de validación y reporte de errores
from .data_validators import (
    FeatureSelectionParams, 
    DataFrameSchema, 
    validate_dataframe, 
    validate_feature_selection_params
)
from .error_reporter import report_dataframe_error, report_parameter_error

def create_numeric_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Crea features numéricas derivadas de columnas existentes.
    
    Args:
        df: DataFrame original
        columns: Lista de columnas numéricas para crear features
        
    Returns:
        DataFrame con features adicionales
    """
    start_time = datetime.now()
    
    # Validar entrada
    if not isinstance(columns, list) or len(columns) == 0:
        raise ValueError("Debe especificar una lista no vacía de columnas")
    
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas no encontradas: {missing_cols}")
    
    # Crear copia del DataFrame
    result_df = df.copy()
    
    # Crear features para cada columna
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            series = df[col].dropna()
            if len(series) > 0:
                # Logaritmo (solo para valores positivos)
                if (series > 0).all():
                    result_df[f'{col}_log'] = np.log(series)
                
                # Cuadrado
                result_df[f'{col}_squared'] = series ** 2
                
                # Raíz cuadrada (solo para valores no negativos)
                if (series >= 0).all():
                    result_df[f'{col}_sqrt'] = np.sqrt(series)
                
                # Recíproco (evitando división por cero)
                if (series != 0).all():
                    result_df[f'{col}_reciprocal'] = 1 / series
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="create_numeric_features",
        step="features",
        parameters={"columns": columns},
        before_metrics={"n_columns": len(df.columns)},
        after_metrics={"n_columns": len(result_df.columns)},
        status="success",
        message=f"Features numéricas creadas para {len(columns)} columnas",
        execution_time=execution_time
    )
    
    return result_df

def encode_categorical(df: pd.DataFrame, columns: List[str], 
                      method: str = 'label') -> pd.DataFrame:
    """
    Codifica variables categóricas usando diferentes métodos.
    
    Args:
        df: DataFrame original
        columns: Lista de columnas categóricas
        method: Método de codificación ('label', 'onehot', 'frequency')
        
    Returns:
        DataFrame con variables codificadas
    """
    start_time = datetime.now()
    
    # Validar entrada
    valid_methods = ['label', 'onehot', 'frequency']
    if method not in valid_methods:
        raise ValueError(f"Método debe ser uno de: {valid_methods}")
    
    if not isinstance(columns, list) or len(columns) == 0:
        raise ValueError("Debe especificar una lista no vacía de columnas")
    
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas no encontradas: {missing_cols}")
    
    # Crear copia del DataFrame
    result_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            if method == 'label':
                # Label encoding
                le = LabelEncoder()
                result_df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                
            elif method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                result_df = pd.concat([result_df, dummies], axis=1)
                
            elif method == 'frequency':
                # Frequency encoding
                freq_map = df[col].value_counts(normalize=True).to_dict()
                result_df[f'{col}_freq'] = df[col].map(freq_map)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="encode_categorical",
        step="features",
        parameters={"columns": columns, "method": method},
        before_metrics={"n_columns": len(df.columns)},
        after_metrics={"n_columns": len(result_df.columns)},
        status="success",
        message=f"Variables categóricas codificadas usando {method}",
        execution_time=execution_time
    )
    
    return result_df

def scale_features(df: pd.DataFrame, columns: List[str], 
                  method: str = 'standard') -> pd.DataFrame:
    """
    Escala features numéricas usando diferentes métodos.
    
    Args:
        df: DataFrame original
        columns: Lista de columnas numéricas a escalar
        method: Método de escalado ('standard', 'minmax', 'robust')
        
    Returns:
        DataFrame con features escaladas
    """
    start_time = datetime.now()
    
    # Validar entrada
    valid_methods = ['standard', 'minmax', 'robust']
    if method not in valid_methods:
        raise ValueError(f"Método debe ser uno de: {valid_methods}")
    
    if not isinstance(columns, list) or len(columns) == 0:
        raise ValueError("Debe especificar una lista no vacía de columnas")
    
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas no encontradas: {missing_cols}")
    
    # Filtrar columnas numéricas
    numeric_cols = [col for col in columns if df[col].dtype in ['int64', 'float64']]
    
    if len(numeric_cols) == 0:
        raise ValueError("No se encontraron columnas numéricas válidas")
    
    # Crear copia del DataFrame
    result_df = df.copy()
    
    # Aplicar escalado
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:  # robust
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    
    # Escalar las columnas
    scaled_data = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=[f'{col}_{method}_scaled' for col in numeric_cols],
                            index=df.index)
    
    # Concatenar con el DataFrame original
    result_df = pd.concat([result_df, scaled_df], axis=1)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="scale_features",
        step="features",
        parameters={"columns": numeric_cols, "method": method},
        before_metrics={"n_columns": len(df.columns)},
        after_metrics={"n_columns": len(result_df.columns)},
        status="success",
        message=f"Features escaladas usando {method}",
        execution_time=execution_time
    )
    
    return result_df

def create_interaction_features(df: pd.DataFrame, columns: List[str], 
                               max_interactions: int = 10) -> pd.DataFrame:
    """
    Crea features de interacción entre columnas numéricas.
    
    Args:
        df: DataFrame original
        columns: Lista de columnas numéricas
        max_interactions: Número máximo de interacciones a crear
        
    Returns:
        DataFrame con features de interacción
    """
    start_time = datetime.now()
    
    # Validar entrada
    if not isinstance(columns, list) or len(columns) < 2:
        raise ValueError("Debe especificar al menos 2 columnas")
    
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas no encontradas: {missing_cols}")
    
    # Filtrar columnas numéricas
    numeric_cols = [col for col in columns if df[col].dtype in ['int64', 'float64']]
    
    if len(numeric_cols) < 2:
        raise ValueError("Se necesitan al menos 2 columnas numéricas")
    
    # Crear copia del DataFrame
    result_df = df.copy()
    
    # Crear interacciones
    interaction_count = 0
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            if interaction_count >= max_interactions:
                break
                
            col1, col2 = numeric_cols[i], numeric_cols[j]
            
            # Multiplicación
            result_df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            
            # División (evitando división por cero)
            if (df[col2] != 0).all():
                result_df[f'{col1}_div_{col2}'] = df[col1] / df[col2]
            
            # Suma
            result_df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
            
            # Diferencia
            result_df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
            
            interaction_count += 1
        
        if interaction_count >= max_interactions:
            break
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="create_interaction_features",
        step="features",
        parameters={"columns": columns, "max_interactions": max_interactions},
        before_metrics={"n_columns": len(df.columns)},
        after_metrics={"n_columns": len(result_df.columns), "interactions_created": interaction_count},
        status="success",
        message=f"Features de interacción creadas: {interaction_count}",
        execution_time=execution_time
    )
    
    return result_df

def select_features(
    df: pd.DataFrame,
    target_column: str,
    method: str = 'correlation',
    n_features: int = 10,
    feature_columns: Optional[List[str]] = None
) -> List[str]:
    """
    Selecciona las mejores features basándose en diferentes métodos estadísticos.
    
    Esta función implementa validación robusta de parámetros de entrada:
    1. Valida que el DataFrame tenga la estructura esperada
    2. Valida que los parámetros sean del tipo y rango correctos
    3. Reporta errores detallados que detienen la aplicación
    4. Registra la ejecución para auditoría
    
    Args:
        df: DataFrame con los datos
        target_column: Nombre de la columna objetivo
        method: Método de selección ('correlation', 'mutual_info', 'f_regression', 'f_classif')
        n_features: Número de features a seleccionar
        feature_columns: Lista de columnas a considerar como features (opcional)
        
    Returns:
        List[str]: Lista de nombres de las mejores features seleccionadas
        
    Raises:
        ValidationError: Si los parámetros o datos son inválidos
    """
    start_time = datetime.now()
    
    # Validar parámetros de entrada usando Pydantic
    try:
        params = FeatureSelectionParams(
            target_column=target_column,
            method=method,
            n_features=n_features,
            feature_columns=feature_columns
        )
    except Exception as e:
        report_parameter_error(
            message=f"Parámetros de selección de features inválidos: {str(e)}",
            context="select_features",
            details={
                "target_column": target_column,
                "method": method,
                "n_features": n_features,
                "feature_columns": feature_columns,
                "validation_error": str(e)
            }
        )
    
    # Validar DataFrame
    schema = DataFrameSchema(
        required_columns=[target_column],
        min_rows=3,
        column_types={target_column: 'object'}  # Permitir tanto numérico como categórico
    )
    
    validation_result = validate_dataframe(df, schema, "select_features")
    if not validation_result.is_valid:
        report_dataframe_error(
            message=f"DataFrame inválido para selección de features: {validation_result.errors[0]}",
            context="select_features",
            details={
                "validation_errors": validation_result.errors,
                "validation_warnings": validation_result.warnings,
                "validation_details": validation_result.details
            }
        )
    
    # Validar parámetros contra el DataFrame
    param_validation = validate_feature_selection_params(params, df, "select_features")
    if not param_validation.is_valid:
        report_parameter_error(
            message=f"Parámetros incompatibles con el DataFrame: {param_validation.errors[0]}",
            context="select_features",
            details={
                "validation_errors": param_validation.errors,
                "validation_warnings": param_validation.warnings,
                "validation_details": param_validation.details
            }
        )
    
    # Si hay warnings, registrarlos pero continuar
    if param_validation.warnings:
        log_action(
            function="select_features",
            step="validation",
            parameters={"target_column": target_column, "method": method, "n_features": n_features},
            status="warning",
            message=f"Warnings de validación: {param_validation.warnings}",
            execution_time=0
        )
    
    # Preparar datos para análisis
    if feature_columns is None:
        feature_cols = [col for col in df.columns if col != target_column]
    else:
        feature_cols = [col for col in feature_columns if col in df.columns and col != target_column]
    
    if len(feature_cols) == 0:
        report_dataframe_error(
            message="No hay columnas de features disponibles para análisis",
            context="select_features",
            details={
                "available_columns": list(df.columns),
                "target_column": target_column,
                "feature_columns": feature_columns
            }
        )
    
    # Preparar datos numéricos
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_column]
    
    if len(X.columns) == 0:
        report_dataframe_error(
            message="No hay columnas numéricas disponibles para análisis de correlación",
            context="select_features",
            details={
                "feature_columns": feature_cols,
                "numeric_columns": list(X.columns),
                "target_column": target_column
            }
        )
    
    # Aplicar selección de features
    selected_features: List[str] = []
    
    if method == 'correlation':
        try:
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(n_features).index.tolist()
        except Exception as e:
            report_dataframe_error(
                message=f"Error al calcular correlaciones: {str(e)}",
                context="select_features(correlation)",
                details={
                    "method": method,
                    "feature_columns": list(X.columns),
                    "target_column": target_column,
                    "error": str(e)
                }
            )
        
    elif method == 'mutual_info':
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            if y.dtype in ['int64', 'object']:
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)
            
            feature_scores = list(zip(feature_cols, mi_scores))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            selected_features = [feature for feature, score in feature_scores[:n_features]]
        except Exception as e:
            report_dataframe_error(
                message=f"Error al calcular información mutua: {str(e)}",
                context="select_features(mutual_info)",
                details={
                    "method": method,
                    "feature_columns": list(X.columns),
                    "target_column": target_column,
                    "error": str(e)
                }
            )
        
    elif method in ['f_regression', 'f_classif']:
        try:
            if method == 'f_regression':
                selector = SelectKBest(score_func=f_regression, k=n_features)
            else:
                selector = SelectKBest(score_func=f_classif, k=n_features)
            
            selector.fit(X, y)
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        except Exception as e:
            report_dataframe_error(
                message=f"Error en selección {method}: {str(e)}",
                context=f"select_features({method})",
                details={
                    "method": method,
                    "feature_columns": list(X.columns),
                    "target_column": target_column,
                    "error": str(e)
                }
            )
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # Registrar ejecución exitosa
    log_action(
        function="select_features",
        step="features",
        parameters={"target_column": target_column, "method": method, "n_features": n_features},
        before_metrics={"n_features": len(feature_cols)},
        after_metrics={"n_selected": len(selected_features)},
        status="success",
        message=f"Features seleccionadas usando {method}",
        execution_time=execution_time
    )
    
    return selected_features

def create_time_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Crea features temporales a partir de una columna de fecha.
    
    Args:
        df: DataFrame original
        date_column: Columna con fechas
        
    Returns:
        DataFrame con features temporales
    """
    start_time = datetime.now()
    
    # Validar entrada
    if date_column not in df.columns:
        raise ValueError(f"La columna {date_column} no existe en el DataFrame")
    
    # Convertir a datetime si es necesario
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            raise ValueError(f"No se pudo convertir {date_column} a datetime: {e}")
    
    # Crear copia del DataFrame
    result_df = df.copy()
    
    # Extraer componentes temporales
    result_df[f'{date_column}_year'] = df[date_column].dt.year
    result_df[f'{date_column}_month'] = df[date_column].dt.month
    result_df[f'{date_column}_day'] = df[date_column].dt.day
    result_df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
    result_df[f'{date_column}_quarter'] = df[date_column].dt.quarter
    result_df[f'{date_column}_is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Crear feature de estación
    def get_season(month: int) -> int:
        if month in [12, 1, 2]:
            return 1  # Invierno
        elif month in [3, 4, 5]:
            return 2  # Primavera
        elif month in [6, 7, 8]:
            return 3  # Verano
        else:
            return 4  # Otoño
    
    result_df[f'{date_column}_season'] = df[date_column].dt.month.apply(get_season)
    
    # Crear feature de día del año
    result_df[f'{date_column}_dayofyear'] = df[date_column].dt.dayofyear
    
    # Crear feature de semana del año
    result_df[f'{date_column}_weekofyear'] = df[date_column].dt.isocalendar().week
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="create_time_features",
        step="features",
        parameters={"date_column": date_column},
        before_metrics={"n_columns": len(df.columns)},
        after_metrics={"n_columns": len(result_df.columns)},
        status="success",
        message=f"Features temporales creadas para {date_column}",
        execution_time=execution_time
    )
    
    return result_df

def create_binning_features(df: pd.DataFrame, columns: List[str], 
                           n_bins: int = 5, method: str = 'equal_width') -> pd.DataFrame:
    """
    Crea features de binning para columnas numéricas.
    
    Args:
        df: DataFrame original
        columns: Lista de columnas numéricas
        n_bins: Número de bins
        method: Método de binning ('equal_width', 'equal_frequency', 'quantile')
        
    Returns:
        DataFrame con features de binning
    """
    start_time = datetime.now()
    
    # Validar entrada
    valid_methods = ['equal_width', 'equal_frequency', 'quantile']
    if method not in valid_methods:
        raise ValueError(f"Método debe ser uno de: {valid_methods}")
    
    if not isinstance(columns, list) or len(columns) == 0:
        raise ValueError("Debe especificar una lista no vacía de columnas")
    
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas no encontradas: {missing_cols}")
    
    # Crear copia del DataFrame
    result_df = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            if method == 'equal_width':
                # Binning de ancho igual
                bins = pd.cut(series, bins=n_bins, labels=False, include_lowest=True)
                result_df[f'{col}_binned'] = bins
            
            elif method == 'equal_frequency':
                # Binning de frecuencia igual
                bins = pd.qcut(series, q=n_bins, labels=False, duplicates='drop')
                result_df[f'{col}_binned'] = bins
            
            elif method == 'quantile':
                # Binning por cuantiles
                quantiles = series.quantile([i/n_bins for i in range(1, n_bins)])
                bins = pd.cut(series, bins=quantiles, labels=False, include_lowest=True)
                result_df[f'{col}_binned'] = bins
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="create_binning_features",
        step="features",
        parameters={"columns": columns, "n_bins": n_bins, "method": method},
        before_metrics={"n_columns": len(df.columns)},
        after_metrics={"n_columns": len(result_df.columns)},
        status="success",
        message=f"Features de binning creadas usando {method}",
        execution_time=execution_time
    )
    
    return result_df
