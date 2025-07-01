"""
Módulo de Estadísticas - Patrón "Reloj Suizo"
============================================

Responsabilidades:
- Estadísticas descriptivas básicas y avanzadas
- Análisis de correlaciones y contingencia
- Pruebas estadísticas (normalidad, t-test, chi2, etc.)
- Regresión lineal y logística
- Logging sistemático de operaciones
- Validación automática de entrada usando decoradores
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from scipy import stats
from scipy.stats import chi2_contingency, shapiro, ttest_ind, f_oneway
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import time

warnings.filterwarnings("ignore")

# Importar logging y validación
from .logging import log_action
from .validation_decorators import (
    validate_summary_stats, 
    validate_correlation, 
    validate_io,
    SummaryStatsSchema,
    CorrelationSchema,
    create_dataframe_schema
)
from .json_logging import LogLevel, LogCategory


def log_stats_operation(func):
    """
    Decorador para logging JSON de operaciones estadísticas.
    
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
                "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns)
            }
        
        try:
            # Log de inicio de operación estadística
            if hasattr(wrapper, 'json_logger'):
                wrapper.json_logger.log_event(
                    level=LogLevel.INFO,
                    message=f"Iniciando análisis estadístico: {operation_name}",
                    module=func.__module__,
                    function=func.__name__,
                    step="statistical_analysis",
                    category=LogCategory.ANALYSIS.value,
                    parameters={"operation": operation_name, "args_count": len(args), "kwargs_count": len(kwargs)},
                    before_metrics=df_info,
                    after_metrics=df_info,
                    execution_time=0.0,
                    tags=["statistical_analysis", operation_name, "start"],
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
                    "result_memory_mb": result.memory_usage(deep=True).sum() / (1024 * 1024)
                })
            elif isinstance(result, dict):
                after_metrics.update({
                    "result_keys": list(result.keys()),
                    "result_type": "dictionary"
                })
            elif isinstance(result, tuple):
                after_metrics.update({
                    "result_length": len(result),
                    "result_types": [type(item).__name__ for item in result]
                })
            
            # Log de éxito
            if hasattr(wrapper, 'json_logger'):
                wrapper.json_logger.log_event(
                    level=LogLevel.INFO,
                    message=f"Análisis estadístico completado: {operation_name}",
                    module=func.__module__,
                    function=func.__name__,
                    step="statistical_analysis",
                    category=LogCategory.ANALYSIS.value,
                    parameters={"operation": operation_name},
                    before_metrics=df_info,
                    after_metrics=after_metrics,
                    execution_time=execution_time,
                    tags=["statistical_analysis", operation_name, "success"],
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
                    context=f"statistical_analysis_{operation_name}",
                    execution_time=execution_time,
                    additional_data={
                        "operation": operation_name,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    }
                )
            
            raise
    
    return wrapper


@log_stats_operation
@validate_summary_stats
def summary_statistics(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Devuelve un DataFrame con media, mediana, std, skew, kurtosis e IQR para cada columna.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga al menos una columna numérica
    2. Valida que las columnas especificadas existan y sean numéricas
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame con los datos
        columns: Lista de columnas numéricas a analizar
        
    Returns:
        DataFrame con estadísticas descriptivas
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    valid_columns = [col for col in columns if col in numeric_cols]
    
    rows = []
    for col in valid_columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        q1, q3 = series.quantile([0.25, 0.75])
        rows.append({
            'variable': col,
            'count': len(series),
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'skew': series.skew(),
            'kurtosis': series.kurtosis(),
            'iqr': q3 - q1,
            'min': series.min(),
            'max': series.max()
        })
    
    result = pd.DataFrame(rows)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # Log adicional específico para estadísticas descriptivas
    if hasattr(summary_statistics, 'json_logger') and summary_statistics.json_logger:
        summary_statistics.json_logger.log_event(
            level=LogLevel.INFO,
            message=f"Estadísticas descriptivas calculadas para {len(result)} columnas",
            module=summary_statistics.__module__,
            function="summary_statistics",
            step="descriptive_statistics",
            category=LogCategory.ANALYSIS.value,
            parameters={"columns": valid_columns, "total_columns": len(columns)},
            before_metrics={"total_columns": len(columns)},
            after_metrics={
                "analyzed_columns": len(result),
                "success_rate": len(result) / len(columns) if len(columns) > 0 else 0,
                "statistics_per_column": len(rows[0]) if rows else 0
            },
            execution_time=execution_time,
            tags=["descriptive_statistics", "summary"],
            metadata={
                "valid_columns": valid_columns,
                "invalid_columns": [col for col in columns if col not in valid_columns]
            }
        )
    
    log_action(
        function="summary_statistics",
        step="stats",
        parameters={"columns": valid_columns},
        before_metrics={"total_columns": len(columns)},
        after_metrics={"analyzed_columns": len(result)},
        status="success",
        message=f"Estadísticas descriptivas calculadas para {len(result)} columnas",
        execution_time=execution_time
    )
    
    return result


@log_stats_operation
@validate_correlation
def compute_correlations(df: pd.DataFrame, columns: List[str], 
                        method: str = 'pearson') -> pd.DataFrame:
    """
    Correlación entre columnas numéricas. method in {'pearson','spearman','kendall'}.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga al menos 2 columnas numéricas
    2. Valida que el método de correlación sea válido
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame con los datos
        columns: Lista de columnas numéricas
        method: Método de correlación ('pearson', 'spearman', 'kendall')
        
    Returns:
        DataFrame con matriz de correlaciones
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    valid_columns = [col for col in columns if col in numeric_cols]
    
    # Calcular correlación
    corr = df[valid_columns].dropna().corr(method=method)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    # Log adicional específico para correlaciones
    if hasattr(compute_correlations, 'json_logger') and compute_correlations.json_logger:
        # Encontrar correlaciones fuertes
        strong_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_value = corr.iloc[i, j]
                if abs(corr_value) >= 0.7:
                    strong_corr.append({
                        "variable1": corr.columns[i],
                        "variable2": corr.columns[j],
                        "correlation": corr_value
                    })
        
        compute_correlations.json_logger.log_event(
            level=LogLevel.INFO,
            message=f"Matriz de correlación calculada ({method})",
            module=compute_correlations.__module__,
            function="compute_correlations",
            step="correlation_analysis",
            category=LogCategory.ANALYSIS.value,
            parameters={"columns": valid_columns, "method": method},
            before_metrics={"n_columns": len(valid_columns)},
            after_metrics={
                "correlation_matrix_size": corr.shape[0],
                "strong_correlations": len(strong_corr),
                "mean_correlation": corr.values[np.triu_indices_from(corr.values, k=1)].mean(),
                "max_correlation": corr.values[np.triu_indices_from(corr.values, k=1)].max()
            },
            execution_time=execution_time,
            tags=["correlation_analysis", method],
            metadata={
                "method": method,
                "strong_correlations": strong_corr,
                "valid_columns": valid_columns
            }
        )
    
    log_action(
        function="compute_correlations",
        step="stats",
        parameters={"columns": valid_columns, "method": method},
        before_metrics={"n_columns": len(valid_columns)},
        after_metrics={"correlation_matrix_size": corr.shape[0]},
        status="success",
        message=f"Matriz de correlación calculada ({method})",
        execution_time=execution_time
    )
    
    return corr


# Esquema específico para análisis de contingencia
class ContingencySchema(create_dataframe_schema(min_rows=1)):
    """Esquema para análisis de contingencia"""
    def validate_dataframe(self, df: pd.DataFrame, context: str):
        """Validación específica para análisis de contingencia"""
        from .data_validators import ValidationResult
        
        # Verificar que hay al menos una columna categórica
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_cols) == 0:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame debe contener al menos una columna categórica para análisis de contingencia"],
                warnings=[],
                details={"categorical_columns": categorical_cols}
            )
        
        return super().validate_dataframe(df, context)


@validate_io(df_schema=ContingencySchema)
def contingency_analysis(df: pd.DataFrame, col1: str, col2: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Devuelve (tabla_frecuencias, stats) donde stats incluye chi2, p-value, cramer_v.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas categóricas
    2. Valida que las columnas especificadas existan
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame con los datos
        col1: Primera columna categórica
        col2: Segunda columna categórica
        
    Returns:
        Tuple: (tabla de contingencia, estadísticas)
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    # Crear tabla de contingencia
    table = pd.crosstab(df[col1], df[col2])
    
    # Calcular estadísticas
    chi2_stat, p, dof, _ = chi2_contingency(table)
    n = table.values.sum()
    min_dim = min(table.shape) - 1
    cramer_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0
    
    stats_result = {
        'chi2': chi2_stat,
        'p_value': p,
        'cramer_v': cramer_v,
        'dof': dof,
        'significant': p < 0.05
    }
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="contingency_analysis",
        step="stats",
        parameters={"col1": col1, "col2": col2},
        before_metrics={"table_shape": table.shape},
        after_metrics=stats_result,
        status="success",
        message=f"Análisis de contingencia completado (p={p:.4f})",
        execution_time=execution_time
    )
    
    return table, stats_result


# Esquema específico para pruebas de normalidad
class NormalitySchema(create_dataframe_schema(min_rows=3)):
    """Esquema para pruebas de normalidad"""
    def validate_dataframe(self, df: pd.DataFrame, context: str):
        """Validación específica para pruebas de normalidad"""
        from .data_validators import ValidationResult
        
        # Verificar que hay al menos una columna numérica
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) == 0:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame debe contener al menos una columna numérica para pruebas de normalidad"],
                warnings=[],
                details={"numeric_columns": numeric_cols}
            )
        
        return super().validate_dataframe(df, context)


@validate_io(df_schema=NormalitySchema)
def normality_test(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Realiza pruebas de normalidad en una columna numérica.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas numéricas
    2. Valida que la columna especificada exista
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame con los datos
        column: Columna numérica a analizar
        
    Returns:
        Diccionario con resultados de las pruebas
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    series = df[column].dropna()
    if len(series) < 3:
        return {
            'error': 'Se necesitan al menos 3 valores no nulos para la prueba de normalidad'
        }
    
    # Realizar pruebas
    shapiro_stat, shapiro_p = shapiro(series)
    ks_stat, ks_p = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
    
    # Interpretar resultados
    is_normal_shapiro = shapiro_p > 0.05
    is_normal_ks = ks_p > 0.05
    
    result = {
        'shapiro_wilk': {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': is_normal_shapiro,
            'interpretation': 'Normal' if is_normal_shapiro else 'No normal'
        },
        'kolmogorov_smirnov': {
            'statistic': ks_stat,
            'p_value': ks_p,
            'is_normal': is_normal_ks,
            'interpretation': 'Normal' if is_normal_ks else 'No normal'
        },
        'overall_conclusion': 'Normal' if (is_normal_shapiro and is_normal_ks) else 'No normal'
    }
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="normality_test",
        step="stats",
        parameters={"column": column},
        before_metrics={"n_values": len(series)},
        after_metrics=result,
        status="success",
        message=f"Prueba de normalidad completada ({result['overall_conclusion']})",
        execution_time=execution_time
    )
    
    return result


# Esquema específico para t-test
class TTestSchema(create_dataframe_schema(min_rows=4)):
    """Esquema para t-test independiente"""
    def validate_dataframe(self, df: pd.DataFrame, context: str):
        """Validación específica para t-test"""
        from .data_validators import ValidationResult
        
        # Verificar que hay al menos una columna numérica
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) == 0:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame debe contener al menos una columna numérica para t-test"],
                warnings=[],
                details={"numeric_columns": numeric_cols}
            )
        
        return super().validate_dataframe(df, context)


@validate_io(df_schema=TTestSchema)
def t_test_independent(df: pd.DataFrame, column: str, group_column: str) -> Dict[str, Any]:
    """
    Realiza t-test de muestras independientes.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas numéricas
    2. Valida que las columnas especificadas existan
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame con los datos
        column: Columna numérica a analizar
        group_column: Columna categórica para agrupar
        
    Returns:
        Diccionario con resultados del t-test
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    # Preparar datos
    data = df[[column, group_column]].dropna()
    if len(data) == 0:
        return {
            'error': 'No hay datos válidos después de eliminar valores faltantes'
        }
    
    groups = data[group_column].unique()
    if len(groups) != 2:
        return {
            'error': f'La columna de grupo debe tener exactamente 2 valores únicos, encontró {len(groups)}'
        }
    
    group1_data = data[data[group_column] == groups[0]][column]
    group2_data = data[data[group_column] == groups[1]][column]
    
    if len(group1_data) < 2 or len(group2_data) < 2:
        return {
            'error': 'Cada grupo debe tener al menos 2 observaciones'
        }
    
    # Realizar t-test
    t_stat, p_value = ttest_ind(group1_data, group2_data)
    
    # Calcular tamaño del efecto (Cohen's d)
    pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                         (len(group2_data) - 1) * group2_data.var()) / 
                        (len(group1_data) + len(group2_data) - 2))
    cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
    
    result = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'group1_mean': group1_data.mean(),
        'group2_mean': group2_data.mean(),
        'group1_std': group1_data.std(),
        'group2_std': group2_data.std(),
        'group1_n': len(group1_data),
        'group2_n': len(group2_data)
    }
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="t_test_independent",
        step="stats",
        parameters={"column": column, "group_column": group_column},
        before_metrics={"total_observations": len(data)},
        after_metrics=result,
        status="success",
        message=f"T-test independiente completado (p={p_value:.4f})",
        execution_time=execution_time
    )
    
    return result


# Esquema específico para regresión lineal
class RegressionSchema(create_dataframe_schema(min_rows=2)):
    """Esquema para regresión lineal"""
    def validate_dataframe(self, df: pd.DataFrame, context: str):
        """Validación específica para regresión lineal"""
        from .data_validators import ValidationResult
        
        # Verificar que hay al menos una columna numérica
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) == 0:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame debe contener al menos una columna numérica para regresión lineal"],
                warnings=[],
                details={"numeric_columns": numeric_cols}
            )
        
        return super().validate_dataframe(df, context)


@validate_io(df_schema=RegressionSchema)
def linear_regression(df: pd.DataFrame, y_column: str, x_columns: List[str]) -> Dict[str, Any]:
    """
    Realiza regresión lineal múltiple.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas numéricas
    2. Valida que las columnas especificadas existan
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame con los datos
        y_column: Variable dependiente
        x_columns: Variables independientes
        
    Returns:
        Diccionario con resultados de la regresión
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    if len(x_columns) == 0:
        return {
            'error': 'Debe especificar al menos una variable independiente'
        }
    
    # Preparar datos
    all_columns = [y_column] + x_columns
    data = df[all_columns].dropna()
    
    if len(data) == 0:
        return {
            'error': 'No hay datos válidos después de eliminar valores faltantes'
        }
    
    if len(data) < len(x_columns) + 1:
        return {
            'error': f'No hay suficientes observaciones ({len(data)}) para el número de variables ({len(x_columns) + 1})'
        }
    
    X = data[x_columns]
    y = data[y_column]
    
    # Realizar regresión
    model = LinearRegression()
    model.fit(X, y)
    
    # Predicciones
    y_pred = model.predict(X)
    
    # Métricas
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Coeficientes
    coefficients = dict(zip(x_columns, model.coef_))
    
    result = {
        'r2_score': r2,
        'mse': mse,
        'rmse': rmse,
        'intercept': model.intercept_,
        'coefficients': coefficients,
        'n_observations': len(data),
        'n_features': len(x_columns)
    }
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="linear_regression",
        step="stats",
        parameters={"y_column": y_column, "x_columns": x_columns},
        before_metrics={"n_observations": len(data)},
        after_metrics={"r2_score": r2, "rmse": rmse},
        status="success",
        message=f"Regresión lineal completada (R²={r2:.4f})",
        execution_time=execution_time
    )
    
    return result


# Esquema específico para tabla de frecuencias
class FrequencySchema(create_dataframe_schema(min_rows=1)):
    """Esquema para tabla de frecuencias"""
    def validate_dataframe(self, df: pd.DataFrame, context: str):
        """Validación específica para tabla de frecuencias"""
        from .data_validators import ValidationResult
        
        # Verificar que hay al menos una columna
        if len(df.columns) == 0:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame debe contener al menos una columna para tabla de frecuencias"],
                warnings=[],
                details={"columns": list(df.columns)}
            )
        
        return super().validate_dataframe(df, context)


@validate_io(df_schema=FrequencySchema)
def frequency_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Crea una tabla de frecuencias para una columna categórica.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas
    2. Valida que la columna especificada exista
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame con los datos
        column: Columna categórica a analizar
        
    Returns:
        DataFrame con tabla de frecuencias
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    # Crear tabla de frecuencias
    freq_table = df[column].value_counts().reset_index()
    freq_table.columns = ['valor', 'frecuencia']
    freq_table['porcentaje'] = (freq_table['frecuencia'] / freq_table['frecuencia'].sum() * 100).round(2)
    freq_table['porcentaje_acumulado'] = freq_table['porcentaje'].cumsum().round(2)
    
    # Ordenar por frecuencia descendente
    freq_table = freq_table.sort_values('frecuencia', ascending=False).reset_index(drop=True)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="frequency_table",
        step="stats",
        parameters={"column": column},
        before_metrics={"unique_values": len(freq_table)},
        after_metrics={"total_frequency": freq_table['frecuencia'].sum()},
        status="success",
        message=f"Tabla de frecuencias creada para {column}",
        execution_time=execution_time
    )
    
    return freq_table


# Esquema específico para detección de outliers
class OutlierSchema(create_dataframe_schema(min_rows=3)):
    """Esquema para detección de outliers"""
    def validate_dataframe(self, df: pd.DataFrame, context: str):
        """Validación específica para detección de outliers"""
        from .data_validators import ValidationResult
        
        # Verificar que hay al menos una columna numérica
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) == 0:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame debe contener al menos una columna numérica para detección de outliers"],
                warnings=[],
                details={"numeric_columns": numeric_cols}
            )
        
        return super().validate_dataframe(df, context)


@validate_io(df_schema=OutlierSchema)
def outlier_detection(df: pd.DataFrame, column: str, method: str = 'iqr') -> Dict[str, Any]:
    """
    Detecta outliers en una columna numérica usando diferentes métodos.
    
    Esta función está protegida por validación automática que:
    1. Verifica que el DataFrame tenga columnas numéricas
    2. Valida que la columna especificada exista
    3. Reporta errores detallados si la validación falla
    
    Args:
        df: DataFrame con los datos
        column: Columna numérica a analizar
        method: Método de detección ('iqr', 'zscore', 'isolation_forest')
        
    Returns:
        Diccionario con información de outliers detectados
    """
    start_time = datetime.now()
    
    # La validación automática ya se encargó de verificar la entrada
    series = df[column].dropna()
    
    if len(series) < 3:
        return {
            'error': 'Se necesitan al menos 3 valores no nulos para detectar outliers'
        }
    
    outliers = []
    
    if method == 'iqr':
        # Método IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        result = {
            'method': 'iqr',
            'q1': Q1,
            'q3': Q3,
            'iqr': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers_count': len(outliers),
            'outliers_percentage': (len(outliers) / len(series) * 100).round(2),
            'outlier_values': outliers.tolist()
        }
        
    elif method == 'zscore':
        # Método Z-score
        z_scores = np.abs(stats.zscore(series))
        threshold = 3
        outliers = series[z_scores > threshold]
        
        result = {
            'method': 'zscore',
            'threshold': threshold,
            'outliers_count': len(outliers),
            'outliers_percentage': (len(outliers) / len(series) * 100).round(2),
            'outlier_values': outliers.tolist(),
            'z_scores': z_scores[z_scores > threshold].tolist()
        }
        
    else:
        return {
            'error': f'Método no válido: {method}. Métodos disponibles: iqr, zscore'
        }
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="outlier_detection",
        step="stats",
        parameters={"column": column, "method": method},
        before_metrics={"n_values": len(series)},
        after_metrics={"outliers_detected": len(outliers)},
        status="success",
        message=f"Detección de outliers completada ({method})",
        execution_time=execution_time
    )
    
    return result
