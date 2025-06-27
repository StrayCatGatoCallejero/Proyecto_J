"""
Módulo de Análisis Estadístico Avanzado
=======================================

Responsabilidades:
- Análisis univariante avanzado con interpretación semántica
- Selección automática de pruebas estadísticas basada en semántica
- Modelos de regresión básicos (lineal, logística)
- Sugerencias automáticas de análisis para el front-end
- Trazabilidad completa de todos los análisis estadísticos
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
from scipy import stats
from scipy.stats import chi2_contingency, shapiro, normaltest, ttest_ind, f_oneway
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import re
from collections import Counter
import logging
warnings.filterwarnings('ignore')

# Importar logging
from .logging import log_action

logger = logging.getLogger(__name__)

def summary_statistics_advanced(
    df: pd.DataFrame, 
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calcula estadísticas univariantes avanzadas con interpretación semántica.
    
    Args:
        df: DataFrame con los datos
        metadata: Metadatos semánticos de las columnas
        
    Returns:
        Diccionario con estadísticas avanzadas por columna
    """
    start_time = datetime.now()
    
    try:
        # Métricas antes del cálculo
        before_metrics = {
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns)
        }
        
        results = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            col_data = df[column].dropna()
            if len(col_data) == 0:
                continue
            
            # Estadísticas básicas
            basic_stats = {
                'count': len(col_data),
                'mean': col_data.mean(),
                'median': col_data.median(),
                'mode': col_data.mode().iloc[0] if len(col_data.mode()) > 0 else np.nan,
                'std': col_data.std(),
                'variance': col_data.var(),
                'min': col_data.min(),
                'max': col_data.max(),
                'range': col_data.max() - col_data.min()
            }
            
            # Estadísticas de forma
            shape_stats = {
                'skewness': stats.skew(col_data),
                'kurtosis': stats.kurtosis(col_data),
                'skewness_interpretation': _interpret_skewness(stats.skew(col_data)),
                'kurtosis_interpretation': _interpret_kurtosis(stats.kurtosis(col_data))
            }
            
            # Percentiles y cuartiles
            percentiles = {
                'q1': col_data.quantile(0.25),
                'q3': col_data.quantile(0.75),
                'iqr': col_data.quantile(0.75) - col_data.quantile(0.25)
            }
            
            # Percentiles adicionales para variables económicas
            if metadata and column.lower() in ['ingresos', 'income', 'salario', 'salary', 'renta', 'rent']:
                percentiles.update({
                    'p10': col_data.quantile(0.10),
                    'p25': col_data.quantile(0.25),
                    'p75': col_data.quantile(0.75),
                    'p90': col_data.quantile(0.90),
                    'p95': col_data.quantile(0.95),
                    'p99': col_data.quantile(0.99)
                })
            
            # Detección de outliers por IQR
            q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_stats = {
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers) / len(col_data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_values': outliers.tolist() if len(outliers) <= 10 else outliers.head(10).tolist()
            }
            
            # Pruebas de normalidad
            normality_tests = {}
            if len(col_data) >= 3 and len(col_data) <= 5000:  # Shapiro-Wilk tiene límites
                try:
                    shapiro_stat, shapiro_p = shapiro(col_data)
                    normality_tests['shapiro_wilk'] = {
                        'statistic': shapiro_stat,
                        'p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05,
                        'interpretation': 'Normal' if shapiro_p > 0.05 else 'No normal'
                    }
                except:
                    pass
            
            # Test de Kolmogorov-Smirnov
            try:
                ks_stat, ks_p = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                normality_tests['kolmogorov_smirnov'] = {
                    'statistic': ks_stat,
                    'p_value': ks_p,
                    'is_normal': ks_p > 0.05,
                    'interpretation': 'Normal' if ks_p > 0.05 else 'No normal'
                }
            except:
                pass
            
            # Interpretación semántica
            semantic_interpretation = _get_semantic_interpretation(
                column, basic_stats, shape_stats, outlier_stats, normality_tests, metadata
            )
            
            results[column] = {
                'basic_stats': basic_stats,
                'shape_stats': shape_stats,
                'percentiles': percentiles,
                'outlier_stats': outlier_stats,
                'normality_tests': normality_tests,
                'semantic_interpretation': semantic_interpretation,
                'recommendations': _get_statistical_recommendations(
                    column, basic_stats, shape_stats, outlier_stats, normality_tests, metadata
                )
            }
        
        # Métricas después del cálculo
        after_metrics = {
            'columns_analyzed': len(results),
            'total_outliers': sum(results[col]['outlier_stats']['outliers_count'] for col in results),
            'non_normal_variables': len([col for col in results if any(
                test.get('is_normal', True) == False for test in results[col]['normality_tests'].values()
            )])
        }
        
        # Registrar acción
        log_action(
            function='summary_statistics_advanced',
            step='statistical_analysis',
            parameters={'metadata_provided': metadata is not None},
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            status='success',
            message=f"Estadísticas avanzadas calculadas para {len(results)} columnas",
            execution_time=(datetime.now() - start_time).total_seconds()
        )
        
        return {
            'column_statistics': results,
            'summary': {
                'total_columns_analyzed': len(results),
                'total_outliers': after_metrics['total_outliers'],
                'non_normal_variables': after_metrics['non_normal_variables'],
                'recommendations_summary': _get_global_recommendations(results)
            }
        }
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        
        log_action(
            function='summary_statistics_advanced',
            step='statistical_analysis',
            parameters={'metadata_provided': metadata is not None},
            before_metrics=before_metrics if 'before_metrics' in locals() else {},
            after_metrics={},
            status='error',
            message=f"Error en estadísticas avanzadas: {str(e)}",
            execution_time=execution_time,
            error_details=str(e)
        )
        
        raise ValueError(f"Error en estadísticas avanzadas: {str(e)}")

def recommend_statistical_tests(
    df: pd.DataFrame,
    metadata: Dict[str, Any] = None,
    semantic_classification: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Recomienda pruebas estadísticas basadas en semántica y distribución de datos.
    
    Args:
        df: DataFrame con los datos
        metadata: Metadatos de las columnas
        semantic_classification: Clasificación semántica previa
        
    Returns:
        Lista de pruebas recomendadas con justificación
    """
    start_time = datetime.now()
    
    try:
        recommendations = []
        
        # Análisis de variables numéricas
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Pruebas de normalidad para variables numéricas
        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) >= 3:
                recommendations.append({
                    'test_name': 'Shapiro-Wilk normality test',
                    'columns': [col],
                    'rationale': f'Verificar normalidad de {col} para determinar pruebas paramétricas vs no paramétricas',
                    'category': 'normality_test',
                    'priority': 'high' if semantic_classification and col in semantic_classification else 'medium'
                })
        
        # Comparaciones de grupos basadas en semántica
        if semantic_classification:
            # Buscar variables de grupo (género, región, etc.)
            group_variables = []
            numeric_variables = []
            
            for col, classification in semantic_classification.items():
                if classification['category'] in ['demographic', 'geographic']:
                    group_variables.append(col)
                elif classification['category'] in ['socioeconomic', 'opinion'] and col in numeric_columns:
                    numeric_variables.append(col)
            
            # t-test o ANOVA para comparar grupos
            for group_var in group_variables:
                if group_var in df.columns:
                    unique_groups = df[group_var].dropna().nunique()
                    
                    for num_var in numeric_variables:
                        if num_var in df.columns:
                            if unique_groups == 2:
                                test_name = 'Independent t-test'
                                rationale = f'Comparar {num_var} entre dos grupos de {group_var}'
                            else:
                                test_name = 'One-way ANOVA'
                                rationale = f'Comparar {num_var} entre {unique_groups} grupos de {group_var}'
                            
                            recommendations.append({
                                'test_name': test_name,
                                'columns': [group_var, num_var],
                                'rationale': rationale,
                                'category': 'group_comparison',
                                'priority': 'high'
                            })
        
        # Chi-cuadrado para variables categóricas
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for i, col1 in enumerate(categorical_columns):
            for col2 in categorical_columns[i+1:]:
                if col1 in df.columns and col2 in df.columns:
                    recommendations.append({
                        'test_name': 'Chi-square test of independence',
                        'columns': [col1, col2],
                        'rationale': f'Verificar independencia entre {col1} y {col2}',
                        'category': 'independence_test',
                        'priority': 'medium'
                    })
        
        # Correlaciones para variables numéricas
        if len(numeric_columns) >= 2:
            for i, col1 in enumerate(numeric_columns):
                for col2 in numeric_columns[i+1:]:
                    recommendations.append({
                        'test_name': 'Pearson correlation',
                        'columns': [col1, col2],
                        'rationale': f'Medir correlación lineal entre {col1} y {col2}',
                        'category': 'correlation',
                        'priority': 'medium'
                    })
        
        # Registrar acción
        log_action(
            function='recommend_statistical_tests',
            step='statistical_analysis',
            parameters={'total_recommendations': len(recommendations)},
            before_metrics={'numeric_columns': len(numeric_columns), 'categorical_columns': len(categorical_columns)},
            after_metrics={'recommendations_count': len(recommendations)},
            status='success',
            message=f"Se generaron {len(recommendations)} recomendaciones de pruebas estadísticas",
            execution_time=(datetime.now() - start_time).total_seconds()
        )
        
        return recommendations
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        
        log_action(
            function='recommend_statistical_tests',
            step='statistical_analysis',
            parameters={},
            before_metrics={},
            after_metrics={},
            status='error',
            message=f"Error al recomendar pruebas: {str(e)}",
            execution_time=execution_time,
            error_details=str(e)
        )
        
        raise ValueError(f"Error al recomendar pruebas: {str(e)}")

def linear_regression_analysis(
    df: pd.DataFrame,
    y_column: str,
    x_columns: List[str],
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Realiza análisis de regresión lineal con interpretación semántica.
    
    Args:
        df: DataFrame con los datos
        y_column: Variable dependiente
        x_columns: Variables independientes
        metadata: Metadatos semánticos
        
    Returns:
        Resultados del análisis de regresión
    """
    start_time = datetime.now()
    
    try:
        # Validar columnas
        if y_column not in df.columns:
            raise ValueError(f"Variable dependiente '{y_column}' no existe")
        
        missing_x = [col for col in x_columns if col not in df.columns]
        if missing_x:
            raise ValueError(f"Variables independientes no encontradas: {missing_x}")
        
        # Preparar datos
        model_data = df[[y_column] + x_columns].dropna()
        if len(model_data) < len(x_columns) + 1:
            raise ValueError("Datos insuficientes para el modelo")
        
        X = model_data[x_columns]
        y = model_data[y_column]
        
        # Ajustar modelo
        model = LinearRegression()
        model.fit(X, y)
        
        # Predicciones
        y_pred = model.predict(X)
        
        # Métricas del modelo
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        # Coeficientes y estadísticas
        coefficients = pd.DataFrame({
            'variable': ['intercept'] + x_columns,
            'coefficient': [model.intercept_] + list(model.coef_),
            'abs_coefficient': [abs(model.intercept_)] + [abs(coef) for coef in model.coef_]
        })
        
        # Interpretación semántica
        semantic_interpretation = _interpret_regression_results(
            y_column, x_columns, coefficients, r2, metadata
        )
        
        # Recomendaciones
        recommendations = _get_regression_recommendations(
            y_column, x_columns, r2, coefficients, metadata
        )
        
        results = {
            'model_type': 'linear_regression',
            'dependent_variable': y_column,
            'independent_variables': x_columns,
            'n_observations': len(model_data),
            'r_squared': r2,
            'adjusted_r_squared': 1 - (1 - r2) * (len(model_data) - 1) / (len(model_data) - len(x_columns) - 1),
            'mse': mse,
            'rmse': rmse,
            'coefficients': coefficients.to_dict('records'),
            'predictions': y_pred.tolist(),
            'residuals': (y - y_pred).tolist(),
            'semantic_interpretation': semantic_interpretation,
            'recommendations': recommendations
        }
        
        # Registrar acción
        log_action(
            function='linear_regression_analysis',
            step='statistical_analysis',
            parameters={'y_column': y_column, 'x_columns': x_columns},
            before_metrics={'n_observations': len(model_data)},
            after_metrics={'r_squared': r2, 'rmse': rmse},
            status='success',
            message=f"Regresión lineal completada: R² = {r2:.3f}",
            execution_time=(datetime.now() - start_time).total_seconds()
        )
        
        return results
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        
        log_action(
            function='linear_regression_analysis',
            step='statistical_analysis',
            parameters={'y_column': y_column, 'x_columns': x_columns},
            before_metrics={},
            after_metrics={},
            status='error',
            message=f"Error en regresión lineal: {str(e)}",
            execution_time=execution_time,
            error_details=str(e)
        )
        
        raise ValueError(f"Error en regresión lineal: {str(e)}")

def suggest_analysis_for_ui(
    df: pd.DataFrame,
    stats_results: Dict[str, Any],
    correlations: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Genera sugerencias de análisis para mostrar en el front-end.
    
    Args:
        df: DataFrame con los datos
        stats_results: Resultados de análisis estadístico
        correlations: Resultados de correlaciones (opcional)
        metadata: Metadatos semánticos (opcional)
        
    Returns:
        Lista de sugerencias con mensajes didácticos
    """
    suggestions = []
    
    # Sugerencias basadas en correlaciones
    if correlations and 'significant_correlations' in correlations:
        for corr in correlations['significant_correlations'][:3]:  # Top 3
            strength = corr['strength']
            if strength == 'strong':
                suggestions.append({
                    'type': 'correlation_insight',
                    'message': f"🔗 **Correlación fuerte detectada**: {corr['variable1']} y {corr['variable2']} tienen una correlación {strength} (r={corr['correlation']:.2f}). ¿Te gustaría ver un gráfico de dispersión con línea de regresión?",
                    'priority': 'high',
                    'action': 'show_scatter_plot',
                    'parameters': {'x': corr['variable1'], 'y': corr['variable2']}
                })
    
    # Sugerencias basadas en normalidad
    if 'column_statistics' in stats_results:
        for col, stats in stats_results['column_statistics'].items():
            normality_tests = stats.get('normality_tests', {})
            for test_name, test_result in normality_tests.items():
                if not test_result.get('is_normal', True):
                    suggestions.append({
                        'type': 'normality_warning',
                        'message': f"📊 **Distribución no normal**: La variable '{col}' no sigue una distribución normal (p={test_result['p_value']:.3f}). Se recomienda usar pruebas no paramétricas o transformar los datos.",
                        'priority': 'medium',
                        'action': 'show_histogram',
                        'parameters': {'column': col}
                    })
                    break
    
    # Sugerencias basadas en outliers
    for col, stats in stats_results.get('column_statistics', {}).items():
        outlier_pct = stats.get('outlier_stats', {}).get('outliers_percentage', 0)
        if outlier_pct > 5:
            suggestions.append({
                'type': 'outlier_alert',
                'message': f"⚠️ **Outliers detectados**: La variable '{col}' tiene {outlier_pct:.1f}% de valores atípicos. Considera revisar estos datos o usar métodos robustos.",
                'priority': 'medium',
                'action': 'show_boxplot',
                'parameters': {'column': col}
            })
    
    # Sugerencias basadas en semántica
    if metadata:
        for col, col_metadata in metadata.items():
            if col_metadata.get('category') == 'socioeconomic' and col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    suggestions.append({
                        'type': 'semantic_insight',
                        'message': f"💰 **Variable económica**: '{col}' es una variable socioeconómica. Considera analizar su distribución por grupos demográficos.",
                        'priority': 'low',
                        'action': 'show_group_analysis',
                        'parameters': {'column': col}
                    })
    
    return suggestions

# Funciones auxiliares

def _interpret_skewness(skewness: float) -> str:
    """Interpreta el valor de asimetría."""
    if abs(skewness) < 0.5:
        return "Aproximadamente simétrica"
    elif abs(skewness) < 1:
        return "Moderadamente asimétrica"
    else:
        return "Altamente asimétrica"

def _interpret_kurtosis(kurtosis: float) -> str:
    """Interpreta el valor de curtosis."""
    if abs(kurtosis) < 0.5:
        return "Mesocúrtica (normal)"
    elif kurtosis > 0.5:
        return "Leptocúrtica (picos agudos)"
    else:
        return "Platicúrtica (picos planos)"

def _get_semantic_interpretation(
    column: str,
    basic_stats: Dict[str, Any],
    shape_stats: Dict[str, Any],
    outlier_stats: Dict[str, Any],
    normality_tests: Dict[str, Any],
    metadata: Dict[str, Any] = None
) -> Dict[str, str]:
    """Genera interpretación semántica de las estadísticas."""
    interpretation = {}
    
    # Interpretación de tendencia central
    mean = basic_stats['mean']
    median = basic_stats['median']
    
    if abs(mean - median) / mean < 0.1:
        interpretation['central_tendency'] = "La media y mediana son similares, sugiriendo simetría"
    else:
        interpretation['central_tendency'] = "La media y mediana difieren, indicando asimetría"
    
    # Interpretación de variabilidad
    cv = basic_stats['std'] / basic_stats['mean'] if basic_stats['mean'] != 0 else 0
    if cv < 0.15:
        interpretation['variability'] = "Baja variabilidad relativa"
    elif cv < 0.35:
        interpretation['variability'] = "Variabilidad moderada"
    else:
        interpretation['variability'] = "Alta variabilidad relativa"
    
    # Interpretación de outliers
    outlier_pct = outlier_stats['outliers_percentage']
    if outlier_pct < 1:
        interpretation['outliers'] = "Pocos outliers, datos relativamente limpios"
    elif outlier_pct < 5:
        interpretation['outliers'] = "Algunos outliers, considerar revisión"
    else:
        interpretation['outliers'] = "Muchos outliers, posible necesidad de limpieza"
    
    return interpretation

def _get_statistical_recommendations(
    column: str,
    basic_stats: Dict[str, Any],
    shape_stats: Dict[str, Any],
    outlier_stats: Dict[str, Any],
    normality_tests: Dict[str, Any],
    metadata: Dict[str, Any] = None
) -> List[str]:
    """Genera recomendaciones estadísticas específicas."""
    recommendations = []
    
    # Recomendaciones basadas en normalidad
    is_normal = all(test.get('is_normal', True) for test in normality_tests.values())
    if not is_normal:
        recommendations.append("Usar pruebas no paramétricas o transformar datos")
    
    # Recomendaciones basadas en outliers
    if outlier_stats['outliers_percentage'] > 5:
        recommendations.append("Considerar métodos robustos o revisar outliers")
    
    # Recomendaciones basadas en semántica
    if metadata and column in metadata:
        category = metadata[column].get('category', '')
        if category == 'socioeconomic':
            recommendations.append("Analizar por grupos demográficos")
        elif category == 'opinion':
            recommendations.append("Considerar escalas de Likert")
    
    return recommendations

def _get_global_recommendations(results: Dict[str, Any]) -> List[str]:
    """Genera recomendaciones globales basadas en todos los resultados."""
    recommendations = []
    
    total_outliers = sum(results[col]['outlier_stats']['outliers_count'] for col in results)
    non_normal_count = len([col for col in results if any(
        not test.get('is_normal', True) for test in results[col]['normality_tests'].values()
    )])
    
    if total_outliers > 0:
        recommendations.append(f"Se detectaron {total_outliers} outliers en total")
    
    if non_normal_count > 0:
        recommendations.append(f"{non_normal_count} variables no siguen distribución normal")
    
    return recommendations

def _interpret_regression_results(
    y_column: str,
    x_columns: List[str],
    coefficients: pd.DataFrame,
    r2: float,
    metadata: Dict[str, Any] = None
) -> Dict[str, str]:
    """Interpreta los resultados de regresión."""
    interpretation = {}
    
    # Interpretación de R²
    if r2 > 0.7:
        interpretation['model_fit'] = "Excelente ajuste del modelo"
    elif r2 > 0.5:
        interpretation['model_fit'] = "Buen ajuste del modelo"
    elif r2 > 0.3:
        interpretation['model_fit'] = "Ajuste moderado del modelo"
    else:
        interpretation['model_fit'] = "Ajuste pobre del modelo"
    
    # Variable más importante
    if len(coefficients) > 1:
        most_important = coefficients.iloc[1:].loc[coefficients['abs_coefficient'].idxmax()]
        interpretation['most_important'] = f"La variable más importante es '{most_important['variable']}'"
    
    return interpretation

def _get_regression_recommendations(
    y_column: str,
    x_columns: List[str],
    r2: float,
    coefficients: pd.DataFrame,
    metadata: Dict[str, Any] = None
) -> List[str]:
    """Genera recomendaciones para regresión."""
    recommendations = []
    
    if r2 < 0.3:
        recommendations.append("Considerar variables adicionales o transformaciones")
    
    if len(x_columns) > 5:
        recommendations.append("Considerar selección de variables para evitar overfitting")
    
    return recommendations

def summarize_survey_structure(df: pd.DataFrame, metadata: Dict) -> Dict:
    """
    Analyze survey structure and provide comprehensive summary.
    
    Args:
        df: Input DataFrame
        metadata: Metadata dictionary with semantic classifications
        
    Returns:
        Dictionary with survey structure summary
    """
    logger.info("Analyzing survey structure")
    
    # Get semantic types from metadata
    semantic_types = metadata.get('semantic_types', {})
    
    # Analyze each column
    columns_analysis = []
    type_counts = {}
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        semantic_type = semantic_types.get(col, 'unknown')
        
        columns_analysis.append({
            'column': col,
            'semantic_type': semantic_type,
            'dtype': dtype,
            'missing_pct': round(missing_pct, 1),
            'unique_values': df[col].nunique(),
            'sample_values': list(df[col].dropna().unique()[:3])
        })
        
        # Count by semantic type
        if semantic_type not in type_counts:
            type_counts[semantic_type] = 0
        type_counts[semantic_type] += 1
    
    # Generate narrative summary
    total_questions = len(df.columns)
    total_missing = sum(1 for col in columns_analysis if col['missing_pct'] > 0)
    avg_missing = sum(col['missing_pct'] for col in columns_analysis) / total_questions
    
    # Build narrative text
    narrative_parts = [f"Esta encuesta tiene {total_questions} preguntas"]
    
    if type_counts:
        type_descriptions = []
        for semantic_type, count in type_counts.items():
            if semantic_type == 'demographic':
                type_descriptions.append(f"{count} demográfica{'s' if count > 1 else ''}")
            elif semantic_type == 'likert':
                type_descriptions.append(f"{count} de escala Likert")
            elif semantic_type == 'categorical':
                type_descriptions.append(f"{count} de opción múltiple")
            elif semantic_type == 'text':
                type_descriptions.append(f"{count} abierta{'s' if count > 1 else ''}")
            elif semantic_type == 'numeric':
                type_descriptions.append(f"{count} numérica{'s' if count > 1 else ''}")
            else:
                type_descriptions.append(f"{count} {semantic_type}")
        
        narrative_parts.append(": " + ", ".join(type_descriptions) + ".")
    
    if total_missing > 0:
        narrative_parts.append(f" El {avg_missing:.1f}% de los campos están vacíos en promedio.")
    
    narrative = "".join(narrative_parts)
    
    return {
        'narrative': narrative,
        'total_questions': total_questions,
        'type_counts': type_counts,
        'columns_analysis': columns_analysis,
        'avg_missing_pct': round(avg_missing, 1),
        'total_missing_columns': total_missing
    }

def frequency_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Generate frequency table for categorical or Likert variables.
    
    Args:
        df: Input DataFrame
        col: Column name
        
    Returns:
        DataFrame with counts and percentages
    """
    logger.info(f"Generating frequency table for column: {col}")
    
    if col not in df.columns:
        logger.warning(f"Column {col} not found in DataFrame")
        return pd.DataFrame()
    
    # Get value counts
    value_counts = df[col].value_counts(dropna=False)
    total = len(df)
    
    # Calculate percentages
    percentages = (value_counts / total) * 100
    
    # Create frequency table
    freq_table = pd.DataFrame({
        'Valor': value_counts.index,
        'Frecuencia': value_counts.values,
        'Porcentaje': percentages.round(1),
        'Porcentaje_Acumulado': percentages.cumsum().round(1)
    })
    
    # Handle missing values
    if freq_table['Valor'].isna().any():
        freq_table.loc[freq_table['Valor'].isna(), 'Valor'] = 'Valor faltante'
    
    return freq_table

def crosstab_summary(df: pd.DataFrame, col: str) -> str:
    """
    Generate textual summary for categorical variable.
    
    Args:
        df: Input DataFrame
        col: Column name
        
    Returns:
        Textual summary string
    """
    logger.info(f"Generating crosstab summary for column: {col}")
    
    if col not in df.columns:
        return "Columna no encontrada en el dataset."
    
    freq_table = frequency_table(df, col)
    
    if freq_table.empty:
        return "No hay datos disponibles para generar resumen."
    
    # Get top 3 most frequent values
    top_values = freq_table.head(3)
    
    summary_parts = []
    
    # Most frequent value
    most_freq = top_values.iloc[0]
    summary_parts.append(f"La opción '{most_freq['Valor']}' fue la más frecuente ({most_freq['Porcentaje']}%)")
    
    # Second most frequent
    if len(top_values) > 1:
        second_freq = top_values.iloc[1]
        summary_parts.append(f", seguida de '{second_freq['Valor']}' ({second_freq['Porcentaje']}%)")
    
    # Third most frequent
    if len(top_values) > 2:
        third_freq = top_values.iloc[2]
        summary_parts.append(f" y '{third_freq['Valor']}' ({third_freq['Porcentaje']}%)")
    
    summary_parts.append(".")
    
    # Add missing data info if any
    missing_row = freq_table[freq_table['Valor'] == 'Valor faltante']
    if not missing_row.empty:
        missing_pct = missing_row.iloc[0]['Porcentaje']
        summary_parts.append(f" El {missing_pct}% de los datos están faltantes.")
    
    return "".join(summary_parts)

def textual_summary(df: pd.DataFrame, col: str) -> Dict:
    """
    Generate comprehensive textual analysis for text columns.
    
    Args:
        df: Input DataFrame
        col: Column name
        
    Returns:
        Dictionary with word cloud, sentiment, and representative sentences
    """
    logger.info(f"Generating textual summary for column: {col}")
    
    if col not in df.columns:
        return {"error": "Columna no encontrada"}
    
    # Get text data
    text_data = df[col].dropna().astype(str)
    
    if text_data.empty:
        return {"error": "No hay datos de texto disponibles"}
    
    # Word frequency analysis
    all_words = []
    for text in text_data:
        # Simple tokenization (can be enhanced with NLTK)
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    top_words = word_freq.most_common(20)
    
    # Simple sentiment analysis (basic approach)
    positive_words = ['bueno', 'excelente', 'mejor', 'positivo', 'satisfecho', 'contento', 'feliz', 'agradable']
    negative_words = ['malo', 'terrible', 'peor', 'negativo', 'insatisfecho', 'triste', 'molesto', 'desagradable']
    
    positive_count = sum(1 for word in all_words if word in positive_words)
    negative_count = sum(1 for word in all_words if word in negative_words)
    total_words = len(all_words)
    
    if total_words > 0:
        positive_pct = (positive_count / total_words) * 100
        negative_pct = (negative_count / total_words) * 100
        neutral_pct = 100 - positive_pct - negative_pct
    else:
        positive_pct = negative_pct = neutral_pct = 0
    
    # Find representative sentences (simple approach)
    representative_sentences = []
    if len(text_data) > 0:
        # Get longest sentences as potentially most informative
        sentences = []
        for text in text_data:
            if len(text) > 10:  # Filter out very short responses
                sentences.append(text)
        
        # Sort by length and take top 3
        sentences.sort(key=len, reverse=True)
        representative_sentences = sentences[:3]
    
    return {
        'word_cloud': dict(top_words),
        'sentiment': {
            'positive_pct': round(positive_pct, 1),
            'negative_pct': round(negative_pct, 1),
            'neutral_pct': round(neutral_pct, 1),
            'total_words': total_words
        },
        'representative_sentences': representative_sentences,
        'total_responses': len(text_data),
        'avg_response_length': round(np.mean([len(str(text)) for text in text_data]), 1)
    }

def generate_data_dictionary(df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
    """
    Generate comprehensive data dictionary for the survey.
    
    Args:
        df: Input DataFrame
        metadata: Metadata dictionary
        
    Returns:
        DataFrame with data dictionary
    """
    logger.info("Generating data dictionary")
    
    semantic_types = metadata.get('semantic_types', {})
    units = metadata.get('units', {})
    
    dictionary_rows = []
    
    for col in df.columns:
        semantic_type = semantic_types.get(col, 'unknown')
        unit = units.get(col, '')
        dtype = str(df[col].dtype)
        
        # Get sample values
        sample_values = df[col].dropna().unique()[:5]
        sample_str = ", ".join([str(val) for val in sample_values])
        
        # Generate description based on semantic type
        if semantic_type == 'demographic':
            description = "Variable demográfica que describe características básicas de la población"
        elif semantic_type == 'likert':
            description = "Escala de Likert para medir actitudes o percepciones"
        elif semantic_type == 'categorical':
            description = "Variable categórica con opciones de respuesta predefinidas"
        elif semantic_type == 'text':
            description = "Respuesta de texto libre o abierta"
        elif semantic_type == 'numeric':
            description = "Variable numérica para análisis cuantitativo"
        else:
            description = "Variable de tipo no especificado"
        
        # Generate treatment suggestions
        if semantic_type == 'demographic':
            treatment = "Usar para segmentación y análisis demográfico"
        elif semantic_type == 'likert':
            treatment = "Analizar con tablas de frecuencia y correlaciones"
        elif semantic_type == 'categorical':
            treatment = "Crear tablas de contingencia y análisis de asociación"
        elif semantic_type == 'text':
            treatment = "Análisis de contenido y minería de texto"
        elif semantic_type == 'numeric':
            treatment = "Análisis estadístico descriptivo e inferencial"
        else:
            treatment = "Revisar y clasificar según el contenido"
        
        dictionary_rows.append({
            'Variable': col,
            'Tipo_Semantico': semantic_type,
            'Tipo_Datos': dtype,
            'Unidad': unit,
            'Descripcion': description,
            'Valores_Ejemplo': sample_str,
            'Valores_Unicos': df[col].nunique(),
            'Valores_Faltantes': df[col].isnull().sum(),
            'Sugerencia_Tratamiento': treatment
        })
    
    return pd.DataFrame(dictionary_rows)

def can_generate_visualizations(df: pd.DataFrame, metadata: Dict) -> bool:
    """
    Check if dataset can generate meaningful visualizations.
    
    Args:
        df: Input DataFrame
        metadata: Metadata dictionary
        
    Returns:
        True if visualizations are possible, False otherwise
    """
    logger.info("Checking if visualizations are possible")
    
    semantic_types = metadata.get('semantic_types', {})
    
    # Check for numeric columns
    numeric_cols = [col for col, sem_type in semantic_types.items() 
                   if sem_type in ['numeric', 'demographic'] and df[col].dtype in ['int64', 'float64']]
    
    if len(numeric_cols) >= 1:
        return True
    
    # Check for categorical columns with sufficient levels
    categorical_cols = [col for col, sem_type in semantic_types.items() 
                       if sem_type in ['categorical', 'likert']]
    
    for col in categorical_cols:
        if col in df.columns and df[col].nunique() >= 2:
            return True
    
    return False

def correlation_analysis_advanced(df: pd.DataFrame, metadata: Dict = None) -> Dict:
    """
    Análisis avanzado de correlaciones con interpretación semántica.
    
    Args:
        df: DataFrame a analizar
        metadata: Metadata del dataset
        
    Returns:
        Diccionario con resultados de correlación
    """
    # Obtener columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return {
            'correlation_matrix': None,
            'significant_correlations': pd.DataFrame(),
            'message': 'Se requieren al menos 2 columnas numéricas para análisis de correlación'
        }
    
    # Calcular correlaciones
    corr_matrix = df[numeric_cols].corr()
    
    # Encontrar correlaciones significativas
    significant_corr = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.3:  # Umbral de significancia
                significant_corr.append({
                    'variable1': numeric_cols[i],
                    'variable2': numeric_cols[j],
                    'correlation': corr_value,
                    'strength': 'fuerte' if abs(corr_value) > 0.7 else 'moderada' if abs(corr_value) > 0.5 else 'débil'
                })
    
    return {
        'correlation_matrix': corr_matrix,
        'significant_correlations': pd.DataFrame(significant_corr),
        'total_correlations': len(significant_corr)
    }

def regression_analysis_advanced(df: pd.DataFrame, target_col: str, feature_cols: List[str] = None) -> Dict:
    """
    Análisis de regresión lineal avanzado.
    
    Args:
        df: DataFrame a analizar
        target_col: Columna objetivo
        feature_cols: Columnas predictoras (opcional)
        
    Returns:
        Diccionario con resultados de regresión
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    
    if target_col not in df.columns:
        return {'error': f'Columna objetivo {target_col} no encontrada'}
    
    # Seleccionar features si no se especifican
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)
    
    if len(feature_cols) == 0:
        return {'error': 'No hay columnas predictoras disponibles'}
    
    # Preparar datos
    X = df[feature_cols].dropna()
    y = df[target_col].dropna()
    
    # Alinear índices
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    if len(X) < 10:
        return {'error': 'Datos insuficientes para regresión (mínimo 10 observaciones)'}
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Coeficientes
    coefficients = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    return {
        'r2_score': r2,
        'rmse': rmse,
        'intercept': model.intercept_,
        'coefficients': coefficients,
        'n_samples': len(X),
        'n_features': len(feature_cols)
    } 