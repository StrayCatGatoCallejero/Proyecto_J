"""
Módulo de Feature Engineering para análisis social.
Incluye rutinas de variables derivadas, ratios, escalado, binning, intervalos, bootstrap y scores compuestos.
Todas las funciones son robustas, con validaciones y contratos claros.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Tuple, Any, Optional
from scipy import stats

# =====================
# Variables derivadas y ratios
# =====================
def compute_ratios(df: pd.DataFrame, numerator_cols: List[str], denominator_cols: List[str]) -> pd.DataFrame:
    """
    Genera nuevas columnas como num_i/den_j, con control de división por cero y NaN seguros.
    Args:
        df: DataFrame de entrada
        numerator_cols: Lista de columnas numerador
        denominator_cols: Lista de columnas denominador
    Returns:
        DataFrame con nuevas columnas ratio (no modifica df original)
    """
    result = df.copy()
    for num in numerator_cols:
        for den in denominator_cols:
            col_name = f"ratio_{num}_over_{den}"
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(result[den] != 0, result[num] / result[den], np.nan)
            result[col_name] = ratio
    return result

def compute_percentage(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    """
    Añade columna % target_col per group_col, calculando proporciones dentro de cada grupo.
    Args:
        df: DataFrame de entrada
        group_col: Columna de agrupación
        target_col: Columna objetivo (conteo)
    Returns:
        DataFrame con columna de porcentaje por grupo
    """
    result = df.copy()
    group_counts = result.groupby(group_col)[target_col].transform('count')
    result[f'perc_{target_col}_in_{group_col}'] = 100 * result[target_col] / group_counts
    return result

# =====================
# Medias ponderadas y agregaciones
# =====================
def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    """
    Calcula la media ponderada, con validación de sum(weights) > 0.
    Args:
        df: DataFrame
        value_col: Columna de valores
        weight_col: Columna de pesos (no negativos)
    Returns:
        Media ponderada (float)
    """
    weights = df[weight_col].fillna(0)
    values = df[value_col].fillna(0)
    if (weights < 0).any():
        raise ValueError("Los pesos no pueden ser negativos.")
    total_weight = weights.sum()
    if total_weight == 0:
        return np.nan
    return np.average(values, weights=weights)

def group_agg(df: pd.DataFrame, group_cols: List[str], aggs: Dict[str, str]) -> pd.DataFrame:
    """
    Agrega estadísticas personalizadas (mean, sum, std, count), retornando un DataFrame multi-index.
    Args:
        df: DataFrame
        group_cols: Lista de columnas para agrupar
        aggs: Diccionario {col: aggfunc}
    Returns:
        DataFrame agrupado
    """
    return df.groupby(group_cols).agg(aggs)

# =====================
# Escalado y normalización
# =====================
def min_max_scale(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Escala cada columna al rango [0,1].
    Args:
        df: DataFrame
        cols: Columnas a escalar
    Returns:
        DataFrame con columnas escaladas
    """
    result = df.copy()
    for col in cols:
        min_val = result[col].min()
        max_val = result[col].max()
        if max_val - min_val == 0:
            result[f'scaled_{col}'] = 0.0
        else:
            result[f'scaled_{col}'] = (result[col] - min_val) / (max_val - min_val)
    return result

def z_score_normalize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Transforma a puntajes Z: (x - μ)/σ.
    Args:
        df: DataFrame
        cols: Columnas a normalizar
    Returns:
        DataFrame con columnas normalizadas
    """
    result = df.copy()
    for col in cols:
        mean = result[col].mean()
        std = result[col].std()
        if std == 0:
            result[f'z_{col}'] = 0.0
        else:
            result[f'z_{col}'] = (result[col] - mean) / std
    return result

def robust_scale(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Escala basado en rango intercuartílico para robustez ante outliers.
    Args:
        df: DataFrame
        cols: Columnas a escalar
    Returns:
        DataFrame con columnas escaladas robustamente
    """
    result = df.copy()
    for col in cols:
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            result[f'robust_{col}'] = 0.0
        else:
            result[f'robust_{col}'] = (result[col] - q1) / iqr
    return result

# =====================
# Binning y discretización
# =====================
def create_bins(df: pd.DataFrame, col: str, bins: List[float], labels: List[str]) -> pd.Series:
    """
    Crea categorías discretas a partir de cortes numéricos.
    Args:
        df: DataFrame
        col: Columna numérica
        bins: Lista de cortes
        labels: Etiquetas para los bins
    Returns:
        Serie categórica
    """
    if len(bins) - 1 != len(labels):
        raise ValueError("El número de etiquetas debe ser igual a len(bins)-1")
    return pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)

def quantile_binning(df: pd.DataFrame, col: str, q: int) -> pd.Series:
    """
    Genera q bins iguales en número de observaciones (percentiles).
    Args:
        df: DataFrame
        col: Columna numérica
        q: Número de bins
    Returns:
        Serie categórica
    """
    return pd.qcut(df[col], q=q, labels=[f"Q{i+1}" for i in range(q)])

# =====================
# Intervalos de confianza y errores estándar
# =====================
def compute_confidence_interval(df: pd.DataFrame, col: str, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calcula el intervalo de confianza (t-student) para la media de la columna.
    Args:
        df: DataFrame
        col: Columna numérica
        alpha: Nivel de significancia
    Returns:
        (limite_inferior, limite_superior)
    """
    data = df[col].dropna()
    n = len(data)
    if n < 2:
        return (np.nan, np.nan)
    mean = data.mean()
    se = data.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - alpha/2, df=n-1)
    return (mean - t*se, mean + t*se)

def standard_error(df: pd.DataFrame, col: str) -> float:
    """
    Devuelve el error estándar de la media.
    Args:
        df: DataFrame
        col: Columna numérica
    Returns:
        Error estándar (float)
    """
    data = df[col].dropna()
    n = len(data)
    if n < 2:
        return np.nan
    return data.std(ddof=1) / np.sqrt(n)

# =====================
# Bootstrap y estimaciones robustas
# =====================
def bootstrap_statistic(df: pd.DataFrame, col: str, func: Callable, n_boot: int = 1000, alpha: float = 0.05) -> Dict:
    """
    Realiza remuestreo bootstrap para estimar distribuciones de medias, medianas u otra estadística func.
    Devuelve media bootstrap, intervalo percentil [α/2, 1–α/2].
    Args:
        df: DataFrame
        col: Columna numérica
        func: Función estadística (ej: np.mean, np.median)
        n_boot: Número de muestras bootstrap
        alpha: Nivel de significancia
    Returns:
        Dict con media, intervalo y distribución bootstrap
    """
    data = df[col].dropna().values
    if len(data) < 2:
        return {'bootstrap_mean': np.nan, 'ci': (np.nan, np.nan), 'distribution': []}
    boot_stats = [func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(boot_stats, 100*alpha/2)
    upper = np.percentile(boot_stats, 100*(1-alpha/2))
    return {
        'bootstrap_mean': np.mean(boot_stats),
        'ci': (lower, upper),
        'distribution': boot_stats
    }

# =====================
# Índices y scores compuestos
# =====================
def composite_index(df: pd.DataFrame, cols: List[str], method: str = 'mean', weights: Optional[List[float]] = None) -> pd.Series:
    """
    Crea un índice compuesto de varias variables con pesos opcionales.
    Args:
        df: DataFrame
        cols: Columnas a combinar
        method: 'mean' o 'sum'
        weights: Pesos opcionales (mismo largo que cols)
    Returns:
        Serie con índice compuesto
    """
    data = df[cols].fillna(0)
    if weights is not None:
        weights = np.array(weights)
        if len(weights) != len(cols):
            raise ValueError("El número de pesos debe coincidir con el número de columnas")
        if (weights < 0).any():
            raise ValueError("Los pesos no pueden ser negativos")
        if method == 'mean':
            return (data * weights).sum(axis=1) / weights.sum()
        elif method == 'sum':
            return (data * weights).sum(axis=1)
        else:
            raise ValueError("Método no soportado")
    else:
        if method == 'mean':
            return data.mean(axis=1)
        elif method == 'sum':
            return data.sum(axis=1)
        else:
            raise ValueError("Método no soportado")

def scale_and_score(df: pd.DataFrame, cols: List[str], reference: Dict[str, float]) -> pd.Series:
    """
    Compara cada observación contra valores de referencia, produciendo un score de distancia o similitud.
    Args:
        df: DataFrame
        cols: Columnas a comparar
        reference: Diccionario {col: valor_referencia}
    Returns:
        Serie con score de similitud (menor es más similar)
    """
    data = df[cols].fillna(0)
    ref_vec = np.array([reference.get(col, 0) for col in cols])
    return ((data - ref_vec)**2).sum(axis=1)**0.5 