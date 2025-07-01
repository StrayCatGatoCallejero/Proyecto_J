"""
Módulo de Análisis de Tendencias Temporales y Comparaciones - Versión Avanzada
=============================================================================

Sistema avanzado para detectar y analizar tendencias temporales, comparaciones
y patrones de evolución en datos con componentes de tiempo.
Incluye suavizado, descomposición estacional, manejo de gaps y frecuencias personalizables.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# Intentar importar statsmodels para descomposición estacional
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    STL_AVAILABLE = True
except ImportError:
    STL_AVAILABLE = False
    print("⚠️ statsmodels no disponible. La descomposición estacional no estará disponible.")

# Palabras clave para detección de tendencias y comparaciones
TREND_KEYWORDS = {
    'tendencia': ['tendencia', 'evolución', 'cambio', 'variación', 'progresión'],
    'comparacion': ['comparar', 'comparación', 'comparacion', 'vs', 'versus', 'frente a'],
    'temporal': ['mes a mes', 'año a año', 'trimestre a trimestre', 'semana a semana', 'diario'],
    'frecuencia': ['mensual', 'trimestral', 'anual', 'semanal', 'diario', 'por mes', 'por año']
}

# Métricas soportadas para análisis temporal
METRICAS_TEMPORALES = {
    'promedio': ['promedio', 'media', 'valor promedio', 'media de'],
    'suma': ['suma', 'total', 'sumatoria', 'acumulado'],
    'contar': ['contar', 'número', 'cantidad', 'frecuencia'],
    'maximo': ['máximo', 'pico', 'valor máximo'],
    'minimo': ['mínimo', 'valor mínimo'],
    'mediana': ['mediana', 'valor central'],
    'desviacion': ['desviación', 'desviacion', 'variabilidad']
}

# Frecuencias temporales soportadas (incluyendo personalizables)
FRECUENCIAS_TEMPORALES = {
    'D': ['diario', 'día a día', 'por día', 'daily'],
    'W': ['semanal', 'semana a semana', 'por semana', 'weekly'],
    'M': ['mensual', 'mes a mes', 'por mes', 'monthly'],
    'Q': ['trimestral', 'trimestre a trimestre', 'por trimestre', 'quarterly'],
    'Y': ['anual', 'año a año', 'por año', 'yearly'],
    'custom': ['personalizado', 'custom', 'específico']
}

# Palabras clave para suavizado y medias móviles
SUAVIZADO_KEYWORDS = {
    'media_movil': ['media móvil', 'media movil', 'moving average', 'ma', 'promedio móvil'],
    'mediana_movil': ['mediana móvil', 'mediana movil', 'rolling median'],
    'suavizado': ['suavizado', 'smooth', 'suave', 'filtrado']
}

# Tipos de ajuste de tendencia
TIPOS_AJUSTE = {
    'lineal': ['lineal', 'linear', 'recta', 'línea'],
    'polinomial': ['polinomial', 'polynomial', 'curva', 'cuadrático'],
    'lowess': ['lowess', 'loess', 'suave', 'local']
}

# Palabras clave para descomposición estacional
DESCOMPOSICION_KEYWORDS = {
    'estacional': ['estacional', 'estacionalidad', 'seasonal', 'componente estacional', 'estacionales'],
    'descomposicion': ['descomposición', 'descomposicion', 'descomponer', 'componentes', 'descomposición'],
    'tendencia': ['tendencia', 'trend', 'componente de tendencia'],
    'residuo': ['residuo', 'residual', 'ruido', 'componente residual', 'residuos']
}

def detectar_analisis_temporal(question: str) -> Dict[str, Any]:
    """
    Detecta si la pregunta requiere análisis temporal y extrae parámetros avanzados.
    
    Args:
        question: Pregunta en lenguaje natural
        
    Returns:
        Dict con parámetros del análisis temporal
    """
    q = question.lower()
    
    # Detectar tipo de análisis
    tipo_analisis = None
    if any(palabra in q for palabra in TREND_KEYWORDS['tendencia']):
        tipo_analisis = 'tendencia'
    elif any(palabra in q for palabra in TREND_KEYWORDS['comparacion']):
        tipo_analisis = 'comparacion'
    
    if not tipo_analisis:
        return None
    
    # Detectar métrica
    metrica = 'promedio'  # Por defecto
    for metrica_nombre, palabras in METRICAS_TEMPORALES.items():
        if any(palabra in q for palabra in palabras):
            metrica = metrica_nombre
            break
    
    # Detectar frecuencia temporal (incluyendo personalizada)
    frecuencia, frecuencia_personalizada = detectar_frecuencia_avanzada(q)
    
    # Detectar período de análisis
    periodo = detectar_periodo_analisis(q)
    
    # Detectar suavizado
    suavizado = detectar_suavizado(q)
    
    # Detectar tipo de ajuste
    tipo_ajuste = detectar_tipo_ajuste(q)
    
    # Detectar descomposición estacional
    descomposicion = detectar_descomposicion_estacional(q)
    
    return {
        'tipo': tipo_analisis,
        'metrica': metrica,
        'frecuencia': frecuencia,
        'frecuencia_personalizada': frecuencia_personalizada,
        'periodo': periodo,
        'suavizado': suavizado,
        'tipo_ajuste': tipo_ajuste,
        'descomposicion': descomposicion,
        'es_temporal': True
    }

def detectar_frecuencia_avanzada(question: str) -> Tuple[str, Optional[str]]:
    """
    Detecta frecuencia temporal incluyendo frecuencias personalizables.
    
    Args:
        question: Pregunta en lenguaje natural
        
    Returns:
        Tuple con (frecuencia, frecuencia_personalizada)
    """
    q = question.lower()
    
    # Buscar frecuencias estándar
    for freq, palabras in FRECUENCIAS_TEMPORALES.items():
        if any(palabra in q for palabra in palabras):
            return freq, None
    
    # Buscar frecuencias personalizables
    # Patrón: "cada X días/semanas/meses"
    patrones_personalizados = [
        r"cada\s+(\d+)\s+días?",
        r"cada\s+(\d+)\s+semanas?",
        r"cada\s+(\d+)\s+meses?",
        r"cada\s+(\d+)\s+años?",
        r"(\d+)\s+días?",
        r"(\d+)\s+semanas?",
        r"(\d+)\s+meses?",
        r"(\d+)\s+años?"
    ]
    
    for patron in patrones_personalizados:
        match = re.search(patron, q)
        if match:
            numero = int(match.group(1))
            if 'días' in match.group(0) or 'día' in match.group(0):
                return 'custom', f'{numero}D'
            elif 'semanas' in match.group(0) or 'semana' in match.group(0):
                return 'custom', f'{numero}W'
            elif 'meses' in match.group(0) or 'mes' in match.group(0):
                return 'custom', f'{numero}M'
            elif 'años' in match.group(0) or 'año' in match.group(0):
                return 'custom', f'{numero}Y'
    
    # Por defecto, mensual
    return 'M', None

def detectar_suavizado(question: str) -> Dict[str, Any]:
    """
    Detecta parámetros de suavizado en la pregunta.
    
    Args:
        question: Pregunta en lenguaje natural
        
    Returns:
        Dict con parámetros de suavizado
    """
    q = question.lower()
    
    # Detectar tipo de suavizado
    tipo_suavizado = None
    for tipo, palabras in SUAVIZADO_KEYWORDS.items():
        if any(palabra in q for palabra in palabras):
            tipo_suavizado = tipo
            break
    
    if not tipo_suavizado:
        return {'aplicar': False}
    
    # Detectar ventana de suavizado
    ventana = 7  # Por defecto
    patrones_ventana = [
        r"(\d+)\s+días?",
        r"(\d+)\s+semanas?",
        r"(\d+)\s+meses?",
        r"ventana\s+(\d+)",
        r"período\s+(\d+)"
    ]
    
    for patron in patrones_ventana:
        match = re.search(patron, q)
        if match:
            ventana = int(match.group(1))
            break
    
    return {
        'aplicar': True,
        'tipo': tipo_suavizado,
        'ventana': ventana
    }

def detectar_tipo_ajuste(question: str) -> str:
    """
    Detecta el tipo de ajuste de tendencia solicitado.
    
    Args:
        question: Pregunta en lenguaje natural
        
    Returns:
        String con el tipo de ajuste
    """
    q = question.lower()
    
    for tipo, palabras in TIPOS_AJUSTE.items():
        if any(palabra in q for palabra in palabras):
            return tipo
    
    return 'lineal'  # Por defecto

def detectar_periodo_analisis(question: str) -> Dict[str, Any]:
    """
    Detecta el período de análisis temporal en la pregunta.
    
    Args:
        question: Pregunta en lenguaje natural
        
    Returns:
        Dict con información del período
    """
    q = question.lower()
    
    # Expresiones temporales comunes
    periodos = {
        'ultimo_año': ['último año', 'ultimo año', 'pasado año', 'año pasado'],
        'ultimos_6_meses': ['últimos 6 meses', 'ultimos 6 meses', '6 meses', 'medio año'],
        'ultimos_3_meses': ['últimos 3 meses', 'ultimos 3 meses', '3 meses', 'trimestre'],
        'ultimo_mes': ['último mes', 'ultimo mes', 'mes pasado'],
        'este_año': ['este año', 'año actual', 'año en curso'],
        'este_mes': ['este mes', 'mes actual'],
        'todo': ['todo el período', 'todos los datos', 'completo']
    }
    
    for periodo_nombre, palabras in periodos.items():
        if any(palabra in q for palabra in palabras):
            return {'tipo': periodo_nombre, 'palabras': palabras}
    
    # Por defecto, usar todo el período disponible
    return {'tipo': 'todo', 'palabras': ['todo el período']}

def extraer_variables_temporales(df: pd.DataFrame, question: str) -> Tuple[Optional[str], List[str]]:
    """
    Extrae la variable temporal y las variables objetivo del DataFrame.
    
    Args:
        df: DataFrame con los datos
        question: Pregunta en lenguaje natural
        
    Returns:
        Tuple con (columna_temporal, variables_objetivo)
    """
    # 1. Buscar columna temporal
    columna_temporal = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            columna_temporal = col
            break
    
    if not columna_temporal:
        # Buscar columnas que contengan palabras temporales
        for col in df.columns:
            if any(palabra in col.lower() for palabra in ['fecha', 'date', 'time', 'dia', 'día']):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if not df[col].isna().all():
                        columna_temporal = col
                        break
                except:
                    continue
    
    # 2. Buscar variables objetivo numéricas
    variables_objetivo = []
    q = question.lower()
    
    # Buscar nombres exactos de columnas numéricas
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and col.lower() in q:
            variables_objetivo.append(col)
    
    # Si no se encontraron, usar todas las columnas numéricas
    if not variables_objetivo:
        variables_objetivo = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    return columna_temporal, variables_objetivo

def filtrar_periodo_temporal(df: pd.DataFrame, columna_temporal: str, periodo: Dict[str, Any]) -> pd.DataFrame:
    """
    Filtra el DataFrame según el período temporal especificado.
    
    Args:
        df: DataFrame con los datos
        columna_temporal: Nombre de la columna temporal
        periodo: Información del período a filtrar
        
    Returns:
        DataFrame filtrado
    """
    if not columna_temporal or columna_temporal not in df.columns:
        return df
    
    # Asegurar que la columna sea datetime
    df_temp = df.copy()
    df_temp[columna_temporal] = pd.to_datetime(df_temp[columna_temporal], errors='coerce')
    
    # Obtener fecha actual para cálculos relativos
    fecha_actual = datetime.now()
    
    if periodo['tipo'] == 'ultimo_año':
        fecha_inicio = fecha_actual - timedelta(days=365)
        df_temp = df_temp[df_temp[columna_temporal] >= fecha_inicio]
    
    elif periodo['tipo'] == 'ultimos_6_meses':
        fecha_inicio = fecha_actual - timedelta(days=180)
        df_temp = df_temp[df_temp[columna_temporal] >= fecha_inicio]
    
    elif periodo['tipo'] == 'ultimos_3_meses':
        fecha_inicio = fecha_actual - timedelta(days=90)
        df_temp = df_temp[df_temp[columna_temporal] >= fecha_inicio]
    
    elif periodo['tipo'] == 'ultimo_mes':
        fecha_inicio = fecha_actual - timedelta(days=30)
        df_temp = df_temp[df_temp[columna_temporal] >= fecha_inicio]
    
    elif periodo['tipo'] == 'este_año':
        fecha_inicio = fecha_actual.replace(month=1, day=1)
        df_temp = df_temp[df_temp[columna_temporal] >= fecha_inicio]
    
    elif periodo['tipo'] == 'este_mes':
        fecha_inicio = fecha_actual.replace(day=1)
        df_temp = df_temp[df_temp[columna_temporal] >= fecha_inicio]
    
    # Para 'todo' no se aplica filtro
    
    return df_temp

@lru_cache(maxsize=128)
def calcular_serie_temporal_cache(df_hash: str, columna_temporal: str, variables_objetivo: tuple, 
                                metrica: str, frecuencia: str) -> pd.DataFrame:
    """
    Versión cacheada del cálculo de serie temporal para mejorar rendimiento.
    
    Args:
        df_hash: Hash del DataFrame para cache
        columna_temporal: Nombre de la columna temporal
        variables_objetivo: Tuple de variables a analizar
        metrica: Métrica a calcular
        frecuencia: Frecuencia temporal
        
    Returns:
        DataFrame con la serie temporal calculada
    """
    # Esta función se implementaría con el DataFrame real
    # Por ahora es un placeholder para la estructura de cache
    return pd.DataFrame()

def calcular_serie_temporal(df: pd.DataFrame, columna_temporal: str, variables_objetivo: List[str], 
                          metrica: str, frecuencia: str, suavizado: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Calcula la serie temporal para las variables objetivo con suavizado opcional.
    
    Args:
        df: DataFrame con los datos
        columna_temporal: Nombre de la columna temporal
        variables_objetivo: Lista de variables a analizar
        metrica: Métrica a calcular ('promedio', 'suma', etc.)
        frecuencia: Frecuencia temporal ('M', 'Q', 'Y', etc.)
        suavizado: Parámetros de suavizado opcional
        
    Returns:
        DataFrame con la serie temporal calculada
    """
    if not columna_temporal or not variables_objetivo:
        return pd.DataFrame()
    
    # Asegurar que la columna temporal sea datetime
    df_temp = df.copy()
    df_temp[columna_temporal] = pd.to_datetime(df_temp[columna_temporal], errors='coerce')
    
    # Eliminar filas con fechas nulas
    df_temp = df_temp.dropna(subset=[columna_temporal])
    
    if df_temp.empty:
        return pd.DataFrame()
    
    # Configurar índice temporal
    df_temp = df_temp.set_index(columna_temporal)
    
    # Mapear métricas a funciones de pandas
    metricas_pandas = {
        'promedio': 'mean',
        'suma': 'sum',
        'contar': 'count',
        'maximo': 'max',
        'minimo': 'min',
        'mediana': 'median',
        'desviacion': 'std'
    }
    
    funcion_metrica = metricas_pandas.get(metrica, 'mean')
    
    # Calcular serie temporal para cada variable
    series_temporales = []
    
    for variable in variables_objetivo:
        if variable in df_temp.columns:
            # Resample y aplicar métrica
            serie = getattr(df_temp[variable].resample(frecuencia), funcion_metrica)()
            
            # Aplicar suavizado si se solicita
            if suavizado and suavizado.get('aplicar', False):
                serie = aplicar_suavizado(serie, suavizado)
            
            serie = serie.reset_index()
            serie['variable'] = variable
            serie['metrica'] = metrica
            series_temporales.append(serie)
    
    if not series_temporales:
        return pd.DataFrame()
    
    # Combinar todas las series
    df_serie = pd.concat(series_temporales, ignore_index=True)
    
    return df_serie

def aplicar_suavizado(serie: pd.Series, suavizado: Dict[str, Any]) -> pd.Series:
    """
    Aplica suavizado a una serie temporal.
    
    Args:
        serie: Serie temporal a suavizar
        suavizado: Parámetros de suavizado
        
    Returns:
        Serie suavizada
    """
    if serie.empty:
        return serie
    
    ventana = suavizado.get('ventana', 7)
    tipo = suavizado.get('tipo', 'media_movil')
    
    # Asegurar que la ventana no sea mayor que la serie
    ventana = min(ventana, len(serie) // 2)
    if ventana < 2:
        return serie
    
    if tipo == 'media_movil':
        return serie.rolling(window=ventana, center=True).mean()
    elif tipo == 'mediana_movil':
        return serie.rolling(window=ventana, center=True).median()
    else:
        return serie

def detectar_descomposicion_estacional(question: str) -> Dict[str, Any]:
    """
    Detecta si se solicita descomposición estacional en la pregunta.
    
    Args:
        question: Pregunta en lenguaje natural
        
    Returns:
        Dict con parámetros de descomposición
    """
    q = question.lower()
    
    # Detectar si se menciona descomposición
    aplicar_descomposicion = False
    for categoria, palabras in DESCOMPOSICION_KEYWORDS.items():
        if any(palabra in q for palabra in palabras):
            aplicar_descomposicion = True
            break
    
    if not aplicar_descomposicion:
        return {'aplicar': False}
    
    # Detectar período estacional
    periodo_estacional = 12  # Por defecto (mensual)
    if any(palabra in q for palabra in ['diario', 'día a día', 'daily']):
        periodo_estacional = 7  # Semanal
    elif any(palabra in q for palabra in ['semanal', 'weekly']):
        periodo_estacional = 4  # Mensual
    elif any(palabra in q for palabra in ['trimestral', 'quarterly']):
        periodo_estacional = 4  # Anual
    
    return {
        'aplicar': True,
        'periodo': periodo_estacional,
        'metodo': 'additive'  # Por defecto aditivo
    }

def generar_grafico_tendencia(df_serie: pd.DataFrame, columna_temporal: str, 
                            variables_objetivo: List[str], metrica: str, 
                            tipo_analisis: str, suavizado: Dict[str, Any] = None,
                            tipo_ajuste: str = 'lineal', descomposicion: Dict[str, Any] = None) -> go.Figure:
    """
    Genera gráfico de tendencia temporal con opciones avanzadas incluyendo descomposición estacional.
    
    Args:
        df_serie: DataFrame con la serie temporal
        columna_temporal: Nombre de la columna temporal
        variables_objetivo: Variables analizadas
        metrica: Métrica calculada
        tipo_analisis: Tipo de análisis ('tendencia' o 'comparacion')
        suavizado: Parámetros de suavizado
        tipo_ajuste: Tipo de ajuste de tendencia
        descomposicion: Parámetros de descomposición estacional
        
    Returns:
        Figura de Plotly
    """
    if df_serie.empty:
        return go.Figure()
    
    # Si se solicita descomposición estacional, crear subplots especiales
    if descomposicion and descomposicion.get('aplicar', False) and len(variables_objetivo) == 1:
        return generar_grafico_descomposicion(df_serie, columna_temporal, variables_objetivo[0], metrica)
    
    # Crear figura normal
    if len(variables_objetivo) > 1:
        # Múltiples variables - usar subplots
        fig = make_subplots(
            rows=len(variables_objetivo), cols=1,
            subplot_titles=[f"{metrica.title()} de {var}" for var in variables_objetivo],
            vertical_spacing=0.1
        )
        
        for i, variable in enumerate(variables_objetivo, 1):
            df_var = df_serie[df_serie['variable'] == variable]
            
            # Línea principal
            fig.add_trace(
                go.Scatter(
                    x=df_var[columna_temporal],
                    y=df_var[variable],
                    mode='lines+markers',
                    name=variable,
                    line=dict(width=2),
                    marker=dict(size=6)
                ),
                row=i, col=1
            )
            
            # Línea de tendencia si es análisis de tendencia
            if tipo_analisis == 'tendencia' and len(df_var) > 2:
                trend_line = calcular_ajuste_tendencia(df_var[columna_temporal], df_var[variable], tipo_ajuste)
                if trend_line is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=df_var[columna_temporal],
                            y=trend_line,
                            mode='lines',
                            name=f'Tendencia {variable}',
                            line=dict(dash='dash', color='red', width=2)
                        ),
                        row=i, col=1
                    )
        
        fig.update_layout(
            height=300 * len(variables_objetivo),
            showlegend=False,
            title=f"Análisis de {tipo_analisis.title()} - {metrica.title()}"
        )
        
    else:
        # Una sola variable - gráfico simple
        variable = variables_objetivo[0]
        df_var = df_serie[df_serie['variable'] == variable]
        
        fig = go.Figure()
        
        # Línea principal
        fig.add_trace(
            go.Scatter(
                x=df_var[columna_temporal],
                y=df_var[variable],
                mode='lines+markers',
                name=f"{metrica.title()} de {variable}",
                line=dict(width=3, color='blue'),
                marker=dict(size=8, color='blue')
            )
        )
        
        # Línea de tendencia si es análisis de tendencia
        if tipo_analisis == 'tendencia' and len(df_var) > 2:
            trend_line = calcular_ajuste_tendencia(df_var[columna_temporal], df_var[variable], tipo_ajuste)
            if trend_line is not None:
                fig.add_trace(
                    go.Scatter(
                        x=df_var[columna_temporal],
                        y=trend_line,
                        mode='lines',
                        name='Tendencia',
                        line=dict(dash='dash', color='red', width=2)
                    )
                )
        
        fig.update_layout(
            title=f"Tendencia de {metrica} de {variable}",
            xaxis_title="Tiempo",
            yaxis_title=f"{metrica.title()} de {variable}",
            hovermode='x unified'
        )
    
    return fig

def calcular_ajuste_tendencia(x: pd.Series, y: pd.Series, tipo_ajuste: str) -> Optional[np.ndarray]:
    """
    Calcula ajuste de tendencia según el tipo especificado.
    
    Args:
        x: Variable independiente (tiempo)
        y: Variable dependiente
        tipo_ajuste: Tipo de ajuste ('lineal', 'polinomial', 'lowess')
        
    Returns:
        Array con valores ajustados o None si no se puede calcular
    """
    if len(x) < 3:
        return None
    
    try:
        x_numeric = np.arange(len(x))
        y_values = y.values
        
        if tipo_ajuste == 'lineal':
            z = np.polyfit(x_numeric, y_values, 1)
            p = np.poly1d(z)
            return p(x_numeric)
        
        elif tipo_ajuste == 'polinomial':
            # Usar grado 2 (cuadrático) por defecto
            z = np.polyfit(x_numeric, y_values, 2)
            p = np.poly1d(z)
            return p(x_numeric)
        
        elif tipo_ajuste == 'lowess':
            # Implementación simple de LOWESS
            from scipy import interpolate
            if len(x_numeric) >= 5:
                # Usar interpolación cúbica como aproximación
                f = interpolate.interp1d(x_numeric, y_values, kind='cubic', fill_value='extrapolate')
                return f(x_numeric)
            else:
                # Fallback a lineal si no hay suficientes puntos
                z = np.polyfit(x_numeric, y_values, 1)
                p = np.poly1d(z)
                return p(x_numeric)
        
        else:
            # Fallback a lineal
            z = np.polyfit(x_numeric, y_values, 1)
            p = np.poly1d(z)
            return p(x_numeric)
    
    except:
        return None

def generar_insights_temporales(df_serie: pd.DataFrame, columna_temporal: str, 
                              variables_objetivo: List[str], metrica: str,
                              suavizado: Dict[str, Any] = None) -> str:
    """
    Genera insights automáticos sobre la serie temporal con análisis avanzado.
    
    Args:
        df_serie: DataFrame con la serie temporal
        columna_temporal: Nombre de la columna temporal
        variables_objetivo: Variables analizadas
        metrica: Métrica calculada
        suavizado: Parámetros de suavizado aplicado
        
    Returns:
        String con insights generados
    """
    if df_serie.empty:
        return "No se pudieron generar insights debido a datos insuficientes."
    
    insights = []
    
    for variable in variables_objetivo:
        df_var = df_serie[df_serie['variable'] == variable]
        
        if len(df_var) < 2:
            continue
        
        valores = df_var[variable].values
        
        # Calcular estadísticas básicas
        valor_actual = valores[-1]
        valor_inicial = valores[0]
        cambio_total = valor_actual - valor_inicial
        cambio_porcentual = (cambio_total / valor_inicial * 100) if valor_inicial != 0 else 0
        
        # Detectar tendencia con R²
        r_cuadrado = None
        if len(valores) >= 3:
            x_numeric = np.arange(len(valores))
            z = np.polyfit(x_numeric, valores, 1)
            p = np.poly1d(z)
            y_pred = p(x_numeric)
            
            # Calcular R²
            ss_res = np.sum((valores - y_pred) ** 2)
            ss_tot = np.sum((valores - np.mean(valores)) ** 2)
            r_cuadrado = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            pendiente = z[0]
            if pendiente > 0:
                tendencia = "creciente"
            elif pendiente < 0:
                tendencia = "decreciente"
            else:
                tendencia = "estable"
        else:
            tendencia = "insuficiente para determinar"
        
        # Detectar valores máximos y mínimos
        valor_max = valores.max()
        valor_min = valores.min()
        
        # Detectar outliers (valores más allá de 2 desviaciones estándar)
        media = np.mean(valores)
        std = np.std(valores)
        outliers = valores[(valores < media - 2*std) | (valores > media + 2*std)]
        
        # Generar insight
        insight = f"**{variable}**: "
        insight += f"El {metrica} actual es {valor_actual:.2f}. "
        
        if cambio_porcentual != 0:
            if cambio_porcentual > 0:
                insight += f"Ha aumentado un {cambio_porcentual:.1f}% desde el inicio del período. "
            else:
                insight += f"Ha disminuido un {abs(cambio_porcentual):.1f}% desde el inicio del período. "
        
        insight += f"La tendencia general es {tendencia}"
        if r_cuadrado is not None:
            insight += f" (R² = {r_cuadrado:.3f})"
        insight += ". "
        
        insight += f"El valor máximo fue {valor_max:.2f} y el mínimo {valor_min:.2f}. "
        
        if len(outliers) > 0:
            insight += f"Se detectaron {len(outliers)} valores atípicos. "
        
        if suavizado and suavizado.get('aplicar', False):
            insight += f"Se aplicó suavizado {suavizado.get('tipo', '')} con ventana de {suavizado.get('ventana', '')} períodos. "
        
        insights.append(insight)
    
    return " ".join(insights)

def generar_grafico_descomposicion(df_serie: pd.DataFrame, columna_temporal: str, 
                                  variable: str, metrica: str) -> go.Figure:
    """
    Genera gráfico de descomposición estacional (STL).
    
    Args:
        df_serie: DataFrame con la serie temporal
        columna_temporal: Nombre de la columna temporal
        variable: Variable a descomponer
        metrica: Métrica calculada
        
    Returns:
        Figura de Plotly con componentes de descomposición
    """
    if not STL_AVAILABLE:
        # Fallback si no hay statsmodels
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ Descomposición estacional no disponible (statsmodels requerido)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig
    
    df_var = df_serie[df_serie['variable'] == variable]
    if len(df_var) < 12:  # Necesitamos suficientes puntos para descomposición
        fig = go.Figure()
        fig.add_annotation(
            text="⚠️ Se necesitan al menos 12 puntos para descomposición estacional",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="orange")
        )
        return fig
    
    # Preparar serie para descomposición
    serie = df_var.set_index(columna_temporal)[variable]
    serie = serie.sort_index()  # Asegurar orden temporal
    
    # Determinar período estacional
    if len(serie) >= 24:
        periodo = 12  # Mensual
    elif len(serie) >= 14:
        periodo = 7   # Semanal
    else:
        periodo = min(4, len(serie) // 3)  # Mínimo razonable
    
    try:
        # Realizar descomposición
        decomposition = seasonal_decompose(serie, model='additive', period=periodo)
        
        # Crear subplots para cada componente
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                f'Serie Original - {metrica.title()} de {variable}',
                'Tendencia',
                'Estacionalidad',
                'Residuos'
            ],
            vertical_spacing=0.08
        )
        
        # Serie original
        fig.add_trace(
            go.Scatter(
                x=serie.index,
                y=serie.values,
                mode='lines',
                name='Original',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Tendencia
        fig.add_trace(
            go.Scatter(
                x=decomposition.trend.index,
                y=decomposition.trend.values,
                mode='lines',
                name='Tendencia',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # Estacionalidad
        fig.add_trace(
            go.Scatter(
                x=decomposition.seasonal.index,
                y=decomposition.seasonal.values,
                mode='lines',
                name='Estacionalidad',
                line=dict(color='green', width=2)
            ),
            row=3, col=1
        )
        
        # Residuos
        fig.add_trace(
            go.Scatter(
                x=decomposition.resid.index,
                y=decomposition.resid.values,
                mode='lines',
                name='Residuos',
                line=dict(color='purple', width=1)
            ),
            row=4, col=1
        )
        
        # Actualizar layout
        fig.update_layout(
            height=800,
            title=f"Descomposición Estacional - {metrica.title()} de {variable}",
            showlegend=False
        )
        
        # Actualizar ejes X
        for i in range(1, 5):
            fig.update_xaxes(title_text="Tiempo", row=i, col=1)
        
        # Actualizar ejes Y
        fig.update_yaxes(title_text="Valor", row=1, col=1)
        fig.update_yaxes(title_text="Tendencia", row=2, col=1)
        fig.update_yaxes(title_text="Estacionalidad", row=3, col=1)
        fig.update_yaxes(title_text="Residuos", row=4, col=1)
        
        return fig
        
    except Exception as e:
        # Fallback en caso de error
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ Error en descomposición: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig

def exportar_resultados_csv(df_serie: pd.DataFrame, resultado: Dict[str, Any], 
                           filename: str = None) -> str:
    """
    Exporta los resultados del análisis temporal a CSV.
    
    Args:
        df_serie: DataFrame con la serie temporal
        resultado: Resultado del análisis temporal
        filename: Nombre del archivo (opcional)
        
    Returns:
        Ruta del archivo CSV generado
    """
    if df_serie.empty:
        return None
    
    # Generar nombre de archivo si no se proporciona
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analisis_temporal_{timestamp}.csv"
    
    # Crear DataFrame de exportación
    df_export = df_serie.copy()
    
    # Añadir metadatos del análisis
    metadatos = {
        'parametro': [
            'tipo_analisis', 'metrica', 'frecuencia', 'periodo',
            'suavizado_aplicado', 'tipo_ajuste', 'descomposicion_aplicada'
        ],
        'valor': [
            resultado.get('tipo_analisis', ''),
            resultado.get('metrica', ''),
            resultado.get('frecuencia', ''),
            str(resultado.get('periodo', {})),
            str(resultado.get('suavizado', {}).get('aplicar', False)),
            resultado.get('tipo_ajuste', ''),
            str(resultado.get('descomposicion', {}).get('aplicar', False))
        ]
    }
    
    df_metadatos = pd.DataFrame(metadatos)
    
    # Calcular métricas adicionales para exportación
    if not df_export.empty:
        # Agrupar por variable si hay múltiples
        if 'variable' in df_export.columns:
            metricas_adicionales = []
            for var in df_export['variable'].unique():
                df_var = df_export[df_export['variable'] == var]
                valores = df_var.iloc[:, 1].values  # Primera columna numérica
                
                if len(valores) > 0:
                    # Calcular estadísticas
                    cambio_total = valores[-1] - valores[0] if len(valores) > 1 else 0
                    cambio_porcentual = (cambio_total / valores[0] * 100) if valores[0] != 0 else 0
                    
                    # Detectar outliers
                    media = np.mean(valores)
                    std = np.std(valores)
                    outliers = valores[(valores < media - 2*std) | (valores > media + 2*std)]
                    
                    metricas_adicionales.append({
                        'variable': var,
                        'valor_inicial': valores[0],
                        'valor_final': valores[-1],
                        'cambio_total': cambio_total,
                        'cambio_porcentual': cambio_porcentual,
                        'valor_maximo': np.max(valores),
                        'valor_minimo': np.min(valores),
                        'media': media,
                        'desviacion_estandar': std,
                        'num_outliers': len(outliers),
                        'registros_analizados': len(valores)
                    })
            
            df_metricas = pd.DataFrame(metricas_adicionales)
        else:
            df_metricas = pd.DataFrame()
    
    # Guardar archivo CSV
    try:
        # Crear directorio de exportación si no existe
        import os
        export_dir = "data/export"
        os.makedirs(export_dir, exist_ok=True)
        
        filepath = os.path.join(export_dir, filename)
        
        # Escribir datos principales
        df_export.to_csv(filepath, index=False, encoding='utf-8')
        
        # Escribir metadatos en archivo separado
        metadatos_filename = filename.replace('.csv', '_metadatos.csv')
        metadatos_filepath = os.path.join(export_dir, metadatos_filename)
        df_metadatos.to_csv(metadatos_filepath, index=False, encoding='utf-8')
        
        # Escribir métricas adicionales si existen
        if not df_metricas.empty:
            metricas_filename = filename.replace('.csv', '_metricas.csv')
            metricas_filepath = os.path.join(export_dir, metricas_filename)
            df_metricas.to_csv(metricas_filepath, index=False, encoding='utf-8')
        
        return filepath
        
    except Exception as e:
        print(f"Error al exportar CSV: {str(e)}")
        return None

def analizar_tendencia_temporal(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    """
    Función principal para analizar tendencias temporales con funcionalidades avanzadas.
    
    Args:
        df: DataFrame con los datos
        question: Pregunta en lenguaje natural
        
    Returns:
        Dict con resultados del análisis temporal
    """
    # 1. Detectar si es análisis temporal
    params_temporal = detectar_analisis_temporal(question)
    if not params_temporal:
        return {'error': 'No se detectó análisis temporal en la pregunta'}
    
    # 2. Extraer variables
    columna_temporal, variables_objetivo = extraer_variables_temporales(df, question)
    
    if not columna_temporal:
        return {'error': 'No se encontró columna temporal en los datos'}
    
    if not variables_objetivo:
        return {'error': 'No se encontraron variables numéricas para analizar'}
    
    # 3. Filtrar período
    df_filtrado = filtrar_periodo_temporal(df, columna_temporal, params_temporal['periodo'])
    
    if df_filtrado.empty:
        return {'error': 'No hay datos en el período especificado'}
    
    # 4. Calcular serie temporal
    df_serie = calcular_serie_temporal(
        df_filtrado, 
        columna_temporal, 
        variables_objetivo, 
        params_temporal['metrica'], 
        params_temporal['frecuencia'],
        params_temporal.get('suavizado')
    )
    
    if df_serie.empty:
        return {'error': 'No se pudo calcular la serie temporal'}
    
    # 5. Generar gráfico
    fig = generar_grafico_tendencia(
        df_serie, 
        columna_temporal, 
        variables_objetivo, 
        params_temporal['metrica'], 
        params_temporal['tipo'],
        params_temporal.get('suavizado'),
        params_temporal.get('tipo_ajuste', 'lineal'),
        params_temporal.get('descomposicion')
    )
    
    # 6. Generar insights
    insights = generar_insights_temporales(
        df_serie, 
        columna_temporal, 
        variables_objetivo, 
        params_temporal['metrica'],
        params_temporal.get('suavizado')
    )
    
    # 7. Preparar datos para exportación
    datos_exportacion = {
        'serie_temporal': df_serie,
        'metadatos': {
            'tipo_analisis': params_temporal['tipo'],
            'metrica': params_temporal['metrica'],
            'frecuencia': params_temporal['frecuencia'],
            'frecuencia_personalizada': params_temporal.get('frecuencia_personalizada'),
            'periodo': params_temporal['periodo'],
            'suavizado': params_temporal.get('suavizado'),
            'tipo_ajuste': params_temporal.get('tipo_ajuste'),
            'descomposicion': params_temporal.get('descomposicion'),
            'variables_analizadas': variables_objetivo,
            'columna_temporal': columna_temporal,
            'registros_analizados': len(df_filtrado),
            'fecha_analisis': datetime.now().isoformat()
        }
    }
    
    return {
        'tipo_analisis': params_temporal['tipo'],
        'metrica': params_temporal['metrica'],
        'frecuencia': params_temporal['frecuencia'],
        'frecuencia_personalizada': params_temporal.get('frecuencia_personalizada'),
        'periodo': params_temporal['periodo'],
        'suavizado': params_temporal.get('suavizado'),
        'tipo_ajuste': params_temporal.get('tipo_ajuste'),
        'descomposicion': params_temporal.get('descomposicion'),
        'variables_analizadas': variables_objetivo,
        'columna_temporal': columna_temporal,
        'datos_serie': df_serie,
        'grafico': fig,
        'insights': insights,
        'registros_analizados': len(df_filtrado),
        'datos_exportacion': datos_exportacion,
        'stl_disponible': STL_AVAILABLE
    } 