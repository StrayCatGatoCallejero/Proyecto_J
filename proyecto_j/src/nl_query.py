"""
Módulo de Consultas en Lenguaje Natural - Versión Avanzada
=========================================================

Sistema robusto para parsear preguntas complejas en lenguaje natural y convertirlas
en consultas de pandas para análisis de datos avanzado.
"""

import re
import pandas as pd
from dateutil import parser as date_parser
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

# Importar módulo de tendencias temporales
try:
    from nl_query_trends import analizar_tendencia_temporal, detectar_analisis_temporal
except ImportError:
    # Si no está disponible, crear funciones dummy
    def analizar_tendencia_temporal(df, question):
        return {'error': 'Módulo de tendencias no disponible'}
    
    def detectar_analisis_temporal(question):
        return None

# Importar módulo de agrupación compleja
try:
    from complex_grouping import ComplexGrouping, detect_complex_grouping_in_question, execute_complex_grouping_from_question
except ImportError:
    # Si no está disponible, crear funciones dummy
    class ComplexGrouping:
        def __init__(self, df, variable_metadata=None):
            pass
        def detect_complex_grouping(self, question):
            return {}
        def execute_complex_grouping(self, params, target_variables=None):
            return pd.DataFrame()
    
    def detect_complex_grouping_in_question(question, df, variable_metadata=None):
        return {}
    
    def execute_complex_grouping_from_question(question, df, target_variables=None, variable_metadata=None):
        return pd.DataFrame(), "Módulo de agrupación compleja no disponible"

# Mapeo de meses en español
MESES_ESPANOL = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
    'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}

# Acciones soportadas expandidas
ACCIONES_SOPORTADAS = {
    # Acciones básicas
    'contar': ['contar', 'cuantos', 'cuántos', 'numero', 'número', 'cantidad', 'total', 'cuántas', 'cuantas'],
    'promedio': ['promedio', 'media', 'promedio de', 'media de', 'valor promedio'],
    'suma': ['suma', 'sumar', 'total de', 'suma de', 'sumatoria'],
    'maximo': ['máximo', 'maximo', 'mayor', 'mas alto', 'más alto', 'valor máximo', 'pico'],
    'minimo': ['mínimo', 'minimo', 'menor', 'mas bajo', 'más bajo', 'valor mínimo'],
    
    # Acciones estadísticas avanzadas
    'mediana': ['mediana', 'valor central', 'valor del medio'],
    'moda': ['moda', 'valor más frecuente', 'más común'],
    'desviacion': ['desviación', 'desviacion', 'desviación estándar', 'desviacion estandar', 'std'],
    'varianza': ['varianza', 'variabilidad'],
    'percentil': ['percentil', 'percentil 25', 'percentil 75', 'q1', 'q3'],
    
    # Acciones compuestas
    'distribucion': ['distribución', 'distribucion', 'distribuir', 'repartir'],
    'tendencia': ['tendencia', 'evolución', 'cambio en el tiempo'],
    'comparar': ['comparar', 'comparación', 'comparacion', 'vs', 'versus'],
    'porcentaje': ['porcentaje', 'por ciento', '%', 'porcentual'],
    
    # Acciones condicionales
    'cuantos_tienen': ['cuántos tienen', 'cuantos tienen', 'cuántas tienen', 'cuantas tienen'],
    'que_porcentaje': ['qué porcentaje', 'que porcentaje', 'qué %', 'que %'],
    'cuantos_por': ['cuántos por', 'cuantos por', 'distribución de', 'distribucion de']
}

# Palabras de conexión para filtros múltiples
CONECTORES = {
    'Y': ['y', 'and', 'además', 'también', 'asimismo'],
    'O': ['o', 'or', 'alternativamente', 'bien'],
    'NO': ['no', 'not', 'excepto', 'excluyendo', 'sin']
}

# Expresiones temporales
EXPRESIONES_TEMPORALES = {
    'hoy': lambda: datetime.now().date(),
    'ayer': lambda: (datetime.now() - timedelta(days=1)).date(),
    'mañana': lambda: (datetime.now() + timedelta(days=1)).date(),
    'esta_semana': lambda: (datetime.now() - timedelta(days=datetime.now().weekday())).date(),
    'semana_pasada': lambda: (datetime.now() - timedelta(days=datetime.now().weekday() + 7)).date(),
    'este_mes': lambda: datetime.now().replace(day=1).date(),
    'mes_pasado': lambda: (datetime.now().replace(day=1) - timedelta(days=1)).replace(day=1).date(),
    'este_año': lambda: datetime.now().replace(month=1, day=1).date(),
    'año_pasado': lambda: datetime.now().replace(year=datetime.now().year-1, month=1, day=1).date(),
    'últimos_3_meses': lambda: (datetime.now() - timedelta(days=90)).date(),
    'últimos_6_meses': lambda: (datetime.now() - timedelta(days=180)).date(),
    'último_año': lambda: (datetime.now() - timedelta(days=365)).date()
}

def parse_and_execute(df: pd.DataFrame, question: str, variable_metadata: Optional[Dict] = None) -> Tuple[pd.DataFrame, Any, pd.DataFrame, str, str]:
    """
    Parsea una pregunta compleja en lenguaje natural y ejecuta la consulta.
    
    Args:
        df: DataFrame con los datos
        question: Pregunta en lenguaje natural
        variable_metadata: Metadatos de variables (opcional)
    
    Returns:
        Tuple con (df_filtrado, resultado, df_grafico, tipo_grafico, interpretacion)
    """
    # Limpiar y normalizar pregunta
    q = question.lower().strip()
    
    # 1. Verificar si es análisis temporal
    params_temporal = detectar_analisis_temporal(question)
    if params_temporal:
        return ejecutar_analisis_temporal(df, question, params_temporal)
    
    # 2. Si no es temporal, continuar con análisis normal
    # Detectar acción principal y secundarias
    accion_principal, acciones_secundarias = detectar_acciones_complejas(q)
    
    # Detectar variables objetivo (múltiples)
    variables_objetivo = detectar_variables_objetivo_avanzado(q, df, variable_metadata)
    
    # Detectar filtros complejos
    filtros = detectar_filtros_avanzados(q, df, variable_metadata)
    
    # Detectar agrupaciones
    agrupaciones = detectar_agrupaciones(q, df, variable_metadata)
    
    # Aplicar filtros
    df_filtrado = aplicar_filtros_avanzados(df, filtros)
    
    # Ejecutar acción principal
    resultado = ejecutar_accion_avanzada(df_filtrado, accion_principal, variables_objetivo, agrupaciones)
    
    # Generar datos para gráfico
    df_grafico, tipo_grafico = generar_grafico_avanzado(df_filtrado, accion_principal, variables_objetivo, filtros, agrupaciones)
    
    # Generar interpretación
    variable_objetivo_str = variables_objetivo[0] if variables_objetivo else None
    interpretacion = interpretar_resultado(accion_principal, resultado, variable_objetivo_str)
    
    return df_filtrado, resultado, df_grafico, tipo_grafico, interpretacion

def ejecutar_analisis_temporal(df: pd.DataFrame, question: str, params_temporal: Dict[str, Any]) -> Tuple[pd.DataFrame, Any, pd.DataFrame, str, str]:
    """
    Ejecuta análisis temporal y retorna resultados en formato compatible.
    
    Args:
        df: DataFrame con los datos
        question: Pregunta original
        params_temporal: Parámetros del análisis temporal
        
    Returns:
        Tuple compatible con parse_and_execute
    """
    # Ejecutar análisis temporal
    resultado_temporal = analizar_tendencia_temporal(df, question)
    
    if 'error' in resultado_temporal:
        # Si hay error, retornar resultado vacío
        return df, None, pd.DataFrame(), 'error', f"Error en análisis temporal: {resultado_temporal['error']}"
    
    # Extraer componentes del resultado temporal
    df_serie = resultado_temporal.get('datos_serie', pd.DataFrame())
    grafico = resultado_temporal.get('grafico')
    insights = resultado_temporal.get('insights', '')
    
    # Convertir gráfico a DataFrame si es necesario
    if grafico and hasattr(grafico, 'data'):
        # Crear DataFrame simple para compatibilidad
        df_grafico = pd.DataFrame({
            'tipo': 'tendencia_temporal',
            'metrica': resultado_temporal.get('metrica', ''),
            'frecuencia': resultado_temporal.get('frecuencia', ''),
            'variables': str(resultado_temporal.get('variables_analizadas', []))
        }, index=[0])
    else:
        df_grafico = df_serie if not df_serie.empty else pd.DataFrame()
    
    # Crear interpretación combinada
    interpretacion = f"**Análisis de {resultado_temporal.get('tipo_analisis', 'tendencia')}**: "
    interpretacion += f"Se analizó la {resultado_temporal.get('metrica', '')} de "
    interpretacion += f"{', '.join(resultado_temporal.get('variables_analizadas', []))} "
    interpretacion += f"con frecuencia {resultado_temporal.get('frecuencia', '')}. "
    interpretacion += insights
    
    return df, df_serie, df_grafico, 'tendencia_temporal', interpretacion

def detectar_acciones_complejas(question: str) -> Tuple[str, List[str]]:
    """Detecta acciones principales y secundarias en la pregunta."""
    q = question.lower()
    acciones_detectadas = []
    
    # Detectar todas las acciones presentes
    for accion, palabras in ACCIONES_SOPORTADAS.items():
        if any(palabra in q for palabra in palabras):
            acciones_detectadas.append(accion)
    
    # Determinar acción principal
    if not acciones_detectadas:
        return 'contar', []
    
    # Priorizar acciones específicas
    prioridad = {
        'comparar': 1,
        'distribucion': 2,
        'tendencia': 3,
        'porcentaje': 4,
        'cuantos_tienen': 5,
        'que_porcentaje': 6,
        'cuantos_por': 7,
        'percentil': 8,
        'mediana': 9,
        'moda': 10,
        'desviacion': 11,
        'varianza': 12,
        'maximo': 13,
        'minimo': 14,
        'promedio': 15,
        'suma': 16,
        'contar': 17
    }
    
    # Ordenar por prioridad
    acciones_ordenadas = sorted(acciones_detectadas, key=lambda x: prioridad.get(x, 999))
    
    accion_principal = acciones_ordenadas[0]
    acciones_secundarias = acciones_ordenadas[1:]
    
    return accion_principal, acciones_secundarias

def detectar_variables_objetivo_avanzado(question: str, df: pd.DataFrame, variable_metadata: Optional[Dict] = None) -> List[str]:
    """Detecta múltiples variables objetivo en la pregunta."""
    variables = []
    q = question.lower()
    
    # 1. Buscar nombres exactos de columnas
    for col in df.columns:
        if col.lower() in q:
            variables.append(col)
    
    # 2. Buscar sinónimos y palabras relacionadas
    sinonimos = {
        'edad': ['años', 'age', 'antigüedad'],
        'ingreso_mensual': ['salario', 'renta', 'income', 'sueldo', 'ganancia', 'ingreso', 'ingresos'],
        'fecha_registro': ['date', 'time', 'dia', 'día', 'momento', 'fecha'],
        'genero': ['sexo', 'gender', 'masculino', 'femenino'],
        'region': ['zona', 'area', 'territorio', 'lugar'],
        'nivel_satisfaccion': ['satisfacción', 'contento', 'felicidad', 'likert', 'satisfaccion'],
        'tipo_cliente': ['cliente', 'tipo', 'categoria', 'categoría']
    }
    
    for col in df.columns:
        if col not in variables:  # Evitar duplicados
            for sinonimo, palabras in sinonimos.items():
                if any(palabra in q for palabra in palabras):
                    variables.append(col)
                    break
    
    # 3. Si no se encontraron variables, usar metadatos para sugerir
    if not variables and variable_metadata:
        # Para métricas numéricas, usar la primera variable numérica
        for col, meta in variable_metadata.items():
            if meta.get('type') == 'numerica':
                variables.append(col)
                break
    
    # 4. Si aún no hay variables, usar la primera columna numérica
    if not variables:
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                variables.append(col)
                break
    
    return variables

def detectar_filtros_avanzados(question: str, df: pd.DataFrame, variable_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Detecta filtros complejos y múltiples."""
    filtros = {
        'categoria': {},
        'fecha': {},
        'numerico': {},
        'texto': {},
        'logica': 'AND'  # AND, OR, NOT
    }
    
    # 1. Detectar conectores lógicos
    q = question.lower()
    for conector, palabras in CONECTORES.items():
        if any(palabra in q for palabra in palabras):
            filtros['logica'] = conector
            break
    
    # 2. Detectar filtros de categoría múltiples
    filtros['categoria'] = detectar_filtros_categoria_avanzado(q, df, variable_metadata)
    
    # 3. Detectar filtros de fecha avanzados
    filtros['fecha'] = detectar_filtros_fecha_avanzado(q, df)
    
    # 4. Detectar filtros numéricos avanzados
    filtros['numerico'] = detectar_filtros_numericos_avanzado(q, df, variable_metadata)
    
    # 5. Detectar filtros de texto
    filtros['texto'] = detectar_filtros_texto(q, df)
    
    return filtros

def detectar_filtros_categoria_avanzado(question: str, df: pd.DataFrame, variable_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Detecta filtros de categoría múltiples y complejos."""
    filtros = {}
    
    # Buscar patrones como "clientes tipo A y B", "regiones Norte o Sur"
    for col in df.columns:
        valores_unicos = df[col].dropna().unique()
        
        # Buscar valores individuales
        valores_encontrados = []
        for valor in valores_unicos:
            if str(valor).lower() in question:
                valores_encontrados.append(valor)
        
        if valores_encontrados:
            filtros[col] = {
                'tipo': 'categoria_multiple',
                'valores': valores_encontrados,
                'operador': 'IN'  # Por defecto IN, se puede cambiar a OR si hay conectores
            }
    
    return filtros

def detectar_filtros_fecha_avanzado(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Detecta filtros de fecha avanzados incluyendo expresiones temporales."""
    filtros = {}
    
    # Buscar columna de fecha
    columna_fecha = None
    for col in df.columns:
        if any(palabra in col.lower() for palabra in ['fecha', 'date', 'time', 'dia', 'día']):
            columna_fecha = col
            break
    
    if not columna_fecha:
        return filtros
    
    # 1. Detectar expresiones temporales
    for expresion, funcion in EXPRESIONES_TEMPORALES.items():
        if expresion.replace('_', ' ') in question:
            fecha = funcion()
            if 'últimos' in expresion or 'pasado' in expresion:
                # Es un rango
                filtros[columna_fecha] = {
                    'tipo': 'rango_fecha',
                    'inicio': fecha,
                    'fin': datetime.now().date()
                }
            else:
                # Es una fecha específica
                filtros[columna_fecha] = {
                    'tipo': 'fecha_especifica',
                    'fecha': fecha
                }
            return filtros
    
    # 2. Detectar mes y año (código existente mejorado)
    date_match = re.search(
        r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+(\d{4})", 
        question
    )
    
    if date_match:
        month_name, year = date_match.groups()
        try:
            dt_start = datetime(int(year), MESES_ESPANOL[month_name], 1)
            dt_end = dt_start + timedelta(days=32)
            dt_end = dt_end.replace(day=1) - timedelta(days=1)
            
            filtros[columna_fecha] = {
                'tipo': 'rango_fecha',
                'inicio': dt_start.date(),
                'fin': dt_end.date()
            }
        except:
            pass
    
    # 3. Detectar año específico
    year_match = re.search(r"(\d{4})", question)
    if year_match and not date_match:
        year = int(year_match.group(1))
        dt_start = datetime(year, 1, 1)
        dt_end = datetime(year, 12, 31)
        
        filtros[columna_fecha] = {
            'tipo': 'rango_fecha',
            'inicio': dt_start.date(),
            'fin': dt_end.date()
        }
    
    return filtros

def detectar_filtros_numericos_avanzado(question: str, df: pd.DataFrame, variable_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Detecta filtros numéricos avanzados."""
    filtros = {}
    
    for col in df.columns:
        # Solo aplicar filtros numéricos a columnas numéricas
        if df[col].dtype not in ['int64', 'float64']:
            continue
        
        # 1. Detectar "mayor a X", "menor a X"
        mayor_match = re.search(r"mayor\s+a\s+(\d+(?:\.\d+)?)", question)
        if mayor_match:
            valor = float(mayor_match.group(1))
            filtros[col] = {'tipo': 'mayor_que', 'valor': valor}
            continue
        
        menor_match = re.search(r"menor\s+a\s+(\d+(?:\.\d+)?)", question)
        if menor_match:
            valor = float(menor_match.group(1))
            filtros[col] = {'tipo': 'menor_que', 'valor': valor}
            continue
        
        # 2. Detectar "entre X y Y"
        entre_match = re.search(r"entre\s+(\d+(?:\.\d+)?)\s+y\s+(\d+(?:\.\d+)?)", question)
        if entre_match:
            min_val = float(entre_match.group(1))
            max_val = float(entre_match.group(2))
            filtros[col] = {'tipo': 'rango', 'min': min_val, 'max': max_val}
            continue
        
        # 3. Detectar "top X%", "bottom X%"
        top_match = re.search(r"top\s+(\d+)%", question)
        if top_match:
            porcentaje = int(top_match.group(1))
            filtros[col] = {'tipo': 'top_percentil', 'porcentaje': porcentaje}
            continue
        
        bottom_match = re.search(r"bottom\s+(\d+)%", question)
        if bottom_match:
            porcentaje = int(bottom_match.group(1))
            filtros[col] = {'tipo': 'bottom_percentil', 'porcentaje': porcentaje}
            continue
        
        # 4. Detectar "mayor que el promedio"
        if "mayor que el promedio" in question or "más alto que la media" in question:
            filtros[col] = {'tipo': 'mayor_que_promedio'}
            continue
        
        # 5. Detectar "menor que el promedio"
        if "menor que el promedio" in question or "más bajo que la media" in question:
            filtros[col] = {'tipo': 'menor_que_promedio'}
            continue
    
    return filtros

def detectar_filtros_texto(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Detecta filtros de texto libre."""
    filtros = {}
    
    # Buscar patrones como "que contengan", "que incluyan"
    for col in df.columns:
        if df[col].dtype == 'object':  # Solo columnas de texto
            # Buscar patrones de texto libre
            if "que contengan" in question or "que incluyan" in question:
                # Extraer el texto después de estas frases
                match = re.search(r"(?:que contengan|que incluyan)\s+([a-zA-ZáéíóúñÁÉÍÓÚÑ]+)", question)
                if match:
                    texto = match.group(1)
                    filtros[col] = {
                        'tipo': 'contiene',
                        'texto': texto
                    }
    
    return filtros

def detectar_agrupaciones(question: str, df: pd.DataFrame, variable_metadata: Optional[Dict] = None) -> List[str]:
    """Detecta variables de agrupación usando el sistema complejo."""
    # Usar el sistema de agrupación compleja
    grouping_params = detect_complex_grouping_in_question(question, df, variable_metadata)
    return grouping_params.get('variables', [])

def aplicar_filtros_avanzados(df: pd.DataFrame, filtros: Dict[str, Any]) -> pd.DataFrame:
    """Aplica filtros complejos al DataFrame."""
    df_filtrado = df.copy()
    
    # Aplicar filtros de categoría
    for columna, filtro in filtros['categoria'].items():
        if columna not in df_filtrado.columns:
            continue
        
        if filtro['tipo'] == 'categoria_multiple':
            df_filtrado = df_filtrado[df_filtrado[columna].isin(filtro['valores'])]
    
    # Aplicar filtros de fecha
    for columna, filtro in filtros['fecha'].items():
        if columna not in df_filtrado.columns:
            continue
        
        df_filtrado[columna] = pd.to_datetime(df_filtrado[columna], errors='coerce')
        
        if filtro['tipo'] == 'rango_fecha':
            df_filtrado = df_filtrado[
                (df_filtrado[columna].dt.date >= filtro['inicio']) & 
                (df_filtrado[columna].dt.date <= filtro['fin'])
            ]
        elif filtro['tipo'] == 'fecha_especifica':
            df_filtrado = df_filtrado[df_filtrado[columna].dt.date == filtro['fecha']]
    
    # Aplicar filtros numéricos
    for columna, filtro in filtros['numerico'].items():
        if columna not in df_filtrado.columns:
            continue
        
        if filtro['tipo'] == 'mayor_que':
            df_filtrado = df_filtrado[df_filtrado[columna] > filtro['valor']]
        elif filtro['tipo'] == 'menor_que':
            df_filtrado = df_filtrado[df_filtrado[columna] < filtro['valor']]
        elif filtro['tipo'] == 'rango':
            df_filtrado = df_filtrado[
                (df_filtrado[columna] >= filtro['min']) & 
                (df_filtrado[columna] <= filtro['max'])
            ]
        elif filtro['tipo'] == 'mayor_que_promedio':
            promedio = df_filtrado[columna].mean()
            df_filtrado = df_filtrado[df_filtrado[columna] > promedio]
        elif filtro['tipo'] == 'menor_que_promedio':
            promedio = df_filtrado[columna].mean()
            df_filtrado = df_filtrado[df_filtrado[columna] < promedio]
        elif filtro['tipo'] in ['top_percentil', 'bottom_percentil']:
            percentil = filtro['porcentaje']
            if filtro['tipo'] == 'top_percentil':
                threshold = df_filtrado[columna].quantile(1 - percentil/100)
                df_filtrado = df_filtrado[df_filtrado[columna] >= threshold]
            else:
                threshold = df_filtrado[columna].quantile(percentil/100)
                df_filtrado = df_filtrado[df_filtrado[columna] <= threshold]
    
    # Aplicar filtros de texto
    for columna, filtro in filtros['texto'].items():
        if columna not in df_filtrado.columns:
            continue
        
        if filtro['tipo'] == 'contiene':
            df_filtrado = df_filtrado[df_filtrado[columna].str.contains(filtro['texto'], case=False, na=False)]
    
    return df_filtrado

def ejecutar_accion_avanzada(df: pd.DataFrame, accion: str, variables_objetivo: List[str], agrupaciones: List[str]) -> Any:
    """Ejecuta acciones avanzadas incluyendo agrupaciones complejas."""
    # Si hay agrupaciones complejas, usar el sistema de agrupación compleja
    if agrupaciones and len(agrupaciones) > 1:
        # Crear pregunta sintética para el sistema de agrupación compleja
        question = f"calcular {accion} de {', '.join(variables_objetivo) if variables_objetivo else 'todos los datos'} agrupado por {', '.join(agrupaciones)}"
        
        try:
            result, insights = execute_complex_grouping_from_question(question, df, variables_objetivo)
            if not result.empty:
                return result
        except Exception as e:
            # Si falla, continuar con el método tradicional
            pass
    
    # Método tradicional para agrupaciones simples
    if accion == 'contar':
        if agrupaciones:
            # Contar por grupos
            return df.groupby(agrupaciones).size().to_dict()
        return len(df)
    
    elif accion in ['promedio', 'suma', 'maximo', 'minimo', 'mediana', 'moda', 'desviacion', 'varianza']:
        if not variables_objetivo:
            return None
        
        variable = variables_objetivo[0]  # Usar la primera variable objetivo
        if variable not in df.columns:
            return None
        
        # Asegurar que la variable sea numérica
        if df[variable].dtype not in ['int64', 'float64']:
            return None
        
        valores = pd.to_numeric(df[variable], errors='coerce').dropna()
        
        if agrupaciones:
            # Calcular por grupos
            if accion == 'promedio':
                return df.groupby(agrupaciones)[variable].mean().to_dict()
            elif accion == 'suma':
                return df.groupby(agrupaciones)[variable].sum().to_dict()
            elif accion == 'maximo':
                return df.groupby(agrupaciones)[variable].max().to_dict()
            elif accion == 'minimo':
                return df.groupby(agrupaciones)[variable].min().to_dict()
            elif accion == 'mediana':
                return df.groupby(agrupaciones)[variable].median().to_dict()
            elif accion == 'moda':
                return df.groupby(agrupaciones)[variable].mode().to_dict()
            elif accion == 'desviacion':
                return df.groupby(agrupaciones)[variable].std().to_dict()
            elif accion == 'varianza':
                return df.groupby(agrupaciones)[variable].var().to_dict()
        else:
            # Calcular sobre toda la columna
            if accion == 'promedio':
                return valores.mean()
            elif accion == 'suma':
                return valores.sum()
            elif accion == 'maximo':
                return valores.max()
            elif accion == 'minimo':
                return valores.min()
            elif accion == 'mediana':
                return valores.median()
            elif accion == 'moda':
                return valores.mode().iloc[0] if not valores.mode().empty else None
            elif accion == 'desviacion':
                return valores.std()
            elif accion == 'varianza':
                return valores.var()
    
    elif accion == 'percentil':
        if not variables_objetivo:
            return None
        
        variable = variables_objetivo[0]
        if variable not in df.columns:
            return None
        
        # Asegurar que la variable sea numérica
        if df[variable].dtype not in ['int64', 'float64']:
            return None
        
        valores = pd.to_numeric(df[variable], errors='coerce').dropna()
        
        # Detectar percentil específico
        percentil_match = re.search(r"percentil\s+(\d+)", question.lower())
        if percentil_match:
            p = int(percentil_match.group(1))
            return valores.quantile(p/100)
        else:
            # Por defecto, percentil 50 (mediana)
            return valores.quantile(0.5)
    
    elif accion == 'distribucion':
        if not variables_objetivo:
            return None
        
        variable = variables_objetivo[0]
        if variable not in df.columns:
            return None
        
        if agrupaciones:
            return df.groupby(agrupaciones)[variable].value_counts().to_dict()
        else:
            return df[variable].value_counts().to_dict()
    
    elif accion == 'porcentaje':
        if not variables_objetivo:
            return None
        
        variable = variables_objetivo[0]
        if variable not in df.columns:
            return None
        
        if agrupaciones:
            return (df.groupby(agrupaciones)[variable].count() / len(df) * 100).to_dict()
        else:
            return len(df) / len(df) * 100  # 100% si no hay filtros
    
    return None

def generar_grafico_avanzado(df: pd.DataFrame, accion: str, variables_objetivo: List[str], filtros: Dict[str, Any], agrupaciones: List[str]) -> Tuple[pd.DataFrame, str]:
    """Genera datos para gráficos avanzados."""
    if accion == 'contar':
        if agrupaciones:
            # Gráfico de barras por grupos
            df_grafico = df.groupby(agrupaciones).size().reset_index(name='conteo')
            return df_grafico, 'barras_grupo'
        else:
            # Gráfico temporal si hay fechas
            columnas_fecha = [col for col in df.columns if any(palabra in col.lower() for palabra in ['fecha', 'date', 'time'])]
            if columnas_fecha:
                col_fecha = columnas_fecha[0]
                df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
                df_grafico = df.groupby(df[col_fecha].dt.date).size().reset_index(name='conteo')
                return df_grafico, 'serie_tiempo'
            else:
                # Gráfico de barras por categorías
                columnas_categoria = df.select_dtypes(include=['object', 'category']).columns
                if len(columnas_categoria) > 0:
                    col_cat = columnas_categoria[0]
                    df_grafico = df[col_cat].value_counts().reset_index()
                    df_grafico.columns = ['categoria', 'conteo']
                    return df_grafico, 'barras'
    
    elif accion in ['promedio', 'suma', 'maximo', 'minimo', 'mediana'] and variables_objetivo:
        variable = variables_objetivo[0]
        if agrupaciones:
            # Gráfico de barras por grupos
            if accion == 'promedio':
                df_grafico = df.groupby(agrupaciones)[variable].mean().reset_index()
            elif accion == 'suma':
                df_grafico = df.groupby(agrupaciones)[variable].sum().reset_index()
            elif accion == 'maximo':
                df_grafico = df.groupby(agrupaciones)[variable].max().reset_index()
            elif accion == 'minimo':
                df_grafico = df.groupby(agrupaciones)[variable].min().reset_index()
            elif accion == 'mediana':
                df_grafico = df.groupby(agrupaciones)[variable].median().reset_index()
            
            df_grafico.columns = list(df_grafico.columns[:-1]) + [accion]
            return df_grafico, 'barras_grupo'
        else:
            # Histograma para distribución
            df_grafico = df[variable].describe().reset_index()
            df_grafico.columns = ['estadistica', 'valor']
            return df_grafico, 'tabla'
    
    elif accion == 'distribucion' and variables_objetivo:
        variable = variables_objetivo[0]
        if agrupaciones:
            df_grafico = df.groupby(agrupaciones)[variable].value_counts().reset_index(name='conteo')
            return df_grafico, 'barras_apiladas'
        else:
            df_grafico = df[variable].value_counts().reset_index()
            df_grafico.columns = ['valor', 'frecuencia']
            return df_grafico, 'barras'
    
    # Por defecto, tabla simple
    return df.head(10), 'tabla'

def interpretar_resultado(accion: str, resultado: Any, variable_objetivo: Optional[str] = None) -> str:
    """Genera una interpretación en lenguaje natural del resultado."""
    if resultado is None:
        return "No se pudo calcular el resultado solicitado."
    
    if accion == 'contar':
        if isinstance(resultado, dict):
            return f"Se encontraron registros distribuidos por grupos: {len(resultado)} grupos diferentes."
        return f"Se encontraron **{resultado:,}** registros que cumplen con los criterios especificados."
    
    elif accion == 'promedio':
        if isinstance(resultado, dict):
            return f"Los promedios por grupo varían entre {min(resultado.values()):.2f} y {max(resultado.values()):.2f}."
        if variable_objetivo:
            return f"El **promedio** de {variable_objetivo} es **{resultado:.2f}**."
        else:
            return f"El **promedio** es **{resultado:.2f}**."
    
    elif accion == 'suma':
        if isinstance(resultado, dict):
            return f"Las sumas por grupo varían entre {min(resultado.values()):,.2f} y {max(resultado.values()):,.2f}."
        if variable_objetivo:
            return f"La **suma total** de {variable_objetivo} es **{resultado:,.2f}**."
        else:
            return f"La **suma total** es **{resultado:,.2f}**."
    
    elif accion == 'maximo':
        if isinstance(resultado, dict):
            return f"Los valores máximos por grupo varían entre {min(resultado.values()):.2f} y {max(resultado.values()):.2f}."
        if variable_objetivo:
            return f"El **valor máximo** de {variable_objetivo} es **{resultado:.2f}**."
        else:
            return f"El **valor máximo** es **{resultado:.2f}**."
    
    elif accion == 'minimo':
        if isinstance(resultado, dict):
            return f"Los valores mínimos por grupo varían entre {min(resultado.values()):.2f} y {max(resultado.values()):.2f}."
        if variable_objetivo:
            return f"El **valor mínimo** de {variable_objetivo} es **{resultado:.2f}**."
        else:
            return f"El **valor mínimo** es **{resultado:.2f}**."
    
    elif accion == 'mediana':
        if variable_objetivo:
            return f"La **mediana** de {variable_objetivo} es **{resultado:.2f}**."
        else:
            return f"La **mediana** es **{resultado:.2f}**."
    
    elif accion == 'moda':
        if variable_objetivo:
            return f"La **moda** de {variable_objetivo} es **{resultado}**."
        else:
            return f"La **moda** es **{resultado}**."
    
    elif accion == 'desviacion':
        if variable_objetivo:
            return f"La **desviación estándar** de {variable_objetivo} es **{resultado:.2f}**."
        else:
            return f"La **desviación estándar** es **{resultado:.2f}**."
    
    elif accion == 'varianza':
        if variable_objetivo:
            return f"La **varianza** de {variable_objetivo} es **{resultado:.2f}**."
        else:
            return f"La **varianza** es **{resultado:.2f}**."
    
    elif accion == 'percentil':
        if variable_objetivo:
            return f"El **percentil** de {variable_objetivo} es **{resultado:.2f}**."
        else:
            return f"El **percentil** es **{resultado:.2f}**."
    
    elif accion == 'distribucion':
        return f"Se encontraron **{len(resultado)}** categorías diferentes en la distribución."
    
    elif accion == 'porcentaje':
        if isinstance(resultado, dict):
            return f"Los porcentajes por grupo varían entre {min(resultado.values()):.1f}% y {max(resultado.values()):.1f}%."
        return f"El **porcentaje** es **{resultado:.1f}%**."
    
    return f"Resultado: {resultado}" 