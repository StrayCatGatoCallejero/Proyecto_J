"""
Módulo de Validación y Enriquecimiento de Datos de Chile
========================================================

Este módulo proporciona funciones para validar y enriquecer datos de encuestas
con información geográfica y demográfica oficial de Chile.
"""

import pandas as pd
import numpy as np
import streamlit as st
import os
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CARGA DE DATOS DE REFERENCIA
# ============================================================================

@st.cache_data
def cargar_referencias() -> Dict[str, pd.DataFrame]:
    """
    Carga todos los archivos de referencia de Chile.
    
    Returns:
        Diccionario con DataFrames de regiones, provincias, comunas e indicadores
    """
    base_path = Path("data/chile")
    
    try:
        # Cargar archivos de referencia forzando dtype string en códigos
        regiones = pd.read_csv(base_path / "regiones.csv", dtype={"region_code": str})
        provincias = pd.read_csv(base_path / "provincias.csv", dtype={"provincia_code": str, "region_code": str})
        comunas = pd.read_csv(base_path / "comunas.csv", dtype={"comuna_code": str, "provincia_code": str})
        indicadores = pd.read_csv(base_path / "indicadores_regiones.csv", dtype={"region_code": str})
        
        # Limpiar espacios en blanco en los códigos
        regiones['region_code'] = regiones['region_code'].str.strip()
        provincias['provincia_code'] = provincias['provincia_code'].str.strip()
        provincias['region_code'] = provincias['region_code'].str.strip()
        comunas['comuna_code'] = comunas['comuna_code'].str.strip()
        comunas['provincia_code'] = comunas['provincia_code'].str.strip()
        indicadores['region_code'] = indicadores['region_code'].str.strip()
        
        # Validar estructura de datos
        validar_estructura_referencias(regiones, provincias, comunas, indicadores)
        
        return {
            'regiones': regiones,
            'provincias': provincias,
            'comunas': comunas,
            'indicadores': indicadores
        }
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error al cargar datos de referencia: {e}")
    except Exception as e:
        raise Exception(f"Error inesperado al cargar referencias: {e}")

def validar_estructura_referencias(regiones: pd.DataFrame, 
                                 provincias: pd.DataFrame,
                                 comunas: pd.DataFrame, 
                                 indicadores: pd.DataFrame) -> None:
    """
    Valida que los archivos de referencia tengan la estructura correcta.
    
    Args:
        regiones: DataFrame de regiones
        provincias: DataFrame de provincias
        comunas: DataFrame de comunas
        indicadores: DataFrame de indicadores
    """
    # Validar columnas requeridas
    columnas_regiones = ['region_code', 'region_nombre', 'poblacion_2023', 'superficie_km2']
    columnas_provincias = ['provincia_code', 'provincia_nombre', 'region_code']
    columnas_comunas = ['comuna_code', 'comuna_nombre', 'provincia_code']
    columnas_indicadores = ['region_code', 'idh_2021', 'pib_per_capita_2021', 'porcentaje_urbano_2022']
    
    # Verificar columnas
    for col in columnas_regiones:
        if col not in regiones.columns:
            raise ValueError(f"Columna '{col}' faltante en regiones.csv")
    
    for col in columnas_provincias:
        if col not in provincias.columns:
            raise ValueError(f"Columna '{col}' faltante en provincias.csv")
    
    for col in columnas_comunas:
        if col not in comunas.columns:
            raise ValueError(f"Columna '{col}' faltante en comunas.csv")
    
    for col in columnas_indicadores:
        if col not in indicadores.columns:
            raise ValueError(f"Columna '{col}' faltante en indicadores_regiones.csv")
    
    # Validar integridad referencial
    regiones_validas = set(regiones['region_code'].astype(str))
    provincias_regiones = set(provincias['region_code'].astype(str))
    
    if not provincias_regiones.issubset(regiones_validas):
        provincias_invalidas = provincias_regiones - regiones_validas
        raise ValueError(f"Provincias con regiones inválidas: {provincias_invalidas}")
    
    provincias_validas = set(provincias['provincia_code'].astype(str))
    comunas_provincias = set(comunas['provincia_code'].astype(str))
    
    if not comunas_provincias.issubset(provincias_validas):
        comunas_invalidas = comunas_provincias - provincias_validas
        raise ValueError(f"Comunas con provincias inválidas: {comunas_invalidas}")

# ============================================================================
# 2. VALIDACIÓN DE DATOS
# ============================================================================

def validar_region(df: pd.DataFrame, columna_region: str) -> Tuple[bool, List[str]]:
    """
    Valida que las regiones en el DataFrame existan en los datos oficiales.
    
    Args:
        df: DataFrame a validar
        columna_region: Nombre de la columna con códigos de región
    
    Returns:
        Tupla con (es_válido, lista_de_errores)
    """
    if columna_region not in df.columns:
        return False, [f"Columna '{columna_region}' no encontrada en el DataFrame"]
    
    referencias = cargar_referencias()
    regiones_validas = set(referencias['regiones']['region_code'].astype(str))
    
    # Obtener regiones únicas del DataFrame
    regiones_df = set(df[columna_region].dropna().astype(str))
    
    # Encontrar regiones inválidas
    regiones_invalidas = regiones_df - regiones_validas
    
    if regiones_invalidas:
        return False, [f"Regiones inválidas encontradas: {list(regiones_invalidas)}"]
    
    return True, []

def validar_comuna(df: pd.DataFrame, columna_comuna: str) -> Tuple[bool, List[str]]:
    """
    Valida que las comunas en el DataFrame existan en los datos oficiales.
    
    Args:
        df: DataFrame a validar
        columna_comuna: Nombre de la columna con códigos de comuna
    
    Returns:
        Tupla con (es_válido, lista_de_errores)
    """
    if columna_comuna not in df.columns:
        return False, [f"Columna '{columna_comuna}' no encontrada en el DataFrame"]
    
    referencias = cargar_referencias()
    comunas_validas = set(referencias['comunas']['comuna_code'].astype(str))
    
    # Obtener comunas únicas del DataFrame
    comunas_df = set(df[columna_comuna].dropna().astype(str))
    
    # Encontrar comunas inválidas
    comunas_invalidas = comunas_df - comunas_validas
    
    if comunas_invalidas:
        return False, [f"Comunas inválidas encontradas: {list(comunas_invalidas)}"]
    
    return True, []

def validar_geografia_chile(df: pd.DataFrame, 
                           columna_region: Optional[str] = None,
                           columna_comuna: Optional[str] = None) -> Dict[str, Union[bool, List[str]]]:
    """
    Valida la geografía de Chile en el DataFrame.
    
    Args:
        df: DataFrame a validar
        columna_region: Nombre de la columna con códigos de región
        columna_comuna: Nombre de la columna con códigos de comuna
    
    Returns:
        Diccionario con resultados de validación
    """
    resultados = {
        'valido': True,
        'errores': [],
        'advertencias': []
    }
    
    # Validar regiones si se especifica
    if columna_region:
        region_valida, errores_region = validar_region(df, columna_region)
        if not region_valida:
            resultados['valido'] = False
            resultados['errores'].extend(errores_region)
    
    # Validar comunas si se especifica
    if columna_comuna:
        comuna_valida, errores_comuna = validar_comuna(df, columna_comuna)
        if not comuna_valida:
            resultados['valido'] = False
            resultados['errores'].extend(errores_comuna)
    
    # Validar consistencia entre región y comuna si ambas están presentes
    if columna_region and columna_comuna:
        if columna_region in df.columns and columna_comuna in df.columns:
            errores_consistencia = validar_consistencia_region_comuna(df, columna_region, columna_comuna)
            if errores_consistencia:
                resultados['advertencias'].extend(errores_consistencia)
    
    return resultados

def validar_consistencia_region_comuna(df: pd.DataFrame, 
                                     columna_region: str, 
                                     columna_comuna: str) -> List[str]:
    """
    Valida que las comunas pertenezcan a las regiones correctas.
    
    Args:
        df: DataFrame a validar
        columna_region: Nombre de la columna con códigos de región
        columna_comuna: Nombre de la columna con códigos de comuna
    
    Returns:
        Lista de errores de consistencia
    """
    referencias = cargar_referencias()
    
    # Crear mapeo comuna -> región
    mapeo_comuna_region = referencias['comunas'].merge(
        referencias['provincias'], on='provincia_code'
    )[['comuna_code', 'region_code']].set_index('comuna_code')['region_code']
    
    errores = []
    
    # Verificar cada fila
    for idx, row in df.dropna(subset=[columna_region, columna_comuna]).iterrows():
        comuna = str(row[columna_comuna])
        region_df = str(row[columna_region])
        
        if comuna in mapeo_comuna_region.index:
            region_esperada = str(mapeo_comuna_region[comuna])
            if region_df != region_esperada:
                errores.append(f"Fila {idx}: Comuna {comuna} debería estar en región {region_esperada}, no en {region_df}")
    
    return errores

# ============================================================================
# 3. ENRIQUECIMIENTO DE DATOS
# ============================================================================

def enriquecer_con_geografia(df: pd.DataFrame, 
                            columna_region: Optional[str] = None,
                            columna_comuna: Optional[str] = None) -> pd.DataFrame:
    """
    Enriquece el DataFrame con información geográfica de Chile.
    
    Args:
        df: DataFrame a enriquecer
        columna_region: Nombre de la columna con códigos de región
        columna_comuna: Nombre de la columna con códigos de comuna
    
    Returns:
        DataFrame enriquecido
    """
    df_enriquecido = df.copy()
    referencias = cargar_referencias()
    
    # Enriquecer con información de región
    if columna_region and columna_region in df.columns:
        # Convertir códigos a string para el merge
        df_enriquecido[columna_region] = df_enriquecido[columna_region].astype(str)
        
        df_enriquecido = df_enriquecido.merge(
            referencias['regiones'][['region_code', 'region_nombre', 'poblacion_2023', 'superficie_km2']],
            left_on=columna_region,
            right_on='region_code',
            how='left',
            suffixes=('', '_region')
        )
        
        # Renombrar columnas para claridad
        df_enriquecido = df_enriquecido.rename(columns={
            'region_nombre': 'nombre_region',
            'poblacion_2023': 'poblacion_region_2023',
            'superficie_km2': 'superficie_region_km2'
        })
    
    # Enriquecer con información de comuna
    if columna_comuna and columna_comuna in df.columns:
        # Convertir códigos a string para el merge
        df_enriquecido[columna_comuna] = df_enriquecido[columna_comuna].astype(str)
        
        # Obtener información de comuna y provincia
        comunas_info = referencias['comunas'].merge(
            referencias['provincias'], on='provincia_code'
        ).merge(
            referencias['regiones'], on='region_code'
        )[['comuna_code', 'comuna_nombre', 'provincia_nombre', 'region_nombre']]
        
        df_enriquecido = df_enriquecido.merge(
            comunas_info,
            left_on=columna_comuna,
            right_on='comuna_code',
            how='left',
            suffixes=('', '_comuna')
        )
        
        # Renombrar columnas para claridad
        df_enriquecido = df_enriquecido.rename(columns={
            'comuna_nombre': 'nombre_comuna',
            'provincia_nombre': 'nombre_provincia'
        })
    
    return df_enriquecido

def enriquecer_con_indicadores(df: pd.DataFrame, 
                              columna_region: str) -> pd.DataFrame:
    """
    Enriquece el DataFrame con indicadores socioeconómicos por región.
    
    Args:
        df: DataFrame a enriquecer
        columna_region: Nombre de la columna con códigos de región
    
    Returns:
        DataFrame enriquecido con indicadores
    """
    if columna_region not in df.columns:
        raise ValueError(f"Columna '{columna_region}' no encontrada en el DataFrame")
    
    referencias = cargar_referencias()
    
    # Convertir códigos a string para el merge
    df_enriquecido = df.copy()
    df_enriquecido[columna_region] = df_enriquecido[columna_region].astype(str)
    
    df_enriquecido = df_enriquecido.merge(
        referencias['indicadores'],
        left_on=columna_region,
        right_on='region_code',
        how='left',
        suffixes=('', '_indicadores')
    )
    
    return df_enriquecido

def enriquecer_datos_chile(df: pd.DataFrame,
                          columna_region: Optional[str] = None,
                          columna_comuna: Optional[str] = None,
                          incluir_indicadores: bool = True) -> pd.DataFrame:
    """
    Enriquece completamente el DataFrame con datos oficiales de Chile.
    
    Args:
        df: DataFrame a enriquecer
        columna_region: Nombre de la columna con códigos de región
        columna_comuna: Nombre de la columna con códigos de comuna
        incluir_indicadores: Si incluir indicadores socioeconómicos
    
    Returns:
        DataFrame completamente enriquecido
    """
    with st.spinner("Enriqueciendo datos con información oficial de Chile..."):
        # Paso 1: Enriquecer con geografía
        df_enriquecido = enriquecer_con_geografia(df, columna_region, columna_comuna)
        
        # Paso 2: Enriquecer con indicadores si se solicita
        if incluir_indicadores and columna_region:
            df_enriquecido = enriquecer_con_indicadores(df_enriquecido, columna_region)
        
        # Paso 3: Calcular métricas adicionales
        df_enriquecido = calcular_metricas_adicionales(df_enriquecido)
        
        return df_enriquecido

def calcular_metricas_adicionales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas adicionales basadas en los datos enriquecidos.
    
    Args:
        df: DataFrame enriquecido
    
    Returns:
        DataFrame con métricas adicionales
    """
    df_metricas = df.copy()
    
    # Densidad poblacional por región
    if 'poblacion_region_2023' in df_metricas.columns and 'superficie_region_km2' in df_metricas.columns:
        df_metricas['densidad_poblacional_region'] = (
            df_metricas['poblacion_region_2023'] / df_metricas['superficie_region_km2']
        )
    
    # Categorización de regiones por desarrollo
    if 'idh_2021' in df_metricas.columns:
        df_metricas['categoria_desarrollo'] = pd.cut(
            df_metricas['idh_2021'],
            bins=[0, 0.8, 0.85, 0.9, 1.0],
            labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'],
            include_lowest=True
        )
    
    # Categorización por PIB per cápita
    if 'pib_per_capita_2021' in df_metricas.columns:
        df_metricas['categoria_pib'] = pd.cut(
            df_metricas['pib_per_capita_2021'],
            bins=[0, 15000000, 20000000, 25000000, float('inf')],
            labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'],
            include_lowest=True
        )
    
    return df_metricas

# ============================================================================
# 4. FUNCIONES DE ANÁLISIS ESPECÍFICAS PARA CHILE
# ============================================================================

def analisis_por_region(df: pd.DataFrame, 
                       variable_analizar: str,
                       columna_region: str = 'region_code') -> pd.DataFrame:
    """
    Realiza análisis estadístico por región.
    
    Args:
        df: DataFrame con datos
        variable_analizar: Variable a analizar
        columna_region: Columna con códigos de región
    
    Returns:
        DataFrame con estadísticas por región
    """
    if variable_analizar not in df.columns:
        raise ValueError(f"Variable '{variable_analizar}' no encontrada")
    
    if columna_region not in df.columns:
        raise ValueError(f"Columna región '{columna_region}' no encontrada")
    
    # Agrupar por región y calcular estadísticas
    stats_por_region = df.groupby(columna_region)[variable_analizar].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).reset_index()
    
    # Agregar información de región si está disponible
    try:
        referencias = cargar_referencias()
        # Convertir códigos a string para el merge
        stats_por_region[columna_region] = stats_por_region[columna_region].astype(str)
        
        stats_por_region = stats_por_region.merge(
            referencias['regiones'][['region_code', 'region_nombre']],
            left_on=columna_region,
            right_on='region_code',
            how='left'
        )
    except:
        pass
    
    return stats_por_region

def comparar_con_promedio_nacional(df: pd.DataFrame,
                                  variable_analizar: str,
                                  columna_region: str = 'region_code') -> pd.DataFrame:
    """
    Compara los valores por región con el promedio nacional.
    
    Args:
        df: DataFrame con datos
        variable_analizar: Variable a analizar
        columna_region: Columna con códigos de región
    
    Returns:
        DataFrame con comparaciones
    """
    # Calcular promedio nacional
    promedio_nacional = df[variable_analizar].mean()
    
    # Análisis por región
    analisis_regional = analisis_por_region(df, variable_analizar, columna_region)
    
    # Calcular diferencias con el promedio nacional
    analisis_regional['diferencia_promedio_nacional'] = (
        analisis_regional['mean'] - promedio_nacional
    )
    
    analisis_regional['porcentaje_diferencia'] = (
        (analisis_regional['diferencia_promedio_nacional'] / promedio_nacional) * 100
    )
    
    # Categorizar diferencias
    analisis_regional['categoria_diferencia'] = pd.cut(
        analisis_regional['porcentaje_diferencia'],
        bins=[float('-inf'), -10, 10, float('inf')],
        labels=['Muy por debajo', 'Cerca del promedio', 'Muy por encima'],
        include_lowest=True
    )
    
    return analisis_regional

# ============================================================================
# 5. FUNCIONES DE UTILIDAD
# ============================================================================

def obtener_lista_regiones() -> pd.DataFrame:
    """
    Obtiene la lista completa de regiones de Chile.
    
    Returns:
        DataFrame con información de regiones
    """
    referencias = cargar_referencias()
    return referencias['regiones'][['region_code', 'region_nombre']]

def obtener_lista_comunas(region_code: Optional[str] = None) -> pd.DataFrame:
    """
    Obtiene la lista de comunas, opcionalmente filtrada por región.
    
    Args:
        region_code: Código de región para filtrar (opcional)
    
    Returns:
        DataFrame con información de comunas
    """
    referencias = cargar_referencias()
    
    comunas_info = referencias['comunas'].merge(
        referencias['provincias'], on='provincia_code'
    ).merge(
        referencias['regiones'], on='region_code'
    )[['comuna_code', 'comuna_nombre', 'provincia_nombre', 'region_nombre', 'region_code']]
    
    if region_code:
        # Convertir a string para comparación
        region_code_str = str(region_code)
        comunas_info = comunas_info[comunas_info['region_code'] == region_code_str]
    
    return comunas_info

def buscar_geografia_chile(termino: str, tipo: str = 'todos') -> pd.DataFrame:
    """
    Busca términos geográficos en los datos de Chile.
    
    Args:
        termino: Término a buscar
        tipo: Tipo de búsqueda ('region', 'comuna', 'todos')
    
    Returns:
        DataFrame con resultados de búsqueda
    """
    referencias = cargar_referencias()
    termino_lower = termino.lower()
    
    resultados = []
    
    if tipo in ['region', 'todos']:
        regiones_match = referencias['regiones'][
            referencias['regiones']['region_nombre'].str.lower().str.contains(termino_lower)
        ]
        for _, row in regiones_match.iterrows():
            resultados.append({
                'tipo': 'Región',
                'codigo': row['region_code'],
                'nombre': row['region_nombre'],
                'descripcion': f"Región de {row['region_nombre']}"
            })
    
    if tipo in ['comuna', 'todos']:
        comunas_match = referencias['comunas'][
            referencias['comunas']['comuna_nombre'].str.lower().str.contains(termino_lower)
        ]
        for _, row in comunas_match.iterrows():
            resultados.append({
                'tipo': 'Comuna',
                'codigo': row['comuna_code'],
                'nombre': row['comuna_nombre'],
                'descripcion': f"Comuna de {row['comuna_nombre']}"
            })
    
    return pd.DataFrame(resultados)

if __name__ == "__main__":
    # Pruebas básicas
    print("Cargando referencias...")
    refs = cargar_referencias()
    print(f"Regiones cargadas: {len(refs['regiones'])}")
    print(f"Provincias cargadas: {len(refs['provincias'])}")
    print(f"Comunas cargadas: {len(refs['comunas'])}")
    print(f"Indicadores cargados: {len(refs['indicadores'])}") 