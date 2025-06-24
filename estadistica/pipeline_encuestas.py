"""
Pipeline Modular de Limpieza y Preprocesamiento de Datos de Encuesta
====================================================================

Este módulo proporciona un conjunto completo de herramientas para el procesamiento
de datos de encuestas, incluyendo carga, exploración, limpieza, imputación,
análisis ponderado y exportación.
"""

import pandas as pd
import numpy as np
import streamlit as st
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Union, List, Dict, Any, Optional, Tuple
import os
import sys
import subprocess
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Imports para análisis estadístico
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Importación opcional de statsmodels
try:
    import statsmodels.api as sm
    from statsmodels.stats.weightstats import DescrStatsW
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels no disponible. Algunas funciones de análisis ponderado pueden no funcionar.")

# Importación del módulo de validación de Chile
try:
    from .validacion_chile import (
        cargar_referencias, validar_geografia_chile, enriquecer_datos_chile,
        analisis_por_region, comparar_con_promedio_nacional,
        obtener_lista_regiones, obtener_lista_comunas, buscar_geografia_chile
    )
    VALIDACION_CHILE_AVAILABLE = True
except ImportError:
    try:
        from validacion_chile import (
            cargar_referencias, validar_geografia_chile, enriquecer_datos_chile,
            analisis_por_region, comparar_con_promedio_nacional,
            obtener_lista_regiones, obtener_lista_comunas, buscar_geografia_chile
        )
        VALIDACION_CHILE_AVAILABLE = True
    except ImportError:
        VALIDACION_CHILE_AVAILABLE = False
        print("Warning: Módulo de validación de Chile no disponible.")

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================

@st.cache_data
def cargar_datos(ruta: str, validar_columnas: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Carga datos desde múltiples formatos con validación de columnas.
    
    Args:
        ruta: Ruta al archivo de datos
        validar_columnas: Lista de columnas que deben estar presentes
    
    Returns:
        DataFrame con los datos cargados
    
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si faltan columnas requeridas
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"El archivo no se encuentra en la ruta: {ruta}")

    filename = os.path.basename(ruta).lower()
    
    try:
        with st.spinner(f"Cargando datos desde {filename}..."):
            if filename.endswith(".sav"):
                import pyreadstat
                df, meta = pyreadstat.read_sav(ruta)
                st.success(f"Archivo SPSS cargado: {len(df)} filas, {len(df.columns)} columnas")
                
            elif filename.endswith(".dta"):
                import pyreadstat
                df, meta = pyreadstat.read_dta(ruta)
                st.success(f"Archivo Stata cargado: {len(df)} filas, {len(df.columns)} columnas")
                
            elif filename.endswith(".csv"):
                df = pd.read_csv(ruta)
                st.success(f"Archivo CSV cargado: {len(df)} filas, {len(df.columns)} columnas")
                
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(ruta)
                st.success(f"Archivo Excel cargado: {len(df)} filas, {len(df.columns)} columnas")
                
            else:
                raise ValueError("Formato de archivo no soportado. Use .sav, .dta, .csv, .xls, o .xlsx.")
        
        # Validar columnas requeridas
        if validar_columnas:
            columnas_faltantes = [col for col in validar_columnas if col not in df.columns]
            if columnas_faltantes:
                raise ValueError(f"Columnas requeridas faltantes: {columnas_faltantes}")
        
        return df
        
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        raise

# ============================================================================
# 1.5. PROCESAMIENTO DE ARCHIVOS GRANDES
# ============================================================================

def cargar_datos_grandes(ruta: str, chunk_size: int = 10000, 
                        validar_columnas: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Carga archivos grandes en chunks para optimizar memoria.
    
    Args:
        ruta: Ruta al archivo de datos
        chunk_size: Número de filas por chunk
        validar_columnas: Lista de columnas que deben estar presentes
    
    Returns:
        DataFrame con los datos cargados
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"El archivo no se encuentra en la ruta: {ruta}")

    filename = os.path.basename(ruta).lower()
    
    try:
        with st.spinner(f"Cargando archivo grande desde {filename} en chunks..."):
            
            # Determinar el método de carga según el formato
            if filename.endswith(".csv"):
                # Para CSV, usar pandas read_csv con chunksize
                chunks = []
                total_rows = 0
                
                for chunk in pd.read_csv(ruta, chunksize=chunk_size):
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    st.progress(min(total_rows / (total_rows + chunk_size), 0.95))
                
                df = pd.concat(chunks, ignore_index=True)
                
            elif filename.endswith((".xlsx", ".xls")):
                # Para Excel, cargar todo (no hay chunks nativos)
                df = pd.read_excel(ruta)
                
            elif filename.endswith(".sav"):
                # Para SPSS, cargar todo
                import pyreadstat
                df, meta = pyreadstat.read_sav(ruta)
                
            elif filename.endswith(".dta"):
                # Para Stata, cargar todo
                import pyreadstat
                df, meta = pyreadstat.read_dta(ruta)
                
            else:
                raise ValueError("Formato de archivo no soportado. Use .sav, .dta, .csv, .xls, o .xlsx.")
            
            st.success(f"Archivo cargado: {len(df):,} filas, {len(df.columns)} columnas")
            
            # Validar columnas requeridas
            if validar_columnas:
                columnas_faltantes = [col for col in validar_columnas if col not in df.columns]
                if columnas_faltantes:
                    raise ValueError(f"Columnas requeridas faltantes: {columnas_faltantes}")
            
            return df
            
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        raise

# ============================================================================
# 1.5. VALIDACIÓN Y ENRIQUECIMIENTO DE DATOS DE CHILE
# ============================================================================

def validar_datos_chile(df: pd.DataFrame, 
                       columna_region: Optional[str] = None,
                       columna_comuna: Optional[str] = None) -> Dict[str, Any]:
    """
    Valida los datos geográficos de Chile en el DataFrame.
    
    Args:
        df: DataFrame a validar
        columna_region: Nombre de la columna con códigos de región
        columna_comuna: Nombre de la columna con códigos de comuna
    
    Returns:
        Diccionario con resultados de validación
    """
    if not VALIDACION_CHILE_AVAILABLE:
        st.warning("Módulo de validación de Chile no disponible")
        return {'valido': False, 'errores': ['Módulo de validación no disponible']}
    
    try:
        with st.spinner("Validando datos geográficos de Chile..."):
            resultados = validar_geografia_chile(df, columna_region, columna_comuna)
            
            # Mostrar resultados
            if resultados['valido']:
                st.success("✅ Validación geográfica exitosa")
            else:
                st.error("❌ Errores encontrados en validación geográfica")
                for error in resultados['errores']:
                    st.error(f"• {error}")
            
            if resultados['advertencias']:
                st.warning("⚠️ Advertencias encontradas:")
                for advertencia in resultados['advertencias']:
                    st.warning(f"• {advertencia}")
            
            return resultados
            
    except Exception as e:
        st.error(f"Error en validación de Chile: {str(e)}")
        return {'valido': False, 'errores': [str(e)]}

def enriquecer_datos_chile_pipeline(df: pd.DataFrame,
                                   columna_region: Optional[str] = None,
                                   columna_comuna: Optional[str] = None,
                                   incluir_indicadores: bool = True) -> pd.DataFrame:
    """
    Enriquece el DataFrame con datos oficiales de Chile.
    
    Args:
        df: DataFrame a enriquecer
        columna_region: Nombre de la columna con códigos de región
        columna_comuna: Nombre de la columna con códigos de comuna
        incluir_indicadores: Si incluir indicadores socioeconómicos
    
    Returns:
        DataFrame enriquecido
    """
    if not VALIDACION_CHILE_AVAILABLE:
        st.warning("Módulo de validación de Chile no disponible")
        return df
    
    try:
        df_enriquecido = enriquecer_datos_chile(
            df, columna_region, columna_comuna, incluir_indicadores
        )
        
        # Mostrar información del enriquecimiento
        columnas_nuevas = set(df_enriquecido.columns) - set(df.columns)
        if columnas_nuevas:
            st.success(f"✅ Datos enriquecidos con {len(columnas_nuevas)} nuevas columnas")
            st.info(f"Columnas agregadas: {', '.join(sorted(columnas_nuevas))}")
        
        return df_enriquecido
        
    except Exception as e:
        st.error(f"Error al enriquecer datos: {str(e)}")
        return df

def analisis_regional_chile(df: pd.DataFrame,
                           variable_analizar: str,
                           columna_region: str = 'region_code') -> pd.DataFrame:
    """
    Realiza análisis estadístico por región de Chile.
    
    Args:
        df: DataFrame con datos
        variable_analizar: Variable a analizar
        columna_region: Columna con códigos de región
    
    Returns:
        DataFrame con estadísticas por región
    """
    if not VALIDACION_CHILE_AVAILABLE:
        st.warning("Módulo de validación de Chile no disponible")
        return pd.DataFrame()
    
    try:
        with st.spinner("Realizando análisis regional..."):
            analisis = analisis_por_region(df, variable_analizar, columna_region)
            
            # Mostrar resultados
            st.subheader("📊 Análisis por Región")
            st.dataframe(analisis)
            
            # Gráfico de barras
            if 'region_nombre' in analisis.columns and 'mean' in analisis.columns:
                fig = px.bar(analisis, x='region_nombre', y='mean',
                            title=f"Promedio de {variable_analizar} por Región",
                            labels={'mean': 'Promedio', 'region_nombre': 'Región'})
                st.plotly_chart(fig, use_container_width=True)
            
            return analisis
            
    except Exception as e:
        st.error(f"Error en análisis regional: {str(e)}")
        return pd.DataFrame()

def comparar_nacional_chile(df: pd.DataFrame,
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
    if not VALIDACION_CHILE_AVAILABLE:
        st.warning("Módulo de validación de Chile no disponible")
        return pd.DataFrame()
    
    try:
        with st.spinner("Comparando con promedio nacional..."):
            comparacion = comparar_con_promedio_nacional(df, variable_analizar, columna_region)
            
            # Mostrar resultados
            st.subheader("🏆 Comparación con Promedio Nacional")
            st.dataframe(comparacion)
            
            # Gráfico de diferencias
            if 'region_nombre' in comparacion.columns and 'porcentaje_diferencia' in comparacion.columns:
                fig = px.bar(comparacion, x='region_nombre', y='porcentaje_diferencia',
                            title=f"Diferencia con Promedio Nacional - {variable_analizar}",
                            labels={'porcentaje_diferencia': 'Diferencia (%)', 'region_nombre': 'Región'},
                            color='categoria_diferencia')
                st.plotly_chart(fig, use_container_width=True)
            
            return comparacion
            
    except Exception as e:
        st.error(f"Error en comparación nacional: {str(e)}")
        return pd.DataFrame()

def mostrar_info_geografia_chile():
    """
    Muestra información sobre la geografía de Chile.
    """
    if not VALIDACION_CHILE_AVAILABLE:
        st.warning("Módulo de validación de Chile no disponible")
        return
    
    try:
        st.subheader("🗺️ Información Geográfica de Chile")
        
        # Información de regiones
        regiones = obtener_lista_regiones()
        st.write("**Regiones de Chile:**")
        st.dataframe(regiones)
        
        # Estadísticas básicas
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Regiones", len(regiones))
        with col2:
            comunas = obtener_lista_comunas()
            st.metric("Total Comunas", len(comunas))
        
        # Búsqueda de geografía
        st.write("**Buscar ubicación geográfica:**")
        termino = st.text_input("Ingrese nombre de región o comuna:")
        if termino:
            resultados = buscar_geografia_chile(termino)
            if not resultados.empty:
                st.dataframe(resultados)
            else:
                st.info("No se encontraron resultados")
                
    except Exception as e:
        st.error(f"Error al mostrar información geográfica: {str(e)}")

# ============================================================================
# 2. EXPLORACIÓN DE DATOS
# ============================================================================

def explorar_datos(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza exploración completa de los datos.
    
    Args:
        df: DataFrame a explorar
    
    Returns:
        Diccionario con estadísticas de exploración
    """
    with st.spinner("Explorando datos..."):
        info = {
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).value_counts().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # Mostrar información básica
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas", info['shape'][0])
        with col2:
            st.metric("Columnas", info['shape'][1])
        with col3:
            st.metric("Filas duplicadas", info['duplicate_rows'])
        
        return info

def mostrar_resumen_datos(df: pd.DataFrame):
    """Muestra un resumen visual de los datos."""
    st.subheader("📊 Resumen de Datos")
    
    # Información básica
    info = explorar_datos(df)
    
    # Mostrar tipos de datos
    st.write("**Tipos de datos:**")
    tipo_counts = pd.Series(info['dtypes'])
    fig = px.bar(x=tipo_counts.index, y=tipo_counts.values, 
                 title="Distribución de Tipos de Datos")
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar valores faltantes
    st.write("**Valores faltantes:**")
    null_df = pd.DataFrame({
        'columna': list(info['null_percentage'].keys()),
        'porcentaje': list(info['null_percentage'].values())
    }).sort_values('porcentaje', ascending=False)
    
    fig = px.bar(null_df, x='columna', y='porcentaje', 
                 title="Porcentaje de Valores Faltantes por Columna")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 3. VISUALIZACIÓN DE VALORES FALTANTES
# ============================================================================

def visualizar_nulos(df: pd.DataFrame, tipo: str = "matrix") -> None:
    """
    Visualiza valores faltantes usando missingno.
    
    Args:
        df: DataFrame a analizar
        tipo: Tipo de visualización ('matrix', 'heatmap', 'dendrogram', 'bar')
    """
    with st.spinner("Generando visualización de valores faltantes..."):
        st.subheader("🔍 Análisis de Valores Faltantes")
        
        # Crear figura y ejes
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if tipo == "matrix":
            msno.matrix(df, ax=ax)
            
        elif tipo == "heatmap":
            msno.heatmap(df, ax=ax)
            
        elif tipo == "dendrogram":
            msno.dendrogram(df, ax=ax)
            
        elif tipo == "bar":
            msno.bar(df, ax=ax)
        
        # Ajustar layout y mostrar
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # Cerrar figura para liberar memoria
        
        # Información adicional
        null_info = df.isnull().sum()
        st.write("**Resumen de valores faltantes:**")
        st.dataframe(null_info[null_info > 0].sort_values(ascending=False))

# ============================================================================
# 4. IMPUTACIÓN DE VALORES FALTANTES
# ============================================================================

@st.cache_data
def imputar_nulos(df: pd.DataFrame, estrategia: str = "knn", 
                  columnas_numericas: Optional[List[str]] = None,
                  columnas_categoricas: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Imputa valores faltantes usando diferentes estrategias.
    
    Args:
        df: DataFrame a procesar
        estrategia: Estrategia de imputación ('media', 'mediana', 'moda', 'knn', 'random_forest')
        columnas_numericas: Columnas numéricas específicas para imputar
        columnas_categoricas: Columnas categóricas específicas para imputar
    
    Returns:
        DataFrame con valores imputados
    """
    df_limpio = df.copy()
    
    with st.spinner(f"Imputando valores faltantes usando estrategia: {estrategia}..."):
        
        # Determinar columnas a procesar
        if columnas_numericas is None:
            columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Si se especificaron columnas numéricas explícitamente, usar solo esas
            columnas_numericas = [col for col in columnas_numericas if col in df.columns]
            
        if columnas_categoricas is None:
            # Solo detectar automáticamente columnas categóricas si NO se especificaron columnas numéricas específicas
            # Esto permite control granular: si especificas columnas numéricas, las categóricas solo se procesan si se especifican explícitamente
            if columnas_numericas is None or len(columnas_numericas) == len(df.select_dtypes(include=[np.number]).columns):
                columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
            else:
                columnas_categoricas = []
        else:
            # Si se especificaron columnas categóricas explícitamente, usar solo esas
            columnas_categoricas = [col for col in columnas_categoricas if col in df.columns]
        
        # Progreso
        progress_bar = st.progress(0)
        total_columns = len(columnas_numericas) + len(columnas_categoricas)
        current_progress = 0
        
        # Imputación para columnas numéricas
        if columnas_numericas and estrategia in ['media', 'mediana', 'knn', 'random_forest']:
            # Mapear estrategias en español a scikit-learn
            estrategias_map = {'media': 'mean', 'mediana': 'median', 'moda': 'most_frequent'}
            estrategia_sklearn = estrategias_map.get(estrategia, estrategia)
            if estrategia in ['media', 'mediana']:
                imputer = SimpleImputer(strategy=estrategia_sklearn)
                df_limpio[columnas_numericas] = imputer.fit_transform(df_limpio[columnas_numericas])
            elif estrategia == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                df_limpio[columnas_numericas] = imputer.fit_transform(df_limpio[columnas_numericas])
            elif estrategia == 'random_forest':
                for col in columnas_numericas:
                    if df_limpio[col].isnull().sum() > 0:
                        # Preparar datos para RF
                        temp_df = df_limpio.dropna(subset=[col])
                        X = temp_df[columnas_numericas].drop(columns=[col])
                        y = temp_df[col]
                        # Entrenar modelo
                        rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf.fit(X, y)
                        # Predecir valores faltantes
                        mask = df_limpio[col].isnull()
                        if mask.sum() > 0:
                            X_pred = df_limpio.loc[mask, columnas_numericas].drop(columns=[col])
                            df_limpio.loc[mask, col] = rf.predict(X_pred)
            current_progress += len(columnas_numericas)
            progress_bar.progress(current_progress / total_columns)
        
        # Imputación para columnas categóricas
        if columnas_categoricas:
            estrategias_map = {'media': 'mean', 'mediana': 'median', 'moda': 'most_frequent'}
            estrategia_sklearn = estrategias_map.get(estrategia, estrategia)
            
            # Para estrategias que no son específicamente para categóricas, usar moda por defecto
            if estrategia in ['media', 'mediana', 'knn', 'random_forest']:
                # Usar moda para columnas categóricas cuando la estrategia no es específica
                imputer = SimpleImputer(strategy='most_frequent')
                df_limpio[columnas_categoricas] = imputer.fit_transform(df_limpio[columnas_categoricas])
            elif estrategia == 'moda':
                imputer = SimpleImputer(strategy=estrategia_sklearn)
                df_limpio[columnas_categoricas] = imputer.fit_transform(df_limpio[columnas_categoricas])
            
            current_progress += len(columnas_categoricas)
            progress_bar.progress(current_progress / total_columns)
        
        progress_bar.progress(1.0)
        st.success("Imputación completada exitosamente")
        
        return df_limpio

# ============================================================================
# 5. ANÁLISIS PONDERADO
# ============================================================================

def analisis_ponderado(df: pd.DataFrame, variable: str, peso: str) -> Dict[str, float]:
    """
    Realiza análisis estadístico ponderado.
    
    Args:
        df: DataFrame con los datos
        variable: Variable a analizar
        peso: Variable de peso
    
    Returns:
        Diccionario con estadísticas ponderadas
    """
    with st.spinner("Calculando estadísticas ponderadas..."):
        # Verificar que las columnas existen
        if variable not in df.columns:
            raise ValueError(f"Variable '{variable}' no encontrada en el DataFrame")
        if peso not in df.columns:
            raise ValueError(f"Variable de peso '{peso}' no encontrada en el DataFrame")
        
        # Filtrar datos válidos
        valid_data = df[[variable, peso]].dropna()
        
        if len(valid_data) == 0:
            raise ValueError("No hay datos válidos para el análisis ponderado")
        
        # Calcular estadísticas ponderadas
        if STATSMODELS_AVAILABLE:
            # Usar statsmodels si está disponible
            weighted_stats = DescrStatsW(valid_data[variable], weights=valid_data[peso])
            
            resultados = {
                'media_ponderada': weighted_stats.mean,
                'varianza_ponderada': weighted_stats.var,
                'desv_est_ponderada': np.sqrt(weighted_stats.var),
                'n_efectivo': weighted_stats.nobs,
                'suma_pesos': valid_data[peso].sum()
            }
        else:
            # Implementación manual de estadísticas ponderadas
            x = valid_data[variable].values
            w = valid_data[peso].values
            
            # Media ponderada
            media_ponderada = np.average(x, weights=w)
            
            # Varianza ponderada
            varianza_ponderada = np.average((x - media_ponderada)**2, weights=w)
            
            # N efectivo
            n_efectivo = (w.sum()**2) / (w**2).sum()
            
            resultados = {
                'media_ponderada': media_ponderada,
                'varianza_ponderada': varianza_ponderada,
                'desv_est_ponderada': np.sqrt(varianza_ponderada),
                'n_efectivo': n_efectivo,
                'suma_pesos': w.sum()
            }
        
        # Mostrar resultados
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Media Ponderada", f"{resultados['media_ponderada']:.2f}")
        with col2:
            st.metric("Desv. Est. Ponderada", f"{resultados['desv_est_ponderada']:.2f}")
        with col3:
            st.metric("N Efectivo", f"{resultados['n_efectivo']:.0f}")
        
        return resultados

# ============================================================================
# 6. MUESTREO ESTRATIFICADO
# ============================================================================

def muestreo_estratificado(df: pd.DataFrame, columna_estratificacion: str, 
                          test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Realiza muestreo estratificado.
    
    Args:
        df: DataFrame a dividir
        columna_estratificacion: Columna para estratificación
        test_size: Proporción del conjunto de prueba
        random_state: Semilla para reproducibilidad
    
    Returns:
        Tupla con (train_df, test_df)
    """
    with st.spinner("Realizando muestreo estratificado..."):
        if columna_estratificacion not in df.columns:
            raise ValueError(f"Columna de estratificación '{columna_estratificacion}' no encontrada")
        
        # Verificar que hay suficientes datos para estratificación
        if df[columna_estratificacion].nunique() > len(df) * test_size:
            st.warning("Demasiadas categorías para estratificación. Usando muestreo aleatorio simple.")
            train_idx, test_idx = train_test_split(
                df.index, test_size=test_size, random_state=random_state
            )
        else:
            train_idx, test_idx = train_test_split(
                df.index, test_size=test_size, 
                stratify=df[columna_estratificacion], 
                random_state=random_state
            )
        
        train_df = df.loc[train_idx].copy()
        test_df = df.loc[test_idx].copy()
        
        # Mostrar información del muestreo
        st.success(f"Muestreo completado: {len(train_df)} train, {len(test_df)} test")
        
        # Visualizar distribución de estratos
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        train_df[columna_estratificacion].value_counts().plot(kind='bar', ax=ax1, title='Train')
        test_df[columna_estratificacion].value_counts().plot(kind='bar', ax=ax2, title='Test')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        return train_df, test_df

# ============================================================================
# 7. EXPORTACIÓN Y REPORTES
# ============================================================================

def exportar_limpio(df: pd.DataFrame, ruta_salida: str, formato: str = "csv") -> str:
    """
    Exporta datos limpios en diferentes formatos.
    
    Args:
        df: DataFrame a exportar
        ruta_salida: Ruta de salida
        formato: Formato de exportación ('csv', 'excel', 'parquet')
    
    Returns:
        Ruta del archivo exportado
    """
    with st.spinner(f"Exportando datos en formato {formato}..."):
        try:
            if formato == "csv":
                df.to_csv(ruta_salida, index=False)
            elif formato == "excel":
                df.to_excel(ruta_salida, index=False)
            elif formato == "parquet":
                df.to_parquet(ruta_salida, index=False)
            else:
                raise ValueError(f"Formato no soportado: {formato}")
            
            st.success(f"Datos exportados exitosamente a: {ruta_salida}")
            return ruta_salida
            
        except Exception as e:
            st.error(f"Error al exportar datos: {str(e)}")
            raise

def generar_reporte_pdf(df: pd.DataFrame, output_path: str = "reporte_encuesta.pdf") -> str:
    """
    Genera un reporte PDF completo con análisis de datos.
    
    Args:
        df: DataFrame con los datos
        output_path: Ruta del archivo PDF de salida
    
    Returns:
        Ruta del archivo PDF generado
    """
    with st.spinner("Generando reporte PDF..."):
        # Crear archivo CSV temporal
        temp_csv_path = "temp_report_data.csv"
        df.to_csv(temp_csv_path, index=False)
        
        # Ruta al generador de reportes
        report_generator_path = os.path.join("tools", "csv-to-pdf-report-generator", "app.py")
        
        if not os.path.exists(report_generator_path):
            st.error(f"Generador de reportes no encontrado en: {report_generator_path}")
            return None
        
        try:
            # Ejecutar generador de reportes
            result = subprocess.run(
                [sys.executable, report_generator_path, temp_csv_path, output_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            st.success(f"Reporte PDF generado exitosamente en: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            st.error(f"Error al generar reporte PDF: {e.stderr}")
            return None
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)

# ============================================================================
# 8. PIPELINE PRINCIPAL
# ============================================================================

def pipeline_completo_encuestas(ruta_archivo: str, 
                               estrategia_imputacion: str = "knn",
                               generar_reporte: bool = True,
                               exportar_datos: bool = True) -> Dict[str, Any]:
    """
    Pipeline completo para procesamiento de datos de encuestas.
    
    Args:
        ruta_archivo: Ruta al archivo de datos
        estrategia_imputacion: Estrategia de imputación
        generar_reporte: Si generar reporte PDF
        exportar_datos: Si exportar datos limpios
    
    Returns:
        Diccionario con resultados del pipeline
    """
    resultados = {}
    
    try:
        # 1. Cargar datos
        st.header("📁 Carga de Datos")
        df = cargar_datos(ruta_archivo)
        resultados['datos_originales'] = df.copy()
        
        # 2. Explorar datos
        st.header("🔍 Exploración de Datos")
        mostrar_resumen_datos(df)
        
        # 3. Visualizar valores faltantes
        st.header("🔍 Análisis de Valores Faltantes")
        visualizar_nulos(df, tipo="matrix")
        
        # 4. Imputar valores faltantes
        st.header("🧹 Limpieza de Datos")
        df_limpio = imputar_nulos(df, estrategia=estrategia_imputacion)
        resultados['datos_limpios'] = df_limpio
        
        # 5. Exportar datos limpios
        if exportar_datos:
            st.header("💾 Exportación de Datos")
            ruta_export = f"datos_limpios_{Path(ruta_archivo).stem}.csv"
            exportar_limpio(df_limpio, ruta_export)
            resultados['ruta_exportacion'] = ruta_export
        
        # 6. Generar reporte
        if generar_reporte:
            st.header("📊 Generación de Reporte")
            ruta_pdf = f"reporte_encuesta_{Path(ruta_archivo).stem}.pdf"
            pdf_path = generar_reporte_pdf(df_limpio, ruta_pdf)
            resultados['ruta_reporte'] = pdf_path
        
        st.success("🎉 Pipeline completado exitosamente!")
        return resultados
        
    except Exception as e:
        st.error(f"Error en el pipeline: {str(e)}")
        raise

# ============================================================================
# 9. FUNCIONES AUXILIARES
# ============================================================================

def validar_archivo(ruta: str) -> bool:
    """Valida que el archivo existe y es del formato correcto."""
    if not os.path.exists(ruta):
        return False
    
    extensiones_validas = ['.csv', '.sav', '.dta', '.xlsx', '.xls']
    return any(ruta.lower().endswith(ext) for ext in extensiones_validas)

def obtener_estadisticas_basicas(df: pd.DataFrame) -> Dict[str, Any]:
    """Obtiene estadísticas básicas del DataFrame."""
    return {
        'filas': len(df),
        'columnas': len(df.columns),
        'memoria_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'valores_faltantes': df.isnull().sum().sum(),
        'filas_duplicadas': df.duplicated().sum()
    }

def procesar_archivo_grande_en_chunks(ruta: str, chunk_size: int = 10000,
                                     estrategia_imputacion: str = "knn",
                                     generar_reporte: bool = True,
                                     exportar_datos: bool = True) -> Dict[str, Any]:
    """
    Procesa archivos grandes en chunks para optimizar memoria y rendimiento.
    
    Args:
        ruta: Ruta al archivo de datos
        chunk_size: Tamaño del chunk para procesamiento
        estrategia_imputacion: Estrategia de imputación
        generar_reporte: Si generar reporte PDF
        exportar_datos: Si exportar datos limpios
    
    Returns:
        Diccionario con resultados del pipeline
    """
    resultados = {}
    
    try:
        # 1. Cargar datos en chunks
        st.header("📁 Carga de Datos (Archivo Grande)")
        df = cargar_datos_grandes(ruta, chunk_size)
        resultados['datos_originales'] = df.copy()
        
        # 2. Exploración básica (sin cargar todo en memoria)
        st.header("🔍 Exploración de Datos")
        info_basica = {
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).value_counts().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        
        # Mostrar métricas básicas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Filas", f"{info_basica['shape'][0]:,}")
        with col2:
            st.metric("Columnas", info_basica['shape'][1])
        with col3:
            st.metric("Memoria (MB)", f"{info_basica['memory_usage_mb']:.1f}")
        with col4:
            st.metric("Valores faltantes", f"{df.isnull().sum().sum():,}")
        
        # 3. Imputación optimizada para archivos grandes
        st.header("🧹 Limpieza de Datos")
        if info_basica['memory_usage_mb'] > 1000:  # Más de 1GB
            st.warning("⚠️ Archivo muy grande detectado. Usando estrategia de imputación optimizada.")
            # Para archivos muy grandes, usar estrategias más simples
            if estrategia_imputacion in ['knn', 'random_forest']:
                estrategia_imputacion = 'media'  # Cambiar a estrategia más simple
                st.info(f"Estrategia cambiada a '{estrategia_imputacion}' para optimizar rendimiento")
        
        df_limpio = imputar_nulos(df, estrategia=estrategia_imputacion)
        resultados['datos_limpios'] = df_limpio
        
        # 4. Exportar datos limpios
        if exportar_datos:
            st.header("💾 Exportación de Datos")
            ruta_export = f"datos_limpios_{Path(ruta).stem}.csv"
            exportar_limpio(df_limpio, ruta_export)
            resultados['ruta_exportacion'] = ruta_export
        
        # 5. Generar reporte simplificado para archivos grandes
        if generar_reporte:
            st.header("📊 Generación de Reporte")
            ruta_pdf = f"reporte_encuesta_{Path(ruta).stem}.pdf"
            pdf_path = generar_reporte_pdf(df_limpio, ruta_pdf)
            resultados['ruta_reporte'] = pdf_path
        
        st.success("🎉 Pipeline para archivo grande completado exitosamente!")
        return resultados
        
    except Exception as e:
        st.error(f"Error en el pipeline para archivo grande: {str(e)}")
        raise

if __name__ == "__main__":
    st.title("📊 Pipeline de Procesamiento de Datos de Encuestas")
    st.write("Este módulo proporciona herramientas completas para el procesamiento de datos de encuestas.") 