# app_front.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
from proyecto_j.src.estadistica import (
    cargar_archivo,
    calcular_media,
    calcular_moda,
    calcular_percentiles,
    generar_histograma,
    calcular_correlacion_pearson,
    calcular_correlacion_spearman,
    generar_heatmap_correlacion,
    obtener_columnas_numericas,
    obtener_columnas_categoricas,
    crear_tabla_contingencia,
    calcular_chi_cuadrado,
    generar_grafico_tabla_contingencia,
    calcular_porcentajes_tabla_contingencia,
    interpretar_chi_cuadrado,
    crear_filtros_dinamicos,
    aplicar_filtros,
    obtener_estadisticas_filtradas,
    generar_estadisticas_descriptivas_completas,
    generar_resumen_correlaciones,
    generar_resumen_tablas_contingencia,
    generar_csv_datos_filtrados,
    generar_excel_completo,
    generar_html_reporte,
    generar_boxplot,
    generar_scatter_plot,
    generar_diagrama_densidad,
    generar_grafico_barras,
    generar_histograma_densidad,
    generar_violin_plot,
    generar_heatmap_correlacion_avanzado,
    generar_panel_visualizaciones,
    generar_scatter_matrix,
)
from proyecto_j.src.ciencias_sociales import (
    clasificar_variable,
    clasificar_variables_avanzado,
    analisis_descriptivo_cs,
    analisis_bivariado_cs,
    analisis_regresion_multiple_cs,
    analisis_clusters_cs,
    calcular_indice_gini,
    calcular_indice_gini_simple,
    calcular_indice_calidad_vida,
    calcular_indice_calidad_vida_simple,
    validar_supuestos_regresion,
    analizar_valores_perdidos,
    sugerir_imputacion,
)
from proyecto_j.src.nl_query import parse_and_execute, interpretar_resultado
from proyecto_j.src.nl_query_trends import (
    analizar_tendencia_temporal, 
    exportar_resultados_csv,
    STL_AVAILABLE,
    detectar_analisis_temporal
)

# Configuración de la página y CSS para wizard visual
st.set_page_config(page_title="🔢 Estadísticas Ninja", layout="wide")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;500;600;700&family=Libre+Baskerville:wght@400;700&display=swap');
    html, body {
        color-scheme: light !important;
        background-color: #FBF7F2 !important;
        color: #333333 !important;
    }
    :root {
        --color-fondo-general: #FBF7F2;
        --color-fondo-secundario: #F5E3D3;
        --color-azul-claro: #C7DCE5;
        --color-azul-profundo: #648DA5;
        --color-texto-principal: #2C3E50;
        --color-texto-secundario: #7F8C8D;
        --color-sombra: rgba(0, 0, 0, 0.08);
        --border-radius: 12px;
        --espaciado: 24px;
        --espaciado-pequeno: 16px;
    }
    .stApp {
        background-color: var(--color-fondo-general) !important;
        color: var(--color-texto-principal) !important;
    }
    .main > div {
        background-color: var(--color-azul-claro) !important;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        box-shadow: 0 2px 8px var(--color-sombra);
    }
    .css-1d391kg {
        background-color: #333333 !important;
        border-right: 1px solid #555555;
        padding: var(--espaciado);
    }
    .css-1d391kg .sidebar-content {
        background-color: #333333 !important;
        color: #FFFFFF !important;
    }
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, 
    .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: #FFFFFF !important;
    }
    .css-1d391kg p, .css-1d391kg div, .css-1d391kg span {
        color: #CCCCCC !important;
    }
    .css-1d391kg .stMarkdown, .css-1d391kg .stText {
        background-color: transparent !important;
        color: #CCCCCC !important;
    }
    .stMarkdown, .stText, .stDataFrame, .stPlotlyChart {
        background-color: transparent !important;
    }
    h1 {
        font-family: 'Raleway', sans-serif;
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--color-azul-profundo);
        text-align: center;
        margin-bottom: var(--espaciado);
    }
    .stButton > button {
        background-color: var(--color-azul-profundo);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: var(--color-azul-claro);
        color: var(--color-texto-principal);
        transform: translateY(-1px);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Wizard visual y navegación clara
st.title("🤖 Asistente de Visualización de Datos")

st.markdown(
    """
Guía paso a paso para crear visualizaciones efectivas de tus datos
"""
)

# ============================================================================
# INICIALIZACIÓN DE SESSION STATE
# ============================================================================

# Inicializar variables de sesión
if "df" not in st.session_state:
    st.session_state["df"] = None

if "filtros_aplicados" not in st.session_state:
    st.session_state["filtros_aplicados"] = {}

if "datos_analisis" not in st.session_state:
    st.session_state["datos_analisis"] = {}

if "variable_seleccionada" not in st.session_state:
    st.session_state["variable_seleccionada"] = None

if "variables_correlacion" not in st.session_state:
    st.session_state["variables_correlacion"] = []

if "tipo_correlacion" not in st.session_state:
    st.session_state["tipo_correlacion"] = "Pearson"

if "variable_contingencia_1" not in st.session_state:
    st.session_state["variable_contingencia_1"] = None

if "variable_contingencia_2" not in st.session_state:
    st.session_state["variable_contingencia_2"] = None

if "tipo_visualizacion" not in st.session_state:
    st.session_state["tipo_visualizacion"] = "📊 Panel Completo de Visualizaciones"

if "variable_visualizacion" not in st.session_state:
    st.session_state["variable_visualizacion"] = None

if "variable_grupo_visualizacion" not in st.session_state:
    st.session_state["variable_grupo_visualizacion"] = None

# Variables para ciencias sociales
if "analisis_cs_variable" not in st.session_state:
    st.session_state["analisis_cs_variable"] = None

if "analisis_cs_variables_bivariado" not in st.session_state:
    st.session_state["analisis_cs_variables_bivariado"] = []

if "analisis_cs_variables_regresion" not in st.session_state:
    st.session_state["analisis_cs_variables_regresion"] = []

if "analisis_cs_variables_clusters" not in st.session_state:
    st.session_state["analisis_cs_variables_clusters"] = []

# Sidebar para navegación
st.sidebar.title("📊 Navegación")
pagina = st.sidebar.selectbox(
    "Selecciona la sección:",
    [
        "📋 Metadatos de Variables",
        "🔍 Filtros",
        "📈 Estadísticas Básicas",
        "🔗 Análisis de Correlaciones",
        "📊 Tablas de Contingencia",
        "📊 Visualizaciones Avanzadas",
        "🎓 Ciencias Sociales",
        "🤖 Consultas en Lenguaje Natural",
        "📤 Exportar Resultados",
    ],
)

# ============================================================================
# CARGA DE DATOS CON PERSISTENCIA
# ============================================================================

archivo = st.file_uploader(
    "📂 Sube tu archivo de datos", 
    type=["sav", "dta", "csv", "xlsx", "xls"],
    help="Formatos soportados: SPSS (.sav), Stata (.dta), Excel (.xlsx, .xls), CSV (.csv)"
)

if archivo is not None:
    import os
    ext = os.path.splitext(archivo.name)[-1].lower()
    temp_path = f"data/temp_file{ext}"
    with open(temp_path, "wb") as f:
        f.write(archivo.getbuffer())
    try:
        df = cargar_archivo(temp_path)
        st.session_state["df"] = df
        st.success("Archivo cargado correctamente 🎉")
        
        # Clasificar variables con metadatos enriquecidos
        with st.spinner("🔍 Analizando estructura de variables..."):
            variable_metadata = clasificar_variables_avanzado(df)
            st.session_state["variable_metadata"] = variable_metadata
        
        # Mostrar metadatos de variables
        st.subheader("📋 Metadatos de Variables")
        st.write("Información detallada sobre la clasificación automática de variables con análisis especializado para ciencias sociales.")
        
        # Verificar si hay metadatos disponibles
        if "variable_metadata" not in st.session_state:
            st.warning("⚠️ No hay metadatos de variables disponibles. Por favor, carga un archivo primero.")
        else:
            variable_metadata = st.session_state["variable_metadata"]
            
            # Crear DataFrame de metadatos para mejor visualización
            df_metadata = pd.DataFrame(variable_metadata).T
            
            # Reorganizar columnas para mejor legibilidad
            column_order = [
                'type', 'dominio', 'es_ordinal', 'es_binaria', 'es_likert', 'es_continua',
                'n_missing', 'pct_missing', 'n_unique', 'pct_unique', 'sample_values'
            ]
            
            # Filtrar columnas que existen
            existing_columns = [col for col in column_order if col in df_metadata.columns]
            df_metadata_display = df_metadata[existing_columns].copy()
            
            # Formatear porcentajes
            if 'pct_missing' in df_metadata_display.columns:
                df_metadata_display['pct_missing'] = df_metadata_display['pct_missing'].apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
                )
            if 'pct_unique' in df_metadata_display.columns:
                df_metadata_display['pct_unique'] = df_metadata_display['pct_unique'].apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
                )
            
            # Mostrar resumen estadístico
            st.subheader("📊 Resumen de Clasificación")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                tipos = df_metadata['type'].value_counts()
                st.metric("📊 Tipos de Variables", len(tipos))
                
            with col2:
                dominios = df_metadata['dominio'].value_counts()
                st.metric("🎯 Dominios", len(dominios))
                
            with col3:
                total_missing = sum(df_metadata['n_missing'])
                st.metric("❌ Valores Faltantes", f"{total_missing:,}")
                
            with col4:
                variables_ordinales = sum(df_metadata.get('es_ordinal', [False]))
                st.metric("📈 Variables Ordinales", variables_ordinales)
            
            # Mostrar tabla completa de metadatos
            st.subheader("📋 Tabla Completa de Metadatos")
            st.dataframe(
                df_metadata_display,
                use_container_width=True,
                height=400
            )
            
            # Mostrar detalles por tipo de variable
            with st.expander("🔍 Detalles por Tipo de Variable"):
                for var_type in df_metadata['type'].unique():
                    st.write(f"**{var_type.upper()}**")
                    vars_of_type = df_metadata[df_metadata['type'] == var_type]
                    st.write(f"- Cantidad: {len(vars_of_type)}")
                    st.write(f"- Variables: {', '.join(vars_of_type.index.tolist())}")
                    st.write("---")
            
            # Mostrar detalles por dominio
            with st.expander("🎯 Detalles por Dominio de Ciencias Sociales"):
                for dominio in df_metadata['dominio'].unique():
                    st.write(f"**{dominio}**")
                    vars_of_domain = df_metadata[df_metadata['dominio'] == dominio]
                    st.write(f"- Cantidad: {len(vars_of_domain)}")
                    st.write(f"- Variables: {', '.join(vars_of_domain.index.tolist())}")
                    st.write("---")
            
            # Mostrar variables especiales
            with st.expander("⭐ Variables Especiales"):
                # Variables ordinales
                ordinal_vars = df_metadata[df_metadata.get('es_ordinal', False) == True]
                if not ordinal_vars.empty:
                    st.write("**📈 Variables Ordinales:**")
                    st.write(f"- {', '.join(ordinal_vars.index.tolist())}")
                    st.write("---")
                
                # Variables Likert
                likert_vars = df_metadata[df_metadata.get('es_likert', False) == True]
                if not likert_vars.empty:
                    st.write("**📊 Variables Likert:**")
                    st.write(f"- {', '.join(likert_vars.index.tolist())}")
                    st.write("---")
                
                # Variables binarias
                binary_vars = df_metadata[df_metadata.get('es_binaria', False) == True]
                if not binary_vars.empty:
                    st.write("**⚖️ Variables Binarias:**")
                    st.write(f"- {', '.join(binary_vars.index.tolist())}")
                    st.write("---")
            
            # Botones de exportación
            st.subheader("📤 Exportar Metadatos")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_metadata = df_metadata.to_csv()
                st.download_button(
                    label="📄 Descargar CSV",
                    data=csv_metadata,
                    file_name="metadatos_variables.csv",
                    mime="text/csv",
                )
            
            with col2:
                # Crear Excel con metadatos
                import io
                from openpyxl import Workbook
                from openpyxl.utils.dataframe import dataframe_to_rows
                
                wb = Workbook()
                ws = wb.active
                ws.title = "Metadatos Variables"
                
                # Agregar datos
                for r in dataframe_to_rows(df_metadata, index=True, header=True):
                    ws.append(r)
                
                # Guardar en buffer
                excel_buffer = io.BytesIO()
                wb.save(excel_buffer)
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📊 Descargar Excel",
                    data=excel_buffer.getvalue(),
                    file_name="metadatos_variables.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        
        # Mostrar resumen estadístico
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            tipos = df_metadata['type'].value_counts()
            st.metric("📊 Tipos de Variables", len(tipos))
            
        with col2:
            dominios = df_metadata['dominio'].value_counts()
            st.metric("🎯 Dominios", len(dominios))
            
        with col3:
            total_missing = sum(df_metadata['n_missing'])
            st.metric("❌ Valores Faltantes", f"{total_missing:,}")
            
        with col4:
            variables_ordinales = sum(df_metadata.get('es_ordinal', [False]))
            st.metric("📈 Variables Ordinales", variables_ordinales)
        
        # Mostrar detalles por tipo de variable
        with st.expander("🔍 Detalles por Tipo de Variable"):
            for var_type in df_metadata['type'].unique():
                st.write(f"**{var_type.upper()}**")
                vars_of_type = df_metadata[df_metadata['type'] == var_type]
                st.write(f"- Cantidad: {len(vars_of_type)}")
                st.write(f"- Variables: {', '.join(vars_of_type.index.tolist())}")
                st.write("---")
        
        # Mostrar detalles por dominio
        with st.expander("🎯 Detalles por Dominio de Ciencias Sociales"):
            for dominio in df_metadata['dominio'].unique():
                st.write(f"**{dominio}**")
                vars_of_domain = df_metadata[df_metadata['dominio'] == dominio]
                st.write(f"- Cantidad: {len(vars_of_domain)}")
                st.write(f"- Variables: {', '.join(vars_of_domain.index.tolist())}")
                st.write("---")
                
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {e}")
        df = None
else:
    df = st.session_state["df"]

# ============================================================================
# BOTÓN PARA LIMPIAR SESIÓN
# ============================================================================

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Limpiar sesión"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ============================================================================
# LÓGICA PRINCIPAL DE LA APLICACIÓN
# ============================================================================

if df is not None:
    if pagina == "📋 Metadatos de Variables":
        st.header("📋 Metadatos de Variables")
        st.write("Información detallada sobre la clasificación automática de variables con análisis especializado para ciencias sociales.")
        
        # Verificar si hay metadatos disponibles
        if "variable_metadata" not in st.session_state:
            st.warning("⚠️ No hay metadatos de variables disponibles. Por favor, carga un archivo primero.")
        else:
            variable_metadata = st.session_state["variable_metadata"]
            
            # Crear DataFrame de metadatos para mejor visualización
            df_metadata = pd.DataFrame(variable_metadata).T
            
            # Reorganizar columnas para mejor legibilidad
            column_order = [
                'type', 'dominio', 'es_ordinal', 'es_binaria', 'es_likert', 'es_continua',
                'n_missing', 'pct_missing', 'n_unique', 'pct_unique', 'sample_values'
            ]
            
            # Filtrar columnas que existen
            existing_columns = [col for col in column_order if col in df_metadata.columns]
            df_metadata_display = df_metadata[existing_columns].copy()
            
            # Formatear porcentajes
            if 'pct_missing' in df_metadata_display.columns:
                df_metadata_display['pct_missing'] = df_metadata_display['pct_missing'].apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
                )
            if 'pct_unique' in df_metadata_display.columns:
                df_metadata_display['pct_unique'] = df_metadata_display['pct_unique'].apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
                )
            
            # Mostrar resumen estadístico
            st.subheader("📊 Resumen de Clasificación")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                tipos = df_metadata['type'].value_counts()
                st.metric("📊 Tipos de Variables", len(tipos))
                
            with col2:
                dominios = df_metadata['dominio'].value_counts()
                st.metric("🎯 Dominios", len(dominios))
                
            with col3:
                total_missing = sum(df_metadata['n_missing'])
                st.metric("❌ Valores Faltantes", f"{total_missing:,}")
                
            with col4:
                variables_ordinales = sum(df_metadata.get('es_ordinal', [False]))
                st.metric("📈 Variables Ordinales", variables_ordinales)
            
            # Mostrar tabla completa de metadatos
            st.subheader("📋 Tabla Completa de Metadatos")
            st.dataframe(
                df_metadata_display,
                use_container_width=True,
                height=400
            )
            
            # Mostrar detalles por tipo de variable
            with st.expander("🔍 Detalles por Tipo de Variable"):
                for var_type in df_metadata['type'].unique():
                    st.write(f"**{var_type.upper()}**")
                    vars_of_type = df_metadata[df_metadata['type'] == var_type]
                    st.write(f"- Cantidad: {len(vars_of_type)}")
                    st.write(f"- Variables: {', '.join(vars_of_type.index.tolist())}")
                    st.write("---")
            
            # Mostrar detalles por dominio
            with st.expander("🎯 Detalles por Dominio de Ciencias Sociales"):
                for dominio in df_metadata['dominio'].unique():
                    st.write(f"**{dominio}**")
                    vars_of_domain = df_metadata[df_metadata['dominio'] == dominio]
                    st.write(f"- Cantidad: {len(vars_of_domain)}")
                    st.write(f"- Variables: {', '.join(vars_of_domain.index.tolist())}")
                    st.write("---")
            
            # Mostrar variables especiales
            with st.expander("⭐ Variables Especiales"):
                # Variables ordinales
                ordinal_vars = df_metadata[df_metadata.get('es_ordinal', False) == True]
                if not ordinal_vars.empty:
                    st.write("**📈 Variables Ordinales:**")
                    st.write(f"- {', '.join(ordinal_vars.index.tolist())}")
                    st.write("---")
                
                # Variables Likert
                likert_vars = df_metadata[df_metadata.get('es_likert', False) == True]
                if not likert_vars.empty:
                    st.write("**📊 Variables Likert:**")
                    st.write(f"- {', '.join(likert_vars.index.tolist())}")
                    st.write("---")
                
                # Variables binarias
                binary_vars = df_metadata[df_metadata.get('es_binaria', False) == True]
                if not binary_vars.empty:
                    st.write("**⚖️ Variables Binarias:**")
                    st.write(f"- {', '.join(binary_vars.index.tolist())}")
                    st.write("---")
            
            # Botones de exportación
            st.subheader("📤 Exportar Metadatos")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_metadata = df_metadata.to_csv()
                st.download_button(
                    label="📄 Descargar CSV",
                    data=csv_metadata,
                    file_name="metadatos_variables.csv",
                    mime="text/csv",
                )
            
            with col2:
                # Crear Excel con metadatos
                import io
                from openpyxl import Workbook
                from openpyxl.utils.dataframe import dataframe_to_rows
                
                wb = Workbook()
                ws = wb.active
                ws.title = "Metadatos Variables"
                
                # Agregar datos
                for r in dataframe_to_rows(df_metadata, index=True, header=True):
                    ws.append(r)
                
                # Guardar en buffer
                excel_buffer = io.BytesIO()
                wb.save(excel_buffer)
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📊 Descargar Excel",
                    data=excel_buffer.getvalue(),
                    file_name="metadatos_variables.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    elif pagina == "🔍 Filtros":
        st.header("🔍 Filtros Dinámicos")
        st.write(
            "Configura filtros para personalizar tu análisis. Los filtros se aplicarán a todas las secciones."
        )

        # Crear información de filtros
        filtros_info = crear_filtros_dinamicos(df)

        if filtros_info:
            st.subheader("📋 Configuración de Filtros")

            # Separar variables numéricas y categóricas
            variables_numericas = [
                col for col, info in filtros_info.items() if info["tipo"] == "numerico"
            ]
            variables_categoricas = [
                col
                for col, info in filtros_info.items()
                if info["tipo"] == "categorico"
            ]

            # Filtros para variables numéricas
            if variables_numericas:
                st.write("**🎯 Filtros por Rango (Variables Numéricas):**")

                # Filtrar variables numéricas válidas
                variables_numericas_validas = []
                for col in variables_numericas:
                    info = filtros_info[col]
                    min_val, max_val = info["min"], info["max"]
                    
                    # Validación robusta de datos numéricos
                    try:
                        # Verificar que min y max sean números válidos y no NaN
                        if (pd.isna(min_val) or pd.isna(max_val) or 
                            not isinstance(min_val, (int, float)) or 
                            not isinstance(max_val, (int, float)) or
                            min_val == max_val or
                            min_val > max_val):
                            
                            st.warning(f"⚠️ La variable '{col}' no tiene datos numéricos válidos para filtrar (valores: min={min_val}, max={max_val})")
                            continue
                        
                        # Verificar que hay suficientes datos para crear un rango útil
                        if max_val - min_val < 1e-10:  # Rango muy pequeño
                            st.info(f"ℹ️ La variable '{col}' tiene un rango muy pequeño ({min_val:.6f} - {max_val:.6f}), no se mostrará filtro")
                            continue
                            
                        variables_numericas_validas.append(col)
                        
                    except Exception as e:
                        st.error(f"❌ Error al validar la variable '{col}': {str(e)}")
                        continue

                # Crear sliders solo para variables válidas
                for col in variables_numericas_validas:
                    info = filtros_info[col]
                    min_val, max_val = info["min"], info["max"]

                    # Obtener valores actuales del session_state con validación
                    filtro_actual = st.session_state["filtros_aplicados"].get(col, {})
                    valor_min_actual = filtro_actual.get("min", min_val)
                    valor_max_actual = filtro_actual.get("max", max_val)
                    
                    # Asegurar que los valores actuales estén dentro del rango válido
                    valor_min_actual = max(min_val, min(valor_min_actual, max_val))
                    valor_max_actual = min(max_val, max(valor_max_actual, min_val))

                    # Calcular step size de forma segura
                    step_size = max((max_val - min_val) / 100, 1e-6)

                    # Crear slider para rango con validación adicional
                    try:
                        rango = st.slider(
                            f"📊 {col}",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=(float(valor_min_actual), float(valor_max_actual)),
                            step=float(step_size),
                            help=f"Selecciona el rango para {col} (min: {min_val:.2f}, max: {max_val:.2f})",
                        )

                        # Guardar filtro en session_state
                        st.session_state["filtros_aplicados"][col] = {
                            "min": rango[0],
                            "max": rango[1],
                        }
                        
                    except Exception as e:
                        st.error(f"❌ Error al crear slider para '{col}': {str(e)}")
                        continue

                # Mostrar mensaje si no hay variables numéricas válidas
                if not variables_numericas_validas:
                    st.info("ℹ️ No se encontraron variables numéricas válidas para filtrar.")

            # Filtros para variables categóricas
            if variables_categoricas:
                st.write("**🏷️ Filtros por Categoría (Variables Categóricas):**")

                for col in variables_categoricas:
                    info = filtros_info[col]
                    categorias = info["categorias"]

                    # Obtener categorías seleccionadas actuales
                    categorias_actuales = st.session_state["filtros_aplicados"].get(
                        col, categorias
                    )

                    # Crear multiselect para categorías
                    categorias_seleccionadas = st.multiselect(
                        f"📋 {col}",
                        options=categorias,
                        default=categorias_actuales,
                        help=f"Selecciona las categorías de {col} que quieres incluir",
                    )

                    # Guardar filtro en session_state
                    st.session_state["filtros_aplicados"][
                        col
                    ] = categorias_seleccionadas

            # Aplicar filtros y mostrar estadísticas
            df_filtrado = aplicar_filtros(df, st.session_state["filtros_aplicados"])
            stats_filtradas = obtener_estadisticas_filtradas(
                df, st.session_state["filtros_aplicados"]
            )

            # Mostrar resumen de filtros aplicados
            st.subheader("📊 Resumen de Filtros")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("📈 Total Original", stats_filtradas["n_original"])

            with col2:
                st.metric("✅ Datos Filtrados", stats_filtradas["n_filtrado"])

            with col3:
                st.metric(
                    "📊 % de Muestra", f"{stats_filtradas['porcentaje_muestra']:.1f}%"
                )

            # Mostrar filtros activos
            if st.session_state["filtros_aplicados"]:
                st.subheader("🔧 Filtros Activos")
                for col, filtro in st.session_state["filtros_aplicados"].items():
                    if isinstance(filtro, dict):
                        st.write(
                            f"• **{col}**: {filtro['min']:.2f} - {filtro['max']:.2f}"
                        )
                    elif isinstance(filtro, list):
                        st.write(f"• **{col}**: {', '.join(filtro)}")

            # Botón para limpiar filtros
            if st.button("🗑️ Limpiar Todos los Filtros"):
                st.session_state["filtros_aplicados"] = {}
                st.rerun()

            # Vista previa de datos filtrados
            st.subheader("👀 Vista Previa de Datos Filtrados")
            st.dataframe(df_filtrado.head(10))

            # Botones de exportación para datos filtrados
            st.subheader("📤 Exportar Datos Filtrados")
            col1, col2 = st.columns(2)

            with col1:
                csv_data = generar_csv_datos_filtrados(
                    df, st.session_state["filtros_aplicados"]
                )
                st.download_button(
                    label="📄 Descargar CSV",
                    data=csv_data,
                    file_name="datos_filtrados.csv",
                    mime="text/csv",
                )

            with col2:
                excel_data = generar_excel_completo(
                    df, st.session_state["filtros_aplicados"]
                )
                st.download_button(
                    label="📊 Descargar Excel",
                    data=excel_data,
                    file_name="datos_filtrados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        else:
            st.warning("⚠️ No se encontraron variables para filtrar.")

    elif pagina == "📈 Estadísticas Básicas":
        st.header("📈 Estadísticas Básicas")

        # Aplicar filtros si existen
        if st.session_state["filtros_aplicados"]:
            df_analisis = aplicar_filtros(df, st.session_state["filtros_aplicados"])
            stats_filtradas = obtener_estadisticas_filtradas(
                df, st.session_state["filtros_aplicados"]
            )

            st.info(
                f"📊 Analizando {stats_filtradas['n_filtrado']} de {stats_filtradas['n_original']} observaciones ({stats_filtradas['porcentaje_muestra']:.1f}% de la muestra)"
            )
        else:
            df_analisis = df
            st.info("📊 Analizando todos los datos (sin filtros aplicados)")

        cols_num = obtener_columnas_numericas(df_analisis)

        # Inicializar columna con None para evitar NameError
        columna = None

        if not cols_num:
            st.warning("⚠️ No hay variables numéricas disponibles para análisis.")
        else:
            # Usar la variable seleccionada anteriormente o la primera disponible
            variable_default = (
                st.session_state["variable_seleccionada"]
                if st.session_state["variable_seleccionada"] in cols_num
                else cols_num[0]
            )

            columna = st.selectbox(
                "🔍 Selecciona columna numérica",
                cols_num,
                index=cols_num.index(variable_default),
            )

            # Guardar la selección en session_state
            st.session_state["variable_seleccionada"] = columna

        if columna:
            st.subheader("📊 Estadísticas básicas")
            st.write(f"• Media: **{calcular_media(df_analisis, columna):.2f}**")
            st.write(
                f"• Moda: **{', '.join(map(str, calcular_moda(df_analisis, columna)))}**"
            )
            pct = calcular_percentiles(df_analisis, columna)
            st.write("• Percentiles:")
            st.write(pct)

            st.subheader("📈 Histograma")
            fig = generar_histograma(df_analisis, columna)
            st.pyplot(fig)

            # Generar estadísticas descriptivas completas para exportación
            estadisticas_completas = generar_estadisticas_descriptivas_completas(
                df_analisis
            )

            # Guardar en session_state para exportación
            st.session_state["datos_analisis"][
                "estadisticas_descriptivas"
            ] = estadisticas_completas

            # Botones de exportación
            st.subheader("📤 Exportar Estadísticas")
            col1, col2 = st.columns(2)

            with col1:
                csv_stats = estadisticas_completas.to_csv(index=False)
                st.download_button(
                    label="📄 Descargar CSV",
                    data=csv_stats,
                    file_name="estadisticas_descriptivas.csv",
                    mime="text/csv",
                )

            with col2:
                excel_stats = generar_excel_completo(
                    df, st.session_state["filtros_aplicados"], estadisticas_completas
                )
                st.download_button(
                    label="📊 Descargar Excel",
                    data=excel_stats,
                    file_name="estadisticas_descriptivas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    elif pagina == "🔗 Análisis de Correlaciones":
        st.header("🔗 Análisis de Correlaciones")

        # Aplicar filtros si existen
        if st.session_state["filtros_aplicados"]:
            df_analisis = aplicar_filtros(df, st.session_state["filtros_aplicados"])
            stats_filtradas = obtener_estadisticas_filtradas(
                df, st.session_state["filtros_aplicados"]
            )

            st.info(
                f"📊 Analizando {stats_filtradas['n_filtrado']} de {stats_filtradas['n_original']} observaciones ({stats_filtradas['porcentaje_muestra']:.1f}% de la muestra)"
            )
        else:
            df_analisis = df
            st.info("📊 Analizando todos los datos (sin filtros aplicados)")

        cols_num = obtener_columnas_numericas(df_analisis)

        if len(cols_num) < 2:
            st.warning(
                "⚠️ Se necesitan al menos 2 variables numéricas para calcular correlaciones."
            )
        else:
            st.subheader("📋 Selección de Variables")
            st.write(
                "Selecciona 2 o más variables numéricas para analizar sus correlaciones:"
            )

            # Usar variables seleccionadas anteriormente o las primeras disponibles
            variables_default = (
                st.session_state["variables_correlacion"]
                if st.session_state["variables_correlacion"]
                and all(
                    v in cols_num for v in st.session_state["variables_correlacion"]
                )
                else cols_num[: min(5, len(cols_num))]
            )

            # Selección múltiple de variables
            variables_seleccionadas = st.multiselect(
                "🔍 Variables a analizar:",
                cols_num,
                default=variables_default,
                help="Selecciona al menos 2 variables para calcular correlaciones",
            )

            # Guardar las variables seleccionadas
            st.session_state["variables_correlacion"] = variables_seleccionadas

            if len(variables_seleccionadas) >= 2:
                st.subheader("📊 Matriz de Correlaciones")

                # Usar tipo de correlación anterior o Pearson por defecto
                tipo_correlacion = st.radio(
                    "🎯 Tipo de correlación:",
                    ["Pearson", "Spearman"],
                    index=0 if st.session_state["tipo_correlacion"] == "Pearson" else 1,
                    horizontal=True,
                    help="Pearson: para relaciones lineales, Spearman: para relaciones monótonas",
                )

                # Guardar el tipo de correlación
                st.session_state["tipo_correlacion"] = tipo_correlacion

                # Calcular correlación según el tipo seleccionado
                if tipo_correlacion == "Pearson":
                    matriz_corr = calcular_correlacion_pearson(
                        df_analisis, variables_seleccionadas
                    )
                    titulo_heatmap = "Matriz de Correlación de Pearson"
                else:
                    matriz_corr = calcular_correlacion_spearman(
                        df_analisis, variables_seleccionadas
                    )
                    titulo_heatmap = "Matriz de Correlación de Spearman"

                # Mostrar matriz de correlación como tabla
                st.write("**Matriz de Correlación:**")
                st.dataframe(
                    matriz_corr.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1)
                )

                # Mostrar heatmap
                st.subheader("🔥 Heatmap de Correlación")
                fig_heatmap = generar_heatmap_correlacion(matriz_corr, titulo_heatmap)
                st.pyplot(fig_heatmap)

                # Generar resumen de correlaciones para exportación
                resumen_correlaciones = generar_resumen_correlaciones(
                    df_analisis, variables_seleccionadas, tipo_correlacion.lower()
                )

                # Guardar en session_state para exportación
                st.session_state["datos_analisis"][
                    "correlaciones"
                ] = resumen_correlaciones

                # Información adicional sobre las correlaciones
                st.subheader("📝 Interpretación")
                st.write(
                    """
                **Guía de interpretación:**
                - **1.0 a 0.7**: Correlación muy fuerte positiva
                - **0.7 a 0.5**: Correlación fuerte positiva  
                - **0.5 a 0.3**: Correlación moderada positiva
                - **0.3 a 0.1**: Correlación débil positiva
                - **0.1 a -0.1**: Sin correlación
                - **-0.1 a -0.3**: Correlación débil negativa
                - **-0.3 a -0.5**: Correlación moderada negativa
                - **-0.5 a -0.7**: Correlación fuerte negativa
                - **-0.7 a -1.0**: Correlación muy fuerte negativa
                """
                )

                # Estadísticas adicionales
                st.subheader("📈 Estadísticas de la Muestra")
                st.write(
                    f"• **Número de observaciones:** {len(df_analisis[variables_seleccionadas].dropna())}"
                )
                st.write(f"• **Variables analizadas:** {len(variables_seleccionadas)}")

                # Mostrar correlaciones más fuertes
                st.subheader("🔍 Correlaciones Destacadas")
                # Obtener pares de correlaciones (sin diagonal)
                correlaciones = []
                for i in range(len(matriz_corr.columns)):
                    for j in range(i + 1, len(matriz_corr.columns)):
                        var1 = matriz_corr.columns[i]
                        var2 = matriz_corr.columns[j]
                        corr_valor = matriz_corr.iloc[i, j]
                        correlaciones.append((var1, var2, corr_valor))

                # Ordenar por valor absoluto de correlación
                correlaciones.sort(key=lambda x: abs(x[2]), reverse=True)

                # Mostrar las 5 correlaciones más fuertes
                st.write("**Top 5 correlaciones más fuertes:**")
                for i, (var1, var2, corr_valor) in enumerate(correlaciones[:5], 1):
                    color = "🟢" if corr_valor > 0 else "🔴"
                    st.write(f"{i}. {color} **{var1}** ↔ **{var2}**: {corr_valor:.3f}")

                # Botones de exportación
                st.subheader("📤 Exportar Correlaciones")
                col1, col2 = st.columns(2)

                with col1:
                    csv_corr = resumen_correlaciones.to_csv(index=False)
                    st.download_button(
                        label="📄 Descargar CSV",
                        data=csv_corr,
                        file_name=f"correlaciones_{tipo_correlacion.lower()}.csv",
                        mime="text/csv",
                    )

                with col2:
                    excel_corr = generar_excel_completo(
                        df,
                        st.session_state["filtros_aplicados"],
                        None,
                        resumen_correlaciones,
                    )
                    st.download_button(
                        label="📊 Descargar Excel",
                        data=excel_corr,
                        file_name=f"correlaciones_{tipo_correlacion.lower()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            else:
                st.warning(
                    "⚠️ Por favor selecciona al menos 2 variables para continuar."
                )

    elif pagina == "📊 Tablas de Contingencia":
        st.header("📊 Tablas de Contingencia y Prueba χ²")

        # Aplicar filtros si existen
        if st.session_state["filtros_aplicados"]:
            df_analisis = aplicar_filtros(df, st.session_state["filtros_aplicados"])
            stats_filtradas = obtener_estadisticas_filtradas(
                df, st.session_state["filtros_aplicados"]
            )

            st.info(
                f"📊 Analizando {stats_filtradas['n_filtrado']} de {stats_filtradas['n_original']} observaciones ({stats_filtradas['porcentaje_muestra']:.1f}% de la muestra)"
            )
        else:
            df_analisis = df
            st.info("📊 Analizando todos los datos (sin filtros aplicados)")

        cols_cat = obtener_columnas_categoricas(df_analisis)

        if len(cols_cat) < 2:
            st.warning(
                "⚠️ Se necesitan al menos 2 variables categóricas para crear tablas de contingencia."
            )
        else:
            st.subheader("📋 Selección de Variables Categóricas")
            st.write("Selecciona dos variables categóricas para analizar su relación:")

            # Usar variables seleccionadas anteriormente o las primeras disponibles
            variable1_default = (
                st.session_state["variable_contingencia_1"]
                if st.session_state["variable_contingencia_1"] in cols_cat
                else cols_cat[0]
            )
            variable2_default = (
                st.session_state["variable_contingencia_2"]
                if st.session_state["variable_contingencia_2"] in cols_cat
                and st.session_state["variable_contingencia_2"] != variable1_default
                else [col for col in cols_cat if col != variable1_default][0]
            )

            # Selección de variables categóricas
            col1, col2 = st.columns(2)
            with col1:
                variable1 = st.selectbox(
                    "🔍 Primera variable:",
                    cols_cat,
                    index=cols_cat.index(variable1_default),
                    help="Selecciona la primera variable categórica",
                )

            with col2:
                variable2 = st.selectbox(
                    "🔍 Segunda variable:",
                    [col for col in cols_cat if col != variable1],
                    index=(
                        [col for col in cols_cat if col != variable1].index(
                            variable2_default
                        )
                        if variable2_default
                        in [col for col in cols_cat if col != variable1]
                        else 0
                    ),
                    help="Selecciona la segunda variable categórica",
                )

            # Guardar las variables seleccionadas
            st.session_state["variable_contingencia_1"] = variable1
            st.session_state["variable_contingencia_2"] = variable2

            if variable1 and variable2:
                st.subheader("📊 Tabla de Contingencia")

                # Crear tabla de contingencia
                tabla_contingencia = crear_tabla_contingencia(
                    df_analisis, variable1, variable2
                )

                # Mostrar tabla de contingencia
                st.write(f"**Tabla de Contingencia: {variable1} vs {variable2}**")
                st.dataframe(tabla_contingencia)

                # Calcular y mostrar porcentajes
                st.subheader("📈 Análisis de Porcentajes")
                porcentajes = calcular_porcentajes_tabla_contingencia(
                    df_analisis, variable1, variable2
                )

                # Tabs para diferentes tipos de porcentajes
                tab1, tab2, tab3 = st.tabs(["Por Fila", "Por Columna", "Del Total"])

                with tab1:
                    st.write(
                        "**Porcentajes por fila** (porcentaje de cada columna dentro de cada fila):"
                    )
                    st.dataframe(porcentajes["porcentajes_fila"].round(2))

                with tab2:
                    st.write(
                        "**Porcentajes por columna** (porcentaje de cada fila dentro de cada columna):"
                    )
                    st.dataframe(porcentajes["porcentajes_columna"].round(2))

                with tab3:
                    st.write(
                        "**Porcentajes del total** (porcentaje de cada celda del total):"
                    )
                    st.dataframe(porcentajes["porcentajes_total"].round(2))

                # Prueba de Chi-cuadrado
                st.subheader("🔬 Prueba de Chi-cuadrado (χ²)")

                # Calcular chi-cuadrado
                resultados_chi = calcular_chi_cuadrado(
                    df_analisis, variable1, variable2
                )

                # Mostrar resultados
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Estadísticas del test:**")
                    st.write(f"• **χ² = {resultados_chi['chi2_statistic']:.4f}**")
                    st.write(f"• **p-valor = {resultados_chi['p_value']:.4f}**")
                    st.write(
                        f"• **Grados de libertad = {resultados_chi['degrees_of_freedom']}**"
                    )
                    st.write(
                        f"• **Tamaño de muestra = {resultados_chi['sample_size']}**"
                    )

                with col2:
                    st.write("**Medidas de asociación:**")
                    st.write(f"• **Cramer's V = {resultados_chi['cramer_v']:.4f}**")
                    st.write(
                        f"• **Coeficiente de contingencia = {resultados_chi['pearson_c']:.4f}**"
                    )

                # Interpretación
                st.subheader("📝 Interpretación")
                interpretacion = interpretar_chi_cuadrado(resultados_chi)
                st.write(interpretacion)

                # Información adicional sobre interpretación
                st.write(
                    """
                **Guía de interpretación:**
                - **p < 0.05**: Existe una relación significativa entre las variables
                - **p ≥ 0.05**: No hay evidencia suficiente de relación entre las variables
                - **Cramer's V < 0.1**: Efecto muy pequeño
                - **Cramer's V 0.1-0.3**: Efecto pequeño
                - **Cramer's V 0.3-0.5**: Efecto moderado
                - **Cramer's V > 0.5**: Efecto grande
                """
                )

                # Visualizaciones
                st.subheader("📊 Visualizaciones")
                fig_visualizacion = generar_grafico_tabla_contingencia(
                    df_analisis, variable1, variable2
                )
                st.pyplot(fig_visualizacion)

                # Información sobre frecuencias esperadas
                st.subheader("📋 Frecuencias Esperadas")
                st.write(
                    "Las frecuencias esperadas bajo la hipótesis de independencia:"
                )
                frecuencias_esperadas = pd.DataFrame(
                    resultados_chi["expected_frequencies"],
                    index=df_analisis[variable1].unique(),
                    columns=df_analisis[variable2].unique(),
                )
                st.dataframe(frecuencias_esperadas.round(2))

                # Generar resumen completo para exportación
                resumen_tablas = generar_resumen_tablas_contingencia(
                    df_analisis, variable1, variable2
                )

                # Guardar en session_state para exportación
                st.session_state["datos_analisis"][
                    "tablas_contingencia"
                ] = resumen_tablas

                # Botones de exportación
                st.subheader("📤 Exportar Análisis de Contingencia")
                col1, col2 = st.columns(2)

                with col1:
                    # Exportar tabla de contingencia como CSV
                    csv_tabla = tabla_contingencia.to_csv()
                    st.download_button(
                        label="📄 Descargar CSV",
                        data=csv_tabla,
                        file_name=f"tabla_contingencia_{variable1}_{variable2}.csv",
                        mime="text/csv",
                    )

                with col2:
                    excel_tabla = generar_excel_completo(
                        df,
                        st.session_state["filtros_aplicados"],
                        None,
                        None,
                        resumen_tablas,
                    )
                    st.download_button(
                        label="📊 Descargar Excel",
                        data=excel_tabla,
                        file_name=f"tabla_contingencia_{variable1}_{variable2}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            else:
                st.warning(
                    "⚠️ Por favor selecciona dos variables categóricas diferentes para continuar."
                )

    elif pagina == "📊 Visualizaciones Avanzadas":
        st.header("📊 Visualizaciones Avanzadas")

        # Aplicar filtros si existen
        if st.session_state["filtros_aplicados"]:
            df_analisis = aplicar_filtros(df, st.session_state["filtros_aplicados"])
            stats_filtradas = obtener_estadisticas_filtradas(
                df, st.session_state["filtros_aplicados"]
            )

            st.info(
                f"📊 Analizando {stats_filtradas['n_filtrado']} de {stats_filtradas['n_original']} observaciones ({stats_filtradas['porcentaje_muestra']:.1f}% de la muestra)"
            )
        else:
            df_analisis = df
            st.info("📊 Analizando todos los datos (sin filtros aplicados)")

        cols_num = obtener_columnas_numericas(df_analisis)
        cols_cat = obtener_columnas_categoricas(df_analisis)

        if not cols_num:
            st.warning(
                "⚠️ No hay variables numéricas disponibles para visualizaciones avanzadas."
            )
        else:
            st.subheader("🎨 Tipos de Visualizaciones")

            # Usar tipo de visualización anterior o el primero por defecto
            tipos_visualizacion = [
                "📊 Panel Completo de Visualizaciones",
                "📦 Boxplot",
                "🔄 Scatter Plot",
                "📈 Diagrama de Densidad",
                "📊 Histograma con Densidad",
                "🎻 Violin Plot",
                "📊 Gráfico de Barras",
                "🔥 Heatmap de Correlación Avanzado",
                "🔗 Matriz de Scatter Plots",
            ]

            tipo_default_index = (
                tipos_visualizacion.index(st.session_state["tipo_visualizacion"])
                if st.session_state["tipo_visualizacion"] in tipos_visualizacion
                else 0
            )

            # Selector de tipo de visualización
            tipo_visualizacion = st.selectbox(
                "🔍 Selecciona el tipo de visualización:",
                tipos_visualizacion,
                index=tipo_default_index,
                help="Elige el tipo de visualización que quieres generar",
            )

            # Guardar el tipo de visualización
            st.session_state["tipo_visualizacion"] = tipo_visualizacion

            if tipo_visualizacion == "📊 Panel Completo de Visualizaciones":
                st.subheader("📊 Panel Completo de Visualizaciones")
                st.write(
                    "Genera un panel completo con múltiples visualizaciones para una variable."
                )

                # Usar variables seleccionadas anteriormente
                variable_principal_default = (
                    st.session_state["variable_visualizacion"]
                    if st.session_state["variable_visualizacion"] in cols_num
                    else cols_num[0]
                )
                variable_grupo_default = (
                    st.session_state["variable_grupo_visualizacion"]
                    if st.session_state["variable_grupo_visualizacion"] in cols_cat
                    else "Ninguna"
                )

                col1, col2 = st.columns(2)
                with col1:
                    variable_principal = st.selectbox(
                        "📊 Variable principal:",
                        cols_num,
                        index=cols_num.index(variable_principal_default),
                    )

                with col2:
                    variable_grupo = st.selectbox(
                        "🏷️ Variable de agrupación (opcional):",
                        ["Ninguna"] + cols_cat,
                        index=(
                            ["Ninguna"] + cols_cat.index(variable_grupo_default)
                            if variable_grupo_default in cols_cat
                            else 0
                        ),
                    )
                    if variable_grupo == "Ninguna":
                        variable_grupo = None

                # Guardar las selecciones
                st.session_state["variable_visualizacion"] = variable_principal
                st.session_state["variable_grupo_visualizacion"] = (
                    variable_grupo if variable_grupo else "Ninguna"
                )

                if st.button("🎨 Generar Panel"):
                    fig_panel = generar_panel_visualizaciones(
                        df_analisis, variable_principal, variable_grupo
                    )
                    st.pyplot(fig_panel)

                    st.write("**Panel incluye:**")
                    st.write("• Histograma con densidad")
                    st.write("• Boxplot")
                    st.write("• Diagrama de densidad")
                    st.write("• Violin plot (si hay grupo) o Q-Q plot (sin grupo)")

            elif tipo_visualizacion == "📦 Boxplot":
                st.subheader("📦 Boxplot")
                st.write(
                    "Visualiza la distribución de una variable numérica, opcionalmente agrupada por una variable categórica."
                )

                # Usar variables seleccionadas anteriormente
                variable_numerica_default = (
                    st.session_state["variable_visualizacion"]
                    if st.session_state["variable_visualizacion"] in cols_num
                    else cols_num[0]
                )
                variable_categorica_default = (
                    st.session_state["variable_grupo_visualizacion"]
                    if st.session_state["variable_grupo_visualizacion"] in cols_cat
                    else "Ninguna"
                )

                col1, col2 = st.columns(2)
                with col1:
                    variable_numerica = st.selectbox(
                        "📊 Variable numérica:",
                        cols_num,
                        index=cols_num.index(variable_numerica_default),
                    )

                with col2:
                    variable_categorica = st.selectbox(
                        "🏷️ Variable de agrupación (opcional):",
                        ["Ninguna"] + cols_cat,
                        index=(
                            ["Ninguna"] + cols_cat.index(variable_categorica_default)
                            if variable_categorica_default in cols_cat
                            else 0
                        ),
                    )
                    if variable_categorica == "Ninguna":
                        variable_categorica = None

                # Guardar las selecciones
                st.session_state["variable_visualizacion"] = variable_numerica
                st.session_state["variable_grupo_visualizacion"] = (
                    variable_categorica if variable_categorica else "Ninguna"
                )

                if st.button("📦 Generar Boxplot"):
                    fig_boxplot = generar_boxplot(
                        df_analisis, variable_numerica, variable_categorica
                    )
                    st.pyplot(fig_boxplot)

            elif tipo_visualizacion == "🔄 Scatter Plot":
                st.subheader("🔄 Scatter Plot")
                st.write(
                    "Visualiza la relación entre dos variables numéricas, opcionalmente coloreado por una variable categórica."
                )

                # Usar variables seleccionadas anteriormente
                variable_x_default = (
                    st.session_state["variable_visualizacion"]
                    if st.session_state["variable_visualizacion"] in cols_num
                    else cols_num[0]
                )
                variable_y_default = cols_num[1] if len(cols_num) > 1 else cols_num[0]
                variable_color_default = (
                    st.session_state["variable_grupo_visualizacion"]
                    if st.session_state["variable_grupo_visualizacion"] in cols_cat
                    else "Ninguna"
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    variable_x = st.selectbox(
                        "📊 Variable X:",
                        cols_num,
                        index=cols_num.index(variable_x_default),
                    )

                with col2:
                    variable_y = st.selectbox(
                        "📊 Variable Y:",
                        cols_num,
                        index=cols_num.index(variable_y_default),
                    )

                with col3:
                    variable_color = st.selectbox(
                        "🎨 Variable de color (opcional):",
                        ["Ninguna"] + cols_cat,
                        index=(
                            ["Ninguna"] + cols_cat.index(variable_color_default)
                            if variable_color_default in cols_cat
                            else 0
                        ),
                    )
                    if variable_color == "Ninguna":
                        variable_color = None

                # Guardar las selecciones
                st.session_state["variable_visualizacion"] = variable_x
                st.session_state["variable_grupo_visualizacion"] = (
                    variable_color if variable_color else "Ninguna"
                )

                if st.button("🔄 Generar Scatter Plot"):
                    fig_scatter = generar_scatter_plot(
                        df_analisis, variable_x, variable_y, variable_color
                    )
                    st.pyplot(fig_scatter)

            elif tipo_visualizacion == "📈 Diagrama de Densidad":
                st.subheader("📈 Diagrama de Densidad")
                st.write(
                    "Visualiza la distribución de densidad de una variable numérica, opcionalmente agrupada."
                )

                # Usar variables seleccionadas anteriormente
                variable_densidad_default = (
                    st.session_state["variable_visualizacion"]
                    if st.session_state["variable_visualizacion"] in cols_num
                    else cols_num[0]
                )
                variable_grupo_densidad_default = (
                    st.session_state["variable_grupo_visualizacion"]
                    if st.session_state["variable_grupo_visualizacion"] in cols_cat
                    else "Ninguna"
                )

                col1, col2 = st.columns(2)
                with col1:
                    variable_densidad = st.selectbox(
                        "📊 Variable:",
                        cols_num,
                        index=cols_num.index(variable_densidad_default),
                    )

                with col2:
                    variable_grupo_densidad = st.selectbox(
                        "🏷️ Variable de agrupación (opcional):",
                        ["Ninguna"] + cols_cat,
                        index=(
                            ["Ninguna"]
                            + cols_cat.index(variable_grupo_densidad_default)
                            if variable_grupo_densidad_default in cols_cat
                            else 0
                        ),
                    )
                    if variable_grupo_densidad == "Ninguna":
                        variable_grupo_densidad = None

                # Guardar las selecciones
                st.session_state["variable_visualizacion"] = variable_densidad
                st.session_state["variable_grupo_visualizacion"] = (
                    variable_grupo_densidad if variable_grupo_densidad else "Ninguna"
                )

                if st.button("📈 Generar Diagrama de Densidad"):
                    fig_densidad = generar_diagrama_densidad(
                        df_analisis, variable_densidad, variable_grupo_densidad
                    )
                    st.pyplot(fig_densidad)

            elif tipo_visualizacion == "📊 Histograma con Densidad":
                st.subheader("📊 Histograma con Densidad")
                st.write(
                    "Combina histograma y curva de densidad para una visualización completa de la distribución."
                )

                # Usar variables seleccionadas anteriormente
                variable_hist_dens_default = (
                    st.session_state["variable_visualizacion"]
                    if st.session_state["variable_visualizacion"] in cols_num
                    else cols_num[0]
                )
                variable_grupo_hist_dens_default = (
                    st.session_state["variable_grupo_visualizacion"]
                    if st.session_state["variable_grupo_visualizacion"] in cols_cat
                    else "Ninguna"
                )

                col1, col2 = st.columns(2)
                with col1:
                    variable_hist_dens = st.selectbox(
                        "📊 Variable:",
                        cols_num,
                        index=cols_num.index(variable_hist_dens_default),
                    )

                with col2:
                    variable_grupo_hist_dens = st.selectbox(
                        "🏷️ Variable de agrupación (opcional):",
                        ["Ninguna"] + cols_cat,
                        index=(
                            ["Ninguna"]
                            + cols_cat.index(variable_grupo_hist_dens_default)
                            if variable_grupo_hist_dens_default in cols_cat
                            else 0
                        ),
                    )
                    if variable_grupo_hist_dens == "Ninguna":
                        variable_grupo_hist_dens = None

                # Guardar las selecciones
                st.session_state["variable_visualizacion"] = variable_hist_dens
                st.session_state["variable_grupo_visualizacion"] = (
                    variable_grupo_hist_dens if variable_grupo_hist_dens else "Ninguna"
                )

                if st.button("📊 Generar Histograma con Densidad"):
                    fig_hist_dens = generar_histograma_densidad(
                        df_analisis, variable_hist_dens, variable_grupo_hist_dens
                    )
                    st.pyplot(fig_hist_dens)

            elif tipo_visualizacion == "🎻 Violin Plot":
                st.subheader("🎻 Violin Plot")
                st.write(
                    "Visualiza la distribución completa de una variable numérica por grupos categóricos."
                )

                if not cols_cat:
                    st.warning(
                        "⚠️ Se necesita al menos una variable categórica para generar violin plots."
                    )
                else:
                    # Usar variables seleccionadas anteriormente
                    variable_numerica_violin_default = (
                        st.session_state["variable_visualizacion"]
                        if st.session_state["variable_visualizacion"] in cols_num
                        else cols_num[0]
                    )
                    variable_categorica_violin_default = (
                        st.session_state["variable_grupo_visualizacion"]
                        if st.session_state["variable_grupo_visualizacion"] in cols_cat
                        else cols_cat[0]
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        variable_numerica_violin = st.selectbox(
                            "📊 Variable numérica:",
                            cols_num,
                            index=cols_num.index(variable_numerica_violin_default),
                        )

                    with col2:
                        variable_categorica_violin = st.selectbox(
                            "🏷️ Variable categórica:",
                            cols_cat,
                            index=cols_cat.index(variable_categorica_violin_default),
                        )

                    # Guardar las selecciones
                    st.session_state["variable_visualizacion"] = (
                        variable_numerica_violin
                    )
                    st.session_state["variable_grupo_visualizacion"] = (
                        variable_categorica_violin
                    )

                    if st.button("🎻 Generar Violin Plot"):
                        fig_violin = generar_violin_plot(
                            df_analisis,
                            variable_numerica_violin,
                            variable_categorica_violin,
                        )
                        st.pyplot(fig_violin)

            elif tipo_visualizacion == "📊 Gráfico de Barras":
                st.subheader("📊 Gráfico de Barras")
                st.write(
                    "Visualiza frecuencias de variables categóricas o promedios de variables numéricas por grupos."
                )

                # Usar variables seleccionadas anteriormente
                variable_categorica_barras_default = (
                    st.session_state["variable_grupo_visualizacion"]
                    if st.session_state["variable_grupo_visualizacion"] in cols_cat
                    else cols_cat[0]
                )
                variable_numerica_barras_default = (
                    st.session_state["variable_visualizacion"]
                    if st.session_state["variable_visualizacion"] in cols_num
                    else "Ninguna"
                )

                col1, col2 = st.columns(2)
                with col1:
                    variable_categorica_barras = st.selectbox(
                        "🏷️ Variable categórica:",
                        cols_cat,
                        index=cols_cat.index(variable_categorica_barras_default),
                    )

                with col2:
                    variable_numerica_barras = st.selectbox(
                        "📊 Variable numérica (opcional):",
                        ["Ninguna"] + cols_num,
                        index=(
                            ["Ninguna"]
                            + cols_num.index(variable_numerica_barras_default)
                            if variable_numerica_barras_default in cols_num
                            else 0
                        ),
                    )
                    if variable_numerica_barras == "Ninguna":
                        variable_numerica_barras = None

                # Guardar las selecciones
                st.session_state["variable_grupo_visualizacion"] = (
                    variable_categorica_barras
                )
                st.session_state["variable_visualizacion"] = (
                    variable_numerica_barras if variable_numerica_barras else "Ninguna"
                )

                if st.button("📊 Generar Gráfico de Barras"):
                    fig_barras = generar_grafico_barras(
                        df_analisis,
                        variable_categorica_barras,
                        variable_numerica_barras,
                    )
                    st.pyplot(fig_barras)

            elif tipo_visualizacion == "🔥 Heatmap de Correlación Avanzado":
                st.subheader("🔥 Heatmap de Correlación Avanzado")
                st.write(
                    "Genera un heatmap de correlación con análisis adicional de las correlaciones más fuertes."
                )

                if len(cols_num) < 2:
                    st.warning(
                        "⚠️ Se necesitan al menos 2 variables numéricas para generar el heatmap."
                    )
                else:
                    # Usar variables seleccionadas anteriormente
                    variables_heatmap_default = (
                        st.session_state["variables_correlacion"]
                        if st.session_state["variables_correlacion"]
                        and all(
                            v in cols_num
                            for v in st.session_state["variables_correlacion"]
                        )
                        else cols_num[: min(6, len(cols_num))]
                    )

                    variables_heatmap = st.multiselect(
                        "🔍 Variables para el heatmap:",
                        cols_num,
                        default=variables_heatmap_default,
                        help="Selecciona las variables para el análisis de correlación",
                    )

                    # Guardar las variables seleccionadas
                    st.session_state["variables_correlacion"] = variables_heatmap

                    if len(variables_heatmap) >= 2:
                        if st.button("🔥 Generar Heatmap Avanzado"):
                            fig_heatmap_avanzado = generar_heatmap_correlacion_avanzado(
                                df_analisis, variables_heatmap
                            )
                            st.pyplot(fig_heatmap_avanzado)
                    else:
                        st.warning("⚠️ Selecciona al menos 2 variables para continuar.")

            elif tipo_visualizacion == "🔗 Matriz de Scatter Plots":
                st.subheader("🔗 Matriz de Scatter Plots")
                st.write(
                    "Genera una matriz de scatter plots para visualizar todas las relaciones entre variables numéricas."
                )

                if len(cols_num) < 2:
                    st.warning(
                        "⚠️ Se necesitan al menos 2 variables numéricas para generar la matriz."
                    )
                else:
                    # Usar variables seleccionadas anteriormente
                    variables_scatter_matrix_default = (
                        st.session_state["variables_correlacion"]
                        if st.session_state["variables_correlacion"]
                        and all(
                            v in cols_num
                            for v in st.session_state["variables_correlacion"]
                        )
                        else cols_num[: min(6, len(cols_num))]
                    )

                    variables_scatter_matrix = st.multiselect(
                        "🔍 Variables para la matriz:",
                        cols_num,
                        default=variables_scatter_matrix_default,
                        help="Selecciona hasta 6 variables para la matriz de scatter plots",
                    )

                    # Guardar las variables seleccionadas
                    st.session_state["variables_correlacion"] = variables_scatter_matrix

                    if len(variables_scatter_matrix) >= 2:
                        if st.button("🔗 Generar Matriz de Scatter Plots"):
                            fig_scatter_matrix = generar_scatter_matrix(
                                df_analisis, variables_scatter_matrix
                            )
                            st.pyplot(fig_scatter_matrix)
                    else:
                        st.warning("⚠️ Selecciona al menos 2 variables para continuar.")

    elif pagina == "🎓 Ciencias Sociales":
        st.header("🎓 Análisis Estadístico para Ciencias Sociales")
        st.write(
            "Herramientas especializadas para investigación en ciencias sociales, demografía y estudios sociales."
        )

        # Aplicar filtros si existen
        if st.session_state["filtros_aplicados"]:
            df_analisis = aplicar_filtros(df, st.session_state["filtros_aplicados"])
            stats_filtradas = obtener_estadisticas_filtradas(
                df, st.session_state["filtros_aplicados"]
            )

            st.info(
                f"📊 Analizando {stats_filtradas['n_filtrado']} de {stats_filtradas['n_original']} observaciones ({stats_filtradas['porcentaje_muestra']:.1f}% de la muestra)"
            )
        else:
            df_analisis = df
            st.info("📊 Analizando todos los datos (sin filtros aplicados)")

        # Tabs para diferentes tipos de análisis
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "🔍 Clasificación Variables",
                "📊 Análisis Descriptivo",
                "🔗 Análisis Bivariado",
                "📈 Regresión Múltiple",
                "🎯 Clustering",
                "📋 Valores Perdidos",
            ]
        )

        with tab1:
            st.subheader("🔍 Clasificación Automática de Variables")
            st.write(
                "Clasifica automáticamente las variables según su tipo y dominio en ciencias sociales."
            )

            # Mostrar clasificación de todas las variables
            clasificaciones = []
            for col in df_analisis.columns:
                clasificacion = clasificar_variable(df_analisis, col)
                clasificaciones.append(clasificacion)

            # Crear DataFrame con clasificaciones
            df_clasificaciones = pd.DataFrame(clasificaciones)

            # Mostrar tabla de clasificaciones
            st.dataframe(
                df_clasificaciones[
                    [
                        "columna",
                        "dominio",
                        "es_continua",
                        "es_categorica",
                        "es_ordinal",
                        "n_unicos",
                        "porcentaje_faltantes",
                    ]
                ]
            )

            # Filtros por dominio
            dominios_unicos = df_clasificaciones["dominio"].unique()
            dominio_seleccionado = st.selectbox(
                "🔍 Filtrar por dominio:", ["Todos"] + list(dominios_unicos)
            )

            if dominio_seleccionado != "Todos":
                df_filtrado = df_clasificaciones[
                    df_clasificaciones["dominio"] == dominio_seleccionado
                ]
                st.write(f"**Variables del dominio: {dominio_seleccionado}**")
                st.dataframe(
                    df_filtrado[
                        [
                            "columna",
                            "es_continua",
                            "es_categorica",
                            "es_ordinal",
                            "n_unicos",
                            "porcentaje_faltantes",
                        ]
                    ]
                )

        with tab2:
            st.subheader("📊 Análisis Descriptivo Especializado")
            st.write(
                "Análisis descriptivo con interpretación específica para ciencias sociales."
            )

            # Selección de variable
            variable_default = (
                st.session_state["analisis_cs_variable"]
                if st.session_state["analisis_cs_variable"] in df_analisis.columns
                else df_analisis.columns[0]
            )
            variable = st.selectbox(
                "🔍 Selecciona la variable:",
                df_analisis.columns,
                index=list(df_analisis.columns).index(variable_default),
            )

            # Guardar la selección
            st.session_state["analisis_cs_variable"] = variable

            if st.button("📊 Realizar Análisis Descriptivo"):
                with st.spinner("Analizando variable..."):
                    resultado = analisis_descriptivo_cs(df_analisis, variable)

                # Mostrar resultados
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**📋 Clasificación de la Variable**")
                    clasif = resultado["clasificacion"]
                    st.write(f"• **Dominio:** {clasif['dominio']}")
                    st.write(
                        f"• **Tipo:** {'Continua' if clasif['es_continua'] else 'Categórica'}"
                    )
                    st.write(f"• **Observaciones:** {clasif['n_total']}")
                    st.write(f"• **Valores únicos:** {clasif['n_unicos']}")
                    st.write(
                        f"• **Valores faltantes:** {clasif['valores_faltantes']} ({clasif['porcentaje_faltantes']:.1f}%)"
                    )

                with col2:
                    st.write("**📈 Estadísticas Básicas**")
                    stats = resultado["estadisticas_basicas"]
                    if clasif["es_continua"]:
                        st.write(f"• **Media:** {stats['media']:.2f}")
                        st.write(f"• **Mediana:** {stats['mediana']:.2f}")
                        st.write(f"• **Desv. Estándar:** {stats['desv_estandar']:.2f}")
                        st.write(
                            f"• **Rango:** {stats['minimo']:.2f} - {stats['maximo']:.2f}"
                        )
                        st.write(f"• **Asimetría:** {stats['asimetria']:.3f}")
                    else:
                        st.write(f"• **Moda:** {stats['moda']}")
                        st.write(f"• **Categorías:** {stats['n_categorias']}")
                        st.write(
                            f"• **Índice de diversidad:** {stats['indice_diversidad']:.3f}"
                        )

                # Interpretación
                st.write("**📝 Interpretación**")
                for key, value in resultado["interpretacion"].items():
                    st.write(f"• **{key.replace('_', ' ').title()}:** {value}")

                # Guardar en session_state para exportación
                st.session_state["datos_analisis"][
                    "analisis_descriptivo_cs"
                ] = resultado

        with tab3:
            st.subheader("🔗 Análisis Bivariado Especializado")
            st.write(
                "Análisis de relaciones entre dos variables con interpretación para ciencias sociales."
            )

            # Selección de variables
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox(
                    "🔍 Primera variable:", df_analisis.columns, index=0
                )
            with col2:
                var2 = st.selectbox(
                    "🔍 Segunda variable:",
                    [col for col in df_analisis.columns if col != var1],
                    index=0,
                )

            # Guardar selecciones
            st.session_state["analisis_cs_variables_bivariado"] = [var1, var2]

            if st.button("🔗 Realizar Análisis Bivariado"):
                with st.spinner("Analizando relación entre variables..."):
                    resultado = analisis_bivariado_cs(df_analisis, var1, var2)

                # Mostrar resultados
                st.write(f"**📊 Análisis entre {var1} y {var2}**")
                st.write(f"• **Observaciones válidas:** {resultado['n_observaciones']}")

                # Mostrar análisis específico
                if "correlacion_continua" in resultado["analisis"]:
                    analisis = resultado["analisis"]
                    st.write("**📈 Correlaciones**")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Pearson:**")
                        st.write(f"• r = {analisis['pearson']['coeficiente']:.3f}")
                        st.write(f"• p = {analisis['pearson']['p_valor']:.3f}")
                        st.write(f"• {analisis['pearson']['fuerza']}")

                    with col2:
                        st.write("**Spearman:**")
                        st.write(f"• ρ = {analisis['spearman']['coeficiente']:.3f}")
                        st.write(f"• p = {analisis['spearman']['p_valor']:.3f}")
                        st.write(f"• {analisis['spearman']['fuerza']}")

                    with col3:
                        st.write("**Kendall:**")
                        st.write(f"• τ = {analisis['kendall']['coeficiente']:.3f}")
                        st.write(f"• p = {analisis['kendall']['p_valor']:.3f}")
                        st.write(f"• {analisis['kendall']['fuerza']}")

                elif "contingencia_categorica" in resultado["analisis"]:
                    analisis = resultado["analisis"]
                    st.write("**📊 Tabla de Contingencia**")

                    # Mostrar tabla
                    tabla = pd.DataFrame(analisis["tabla_contingencia"])
                    st.dataframe(tabla)

                    st.write("**🔬 Prueba Chi-cuadrado**")
                    chi2 = analisis["chi_cuadrado"]
                    st.write(f"• **χ² = {chi2['estadistico']:.3f}**")
                    st.write(f"• **p-valor = {chi2['p_valor']:.3f}**")
                    st.write(f"• **Grados de libertad = {chi2['grados_libertad']}**")
                    st.write(f"• **Cramer's V = {analisis['cramer_v']:.3f}**")

                elif "grupos_continua" in resultado["analisis"]:
                    analisis = resultado["analisis"]
                    st.write("**📊 Análisis por Grupos**")

                    # Mostrar estadísticas por grupo
                    for grupo, stats in analisis["estadisticas_grupos"].items():
                        st.write(
                            f"**{grupo}:** n={stats['n']}, Media={stats['media']:.2f}, DE={stats['desv_estandar']:.2f}"
                        )

                    st.write("**🔬 ANOVA**")
                    anova = analisis["anova"]
                    st.write(f"• **F = {anova['f_statistico']:.3f}**")
                    st.write(f"• **p-valor = {anova['p_valor']:.3f}**")

                # Interpretación
                st.write("**📝 Interpretación**")
                for key, value in resultado["interpretacion"].items():
                    st.write(f"• **{key.replace('_', ' ').title()}:** {value}")

                # Guardar en session_state para exportación
                st.session_state["datos_analisis"]["analisis_bivariado_cs"] = resultado

        with tab4:
            st.subheader("📈 Regresión Múltiple")
            st.write("Análisis de regresión múltiple con validación de supuestos.")

            # Selección de variables
            variable_dependiente = st.selectbox(
                "🎯 Variable dependiente:", df_analisis.columns, index=0
            )

            variables_independientes = st.multiselect(
                "📊 Variables independientes:",
                [col for col in df_analisis.columns if col != variable_dependiente],
                default=st.session_state["analisis_cs_variables_regresion"],
            )

            # Guardar selecciones
            st.session_state["analisis_cs_variables_regresion"] = (
                variables_independientes
            )

            if len(variables_independientes) >= 1:
                if st.button("📈 Realizar Regresión Múltiple"):
                    with st.spinner("Calculando regresión múltiple..."):
                        resultado = analisis_regresion_multiple_cs(
                            df_analisis, variable_dependiente, variables_independientes
                        )

                    if "error" not in resultado:
                        # Mostrar resultados
                        st.write("**📊 Resultados del Modelo**")
                        st.write(f"• **R² = {resultado['r_cuadrado']:.3f}**")
                        st.write(
                            f"• **R² ajustado = {resultado['r_cuadrado_ajustado']:.3f}**"
                        )
                        st.write(
                            f"• **Observaciones = {resultado['n_observaciones']}**"
                        )
                        st.write(f"• **Variables = {resultado['n_variables']}**")

                        st.write("**📈 Coeficientes**")
                        for var, coef in resultado["coeficientes"].items():
                            st.write(
                                f"• **{var}:** {coef['coeficiente']:.3f} (estandarizado: {coef['coeficiente_estandarizado']:.3f})"
                            )

                        # Validación de supuestos
                        st.write("**🔬 Validación de Supuestos**")
                        supuestos = validar_supuestos_regresion(
                            df_analisis, variable_dependiente, variables_independientes
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Normalidad de residuos:**")
                            norm = supuestos["normalidad_residuos"]
                            st.write(f"• p-valor = {norm['p_valor']:.3f}")
                            st.write(
                                f"• Cumple: {'✅' if norm['cumple_supuesto'] else '❌'}"
                            )

                        with col2:
                            st.write("**Homocedasticidad:**")
                            hom = supuestos["homocedasticidad"]
                            if hom["p_valor"] is not None:
                                st.write(f"• p-valor = {hom['p_valor']:.3f}")
                                st.write(
                                    f"• Cumple: {'✅' if hom['cumple_supuesto'] else '❌'}"
                                )
                            else:
                                st.write("• No se pudo calcular")

                        # Guardar en session_state para exportación
                        st.session_state["datos_analisis"][
                            "regresion_multiple_cs"
                        ] = resultado
                        st.session_state["datos_analisis"][
                            "supuestos_regresion"
                        ] = supuestos
                    else:
                        st.error(f"❌ Error: {resultado['error']}")
            else:
                st.warning("⚠️ Selecciona al menos una variable independiente.")

        with tab5:
            st.subheader("🎯 Análisis de Clusters")
            st.write("Análisis de conglomerados para identificar grupos en los datos.")

            # Selección de variables
            variables_clusters = st.multiselect(
                "📊 Variables para clustering:",
                df_analisis.columns,
                default=st.session_state["analisis_cs_variables_clusters"],
            )

            # Número de clusters
            n_clusters = st.slider("🎯 Número de clusters:", 2, 10, 3)

            # Guardar selecciones
            st.session_state["analisis_cs_variables_clusters"] = variables_clusters

            if len(variables_clusters) >= 2:
                if st.button("🎯 Realizar Clustering"):
                    with st.spinner("Calculando clusters..."):
                        resultado = analisis_clusters_cs(
                            df_analisis, variables_clusters, n_clusters
                        )

                    if "error" not in resultado:
                        # Mostrar resultados
                        st.write("**📊 Resultados del Clustering**")
                        st.write(f"• **Número de clusters:** {resultado['n_clusters']}")
                        st.write(f"• **Observaciones:** {resultado['n_observaciones']}")
                        st.write(f"• **Inercia:** {resultado['inercia']:.2f}")

                        st.write("**📈 Distribución de Clusters**")
                        for cluster, stats in resultado[
                            "estadisticas_clusters"
                        ].items():
                            st.write(
                                f"• **{cluster}:** {stats['n']} observaciones ({stats['porcentaje']:.1f}%)"
                            )

                        # Mostrar características de cada cluster
                        st.write("**🔍 Características por Cluster**")
                        for cluster, stats in resultado[
                            "estadisticas_clusters"
                        ].items():
                            st.write(f"**{cluster}:**")
                            for var in variables_clusters:
                                if f"media_{var}" in stats:
                                    st.write(f"  • {var}: {stats[f'media_{var}']:.2f}")

                        # Guardar en session_state para exportación
                        st.session_state["datos_analisis"]["clustering_cs"] = resultado
                    else:
                        st.error(f"❌ Error: {resultado['error']}")
            else:
                st.warning("⚠️ Selecciona al menos 2 variables para el clustering.")

        with tab6:
            st.subheader("📋 Análisis de Valores Perdidos")
            st.write(
                "Análisis de patrones de valores perdidos y sugerencias de imputación."
            )

            # Análisis general de valores perdidos
            if st.button("📋 Analizar Valores Perdidos"):
                with st.spinner("Analizando valores perdidos..."):
                    resultado = analizar_valores_perdidos(df_analisis)

                # Mostrar resultados generales
                st.write("**📊 Resumen de Valores Perdidos**")
                st.write(
                    f"• **Total de valores perdidos:** {resultado['total_valores_perdidos']}"
                )
                st.write(
                    f"• **Porcentaje total perdido:** {resultado['porcentaje_total_perdidos']:.1f}%"
                )

                # Mostrar variables con valores perdidos
                st.write("**📈 Variables con Valores Perdidos**")
                df_perdidos = pd.DataFrame(
                    {
                        "Variable": list(resultado["conteo_por_variable"].keys()),
                        "Valores Perdidos": list(
                            resultado["conteo_por_variable"].values()
                        ),
                        "Porcentaje": list(
                            resultado["porcentajes_por_variable"].values()
                        ),
                    }
                )
                df_perdidos = df_perdidos[
                    df_perdidos["Valores Perdidos"] > 0
                ].sort_values("Valores Perdidos", ascending=False)
                st.dataframe(df_perdidos)

                # Sugerencias de imputación
                st.write("**💡 Sugerencias de Imputación**")
                for var in df_perdidos["Variable"]:
                    sugerencia = sugerir_imputacion(df_analisis, var)
                    st.write(f"**{var}:**")
                    st.write(
                        f"  • Métodos recomendados: {', '.join(sugerencia['metodos_recomendados'])}"
                    )
                    if "advertencia" in sugerencia:
                        st.write(f"  • ⚠️ {sugerencia['advertencia']}")

                # Guardar en session_state para exportación
                st.session_state["datos_analisis"]["valores_perdidos"] = resultado

    elif pagina == "🤖 Consultas en Lenguaje Natural":
        st.header("🤖 Consultas en Lenguaje Natural")
        st.write(
            "Haz preguntas en español y obtén análisis automáticos. El sistema detecta automáticamente el tipo de análisis necesario."
        )

        # Verificar si hay datos cargados
        if st.session_state["df"] is None:
            st.warning("⚠️ Por favor, carga un archivo de datos primero en la sección de carga de datos.")
            st.stop()
        
        # Definir df_analisis (aplicar filtros si existen)
        df = st.session_state["df"]
        if st.session_state["filtros_aplicados"]:
            df_analisis = aplicar_filtros(df, st.session_state["filtros_aplicados"])
        else:
            df_analisis = df

        # Sección específica para análisis temporal
        with st.expander("📈 Análisis Temporal Avanzado", expanded=True):
            st.write("**Funcionalidades disponibles para análisis temporal:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Tipos de Análisis:**")
                st.write("• Tendencia de variables en el tiempo")
                st.write("• Comparación entre variables")
                st.write("• Análisis estacional")
                st.write("• Medias móviles y suavizado")
                
                st.write("**📊 Métricas Soportadas:**")
                st.write("• Promedio, suma, contar")
                st.write("• Máximo, mínimo, mediana")
                st.write("• Desviación estándar")
            
            with col2:
                st.write("**⏰ Frecuencias:**")
                st.write("• Diario, semanal, mensual")
                st.write("• Trimestral, anual")
                st.write("• Personalizada (cada X días)")
                
                st.write("**🔧 Opciones Avanzadas:**")
                st.write("• Suavizado con medias móviles")
                st.write("• Ajuste de tendencia (lineal/polinomial)")
                st.write("• Descomposición estacional (STL)")
            
            # Ejemplos de preguntas temporales
            st.write("**💡 Ejemplos de preguntas:**")
            ejemplos_temporales = [
                "Mostrar tendencia de ventas mensual",
                "Comparar gastos y ventas cada semana",
                "Análisis de márgenes con media móvil 7 días",
                "Tendencia de ventas en el último año",
                "Descomposición estacional de ventas",
                "Análisis de gastos cada 15 días con suavizado"
            ]
            
            for ejemplo in ejemplos_temporales:
                if st.button(f"💬 {ejemplo}", key=f"ejemplo_{ejemplo}"):
                    st.session_state["pregunta_nl"] = ejemplo
                    st.rerun()
        
        # Consulta principal
        question = st.text_area(
            "💬 Escribe tu pregunta:",
            value=st.session_state.get("pregunta_nl", ""),
            placeholder="Ejemplo: 'Mostrar tendencia de ventas mensual' o '¿Cuántos registros tienen edad mayor a 30?'",
            height=100,
        )

        # Procesar consulta
        if st.button("🔍 Analizar Consulta"):
            if question.strip():
                try:
                    with st.spinner("🔍 Procesando consulta..."):
                        # Verificar si es análisis temporal
                        params_temporal = detectar_analisis_temporal(question)
                        
                        if params_temporal:
                            # Es análisis temporal
                            resultado = analizar_tendencia_temporal(df_analisis, question)
                            
                            if 'error' in resultado:
                                st.error(f"❌ Error en análisis temporal: {resultado['error']}")
                            else:
                                # Guardar resultados en session_state
                                st.session_state["nl_results"] = {
                                    "question": question,
                                    "df_filtrado": df_analisis,  # Usar datos originales
                                    "resultado": resultado,
                                    "df_grafico": resultado.get('datos_serie', pd.DataFrame()),
                                    "tipo_grafico": "tendencia_temporal",
                                    "accion": "análisis_temporal"
                                }
                                st.success("✅ Análisis temporal completado")
                        else:
                            # Es consulta normal
                            df_filtrado, resultado, df_grafico, tipo_grafico, interpretacion = parse_and_execute(
                                df_analisis, question, st.session_state.get("variable_metadata")
                            )
                            
                            # Guardar resultados en session_state
                            st.session_state["nl_results"] = {
                                "question": question,
                                "df_filtrado": df_filtrado,
                                "resultado": resultado,
                                "df_grafico": df_grafico,
                                "tipo_grafico": tipo_grafico,
                                "accion": "análisis_general"
                            }
                            st.success("✅ Consulta procesada correctamente")
                    
                except Exception as e:
                    st.error(f"❌ Error al procesar la consulta: {str(e)}")
                    st.session_state["nl_results"] = None
            else:
                st.warning("⚠️ Por favor, escribe una pregunta")
        
        # Mostrar resultados
        if "nl_results" in st.session_state and st.session_state["nl_results"] is not None:
            results = st.session_state["nl_results"]
            
            st.subheader("📊 Resultados")
            
            # Interpretación del resultado
            if results['accion'] == 'análisis_temporal':
                st.markdown("**💬 Análisis Temporal Detectado**")
                if hasattr(results['resultado'], 'insights'):
                    st.write(results['resultado']['insights'])
            else:
                st.markdown(f"**💬 {interpretar_resultado(results['accion'], results['resultado'])}**")
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                if results['accion'] == 'análisis_temporal':
                    st.metric("📈 Registros Analizados", f"{results['resultado'].get('registros_analizados', 0):,}")
                else:
                    st.metric("📈 Registros Encontrados", f"{results['resultado']:,}")
            with col2:
                st.metric("📊 Total Original", f"{len(df_analisis):,}")
            with col3:
                if results['accion'] == 'análisis_temporal':
                    porcentaje = (results['resultado'].get('registros_analizados', 0) / len(df_analisis)) * 100 if len(df_analisis) > 0 else 0
                else:
                    porcentaje = (results['resultado'] / len(df_analisis)) * 100 if len(df_analisis) > 0 else 0
                st.metric("📋 % de Muestra", f"{porcentaje:.1f}%")
            
            # Tabs para diferentes vistas
            tab1, tab2, tab3 = st.tabs(["📋 Datos Filtrados", "📊 Visualización", "🔍 Detalles"])
            
            with tab1:
                st.write("**📋 Datos que cumplen con los criterios:**")
                st.dataframe(
                    results['df_filtrado'],
                    use_container_width=True,
                    height=400
                )
                
                # Botón de descarga
                csv_data = results['df_filtrado'].to_csv(index=False)
                st.download_button(
                    label="📄 Descargar CSV",
                    data=csv_data,
                    file_name="consulta_resultados.csv",
                    mime="text/csv",
                )
            
            with tab2:
                st.write("**📊 Visualización de Resultados:**")
                
                # Verificar si es análisis temporal
                if results['tipo_grafico'] == 'tendencia_temporal':
                    # Mostrar gráfico de tendencia temporal
                    if hasattr(results['resultado'], 'grafico'):
                        st.plotly_chart(results['resultado']['grafico'], use_container_width=True)
                    
                    # Mostrar insights si están disponibles
                    if hasattr(results['resultado'], 'insights'):
                        st.markdown("**💡 Insights Automáticos:**")
                        st.write(results['resultado']['insights'])
                    
                    # Opciones adicionales para análisis temporal
                    st.subheader("🔧 Opciones Avanzadas")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Exportar a CSV
                        if st.button("📄 Exportar Análisis Temporal"):
                            try:
                                if hasattr(results['resultado'], 'datos_exportacion'):
                                    filepath = exportar_resultados_csv(
                                        results['resultado']['datos_serie'],
                                        results['resultado']['datos_exportacion']['metadatos']
                                    )
                                    if filepath:
                                        st.success(f"✅ Análisis exportado a: {filepath}")
                                        
                                        # Leer y mostrar archivo para descarga
                                        with open(filepath, 'r', encoding='utf-8') as f:
                                            csv_data = f.read()
                                        
                                        st.download_button(
                                            label="📄 Descargar CSV",
                                            data=csv_data,
                                            file_name=f"analisis_temporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                            mime="text/csv"
                                        )
                                    else:
                                        st.error("❌ Error al exportar el análisis")
                                else:
                                    st.warning("⚠️ No hay datos de exportación disponibles")
                            except Exception as e:
                                st.error(f"❌ Error en exportación: {str(e)}")
                    
                    with col2:
                        # Información sobre STL
                        if hasattr(results['resultado'], 'stl_disponible'):
                            if results['resultado']['stl_disponible']:
                                st.success("✅ Descomposición estacional disponible")
                                st.info("💡 Para ver descomposición estacional, incluye palabras como 'estacional', 'descomposición' o 'componentes' en tu pregunta")
                            else:
                                st.warning("⚠️ Descomposición estacional no disponible")
                                st.info("💡 Instala statsmodels: `pip install statsmodels`")
                    
                    # Mostrar metadatos del análisis temporal
                    if hasattr(results['resultado'], 'datos_exportacion'):
                        with st.expander("📋 Metadatos del Análisis"):
                            metadatos = results['resultado']['datos_exportacion']['metadatos']
                            st.write(f"**Tipo de análisis:** {metadatos.get('tipo_analisis', 'N/A')}")
                            st.write(f"**Métrica:** {metadatos.get('metrica', 'N/A')}")
                            st.write(f"**Frecuencia:** {metadatos.get('frecuencia', 'N/A')}")
                            if metadatos.get('frecuencia_personalizada'):
                                st.write(f"**Frecuencia personalizada:** {metadatos['frecuencia_personalizada']}")
                            st.write(f"**Variables analizadas:** {', '.join(metadatos.get('variables_analizadas', []))}")
                            st.write(f"**Registros analizados:** {metadatos.get('registros_analizados', 0):,}")
                            st.write(f"**Fecha de análisis:** {metadatos.get('fecha_analisis', 'N/A')}")
                            
                            # Información de suavizado
                            if metadatos.get('suavizado'):
                                suavizado = metadatos['suavizado']
                                if suavizado.get('aplicar'):
                                    st.write(f"**Suavizado aplicado:** {suavizado.get('tipo', 'N/A')} con ventana {suavizado.get('ventana', 'N/A')}")
                            
                            # Información de descomposición
                            if metadatos.get('descomposicion'):
                                descomp = metadatos['descomposicion']
                                if descomp.get('aplicar'):
                                    st.write(f"**Descomposición estacional:** Período {descomp.get('periodo', 'N/A')}")
                
                elif results['tipo_grafico'] == 'serie_tiempo':
                    import plotly.express as px
                    fig = px.line(
                        results['df_grafico'], 
                        x=results['df_grafico'].columns[0], 
                        y='conteo',
                        title="Evolución Temporal",
                        labels={'conteo': 'Conteo', 'index': 'Fecha'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif results['tipo_grafico'] == 'barras':
                    import plotly.express as px
                    fig = px.bar(
                        results['df_grafico'],
                        x='categoria',
                        y='conteo',
                        title="Distribución por Categorías",
                        labels={'conteo': 'Conteo', 'categoria': 'Categoría'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif results['tipo_grafico'] == 'tabla':
                    st.dataframe(results['df_grafico'])
                    
                else:
                    st.write("No se pudo generar una visualización específica.")
            
            with tab3:
                st.write("**🔍 Detalles de la Consulta:**")
                st.write(f"**Pregunta:** {results['question']}")
                st.write(f"**Tipo de gráfico generado:** {results['tipo_grafico']}")
                st.write(f"**Columnas en el resultado:** {list(results['df_filtrado'].columns)}")
                
                # Mostrar filtros aplicados (si los hay)
                if len(results['df_filtrado']) < len(df_analisis):
                    st.write("**Filtros aplicados:** Se aplicaron filtros automáticos basados en tu pregunta.")
                else:
                    st.write("**Filtros aplicados:** Ninguno (se analizaron todos los datos).")

    elif pagina == "📤 Exportar Resultados":
        st.header("📤 Exportar Resultados Completos")
        st.write("Genera reportes completos con todos los análisis realizados.")

        # Verificar si hay datos de análisis disponibles
        if not st.session_state["datos_analisis"]:
            st.warning(
                "⚠️ No hay análisis disponibles para exportar. Realiza algunos análisis primero."
            )
        else:
            st.subheader("📋 Resumen de Análisis Disponibles")

            analisis_disponibles = []
            if "estadisticas_descriptivas" in st.session_state["datos_analisis"]:
                analisis_disponibles.append("📈 Estadísticas Descriptivas")

            if "correlaciones" in st.session_state["datos_analisis"]:
                analisis_disponibles.append("🔗 Análisis de Correlaciones")

            if "tablas_contingencia" in st.session_state["datos_analisis"]:
                analisis_disponibles.append("📊 Tablas de Contingencia")

            for analisis in analisis_disponibles:
                st.write(f"✅ {analisis}")

            st.subheader("📤 Opciones de Exportación")

            # Exportar Excel completo
            col1, col2 = st.columns(2)

            with col1:
                st.write("**📊 Reporte Excel Completo**")
                st.write(
                    "Incluye todas las hojas con datos filtrados, estadísticas, correlaciones y tablas de contingencia."
                )

                excel_completo = generar_excel_completo(
                    df,
                    st.session_state["filtros_aplicados"],
                    st.session_state["datos_analisis"].get("estadisticas_descriptivas"),
                    st.session_state["datos_analisis"].get("correlaciones"),
                    st.session_state["datos_analisis"].get("tablas_contingencia"),
                )

                st.download_button(
                    label="📊 Descargar Excel Completo",
                    data=excel_completo,
                    file_name="reporte_analisis_completo.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            with col2:
                st.write("**📄 Reporte HTML**")
                st.write(
                    "Genera un reporte HTML formateado con todos los análisis y resultados."
                )

                html_reporte = generar_html_reporte(
                    df,
                    st.session_state["filtros_aplicados"],
                    st.session_state["datos_analisis"].get("estadisticas_descriptivas"),
                    st.session_state["datos_analisis"].get("correlaciones"),
                    st.session_state["datos_analisis"].get("tablas_contingencia"),
                )

                st.download_button(
                    label="📄 Descargar HTML",
                    data=html_reporte,
                    file_name="reporte_analisis.html",
                    mime="text/html",
                )

            # Información adicional
            st.subheader("ℹ️ Información sobre los Formatos")

            with st.expander("📊 Formato Excel"):
                st.write(
                    """
                **Ventajas del formato Excel:**
                - Múltiples hojas organizadas
                - Fácil de manipular y analizar
                - Compatible con la mayoría de software estadístico
                - Incluye todos los datos y resultados
                """
                )

            with st.expander("📄 Formato HTML"):
                st.write(
                    """
                **Ventajas del formato HTML:**
                - Formato profesional y legible
                - Fácil de compartir por email
                - Se puede abrir en cualquier navegador
                - Incluye interpretaciones y guías
                """
                )

            # Botón para limpiar datos de análisis
            if st.button("🗑️ Limpiar Datos de Análisis"):
                st.session_state["datos_analisis"] = {}
                st.rerun()
