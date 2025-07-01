# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os
import time
from datetime import datetime

# Importar el pipeline demográfico
from proyecto_j.src.pipeline_demografico import (
    PipelineDemografico,
    create_streamlit_pipeline,
    load_data_with_progress,
    run_pipeline_with_progress,
)

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="📊 Pipeline Demográfico Modular - Proyecto J",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;500;600;700&display=swap');
    
    html, body {
        color-scheme: light !important;
        background-color: #FBF7F2 !important;
        color: #333333 !important;
    }
    
    :root {
        --color-fondo-general: #FBF7F2;
        --color-azul-claro: #C7DCE5;
        --color-azul-profundo: #648DA5;
        --color-texto-principal: #2C3E50;
        --color-sombra: rgba(0, 0, 0, 0.08);
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
        padding: 24px;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        color: #FFFFFF !important;
    }
    
    .css-1d391kg p, .css-1d391kg div, .css-1d391kg span {
        color: #CCCCCC !important;
    }
    
    h1 {
        font-family: 'Raleway', sans-serif;
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--color-azul-profundo);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px var(--color-sombra);
        border-left: 4px solid var(--color-azul-profundo);
    }
    
    .success-card {
        border-left-color: #66BB6A;
    }
    
    .warning-card {
        border-left-color: #FFA726;
    }
    
    .error-card {
        border-left-color: #EF5350;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# INICIALIZACIÓN DE SESSION STATE
# ============================================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "data" not in st.session_state:
    st.session_state.data = None

if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None

if "transformed_data" not in st.session_state:
    st.session_state.transformed_data = None

if "models" not in st.session_state:
    st.session_state.models = {}

if "figures" not in st.session_state:
    st.session_state.figures = {}

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================


def guardar_archivo_temporal(uploaded_file):
    """Guarda el archivo subido en una ubicación temporal."""
    try:
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)

        temp_path = temp_dir / f"upload_{int(time.time())}_{uploaded_file.name}"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return str(temp_path)
    except Exception as e:
        st.error(f"Error al guardar archivo: {e}")
        return None


def mostrar_estadisticas_descriptivas(data, target_column=None):
    """Muestra estadísticas descriptivas de los datos."""
    st.subheader("📈 Estadísticas Descriptivas")

    if target_column and target_column in data.columns:
        # Estadísticas de la variable objetivo
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Estadísticas de '{target_column}':**")
            stats = data[target_column].describe()
            st.dataframe(stats, use_container_width=True)

        with col2:
            # Gráfico de distribución
            fig = px.histogram(
                data, x=target_column, title=f"Distribución de {target_column}"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Estadísticas generales
    st.write("**Estadísticas generales del dataset:**")
    st.dataframe(data.describe(), use_container_width=True)


def mostrar_informacion_columnas(data):
    """Muestra información detallada de las columnas."""
    st.subheader("📋 Información de Columnas")

    # Crear DataFrame con información de columnas
    column_info = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        missing = data[col].isnull().sum()
        missing_pct = (missing / len(data)) * 100
        unique = data[col].nunique()

        if data[col].dtype in ["object", "category"]:
            sample_values = ", ".join(data[col].dropna().unique()[:3].astype(str))
        else:
            sample_values = f"Min: {data[col].min():.2f}, Max: {data[col].max():.2f}"

        column_info.append(
            {
                "Columna": col,
                "Tipo": dtype,
                "Valores Faltantes": missing,
                "% Faltantes": f"{missing_pct:.1f}%",
                "Valores Únicos": unique,
                "Muestra": sample_values,
            }
        )

    st.dataframe(pd.DataFrame(column_info), use_container_width=True)


def mostrar_correlaciones(data):
    """Muestra matriz de correlación para variables numéricas."""
    numeric_data = data.select_dtypes(include=[np.number])

    if len(numeric_data.columns) > 1:
        st.subheader("🔗 Matriz de Correlación")

        corr_matrix = numeric_data.corr()
        fig = px.imshow(
            corr_matrix,
            title="Matriz de Correlación",
            color_continuous_scale="RdBu",
            aspect="auto",
        )
        fig.update_layout(xaxis_title="Variables", yaxis_title="Variables")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Se necesitan al menos 2 variables numéricas para mostrar correlaciones."
        )


def mostrar_resultados_modelos(models):
    """Muestra resultados de los modelos ajustados."""
    if not models:
        return

    st.subheader("🤖 Resultados de Modelos")

    # Crear tabla de comparación
    model_results = []
    for name, info in models.items():
        model_results.append(
            {
                "Modelo": name.title(),
                "R²": f"{info['r2']:.4f}",
                "RMSE": f"{info['rmse']:.4f}",
                "Estado": "✅ Ajustado",
            }
        )

    st.dataframe(pd.DataFrame(model_results), use_container_width=True)

    # Gráfico de comparación de R²
    fig = px.bar(
        x=[r["Modelo"] for r in model_results],
        y=[float(r["R²"]) for r in model_results],
        title="Comparación de R² entre Modelos",
        labels={"x": "Modelo", "y": "R²"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.title("📊 Pipeline Demográfico Modular")

# Sidebar con información
st.sidebar.title("📋 Información")
st.sidebar.markdown(
    """
### ¿Cómo funciona?

1. **Sube tu archivo** (CSV, Excel, SPSS, STATA)
2. **Selecciona la variable objetivo** para análisis
3. **Configura las opciones** de limpieza y transformación
4. **Ejecuta el análisis** completo
5. **Explora los resultados** y visualizaciones

### Formatos Soportados:
- ✅ CSV (.csv)
- ✅ Excel (.xlsx, .xls)
- ✅ SPSS (.sav)
- ✅ STATA (.dta)

### Características:
- ✅ Limpieza automática de datos
- ✅ Imputación inteligente de valores faltantes
- ✅ Análisis estadístico completo
- ✅ Modelado predictivo
- ✅ Visualizaciones interactivas
- ✅ Generación de reportes PDF
"""
)

# ============================================================================
# SECCIÓN DE CARGA DE ARCHIVOS
# ============================================================================

st.header("📁 Carga de Archivos")

# File uploader con múltiples formatos
uploaded_file = st.file_uploader(
    "Selecciona un archivo de datos para procesar",
    type=["csv", "xlsx", "xls", "sav", "dta"],
    help="Soporta CSV, Excel, SPSS y STATA",
)

if uploaded_file is not None:
    # Mostrar información del archivo
    file_size = uploaded_file.size / (1024 * 1024)  # MB

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre", uploaded_file.name)
    with col2:
        st.metric("Tamaño", f"{file_size:.1f} MB")
    with col3:
        st.metric("Tipo", uploaded_file.type or "Desconocido")

    # Validar tamaño
    if file_size > 500:
        st.error("❌ El archivo es demasiado grande. Máximo 500 MB.")
        st.stop()

    # Guardar archivo temporal y cargar datos
    if st.session_state.data is None:
        with st.spinner("Cargando datos..."):
            temp_path = guardar_archivo_temporal(uploaded_file)
            if temp_path:
                try:
                    # Crear pipeline y cargar datos
                    pipeline = PipelineDemografico()
                    data = pipeline.load_data(temp_path)

                    st.session_state.pipeline = pipeline
                    st.session_state.data = data

                    st.success(
                        f"✅ Datos cargados exitosamente: {data.shape[0]} filas, {data.shape[1]} columnas"
                    )

                except Exception as e:
                    st.error(f"❌ Error al cargar el archivo: {e}")
                    st.stop()

    # Mostrar datos cargados
    if st.session_state.data is not None:
        data = st.session_state.data

        # Vista previa
        st.subheader("👀 Vista Previa de Datos")
        st.dataframe(data.head(10), use_container_width=True)

        # Información básica
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Filas", f"{len(data):,}")
        with col2:
            st.metric("Columnas", len(data.columns))
        with col3:
            st.metric("Valores Faltantes", f"{data.isnull().sum().sum():,}")
        with col4:
            st.metric(
                "Memoria", f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
            )

        # Mostrar información de columnas
        mostrar_informacion_columnas(data)

        # Selección de variable objetivo
        st.subheader("🎯 Selección de Variable Objetivo")

        # Filtrar columnas numéricas para variable objetivo
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_columns:
            target_column = st.selectbox(
                "Selecciona la variable objetivo para el análisis:",
                numeric_columns,
                help="Esta será la variable que se intentará predecir o analizar",
            )

            # Configuración del análisis
            st.subheader("⚙️ Configuración del Análisis")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Opciones de Limpieza:**")
                handle_missing = st.selectbox(
                    "Manejo de valores faltantes:",
                    ["auto", "drop", "impute", "llm"],
                    help="auto: automático, drop: eliminar, impute: imputar, llm: imputación avanzada",
                )

                remove_duplicates = st.checkbox("Eliminar duplicados", value=True)
                handle_outliers = st.checkbox("Manejar outliers", value=True)

            with col2:
                st.write("**Opciones de Transformación:**")
                categorical_handling = st.selectbox(
                    "Manejo de variables categóricas:",
                    ["auto", "label", "onehot", "drop"],
                    help="auto: automático, label: codificación, onehot: variables dummy, drop: eliminar",
                )

                normalize = st.checkbox("Normalizar variables numéricas", value=False)

                model_types = st.multiselect(
                    "Tipos de modelos a ajustar:",
                    ["linear", "polynomial", "exponential"],
                    default=["linear"],
                    help="Selecciona los tipos de modelos que quieres ajustar",
                )

            # Botón para ejecutar análisis completo
            st.subheader("🚀 Ejecutar Análisis Completo")

            if st.button(
                "📊 Ejecutar Pipeline Completo",
                type="primary",
                use_container_width=True,
            ):
                if st.session_state.pipeline and st.session_state.data is not None:
                    # Crear barra de progreso
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        pipeline = st.session_state.pipeline
                        data = st.session_state.data

                        # Paso 1: Limpieza
                        status_text.text("Limpiando datos...")
                        progress_bar.progress(20)

                        cleaned_data = pipeline.clean_data(
                            handle_missing=handle_missing,
                            remove_duplicates=remove_duplicates,
                            handle_outliers=handle_outliers,
                        )
                        st.session_state.cleaned_data = cleaned_data

                        # Paso 2: Transformación
                        status_text.text("Transformando datos...")
                        progress_bar.progress(40)

                        transformed_data = pipeline.transform_data(
                            categorical_handling=categorical_handling,
                            normalize=normalize,
                            target_column=target_column,
                        )
                        st.session_state.transformed_data = transformed_data

                        # Paso 3: Ajuste de modelos
                        status_text.text("Ajustando modelos...")
                        progress_bar.progress(60)

                        models = pipeline.fit_models(
                            target_column=target_column, model_types=model_types
                        )
                        st.session_state.models = models

                        # Paso 4: Visualizaciones
                        status_text.text("Creando visualizaciones...")
                        progress_bar.progress(80)

                        figures = pipeline.create_visualizations(target_column)
                        st.session_state.figures = figures

                        # Paso 5: Completado
                        status_text.text("¡Análisis completado!")
                        progress_bar.progress(100)

                        st.session_state.analysis_complete = True

                        st.success("🎉 ¡Análisis completado exitosamente!")

                    except Exception as e:
                        st.error(f"❌ Error durante el análisis: {e}")
                        progress_bar.empty()
                        status_text.empty()
                else:
                    st.error("❌ No hay datos cargados para analizar")
        else:
            st.warning(
                "⚠️ No se encontraron columnas numéricas en el dataset. El análisis predictivo requiere variables numéricas."
            )

# ============================================================================
# SECCIÓN DE RESULTADOS
# ============================================================================

if st.session_state.analysis_complete:
    st.header("📈 Resultados del Análisis")

    # Información del dataset limpio
    if st.session_state.cleaned_data is not None:
        cleaned_data = st.session_state.cleaned_data

        st.subheader("🧹 Datos Limpios")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas Originales", f"{len(st.session_state.data):,}")
        with col2:
            st.metric("Filas Limpias", f"{len(cleaned_data):,}")
        with col3:
            st.metric("Valores Faltantes", f"{cleaned_data.isnull().sum().sum():,}")

        # Estadísticas descriptivas
        mostrar_estadisticas_descriptivas(
            cleaned_data, target_column if "target_column" in locals() else None
        )

        # Correlaciones
        mostrar_correlaciones(cleaned_data)

        # Resultados de modelos
        if st.session_state.models:
            mostrar_resultados_modelos(st.session_state.models)

        # Visualizaciones
        if st.session_state.figures:
            st.subheader("📊 Visualizaciones")

            # Crear pestañas para diferentes visualizaciones
            tab_names = list(st.session_state.figures.keys())
            tabs = st.tabs([name.title() for name in tab_names])

            for i, (name, fig) in enumerate(st.session_state.figures.items()):
                with tabs[i]:
                    st.plotly_chart(fig, use_container_width=True)

        # Generación de reporte
        st.subheader("📄 Generar Reporte")

        if st.button("📋 Generar Reporte PDF", type="secondary"):
            if st.session_state.pipeline and "target_column" in locals():
                with st.spinner("Generando reporte PDF..."):
                    try:
                        report_path = st.session_state.pipeline.generate_report(
                            target_column=target_column,
                            output_path=f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        )

                        if report_path and os.path.exists(report_path):
                            st.success(f"✅ Reporte generado: {report_path}")

                            # Botón para descargar
                            with open(report_path, "rb") as file:
                                st.download_button(
                                    label="📥 Descargar Reporte PDF",
                                    data=file.read(),
                                    file_name=f"reporte_demografico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                )
                        else:
                            st.error("❌ Error al generar el reporte")
                    except Exception as e:
                        st.error(f"❌ Error al generar reporte: {e}")

        # Botón para limpiar y empezar de nuevo
        if st.button("🔄 Nuevo Análisis"):
            # Limpiar session state
            for key in [
                "pipeline",
                "data",
                "cleaned_data",
                "transformed_data",
                "models",
                "figures",
                "analysis_complete",
            ]:
                if key in st.session_state:
                    del st.session_state[key]

            # Limpiar archivos temporales
            temp_dir = Path("./temp")
            if temp_dir.exists():
                for file in temp_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass

            st.rerun()

# ============================================================================
# SECCIÓN DE AYUDA
# ============================================================================

with st.expander("❓ ¿Necesitas ayuda?"):
    st.markdown(
        """
    ### Preguntas Frecuentes
    
    **Q: ¿Qué tipos de archivos puedo procesar?**
    A: CSV, Excel (.xlsx, .xls), SPSS (.sav), STATA (.dta)
    
    **Q: ¿Cuál es el tamaño máximo de archivo?**
    A: 500 MB para garantizar un procesamiento eficiente.
    
    **Q: ¿Qué hace el pipeline automáticamente?**
    A: Limpieza de datos, imputación de valores faltantes, manejo de outliers, transformación de variables categóricas, ajuste de modelos predictivos y generación de visualizaciones.
    
    **Q: ¿Qué modelos se ajustan?**
    A: Regresión lineal, polinomial y exponencial según la configuración seleccionada.
    
    **Q: ¿Se guardan mis datos permanentemente?**
    A: No, los archivos temporales se eliminan al limpiar la sesión.
    
    **Q: ¿Puedo descargar los resultados?**
    A: Sí, se genera un reporte PDF con todos los análisis y visualizaciones.
    
    ### Características Avanzadas
    - **Imputación LLM:** Usa modelos de machine learning para imputar valores faltantes
    - **Manejo de Outliers:** Detecta y ajusta valores atípicos automáticamente
    - **Visualizaciones Interactivas:** Gráficos interactivos con Plotly
    - **Análisis de Correlación:** Matriz de correlación automática
    - **Comparación de Modelos:** Métricas de rendimiento comparativas
    """
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        📊 Pipeline Demográfico Modular - Proyecto J | 
        Desarrollado con Streamlit y Análisis Estadístico Avanzado
    </div>
    """,
    unsafe_allow_html=True,
)
