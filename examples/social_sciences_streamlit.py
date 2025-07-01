"""
Aplicación Streamlit para Análisis de Ciencias Sociales
======================================================

Interfaz web para el análisis especializado de datos de ciencias sociales,
encuestas y estudios demográficos.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from social_sciences_analyzer import SocialSciencesAnalyzer, analyze_survey_data

# Configuración de la página
    st.set_page_config(
    page_title="Análisis de Ciencias Sociales", page_icon="🎓", layout="wide"
    )
    
# Título principal
st.title("🎓 Analizador Avanzado de Datos de Ciencias Sociales")
    st.markdown("---")

# Inicializar session state
if "analyzer" not in st.session_state:
    st.session_state.analyzer = SocialSciencesAnalyzer()
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None


def main():
    """Función principal de la aplicación."""

    # Sidebar para navegación
    st.sidebar.title("📊 Navegación")
    page = st.sidebar.selectbox(
        "Selecciona la sección:",
        [
            "📁 Cargar Datos",
            "🔍 Análisis Semántico",
            "📏 Escalas Likert",
            "🔄 Normalización",
            "✅ Validación",
            "📈 Visualizaciones",
            "📤 Exportar",
        ],
    )

    if page == "📁 Cargar Datos":
        load_data_page()
    elif page == "🔍 Análisis Semántico":
        semantic_analysis_page()
    elif page == "📏 Escalas Likert":
        likert_scales_page()
    elif page == "🔄 Normalización":
        normalization_page()
    elif page == "✅ Validación":
        validation_page()
    elif page == "📈 Visualizaciones":
        visualizations_page()
    elif page == "📤 Exportar":
        export_page()


def load_data_page():
    """Página para cargar datos."""
    st.header("📁 Cargar Datos")
    
    uploaded_file = st.file_uploader(
        "Selecciona un archivo de datos:",
        type=["csv", "xlsx", "xls"],
        help="Soporta archivos CSV, Excel (.xlsx, .xls)",
    )
    
    if uploaded_file is not None:
        try:
            # Cargar datos según el tipo de archivo
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.uploaded_data = df

            # Mostrar información básica
            st.success(
                f"✅ Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas"
            )

            # Resumen de datos
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Filas", df.shape[0])
            with col2:
                st.metric("Columnas", df.shape[1])
            with col3:
                st.metric("Valores faltantes", df.isnull().sum().sum())

            # Mostrar primeras filas
            st.subheader("📋 Vista previa de los datos")
            st.dataframe(df.head())

            # Información de tipos de datos
            st.subheader("🔍 Tipos de datos")
            dtype_info = pd.DataFrame(
                {
                    "Columna": df.columns,
                    "Tipo": df.dtypes,
                    "Valores únicos": df.nunique(),
                    "Valores faltantes": df.isnull().sum(),
                }
            )
            st.dataframe(dtype_info)

        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {str(e)}")


def semantic_analysis_page():
    """Página de análisis semántico."""
    st.header("🔍 Análisis Semántico")

    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Por favor carga datos primero en la sección 'Cargar Datos'")
        return

    df = st.session_state.uploaded_data

    # Configuración del análisis
    st.subheader("⚙️ Configuración")
    confidence_threshold = st.slider(
        "Umbral de confianza para clasificación:",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Confianza mínima para clasificar una columna",
    )

    if st.button("🔍 Ejecutar Análisis Semántico"):
        with st.spinner("Analizando datos..."):
            # Realizar análisis semántico
            results = st.session_state.analyzer.analyze_survey_data(df)
            st.session_state.analysis_results = results

            # Mostrar resultados
            st.success("✅ Análisis semántico completado")

            # Clasificación semántica
            st.subheader("📊 Clasificación Semántica")
            classification = results["semantic_classification"]

            # Crear DataFrame para mostrar resultados
            classification_df = pd.DataFrame(
                [
                    {"Columna": col, "Clasificación": cls}
                    for col, cls in classification.items()
                ]
            )

            st.dataframe(classification_df)

            # Estadísticas de clasificación
            st.subheader("📈 Estadísticas de Clasificación")
            cls_counts = pd.Series(classification.values()).value_counts()

    fig = px.bar(
                x=cls_counts.index,
                y=cls_counts.values,
                title="Distribución de Clasificaciones Semánticas",
                labels={"x": "Tipo de Clasificación", "y": "Cantidad"},
            )
            st.plotly_chart(fig)


def likert_scales_page():
    """Página de detección de escalas Likert."""
    st.header("📏 Detección de Escalas Likert")

    if st.session_state.analysis_results is None:
        st.warning("⚠️ Por favor ejecuta el análisis semántico primero")
        return
    
    results = st.session_state.analysis_results
    likert_scales = results.get("likert_scales", {})

    if not likert_scales:
        st.info("ℹ️ No se detectaron escalas Likert en los datos")
        return
    
    st.success(f"✅ Se detectaron {len(likert_scales)} columnas con escalas Likert")

    # Mostrar escalas detectadas
    for col, scales in likert_scales.items():
        with st.expander(f"📏 {col} - Escalas detectadas"):
            st.write("**Escalas encontradas:**")
            for scale in scales:
                st.write(f"• {scale}")

            # Mostrar distribución de valores
            df = st.session_state.uploaded_data
            if col in df.columns:
                value_counts = df[col].value_counts()
                    
                    fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Distribución de valores en {col}",
                    labels={"x": "Valor", "y": "Frecuencia"},
                )
                st.plotly_chart(fig)


def normalization_page():
    """Página de normalización de categorías."""
    st.header("🔄 Normalización de Categorías")

    if st.session_state.analysis_results is None:
        st.warning("⚠️ Por favor ejecuta el análisis semántico primero")
        return
    
    results = st.session_state.analysis_results
    normalization = results.get("normalized_categories", {})

    if not normalization:
        st.info("ℹ️ No se encontraron categorías para normalizar")
        return

    st.success(
        f"✅ Se encontraron {len(normalization)} columnas con categorías para normalizar"
    )

    # Configuración de normalización
    st.subheader("⚙️ Configuración de Normalización")
    similarity_threshold = st.slider(
        "Umbral de similitud:",
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.1,
        help="Umbral de similitud para agrupar categorías",
    )

    # Mostrar mapeos de normalización
    for col, mapping in normalization.items():
        with st.expander(f"🔄 {col} - Mapeos de normalización"):
            st.write("**Mapeos encontrados:**")
            for original, normalized in mapping.items():
                st.write(f"• '{original}' → '{normalized}'")


def validation_page():
    """Página de validación de consistencia."""
    st.header("✅ Validación de Consistencia")

    if st.session_state.analysis_results is None:
        st.warning("⚠️ Por favor ejecuta el análisis semántico primero")
        return

    results = st.session_state.analysis_results
    inconsistencies = results.get("consistency_validation", {})

    if not inconsistencies:
        st.success("✅ No se encontraron inconsistencias en los datos")
        return

    st.warning(f"⚠️ Se encontraron {len(inconsistencies)} tipos de inconsistencias")

    # Mostrar inconsistencias
    for issue_type, issues in inconsistencies.items():
        with st.expander(f"❌ {issue_type}"):
            st.write(f"**Problemas encontrados ({len(issues)}):**")
            for issue in issues:
                st.write(f"• {issue}")


def visualizations_page():
    """Página de visualizaciones sugeridas."""
    st.header("📈 Visualizaciones Sugeridas")

    if st.session_state.analysis_results is None:
        st.warning("⚠️ Por favor ejecuta el análisis semántico primero")
        return

            results = st.session_state.analysis_results
    suggestions = results.get("visualization_suggestions", {})

    if not suggestions:
        st.info("ℹ️ No hay sugerencias de visualización disponibles")
        return

    st.success(f"✅ Se generaron sugerencias para {len(suggestions)} columnas")

    # Mostrar sugerencias por columna
    for col, viz_suggestions in suggestions.items():
        with st.expander(f"📊 {col} - Visualizaciones sugeridas"):
            st.write("**Visualizaciones recomendadas:**")
            for viz in viz_suggestions:
                st.write(f"• {viz}")

            # Crear visualización de ejemplo
            df = st.session_state.uploaded_data
            if col in df.columns:
                try:
                    if viz_suggestions and "histogram" in viz_suggestions:
                        fig = px.histogram(df, x=col, title=f"Histograma de {col}")
                        st.plotly_chart(fig)
                    elif viz_suggestions and "bar" in viz_suggestions:
                        value_counts = df[col].value_counts()
                        fig = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            title=f"Distribución de {col}",
                        )
                        st.plotly_chart(fig)
                except Exception as e:
                    st.warning(f"No se pudo crear visualización: {str(e)}")


def export_page():
    """Página de exportación de resultados."""
    st.header("📤 Exportar Resultados")

    if st.session_state.analysis_results is None:
        st.warning("⚠️ No hay resultados para exportar")
        return

    results = st.session_state.analysis_results

    # Opciones de exportación
    st.subheader("📋 Seleccionar datos para exportar")

    export_options = st.multiselect(
        "Selecciona qué exportar:",
        [
            "Clasificación Semántica",
            "Escalas Likert",
            "Normalización",
            "Validación de Consistencia",
            "Sugerencias de Visualización",
        ],
        default=["Clasificación Semántica"],
    )

    if st.button("📤 Exportar Resultados"):
        # Crear DataFrame con resultados seleccionados
        export_data = {}

        if "Clasificación Semántica" in export_options:
            classification = results.get("semantic_classification", {})
            export_data["clasificacion_semantica"] = pd.DataFrame(
                [
                    {"columna": col, "clasificacion": cls}
                    for col, cls in classification.items()
                ]
            )

        if "Escalas Likert" in export_options:
            likert_scales = results.get("likert_scales", {})
            likert_data = []
            for col, scales in likert_scales.items():
                for scale in scales:
                    likert_data.append({"columna": col, "escala": scale})
            if likert_data:
                export_data["escalas_likert"] = pd.DataFrame(likert_data)

        # Exportar como CSV
        if export_data:
            csv_data = pd.concat(export_data.values(), keys=export_data.keys(), axis=0)
                st.download_button(
                label="📄 Descargar CSV",
                data=csv_data.to_csv(index=False),
                file_name="analisis_ciencias_sociales.csv",
                mime="text/csv",
            )

            st.success("✅ Resultados listos para descargar")


if __name__ == "__main__":
    main()
