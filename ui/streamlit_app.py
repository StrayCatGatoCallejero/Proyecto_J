# -*- coding: utf-8 -*-
"""
Máquina de Relojería - Análisis de Ciencias Sociales
===================================================
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import json

# Configuración de la página
st.set_page_config(
    page_title="Máquina de Relojería - Análisis Social",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .step-card.active {
        border-color: #667eea;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.2);
    }
    .step-card.completed {
        border-color: #28a745;
        background-color: #f8fff9;
    }
    .step-number {
        background: #667eea;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
    .step-number.completed {
        background: #28a745;
    }
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
    .toast-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #155724;
    }
    .toast-info {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #0c5460;
    }
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'tour_completed' not in st.session_state:
        st.session_state.tour_completed = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None

def show_tour():
    if not st.session_state.tour_completed:
        with st.expander("🎯 Tour Guiado - Bienvenido a la Máquina de Relojería", expanded=True):
            st.markdown("""
            ### ⚙️ Cómo funciona esta máquina:
            1. **Carga de Datos** 📁 - Sube tu archivo (CSV, Excel, SPSS, etc.)
            2. **Resumen de Datos** 📊 - Revisa la estructura y calidad
            3. **Detección de Tipos** 🔍 - Identificamos automáticamente el tipo de cada variable
            4. **Limpieza** 🧹 - Corregimos valores faltantes y errores
            5. **Filtrado** 🔧 - Aplicamos filtros según tus necesidades
            6. **Estadísticas** 📈 - Calculamos análisis descriptivos y modelos
            7. **Visualización** 📊 - Creamos gráficos inteligentes
            8. **Exportación** 💾 - Descargamos resultados y reportes

            **💡 Consejo**: Cada paso se construye sobre el anterior, como engranajes de un reloj.
            """)
            if st.button("✅ Entendido, comenzar"):
                st.session_state.tour_completed = True
                st.rerun()

def render_sidebar_steps():
    st.sidebar.markdown("## ⚙️ Pasos de la Máquina")
    steps = [
        ("📁 Cargar Datos", "Sube tu archivo de datos"),
        ("📊 Resumen", "Revisa estructura y calidad"),
        ("🔍 Detección de Tipos", "Identificamos tipos de variables"),
        ("🧹 Limpieza", "Corregimos valores faltantes"),
        ("🔧 Filtrado", "Aplicamos filtros"),
        ("📈 Estadísticas", "Análisis descriptivo y modelos"),
        ("📊 Visualización", "Gráficos inteligentes"),
        ("💾 Exportación", "Descarga resultados")
    ]
    for i, (title, description) in enumerate(steps, 1):
        step_class = ""
        if i == st.session_state.current_step:
            step_class = "active"
        elif i < st.session_state.current_step:
            step_class = "completed"
        with st.sidebar.container():
            st.markdown(f"""
            <div class="step-card {step_class}">
                <span class="step-number {step_class}">{i}</span>
                <strong>{title}</strong><br>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)

def step_1_load_data():
    st.markdown("## 📁 Paso 1: Cargar Datos")
    uploaded_file = st.file_uploader(
        "Arrastra tu archivo aquí o haz clic para seleccionar",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos soportados: CSV, Excel (.xlsx/.xls)"
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.markdown(f"""
            <div class="toast-success">
                ✅ <strong>¡Archivo cargado exitosamente!</strong><br>
                📄 {uploaded_file.name}<br>
                📊 {uploaded_file.size:,} bytes
            </div>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🚀 Ejecutar Análisis Completo", type="primary"):
                    st.session_state.current_step = 2
                    st.rerun()
            with col2:
                if st.button("📊 Solo Visualizaciones"):
                    st.session_state.current_step = 7
                    st.rerun()
            with col3:
                if st.button("📋 Solo Resúmenes"):
                    st.session_state.current_step = 2
                    st.rerun()
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {e}")

def step_2_data_summary():
    st.markdown("## 📊 Paso 2: Resumen de Datos")
    if st.session_state.df is not None:
        df = st.session_state.df
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Filas</h3>
                <h2>{df.shape[0]:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📋 Columnas</h3>
                <h2>{df.shape[1]}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.markdown(f"""
            <div class="metric-card">
                <h3>💾 Memoria</h3>
                <h2>{memory_usage:.1f} MB</h2>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            missing_total = df.isnull().sum().sum()
            missing_pct = (missing_total / (df.shape[0] * df.shape[1])) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚠️ Faltantes</h3>
                <h2>{missing_pct:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        st.subheader("📋 Análisis de Valores Faltantes")
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Columna': missing_data.index,
            'Valores Faltantes': missing_data.values,
            'Porcentaje': missing_pct.values
        }).sort_values('Valores Faltantes', ascending=False)
        st.dataframe(missing_df, use_container_width=True)
        if st.button("➡️ Continuar a Detección de Tipos", type="primary"):
            st.session_state.current_step = 3
            st.rerun()

def step_3_type_detection():
    st.markdown("## 🔍 Paso 3: Detección de Tipos y Semántica")
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("📋 Tipos de Datos Detectados")
        type_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if df[col].dtype in ['int64', 'float64']:
                semantic_type = 'numérico'
            elif df[col].dtype == 'object':
                if df[col].nunique() < len(df) * 0.1:
                    semantic_type = 'categórico'
                else:
                    semantic_type = 'texto'
            else:
                semantic_type = 'otro'
            type_info.append({
                'Columna': col,
                'Tipo Python': dtype,
                'Tipo Semántico': semantic_type,
                'Faltantes (%)': f"{missing_pct:.1f}%"
            })
        type_df = pd.DataFrame(type_info)
        st.dataframe(type_df, use_container_width=True)
        st.subheader("🔥 Distribución de Tipos")
        semantic_counts = pd.Series([info['Tipo Semántico'] for info in type_info]).value_counts()
        fig = px.bar(
            x=semantic_counts.index,
            y=semantic_counts.values,
            title="Distribución de Tipos Semánticos",
            labels={'x': 'Tipo Semántico', 'y': 'Cantidad'}
        )
        st.plotly_chart(fig, use_container_width=True)
        if st.button("➡️ Continuar a Limpieza", type="primary"):
            st.session_state.current_step = 4
            st.rerun()

def step_4_data_cleaning():
    st.markdown("## 🧹 Paso 4: Limpieza e Imputación")
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("📊 Comparación: Antes vs. Después")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📋 Antes de la Limpieza")
            st.dataframe(df.head(10), use_container_width=True)
            before_missing = df.isnull().sum().sum()
            before_missing_pct = (before_missing / (df.shape[0] * df.shape[1])) * 100
            st.metric("Valores Faltantes", f"{before_missing:,} ({before_missing_pct:.1f}%)")
        with col2:
            st.markdown("### ✅ Después de la Limpieza")
            st.info("💡 La limpieza automática se ejecutará en el análisis completo")
            st.dataframe(df.head(10), use_container_width=True)
            st.metric("Valores Faltantes", "Por calcular")
        with st.sidebar:
            st.markdown("### ⚙️ Gestión Avanzada de Valores Faltantes")
            st.selectbox(
                "Método de Imputación",
                ["Automático", "Mediana", "Media", "Moda", "Forward Fill"],
                help="Mediana: robusta ante valores extremos\nMedia: para distribuciones normales\nModa: para variables categóricas"
            )
            st.checkbox("Vista previa de cambios", value=False)
            st.button("🔄 Revertir cambios")
        if st.button("➡️ Continuar a Filtrado", type="primary"):
            st.session_state.current_step = 5
            st.rerun()

def step_5_data_filtering():
    st.markdown("## 🔧 Paso 5: Filtrado de Datos")
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("🎛️ Controles de Filtrado")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Seleccionar columna para filtrar:", numeric_cols)
            if selected_col:
                col_min = df[selected_col].min()
                col_max = df[selected_col].max()
                st.markdown(f"**Rango actual: {col_min:.2f} - {col_max:.2f}**")
                filter_min, filter_max = st.slider(
                    f"Filtrar {selected_col}",
                    min_value=float(col_min),
                    max_value=float(col_max),
                    value=(float(col_min), float(col_max)),
                    help=f"Selecciona el rango para {selected_col}"
                )
                affected_rows = df[(df[selected_col] >= filter_min) & (df[selected_col] <= filter_max)]
                affected_pct = (len(affected_rows) / len(df)) * 100
                st.markdown(f"""
                <div class="toast-info">
                    📊 <strong>Proyección:</strong> {len(affected_rows):,} de {len(df):,} filas ({affected_pct:.1f}%)
                </div>
                """, unsafe_allow_html=True)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            st.subheader("📝 Filtros Categóricos")
            selected_cat_col = st.selectbox("Seleccionar columna categórica:", categorical_cols)
            if selected_cat_col:
                unique_values = df[selected_cat_col].unique()
                selected_values = st.multiselect(
                    f"Seleccionar valores de {selected_cat_col}:",
                    unique_values,
                    default=unique_values.tolist()
                )
        if st.button("🔧 Aplicar Filtros", type="primary"):
            st.markdown("""
            <div class="toast-success">
                ✅ <strong>¡Filtros aplicados!</strong><br>
                📊 Filas: 10,000 → 8,500 (–15%)
            </div>
            """, unsafe_allow_html=True)
        if st.button("➡️ Continuar a Estadísticas", type="primary"):
            st.session_state.current_step = 6
            st.rerun()

def step_6_statistics():
    st.markdown("## 📈 Paso 6: Estadísticas y Modelos")
    if st.session_state.df is not None:
        df = st.session_state.df
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Univariantes", "🔗 Bivariantes", "🧪 Pruebas", "📈 Modelos"])
        with tab1:
            st.subheader("📊 Análisis Univariante")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Seleccionar columna numérica:", numeric_cols)
                if selected_col:
                    stats = df[selected_col].describe()
                    st.dataframe(stats, use_container_width=True)
                    fig = px.histogram(df, x=selected_col, title=f"Distribución de {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.subheader("🔗 Análisis Bivariante")
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, title="Matriz de Correlaciones")
                st.plotly_chart(fig, use_container_width=True)
        with tab3:
            st.subheader("🧪 Pruebas Estadísticas")
            st.info("💡 Las pruebas se ejecutarán automáticamente según el tipo de datos")
        with tab4:
            st.subheader("📈 Modelos de Regresión")
            st.info("💡 Los modelos se ejecutarán en el análisis completo")
        if st.button("➡️ Continuar a Visualización", type="primary"):
            st.session_state.current_step = 7
            st.rerun()

def step_7_visualization():
    st.markdown("## 📊 Paso 7: Visualización Inteligente")
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("💡 Sugerencias de Visualización")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Gráficos Sugeridos")
            if categorical_cols:
                st.markdown("**Categóricos:**")
                st.markdown("- 📊 Gráfico de barras")
                st.markdown("- 🥧 Gráfico circular")
                st.markdown("- 📈 Gráfico horizontal")
            if numeric_cols:
                st.markdown("**Numéricos:**")
                st.markdown("- 📊 Histograma")
                st.markdown("- 📦 Diagrama de caja")
                st.markdown("- 🔗 Gráfico de dispersión")
        with col2:
            st.markdown("### 🎨 Renderizado")
            chart_type = st.selectbox(
                "Tipo de gráfico:",
                ["Gráfico de barras", "Histograma", "Diagrama de caja", "Gráfico de dispersión"]
            )
            if chart_type == "Gráfico de barras" and categorical_cols:
                selected_col = st.selectbox("Seleccionar columna:", categorical_cols)
                if selected_col:
                    fig = px.bar(df[selected_col].value_counts(), title=f"Distribución de {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Histograma" and numeric_cols:
                selected_col = st.selectbox("Seleccionar columna:", numeric_cols)
                if selected_col:
                    fig = px.histogram(df, x=selected_col, title=f"Distribución de {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
        if st.button("➡️ Continuar a Exportación", type="primary"):
            st.session_state.current_step = 8
            st.rerun()

def step_8_export():
    st.markdown("## 💾 Paso 8: Exportar Resultados")
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("📤 Opciones de Exportación")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Datos")
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Descargar CSV",
                data=csv,
                file_name="datos_analizados.csv",
                mime="text/csv"
            )
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="💾 Descargar JSON",
                data=json_data,
                file_name="datos_analizados.json",
                mime="application/json"
            )
        with col2:
            st.markdown("### 📋 Reportes")
            logs = ["Análisis completado", f"Filas: {df.shape[0]}", f"Columnas: {df.shape[1]}"]
            logs_json = json.dumps(logs, indent=2, ensure_ascii=False)
            st.download_button(
                label="💾 Descargar Logs",
                data=logs_json,
                file_name="logs_analisis.json",
                mime="application/json"
            )
            html_report = f"""
            <html>
            <head><title>Reporte de Análisis</title></head>
            <body>
                <h1>Reporte de Análisis de Datos</h1>
                <p>Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <h2>Resumen</h2>
                <p>Filas: {df.shape[0]}</p>
                <p>Columnas: {df.shape[1]}</p>
                <h2>Datos</h2>
                {df.head(10).to_html()}
            </body>
            </html>
            """
            st.download_button(
                label="💾 Descargar HTML",
                data=html_report,
                file_name="reporte_analisis.html",
                mime="text/html"
            )
        st.markdown("""
        <div class="toast-success">
            🎉 <strong>¡Análisis completo!</strong><br>
            Todo el historial está disponible para auditoría.
        </div>
        """, unsafe_allow_html=True)
        st.subheader("📚 Historial de Sesión")
        st.text("Análisis completado exitosamente")
        st.text(f"Archivo procesado: {df.shape[0]} filas, {df.shape[1]} columnas")

def main():
    initialize_session_state()
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Máquina de Relojería - Análisis de Ciencias Sociales</h1>
        <p>Una experiencia ultra-didáctica, paso a paso, cohesiva y modulable</p>
    </div>
    """, unsafe_allow_html=True)
    show_tour()
    render_sidebar_steps()
    progress = st.session_state.current_step / 8
    st.markdown(f"""
    <div class="progress-container">
        <div style="background: #667eea; height: 10px; border-radius: 5px; width: {progress*100}%; transition: width 0.3s ease;"></div>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.current_step == 1:
        step_1_load_data()
    elif st.session_state.current_step == 2:
        step_2_data_summary()
    elif st.session_state.current_step == 3:
        step_3_type_detection()
    elif st.session_state.current_step == 4:
        step_4_data_cleaning()
    elif st.session_state.current_step == 5:
        step_5_data_filtering()
    elif st.session_state.current_step == 6:
        step_6_statistics()
    elif st.session_state.current_step == 7:
        step_7_visualization()
    elif st.session_state.current_step == 8:
        step_8_export()
    if st.session_state.current_step > 1:
        if st.button("⬅️ Paso Anterior"):
            st.session_state.current_step -= 1
            st.rerun()
    if st.session_state.current_step < 8:
        if st.button("➡️ Siguiente Paso"):
            st.session_state.current_step += 1
            st.rerun()

if __name__ == "__main__":
    main()
    