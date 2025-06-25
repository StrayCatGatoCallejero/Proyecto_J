# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
from datetime import datetime
import os
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from collections import Counter

# =====================
# ESTILOS Y CONFIGURACIÓN
# =====================
st.set_page_config(
    page_title="Asistente de Visualización de Datos",
    page_icon=":bar_chart:",
    layout="wide"
)

st.markdown('''<style>
body, .main, .block-container {
    background-color: #f9f6f2 !important;
}
section[data-testid="stSidebar"] {
    background-color: #f4e3d7 !important;
}
.sidebar-content {
    background-color: #f4e3d7 !important;
}
.css-1d391kg {background-color: #f4e3d7 !important;}
.stButton>button, .stDownloadButton>button {
    background-color: #4f8cff;
    color: white;
    border-radius: 6px;
    padding: 0.5em 1.5em;
    font-weight: 600;
    border: none;
    margin: 0.5em 0.2em;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background-color: #2563eb;
}
.stAlert, .stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 6px;
}
.stDataFrame, .stTable {
    background-color: #fff !important;
    border-radius: 6px;
}
</style>''', unsafe_allow_html=True)

# =====================
# SIDEBAR WIZARD
# =====================
steps = [
    "Cargar archivo",
    "Resumen de datos",
    "Detección de tipos",
    "Sugerencias",
    "Selección de gráfico",
    "Visualización",
    "Exportar resultados"
]

if 'wizard_step' not in st.session_state:
    st.session_state.wizard_step = 0
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_types' not in st.session_state:
    st.session_state.data_types = None
if 'selected_var' not in st.session_state:
    st.session_state.selected_var = None
if 'suggested_charts' not in st.session_state:
    st.session_state.suggested_charts = []
if 'chart_type' not in st.session_state:
    st.session_state.chart_type = None

with st.sidebar:
    st.markdown("<h3 style='margin-bottom:0.5em;'>🧭 Navegación</h3>", unsafe_allow_html=True)
    st.markdown("<ul style='list-style:none;padding-left:0;'>" +
        ''.join([
            f"<li style='margin-bottom:0.5em;{('font-weight:bold;color:#2563eb;' if i==st.session_state.wizard_step else '')}'>"
            f"{i+1}. {step}" + ("<div style='height:4px;width:80%;background:#2563eb;border-radius:2px;margin:2px 0;' ></div>" if i==st.session_state.wizard_step else "") +
            "</li>" for i, step in enumerate(steps)
        ]) + "</ul>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-top:2em;'><b>Tip:</b> En el futuro podrás crear visualizaciones que relacionen dos variables.</div>
    <div style='margin-top:2em;'>
        <form action="#" method="post">
            <button type="submit" style="background:#fff;border:1px solid #2563eb;color:#2563eb;padding:0.5em 1.5em;border-radius:6px;font-weight:600;cursor:pointer;" onclick="window.location.reload();return false;">🔄 Reiniciar Asistente</button>
        </form>
    </div>
    """, unsafe_allow_html=True)

# =====================
# PASO 1: CARGA DE ARCHIVO
# =====================
def step_1():
    st.markdown("""
    <h1 style='margin-bottom:0.2em;'>🤖 Asistente de Visualización de Datos</h1>
    <div style='font-size:1.1em;color:#444;'>Guía paso a paso para crear visualizaciones efectivas de tus datos</div>
    <br>
    <h2>📁 Paso 1: Cargar archivo de datos</h2>
    <div>Guía tu archivo de datos</div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        uploaded_file = st.file_uploader(
            "Carga tu archivo de datos",
            type=["csv", "xlsx", "xls", "sav", "dta"],
            help="Limit 200MB per file - CSV, XLSX, XLS, SAV, DTA"
        )
        if uploaded_file:
            temp_dir = Path('./temp')
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / f"upload_{int(time.time())}_{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            ext = temp_path.suffix.lower()
            try:
                if ext == '.csv':
                    data = pd.read_csv(temp_path)
                elif ext in ['.xlsx', '.xls']:
                    data = pd.read_excel(temp_path, engine='openpyxl')
                elif ext == '.sav':
                    import pyreadstat
                    data, _ = pyreadstat.read_sav(temp_path)
                elif ext == '.dta':
                    data = pd.read_stata(temp_path)
                else:
                    st.error("Tipo de archivo no soportado")
                    return
                st.session_state.data = data
                st.success("Archivo cargado correctamente")
                st.info(f"Datos: {data.shape[0]} filas × {data.shape[1]} columnas")
                if st.button("Continuar al siguiente paso"):
                    st.session_state.wizard_step = 1
                    st.rerun()
            except Exception as e:
                st.error(f"Error al cargar archivo: {e}")
        else:
            st.info("Por favor, sube un archivo de datos para comenzar.")
    with col2:
        st.markdown("""
        <div style='margin-top:2em;'>
        <b>Formatos soportados:</b><br>
        <ul style='margin:0 0 0 1em;padding:0;'>
        <li>CSV (.csv)</li>
        <li>Excel (.xlsx, .xls)</li>
        <li>SPSS (.sav)</li>
        <li>Stata (.dta)</li>
        </ul>
        <div style='margin-top:1em;font-size:0.95em;color:#b8860b;'>
        💡 <b>Consejo:</b> Para mejores resultados, asegúrate de que tu archivo tenga encabezados en la primera fila.
        </div>
        </div>
        """, unsafe_allow_html=True)

# =====================
# PASO 2: RESUMEN DE DATOS
# =====================
def step_2():
    st.markdown("""
    <h1>🤖 Asistente de Visualización de Datos</h1>
    <h2>📊 Paso 2: Resumen automático de los datos</h2>
    <div style='font-size:1.1em;color:#444;'>Guía paso a paso para crear visualizaciones efectivas de tus datos</div>
    <br>
    """, unsafe_allow_html=True)
    data = st.session_state.data
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("<h3>Información general</h3>", unsafe_allow_html=True)
        st.write(f"Filas: {data.shape[0]}")
        st.write(f"Columnas: {data.shape[1]}")
        st.write(f"Memoria utilizada: {data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        st.markdown("<h4>🔍 Primeras filas</h4>", unsafe_allow_html=True)
        st.dataframe(data.head(5), use_container_width=True)
    with col2:
        st.markdown("<h3>Tipos de datos</h3>", unsafe_allow_html=True)
        types_df = pd.DataFrame({
            'Tipo': data.dtypes.astype(str),
            'No nulos': data.notnull().sum(),
            '% Completo': (data.notnull().sum() / len(data) * 100).round(1)
        })
        st.dataframe(types_df, use_container_width=True)
    st.markdown("<h4>⚠️ Valores faltantes</h4>", unsafe_allow_html=True)
    missing = data.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.dataframe(missing.reset_index().rename(columns={'index':'Columna',0:'Faltantes'}), use_container_width=True)
    else:
        st.success("No hay valores faltantes en el dataset.")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Volver"):
            st.session_state.wizard_step = 0
            st.rerun()
    with col2:
        if st.button("Continuar"):
            st.session_state.wizard_step = 2
            st.rerun()

# =====================
# PASO 3: DETECCIÓN DE TIPOS
# =====================
def step_3():
    st.markdown("""
    <h1>🤖 Asistente de Visualización de Datos</h1>
    <h2>🔍 Paso 3: Detección automática de tipos de variables</h2>
    """, unsafe_allow_html=True)
    data = st.session_state.data
    # Detección simple de tipos
    tipos = []
    for col in data.columns:
        vals = data[col].dropna()
        if len(vals) == 0:
            tipo = 'vacía'
            detalles = 'Todos los valores son NaN'
        elif data[col].dtype in [np.float64, np.int64]:
            tipo = 'numérico'
            detalles = f"{vals.nunique()} valores únicos"
        else:
            nunicos = vals.nunique()
            if nunicos < 20:
                tipo = 'categórico'
                detalles = f"{nunicos} valores únicos"
            else:
                tipo = 'texto'
                detalles = f"{nunicos} valores únicos"
        tipos.append({'columna': col, 'tipo_detectado': tipo, 'detalles': detalles})
    tipos_df = pd.DataFrame(tipos)
    st.session_state.data_types = tipos_df
    st.markdown("<h3>Resultados del análisis</h3>", unsafe_allow_html=True)
    st.dataframe(tipos_df, use_container_width=True)
    st.markdown("<h3>Distribución de tipos de variables</h3>", unsafe_allow_html=True)
    pie = tipos_df['tipo_detectado'].value_counts().reset_index()
    pie.columns = ['Tipo','Cantidad']
    fig = px.pie(pie, names='Tipo', values='Cantidad', color_discrete_sequence=px.colors.sequential.Blues)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Los tipos detectados automáticamente te ayudarán a elegir las mejores visualizaciones.")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Volver"):
            st.session_state.wizard_step = 1
            st.rerun()
    with col2:
        if st.button("Continuar"):
            st.session_state.wizard_step = 3
            st.rerun()

# =====================
# PASO 4: SUGERENCIAS DE VISUALIZACIÓN
# =====================
def step_4():
    st.markdown("""
    <h1>🤖 Asistente de Visualización de Datos</h1>
    <h2>💡 Paso 4: Sugerencias de visualización</h2>
    """, unsafe_allow_html=True)
    tipos_df = st.session_state.data_types
    data = st.session_state.data
    var = st.selectbox("Selecciona la variable que quieres visualizar:", data.columns)
    st.session_state.selected_var = var
    tipo = tipos_df[tipos_df['columna'] == var]['tipo_detectado'].values[0]
    st.markdown(f"<h3>Variable: {var}</h3>", unsafe_allow_html=True)
    st.info(f"Tipo detectado: {tipo}")
    # Sugerencias simples
    sugerencias = []
    if tipo == 'categórico':
        sugerencias = ["Gráfico de barras", "Gráfico de torta", "Tabla de frecuencias"]
    elif tipo == 'numérico':
        sugerencias = ["Histograma", "Boxplot", "Gráfico de dispersión"]
    elif tipo == 'vacía':
        sugerencias = ["No visualizable"]
    else:
        sugerencias = ["Tabla de frecuencias", "Nube de palabras"]
    st.session_state.suggested_charts = sugerencias
    st.markdown("<h4>Visualizaciones sugeridas</h4>", unsafe_allow_html=True)
    for i, sug in enumerate(sugerencias, 1):
        st.write(f"{i}. {sug}")
    st.info("Estas sugerencias están basadas en el tipo de variable detectado.")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Volver"):
            st.session_state.wizard_step = 2
            st.rerun()
    with col2:
        if st.button("Continuar"):
            st.session_state.wizard_step = 4
            st.rerun()

# =====================
# PASO 5: SELECCIÓN DE GRÁFICO
# =====================
def step_5():
    st.markdown("""
    <h1>🤖 Asistente de Visualización de Datos</h1>
    <h2>📊 Paso 5: Selección de gráfico</h2>
    """, unsafe_allow_html=True)
    sugerencias = st.session_state.suggested_charts
    chart_type = st.selectbox("Selecciona el tipo de gráfico que deseas generar:", sugerencias)
    st.session_state.chart_type = chart_type
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Volver"):
            st.session_state.wizard_step = 3
            st.rerun()
    with col2:
        if st.button("Continuar"):
            st.session_state.wizard_step = 5
            st.rerun()

# =====================
# PASO 6: VISUALIZACIÓN
# =====================
def step_6():
    st.markdown("""
    <h1>🤖 Asistente de Visualización de Datos</h1>
    <h2>📊 Paso 6: Visualización</h2>
    """, unsafe_allow_html=True)
    data = st.session_state.data
    var = st.session_state.selected_var
    chart_type = st.session_state.chart_type
    st.markdown(f"<h3>Visualización de: {var}</h3>", unsafe_allow_html=True)
    if chart_type == "Gráfico de barras":
        counts = data[var].value_counts()
        fig = px.bar(x=counts.index.astype(str), y=counts.values, labels={'x':var,'y':'Frecuencia'}, title=f"Gráfico de barras de {var}")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Gráfico de torta":
        counts = data[var].value_counts()
        fig = px.pie(names=counts.index.astype(str), values=counts.values, title=f"Gráfico de torta de {var}")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Tabla de frecuencias":
        st.dataframe(data[var].value_counts().reset_index().rename(columns={'index':var, var:'Frecuencia'}), use_container_width=True)
    elif chart_type == "Histograma":
        fig = px.histogram(data, x=var, title=f"Histograma de {var}")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Boxplot":
        fig = px.box(data, y=var, title=f"Boxplot de {var}")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Gráfico de dispersión":
        # Permitir elegir otra variable numérica
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != var]
        if numeric_cols:
            x_var = st.selectbox("Variable X:", numeric_cols)
            fig = px.scatter(data, x=x_var, y=var, title=f"Dispersión: {x_var} vs {var}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay otra variable numérica para comparar.")
    elif chart_type == "Nube de palabras":
        st.info("Funcionalidad de nube de palabras próximamente disponible.")
    else:
        st.info("No hay visualización disponible para este tipo de variable.")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Volver"):
            st.session_state.wizard_step = 4
            st.rerun()
    with col2:
        if st.button("Continuar"):
            st.session_state.wizard_step = 6
            st.rerun()

# =====================
# PASO 7: EXPORTAR RESULTADOS
# =====================
def step_7():
    st.markdown("""
    <h1>🤖 Asistente de Visualización de Datos</h1>
    <h2>📤 Paso 7: Exportar resultados</h2>
    """, unsafe_allow_html=True)
    st.success("¡Visualización generada con éxito!")
    st.info("Puedes descargar la tabla de frecuencias o la imagen del gráfico generado.")
    # Exportar tabla si corresponde
    data = st.session_state.data
    var = st.session_state.selected_var
    chart_type = st.session_state.chart_type
    if chart_type == "Tabla de frecuencias":
        freq_df = data[var].value_counts().reset_index().rename(columns={'index':var, var:'Frecuencia'})
        st.download_button("Descargar tabla de frecuencias", freq_df.to_csv(index=False).encode('utf-8'), file_name=f"frecuencias_{var}.csv", mime="text/csv")
    # Exportar imagen (sólo para gráficos)
    st.info("Para descargar la imagen, haz clic derecho sobre el gráfico y selecciona 'Guardar imagen como...'")
    if st.button("Volver"):
        st.session_state.wizard_step = 5
        st.rerun()
    if st.button("Reiniciar Asistente"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# =====================
# CONTROLADOR DE PASOS
# =====================
if st.session_state.wizard_step == 0:
    step_1()
elif st.session_state.wizard_step == 1:
    step_2()
elif st.session_state.wizard_step == 2:
    step_3()
elif st.session_state.wizard_step == 3:
    step_4()
elif st.session_state.wizard_step == 4:
    step_5()
elif st.session_state.wizard_step == 5:
    step_6()
elif st.session_state.wizard_step == 6:
    step_7()

# Sección de ayuda
with st.expander("❓ ¿Necesitas ayuda?"):
    st.markdown("""
    ### Preguntas Frecuentes
    
    **Q: ¿Qué tipos de archivos puedo procesar?**
    A: CSV, Excel (.xlsx, .xls), SPSS (.sav), STATA (.dta)
    
    **Q: ¿Cuál es el tamaño máximo de archivo?**
    A: 500 MB para garantizar un procesamiento eficiente.
    
    **Q: ¿Qué hace la limpieza automática?**
    A: Elimina duplicados y maneja valores faltantes según la configuración seleccionada.
    
    **Q: ¿Cómo interpreto la matriz de correlación?**
    A: Los valores van de -1 a 1. Valores cercanos a 1 indican correlación positiva fuerte, cercanos a -1 correlación negativa fuerte, y cercanos a 0 poca correlación.
    
    **Q: ¿Qué modelos se ajustan?**
    A: Regresión lineal, polinomial (grado 2) y exponencial según la configuración seleccionada.
    
    **Q: ¿Qué significan R² y RMSE?**
    A: R² mide qué tan bien el modelo explica la variabilidad (0-1, más alto es mejor). RMSE mide el error promedio de predicción (más bajo es mejor).
    
    **Q: ¿Se guardan mis datos permanentemente?**
    A: No, los archivos temporales se eliminan al cerrar la sesión.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        📊 Pipeline Demográfico Modular - Proyecto J | 
        Desarrollado con Streamlit y Análisis Estadístico Avanzado
    </div>
    """,
    unsafe_allow_html=True
)

print('Archivo limpio y codificación correcta') 