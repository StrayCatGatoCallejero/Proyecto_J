"""
Aplicación Streamlit para Pipeline de Procesamiento de Datos de Encuestas
=======================================================================

Interfaz de usuario completa para el procesamiento de datos de encuestas
con todas las funcionalidades del pipeline modular.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict, Any
import time

# Importar nuestro pipeline
sys.path.append('estadistica')
from pipeline_encuestas import (
    cargar_datos, explorar_datos, visualizar_nulos, imputar_nulos,
    analisis_ponderado, muestreo_estratificado, exportar_limpio,
    generar_reporte_pdf, pipeline_completo_encuestas, validar_archivo,
    # Nuevas funciones de validación de Chile
    validar_datos_chile, enriquecer_datos_chile_pipeline, 
    analisis_regional_chile, comparar_nacional_chile, mostrar_info_geografia_chile,
    # Nueva función para archivos grandes
    procesar_archivo_grande_en_chunks
)

# Configuración de la página
st.set_page_config(
    page_title="Proyecto J - Análisis de Encuestas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/tu-usuario/proyecto-j',
        'Report a bug': 'https://github.com/tu-usuario/proyecto-j/issues',
        'About': 'Proyecto J - Herramienta amigable para análisis de datos de encuestas'
    }
)

# CSS personalizado mejorado según guía de diseño UX/UI
st.markdown("""
<style>
    /* Importar fuentes */
    @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;500;600;700&family=Libre+Baskerville:wght@400;700&display=swap');
    
    /* FORZAR MODO CLARO - CSS adicional para garantizar modo claro */
    html, body {
        color-scheme: light !important;
        background-color: #FBF7F2 !important;
        color: #333333 !important;
    }
    
    /* Variables CSS para la paleta de colores actualizada */
    :root {
        --color-fondo-general: #FBF7F2;      /* Fondo general muy claro */
        --color-fondo-secundario: #F5E3D3;   /* Crema para tarjetas */
        --color-crema: #F5E3D3;              /* Crema más profundo */
        --color-durazno: #FFD9AE;            /* Durazno original */
        --color-arena: #D4A476;              /* Arena */
        --color-azul-claro: #C7DCE5;         /* Azul muy claro para área de contenido */
        --color-azul-profundo: #648DA5;      /* Azul profundo */
        --color-texto-principal: #2C3E50;    /* Texto principal */
        --color-texto-secundario: #7F8C8D;   /* Texto secundario */
        --color-blanco-suave: #FBF7F2;       /* Blanco suave */
        --color-sombra: rgba(0, 0, 0, 0.08);
        --color-sombra-hover: rgba(0, 0, 0, 0.12);
        --border-radius: 12px;
        --border-radius-botones: 8px;
        --espaciado: 24px;
        --espaciado-pequeno: 16px;
        --espaciado-grande: 32px;
    }
    
    /* Estilos generales */
    .main {
        background-color: var(--color-fondo-general) !important;
        font-family: 'Helvetica', sans-serif;
        color: var(--color-texto-principal);
        line-height: 1.6;
    }
    
    /* Forzar modo claro en elementos de Streamlit */
    .stApp {
        background-color: var(--color-fondo-general) !important;
        color: var(--color-texto-principal) !important;
    }
    
    .stApp > header {
        background-color: var(--color-fondo-general) !important;
    }
    
    .stApp > footer {
        background-color: var(--color-fondo-general) !important;
    }
    
    /* ÁREA DE CONTENIDO PRINCIPAL - Fondo azul claro */
    .main > div {
        background-color: var(--color-azul-claro) !important;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        box-shadow: 0 2px 8px var(--color-sombra);
    }
    
    /* Elementos de Streamlit en el área principal */
    .stMarkdown, .stText, .stDataFrame, .stPlotlyChart {
        background-color: transparent !important;
    }
    
    /* PANEL LATERAL IZQUIERDO - Fondo oscuro */
    .css-1d391kg {
        background-color: #333333 !important;
        border-right: 1px solid #555555;
        padding: var(--espaciado);
    }
    
    .css-1d391kg .sidebar-content {
        background-color: #333333 !important;
        color: #FFFFFF !important;
    }
    
    /* Texto en el sidebar */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, 
    .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: #FFFFFF !important;
    }
    
    .css-1d391kg p, .css-1d391kg div, .css-1d391kg span {
        color: #CCCCCC !important;
    }
    
    /* Elementos de Streamlit en el sidebar */
    .css-1d391kg .stMarkdown, .css-1d391kg .stText {
        background-color: transparent !important;
        color: #CCCCCC !important;
    }
    
    /* Header principal */
    .main-header {
        font-family: 'Raleway', sans-serif;
        font-size: 2.8rem;
        font-weight: 600;
        color: var(--color-azul-profundo);
        text-align: center;
        margin-bottom: var(--espaciado-grande);
        background: linear-gradient(135deg, var(--color-azul-profundo), var(--color-azul-claro));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    
    /* Contenedores principales */
    .main-container {
        background-color: var(--color-fondo-secundario) !important;
        border-radius: var(--border-radius);
        padding: var(--espaciado);
        margin: var(--espaciado-pequeno) 0;
        box-shadow: 0 2px 8px var(--color-sombra);
        border: 1px solid var(--color-durazno);
    }
    
    /* Tarjetas de información */
    .info-card {
        background-color: var(--color-azul-claro);
        border-radius: var(--border-radius);
        padding: var(--espaciado);
        margin: var(--espaciado-pequeno) 0;
        border: 1px solid var(--color-azul-profundo);
        box-shadow: 0 2px 4px var(--color-sombra);
    }
    
    /* Botones principales */
    .stButton > button {
        background-color: var(--color-azul-claro);
        color: var(--color-texto-principal);
        border: none;
        border-radius: var(--border-radius-botones);
        padding: 12px 24px;
        font-family: 'Helvetica', sans-serif;
        font-weight: 500;
        font-size: 14px;
        box-shadow: 0 2px 4px var(--color-sombra);
        transition: all 0.3s ease;
        width: 100%;
        margin: 8px 0;
    }
    
    .stButton > button:hover {
        background-color: var(--color-azul-profundo);
        color: var(--color-blanco-suave);
        box-shadow: 0 4px 8px var(--color-sombra-hover);
        transform: translateY(-1px);
    }
    
    .stButton > button:disabled {
        background-color: var(--color-arena);
        color: var(--color-texto-secundario);
        opacity: 0.6;
        cursor: not-allowed;
        transform: none;
    }
    
    /* Botón primario */
    .stButton > button[data-baseweb="button"] {
        background-color: var(--color-azul-profundo);
        color: var(--color-blanco-suave);
    }
    
    .stButton > button[data-baseweb="button"]:hover {
        background-color: var(--color-azul-claro);
        color: var(--color-texto-principal);
    }
    
    /* Mensajes de estado */
    .success-message {
        background-color: #E8F5E8;
        color: #2E7D32;
        padding: var(--espaciado-pequeno);
        border-radius: var(--border-radius);
        border: 1px solid #A5D6A7;
        margin: var(--espaciado-pequeno) 0;
        font-family: 'Helvetica', sans-serif;
        box-shadow: 0 2px 4px var(--color-sombra);
    }
    
    .error-message {
        background-color: #FFEBEE;
        color: #C62828;
        padding: var(--espaciado-pequeno);
        border-radius: var(--border-radius);
        border: 1px solid #EF9A9A;
        margin: var(--espaciado-pequeno) 0;
        font-family: 'Helvetica', sans-serif;
        box-shadow: 0 2px 4px var(--color-sombra);
    }
    
    .warning-message {
        background-color: #FFF8E1;
        color: #F57C00;
        padding: var(--espaciado-pequeno);
        border-radius: var(--border-radius);
        border: 1px solid var(--color-durazno);
        margin: var(--espaciado-pequeno) 0;
        font-family: 'Helvetica', sans-serif;
        box-shadow: 0 2px 4px var(--color-sombra);
    }
    
    .info-message {
        background-color: #E3F2FD;
        color: #1565C0;
        padding: var(--espaciado-pequeno);
        border-radius: var(--border-radius);
        border: 1px solid #90CAF9;
        margin: var(--espaciado-pequeno) 0;
        font-family: 'Helvetica', sans-serif;
        box-shadow: 0 2px 4px var(--color-sombra);
    }
    
    /* Títulos de sección */
    .section-title {
        font-family: 'Raleway', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--color-azul-profundo);
        margin: var(--espaciado) 0 var(--espaciado-pequeno) 0;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--color-durazno);
    }
    
    .subsection-title {
        font-family: 'Raleway', sans-serif;
        font-size: 1.4rem;
        font-weight: 500;
        color: var(--color-azul-profundo);
        margin: var(--espaciado-pequeno) 0 12px 0;
    }
    
    /* Textos de instrucción */
    .instruction-text {
        font-family: 'Helvetica', sans-serif;
        font-size: 14px;
        color: var(--color-texto-secundario);
        line-height: 1.5;
        margin: 8px 0;
        padding: 12px;
        background-color: var(--color-crema);
        border-radius: var(--border-radius);
        border-left: 4px solid var(--color-azul-claro);
    }
    
    /* Métricas y estadísticas */
    .metric-card {
        background-color: var(--color-fondo-secundario) !important;
        padding: var(--espaciado-pequeno);
        border-radius: var(--border-radius);
        border: 1px solid var(--color-durazno);
        box-shadow: 0 2px 4px var(--color-sombra);
        text-align: center;
    }
    
    .metric-value {
        font-family: 'Raleway', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: var(--color-azul-profundo);
        margin: 8px 0;
    }
    
    .metric-label {
        font-family: 'Helvetica', sans-serif;
        font-size: 12px;
        color: var(--color-texto-secundario);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Contenedor de progreso */
    .progress-container {
        background-color: var(--color-fondo-secundario);
        border-radius: var(--border-radius);
        padding: var(--espaciado-pequeno);
        margin: var(--espaciado-pequeno) 0;
        border: 1px solid var(--color-durazno);
    }
    
    /* Inputs y selectores */
    .stSelectbox, .stTextInput, .stFileUploader {
        background-color: var(--color-fondo-secundario) !important;
        border-radius: var(--border-radius-botones);
        border: 1px solid var(--color-durazno);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: var(--color-azul-claro);
        color: var(--color-texto-principal);
        border-radius: var(--border-radius-botones);
        font-family: 'Raleway', sans-serif;
        font-weight: 500;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: 0 2px 4px var(--color-sombra);
    }
    
    /* Gráficos */
    .stPlotlyChart {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: 0 2px 4px var(--color-sombra);
        margin: var(--espaciado-pequeno) 0;
    }
    
    /* Separadores */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--color-arena), transparent);
        margin: var(--espaciado) 0;
    }
    
    /* Microtextos */
    .micro-text {
        font-family: 'Libre Baskerville', serif;
        font-size: 11px;
        color: var(--color-texto-secundario);
        font-style: italic;
    }
    
    /* Animaciones sutiles */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .section-title {
            font-size: 1.5rem;
        }
        
        .main-container {
            padding: var(--espaciado-pequeno);
        }
    }
</style>
""", unsafe_allow_html=True)

def show_success_message(message: str):
    """Muestra un mensaje de éxito con el nuevo estilo."""
    st.markdown(f'<div class="success-message fade-in">✅ {message}</div>', unsafe_allow_html=True)

def show_error_message(message: str):
    """Muestra un mensaje de error con el nuevo estilo."""
    st.markdown(f'<div class="error-message fade-in">❌ {message}</div>', unsafe_allow_html=True)

def show_warning_message(message: str):
    """Muestra un mensaje de advertencia con el nuevo estilo."""
    st.markdown(f'<div class="warning-message fade-in">⚠️ {message}</div>', unsafe_allow_html=True)

def show_info_message(message: str):
    """Muestra un mensaje informativo con el nuevo estilo."""
    st.markdown(f'<div class="info-message fade-in">ℹ️ {message}</div>', unsafe_allow_html=True)

def show_instruction_text(message: str):
    """Muestra un texto de instrucción amigable."""
    st.markdown(f'<div class="instruction-text fade-in">💡 {message}</div>', unsafe_allow_html=True)

def show_section_title(title: str):
    """Muestra un título de sección con el nuevo estilo."""
    st.markdown(f'<h2 class="section-title fade-in">{title}</h2>', unsafe_allow_html=True)

def show_subsection_title(title: str):
    """Muestra un título de subsección con el nuevo estilo."""
    st.markdown(f'<h3 class="subsection-title fade-in">{title}</h3>', unsafe_allow_html=True)

def show_micro_text(text: str):
    """Muestra un microtexto con el estilo Libre Baskerville."""
    st.markdown(f'<p class="micro-text fade-in">{text}</p>', unsafe_allow_html=True)

def validate_file_format(file_path: str) -> bool:
    """Valida que el archivo tenga un formato soportado."""
    supported_formats = ['.csv', '.sav', '.dta', '.xlsx', '.xls']
    return any(file_path.lower().endswith(fmt) for fmt in supported_formats)

def get_file_size_mb(file_path: str) -> float:
    """Obtiene el tamaño del archivo en MB."""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except:
        return 0.0

def main():
    """Función principal de la aplicación con nuevo diseño UX/UI."""
    
    # Header principal con nuevo estilo
    st.markdown('<h1 class="main-header">📊 Proyecto J - Análisis de Encuestas</h1>', 
                unsafe_allow_html=True)
    
    # Texto de bienvenida amigable
    st.markdown("""
    <div class="instruction-text fade-in">
        🎯 <strong>Bienvenido a Proyecto J</strong><br>
        Tu herramienta amigable para analizar datos de encuestas. 
        Carga tu archivo y descubre insights valiosos de forma sencilla.
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar variables de estado en session_state
    if 'pipeline_running' not in st.session_state:
        st.session_state.pipeline_running = False
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    # Sidebar con nuevo diseño
    with st.sidebar:
        st.markdown('<h2 class="section-title">⚙️ Configuración</h2>', unsafe_allow_html=True)
        
        # Sección de carga de archivo
        st.markdown('<h3 class="subsection-title">📁 Carga de Archivo</h3>', unsafe_allow_html=True)
        
        # Opciones de carga de archivo
        opcion_carga = st.radio(
            "¿Cómo quieres cargar tu archivo?",
            ["Subir archivo", "Archivo existente", "Ruta local"],
            help="Elige la opción que mejor se adapte a tu archivo"
        )
        
        archivo_subido = None
        archivo_existente = None
        ruta_local = None
        
        if opcion_carga == "Subir archivo":
            show_instruction_text("Carga un archivo en formato .CSV, .SAV o .DTA para comenzar. Tamaño máximo: 200MB.")
            archivo_subido = st.file_uploader(
                "Selecciona tu archivo de datos",
                type=['csv', 'sav', 'dta', 'xlsx', 'xls'],
                help="Formatos soportados: CSV, SPSS (.sav), Stata (.dta), Excel (.xlsx, .xls)"
            )
            
        elif opcion_carga == "Archivo existente":
            show_instruction_text("Selecciona un archivo que ya esté en la carpeta de datos del proyecto.")
            archivo_existente = st.selectbox(
                "Archivos disponibles",
                ["Ninguno"] + [f for f in os.listdir("data") if f.endswith(('.csv', '.sav', '.dta', '.xlsx', '.xls'))],
                help="Archivos de ejemplo y datos guardados"
            )
            
        else:  # Ruta local
            show_instruction_text("Para archivos grandes (>200MB): Ingresa la ruta completa al archivo. No hay límite de tamaño.")
            st.markdown("""
            <div class="info-card fade-in">
                <strong>💡 Ejemplo de ruta:</strong><br>
                C:/Users/TuUsuario/Documents/encuesta.dta<br>
                <em class="micro-text">No hay límite de tamaño para archivos locales</em>
            </div>
            """, unsafe_allow_html=True)
            
            ruta_local = st.text_input(
                "Ruta completa del archivo",
                placeholder="C:/ruta/al/archivo.dta",
                help="Ruta completa al archivo de datos en tu computadora"
            )
            
            if ruta_local:
                if os.path.exists(ruta_local):
                    file_size = get_file_size_mb(ruta_local)
                    st.success(f"✅ Archivo encontrado: {file_size:.2f} MB")
                else:
                    st.error("❌ Archivo no encontrado en la ruta especificada")
        
        # Mostrar información del archivo seleccionado
        if archivo_subido is not None:
            file_size = len(archivo_subido.getbuffer()) / (1024 * 1024)  # MB
            st.markdown(f"""
            <div class="info-card fade-in">
                <strong>📄 Archivo:</strong> {archivo_subido.name}<br>
                <strong>📏 Tamaño:</strong> {file_size:.2f} MB<br>
                <strong>📅 Tipo:</strong> {archivo_subido.type}
            </div>
            """, unsafe_allow_html=True)
            
            if file_size > 200:
                show_warning_message("Archivo muy grande (>200MB). Considera usar 'Ruta local' para archivos grandes.")
        
        elif archivo_existente and archivo_existente != "Ninguno":
            file_path = os.path.join("data", archivo_existente)
            file_size = get_file_size_mb(file_path)
            st.markdown(f"""
            <div class="info-card fade-in">
                <strong>📄 Archivo:</strong> {archivo_existente}<br>
                <strong>📏 Tamaño:</strong> {file_size:.2f} MB
            </div>
            """, unsafe_allow_html=True)
            
        elif ruta_local and os.path.exists(ruta_local):
            file_size = get_file_size_mb(ruta_local)
            st.markdown(f"""
            <div class="info-card fade-in">
                <strong>📄 Archivo:</strong> {os.path.basename(ruta_local)}<br>
                <strong>📏 Tamaño:</strong> {file_size:.2f} MB
            </div>
            """, unsafe_allow_html=True)
            
            if file_size > 500:
                st.warning(f"⚠️ Archivo muy grande ({file_size:.1f} MB). El procesamiento puede tardar varios minutos.")
        
        # Configuración de procesamiento
        st.markdown('<h3 class="subsection-title">🔧 Opciones de Procesamiento</h3>', unsafe_allow_html=True)
        
        show_instruction_text("Elige cómo quieres que se procesen los datos faltantes en tu archivo.")
        
        estrategia_imputacion = st.selectbox(
            "Método para datos faltantes",
            ["media", "mediana", "moda", "knn", "random_forest"],
            help="Estrategia para completar datos faltantes",
            index=0
        )
        
        # Mostrar descripción de la estrategia
        estrategias_desc = {
            "media": "Reemplaza con el promedio de la columna (recomendado para números)",
            "mediana": "Reemplaza con el valor del medio (bueno para datos con valores extremos)",
            "moda": "Reemplaza con el valor más frecuente (ideal para categorías)",
            "knn": "Usa valores similares para completar (más preciso pero más lento)",
            "random_forest": "Predice valores usando inteligencia artificial (más preciso, más lento)"
        }
        st.caption(f"💡 {estrategias_desc[estrategia_imputacion]}")
        
        generar_reporte = st.checkbox(
            "Generar reporte PDF",
            value=True,
            help="Crear un reporte PDF con los resultados del análisis"
        )
        
        exportar_datos = st.checkbox(
            "Exportar datos procesados",
            value=True,
            help="Guardar los datos limpios en un archivo"
        )
        
        # Opciones avanzadas
        with st.expander("🔬 Opciones Avanzadas"):
            show_instruction_text("Estas opciones son recomendadas solo si tienes experiencia técnica. Si no estás seguro, deja los valores por defecto.")
            
            # Opciones para archivos grandes
            if ruta_local and os.path.exists(ruta_local):
                file_size = get_file_size_mb(ruta_local)
                if file_size > 100:  # Solo mostrar para archivos grandes
                    st.markdown('<h4 class="subsection-title">📦 Procesamiento de Archivos Grandes</h4>', unsafe_allow_html=True)
                    
                    procesar_en_chunks = st.checkbox(
                        "Procesar en partes (recomendado para archivos >100MB)",
                        value=True,
                        help="Divide el archivo en partes para procesamiento más eficiente"
                    )
                    
                    # Guardar en session_state
                    st.session_state.procesar_en_chunks = procesar_en_chunks
                    
                    if procesar_en_chunks:
                        chunk_size = st.number_input(
                            "Tamaño de cada parte (filas)",
                            min_value=1000,
                            max_value=100000,
                            value=10000,
                            step=1000,
                            help="Número de filas a procesar por vez"
                        )
                        
                        # Guardar en session_state
                        st.session_state.chunk_size = chunk_size
                        
                        st.info(f"💡 El archivo se procesará en partes de {chunk_size:,} filas")
            
            test_size = st.slider(
                "Tamaño del conjunto de prueba",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Proporción de datos para el conjunto de prueba"
            )
            
            random_state = st.number_input(
                "Semilla aleatoria",
                min_value=0,
                max_value=1000,
                value=42,
                help="Semilla para reproducibilidad"
            )
            
            # Opciones de visualización
            st.markdown('<h4 class="subsection-title">📊 Visualización</h4>', unsafe_allow_html=True)
            mostrar_correlaciones = st.checkbox(
                "Mostrar correlaciones",
                value=True,
                help="Generar matriz de correlaciones"
            )
            
            mostrar_distribuciones = st.checkbox(
                "Mostrar distribuciones",
                value=True,
                help="Generar histogramas de variables numéricas"
            )
    
    # Contenido principal
    if archivo_subido is not None or (archivo_existente and archivo_existente != "Ninguno") or (ruta_local and os.path.exists(ruta_local)):
        
        # Determinar ruta del archivo
        if archivo_subido is not None:
            # Guardar archivo subido temporalmente
            temp_path = f"temp_{archivo_subido.name}"
            with open(temp_path, "wb") as f:
                f.write(archivo_subido.getbuffer())
            ruta_archivo = temp_path
            st.session_state.current_file = archivo_subido.name
        elif archivo_existente and archivo_existente != "Ninguno":
            ruta_archivo = os.path.join("data", archivo_existente)
            st.session_state.current_file = archivo_existente
        else:
            ruta_archivo = ruta_local
            st.session_state.current_file = os.path.basename(ruta_archivo)
        
        # Mostrar información del archivo
        show_info_message(f"Archivo seleccionado: **{Path(ruta_archivo).name}**")
        
        # Validar archivo
        if not validate_file_format(ruta_archivo):
            show_error_message("Formato de archivo no soportado. Use CSV, SAV, DTA, XLSX o XLS.")
            return
        
        # Sección de análisis principal
        show_section_title("📊 Análisis de Datos")
        show_instruction_text("Aquí puedes ver un resumen rápido de tu archivo y explorar su contenido.")
        
        # Botones principales con nuevo diseño
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Ejecutar Análisis Completo", 
                        type="primary", 
                        disabled=st.session_state.pipeline_running,
                        help="Ejecuta todo el proceso de limpieza y análisis de tus datos"):
                
                st.session_state.pipeline_running = True
                
                try:
                    with st.spinner("Ejecutando análisis completo..."):
                        # Mostrar progreso
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Determinar si usar procesamiento de archivos grandes
                        file_size = get_file_size_mb(ruta_archivo)
                        usar_procesamiento_grande = False
                        chunk_size = 10000
                        
                        # Verificar si se debe usar procesamiento de archivos grandes
                        if ruta_local and os.path.exists(ruta_local) and file_size > 100:
                            # Obtener configuración de chunks desde session_state o usar valores por defecto
                            procesar_en_chunks = st.session_state.get('procesar_en_chunks', True)
                            if procesar_en_chunks:
                                chunk_size = st.session_state.get('chunk_size', 10000)
                                usar_procesamiento_grande = True
                        
                        # Ejecutar pipeline apropiado
                        if usar_procesamiento_grande:
                            st.info(f"📦 Usando procesamiento optimizado para archivo grande ({file_size:.1f} MB)")
                            resultados = procesar_archivo_grande_en_chunks(
                                ruta_archivo=ruta_archivo,
                                chunk_size=chunk_size,
                                estrategia_imputacion=estrategia_imputacion,
                                generar_reporte=generar_reporte,
                                exportar_datos=exportar_datos
                            )
                        else:
                            resultados = pipeline_completo_encuestas(
                                ruta_archivo=ruta_archivo,
                                estrategia_imputacion=estrategia_imputacion,
                                generar_reporte=generar_reporte,
                                exportar_datos=exportar_datos
                            )
                        
                        # Actualizar progreso
                        progress_bar.progress(100)
                        status_text.text("¡Análisis completado!")
                        
                        # Guardar resultados
                        st.session_state.results = resultados
                        
                        # Mostrar mensaje de éxito
                        show_success_message("Análisis ejecutado exitosamente")
                        
                        # Mostrar toast
                        st.toast("🎉 Análisis completado", icon="✅")
                        
                except Exception as e:
                    show_error_message(f"Error en el análisis: {str(e)}")
                    st.toast("❌ Error en el análisis", icon="❌")
                finally:
                    st.session_state.pipeline_running = False
                    progress_bar.empty()
                    status_text.empty()
        
        with col2:
            if st.button("📊 Explorar Datos", 
                        disabled=st.session_state.pipeline_running,
                        help="Ver un resumen rápido de tu archivo"):
                try:
                    with st.spinner("Cargando datos..."):
                        df = cargar_datos(ruta_archivo)
                        mostrar_exploracion_avanzada(df)
                    show_success_message("Exploración completada")
                except Exception as e:
                    show_error_message(f"Error al explorar datos: {str(e)}")
        
        with col3:
            if st.button("🔍 Analizar Faltantes", 
                        disabled=st.session_state.pipeline_running,
                        help="Ver qué datos faltan en tu archivo"):
                try:
                    with st.spinner("Analizando valores faltantes..."):
                        df = cargar_datos(ruta_archivo)
                        visualizar_nulos(df, tipo="matrix")
                    show_success_message("Análisis de faltantes completado")
                except Exception as e:
                    show_error_message(f"Error al analizar valores faltantes: {str(e)}")
        
        # Mostrar resultados si existen
        if st.session_state.results:
            mostrar_resultados_mejorados(st.session_state.results)
        
        # Herramientas individuales
        show_section_title("🔧 Herramientas Individuales")
        show_instruction_text("Selecciona una variable para generar gráficos o explorar sus características.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Limpiar Datos", 
                        disabled=st.session_state.pipeline_running,
                        help="Completar datos faltantes en tu archivo"):
                try:
                    with st.spinner("Limpiando datos..."):
                        df = cargar_datos(ruta_archivo)
                        df_limpio = imputar_nulos(df, estrategia=estrategia_imputacion)
                        mostrar_comparacion_datos(df, df_limpio)
                    show_success_message("Datos limpiados exitosamente")
                except Exception as e:
                    show_error_message(f"Error al limpiar datos: {str(e)}")
            
            if st.button("📈 Análisis Ponderado", 
                        disabled=st.session_state.pipeline_running,
                        help="Analizar datos con pesos de encuesta"):
                try:
                    with st.spinner("Preparando análisis ponderado..."):
                        df = cargar_datos(ruta_archivo)
                        mostrar_analisis_ponderado(df)
                except Exception as e:
                    show_error_message(f"Error en análisis ponderado: {str(e)}")
        
        with col2:
            if st.button("📋 Muestreo Estratificado", 
                        disabled=st.session_state.pipeline_running,
                        help="Dividir datos en grupos para análisis"):
                try:
                    with st.spinner("Realizando muestreo..."):
                        df = cargar_datos(ruta_archivo)
                        mostrar_muestreo_estratificado(df)
                except Exception as e:
                    show_error_message(f"Error en muestreo estratificado: {str(e)}")
            
            if st.button("💾 Exportar Datos", 
                        disabled=st.session_state.pipeline_running,
                        help="Guardar los datos procesados"):
                try:
                    with st.spinner("Exportando datos..."):
                        df = cargar_datos(ruta_archivo)
                        df_limpio = imputar_nulos(df, estrategia=estrategia_imputacion)
                        ruta_export = f"datos_limpios_{Path(ruta_archivo).stem}.csv"
                        exportar_limpio(df_limpio, ruta_export)
                        show_success_message(f"Datos exportados a: {ruta_export}")
                except Exception as e:
                    show_error_message(f"Error al exportar datos: {str(e)}")
        
        # Separador visual
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # ============================================================================
        # NUEVAS FUNCIONALIDADES DE VALIDACIÓN DE CHILE
        # ============================================================================
        
        show_section_title("🇨🇱 Validación de Datos de Chile")
        show_instruction_text("Valida y enriquece tus datos con información oficial de Chile.")
        
        # Pestañas para las funcionalidades de Chile
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📍 Información Geográfica", 
            "✅ Validación", 
            "📊 Enriquecimiento", 
            "📈 Análisis Regional", 
            "🏛️ Comparación Nacional"
        ])
        
        with tab1:
            mostrar_info_geografia_chile()
        
        with tab2:
            if st.button("🔍 Validar Datos de Chile", disabled=st.session_state.pipeline_running):
                try:
                    with st.spinner("Validando datos geográficos..."):
                        df = cargar_datos(ruta_archivo)
                        
                        # Permitir al usuario seleccionar columnas
                        col1, col2 = st.columns(2)
                        with col1:
                            columna_region = st.selectbox(
                                "Columna con códigos de región",
                                ["Ninguna"] + list(df.columns),
                                help="Selecciona la columna que contiene los códigos de región"
                            )
                        with col2:
                            columna_comuna = st.selectbox(
                                "Columna con códigos de comuna",
                                ["Ninguna"] + list(df.columns),
                                help="Selecciona la columna que contiene los códigos de comuna"
                            )
                        
                        if columna_region != "Ninguna" or columna_comuna != "Ninguna":
                            region_col = columna_region if columna_region != "Ninguna" else None
                            comuna_col = columna_comuna if columna_comuna != "Ninguna" else None
                            
                            resultados_validacion = validar_datos_chile(df, region_col, comuna_col)
                            
                            if resultados_validacion['valido']:
                                show_success_message("✅ Validación geográfica exitosa")
                            else:
                                show_error_message("❌ Errores encontrados en validación geográfica")
                                
                except Exception as e:
                    show_error_message(f"Error en validación: {str(e)}")
        
        with tab3:
            if st.button("📊 Enriquecer Datos", disabled=st.session_state.pipeline_running):
                try:
                    with st.spinner("Enriqueciendo datos..."):
                        df = cargar_datos(ruta_archivo)
                        
                        # Permitir al usuario seleccionar columnas
                        col1, col2 = st.columns(2)
                        with col1:
                            columna_region = st.selectbox(
                                "Columna región (enriquecimiento)",
                                ["Ninguna"] + list(df.columns),
                                key="enriquecimiento_region"
                            )
                        with col2:
                            columna_comuna = st.selectbox(
                                "Columna comuna (enriquecimiento)",
                                ["Ninguna"] + list(df.columns),
                                key="enriquecimiento_comuna"
                            )
                        
                        if columna_region != "Ninguna" or columna_comuna != "Ninguna":
                            region_col = columna_region if columna_region != "Ninguna" else None
                            comuna_col = columna_comuna if columna_comuna != "Ninguna" else None
                            
                            df_enriquecido = enriquecer_datos_chile_pipeline(df, region_col, comuna_col)
                            st.session_state.datos_enriquecidos = df_enriquecido
                            
                            show_success_message("✅ Datos enriquecidos exitosamente")
                            
                            # Mostrar nuevas columnas agregadas
                            if 'datos_enriquecidos' in st.session_state:
                                columnas_nuevas = set(df_enriquecido.columns) - set(df.columns)
                                if columnas_nuevas:
                                    st.markdown(f"""
                                    <div class="info-card fade-in">
                                        <strong>📊 Nuevas columnas agregadas:</strong><br>
                                        {', '.join(sorted(columnas_nuevas))}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                except Exception as e:
                    show_error_message(f"Error en enriquecimiento: {str(e)}")
        
        with tab4:
            if st.button("📈 Analizar por Región", disabled=st.session_state.pipeline_running):
                try:
                    if 'datos_enriquecidos' in st.session_state:
                        df_enriquecido = st.session_state.datos_enriquecidos
                        
                        # Permitir al usuario seleccionar variable a analizar
                        variable_analizar = st.selectbox(
                            "Variable a analizar",
                            df_enriquecido.select_dtypes(include=[np.number]).columns.tolist(),
                            help="Selecciona una variable numérica para analizar por región"
                        )
                        
                        if variable_analizar:
                            with st.spinner("Analizando por región..."):
                                analisis_regional = analisis_regional_chile(df_enriquecido, variable_analizar)
                                st.dataframe(analisis_regional)
                                show_success_message("✅ Análisis regional completado")
                    else:
                        show_warning_message("⚠️ Primero debes enriquecer los datos en la pestaña 'Enriquecimiento'")
                        
                except Exception as e:
                    show_error_message(f"Error en análisis regional: {str(e)}")
        
        with tab5:
            if st.button("🏛️ Comparar con Nacional", disabled=st.session_state.pipeline_running):
                try:
                    if 'datos_enriquecidos' in st.session_state:
                        df_enriquecido = st.session_state.datos_enriquecidos
                        
                        # Permitir al usuario seleccionar variable a analizar
                        variable_analizar = st.selectbox(
                            "Variable a comparar",
                            df_enriquecido.select_dtypes(include=[np.number]).columns.tolist(),
                            key="comparacion_variable",
                            help="Selecciona una variable numérica para comparar con promedios nacionales"
                        )
                        
                        if variable_analizar:
                            with st.spinner("Comparando con promedios nacionales..."):
                                comparacion = comparar_nacional_chile(df_enriquecido, variable_analizar)
                                st.dataframe(comparacion)
                                show_success_message("✅ Comparación nacional completada")
                    else:
                        show_warning_message("⚠️ Primero debes enriquecer los datos en la pestaña 'Enriquecimiento'")
                        
                except Exception as e:
                    show_error_message(f"Error en comparación nacional: {str(e)}")
    
    else:
        # Estado inicial - mostrar instrucciones
        st.markdown("""
        <div class="main-container fade-in">
            <h2 class="section-title">🎯 ¿Cómo empezar?</h2>
            <div class="instruction-text">
                <strong>Paso 1:</strong> En el panel lateral izquierdo, selecciona cómo quieres cargar tu archivo.<br>
                <strong>Paso 2:</strong> Elige las opciones de procesamiento que mejor se adapten a tus datos.<br>
                <strong>Paso 3:</strong> Haz clic en "Ejecutar Análisis Completo" para comenzar.<br><br>
                <em class="micro-text">No te preocupes por los términos técnicos, la aplicación te guiará en cada paso.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar características principales
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card fade-in">
                <div class="metric-value">📊</div>
                <div class="metric-label">Análisis Completo</div>
                <p style="font-size: 12px; margin-top: 8px;">Explora, limpia y analiza tus datos de encuestas de forma automática</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card fade-in">
                <div class="metric-value">🇨🇱</div>
                <div class="metric-label">Datos de Chile</div>
                <p style="font-size: 12px; margin-top: 8px;">Valida y enriquece con información oficial geográfica y demográfica</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card fade-in">
                <div class="metric-value">📈</div>
                <div class="metric-label">Visualizaciones</div>
                <p style="font-size: 12px; margin-top: 8px;">Genera gráficos y reportes profesionales de tus resultados</p>
            </div>
            """, unsafe_allow_html=True)

def mostrar_resultados_mejorados(resultados: Dict[str, Any]):
    """Muestra los resultados del análisis con el nuevo diseño UX/UI."""
    
    show_section_title("📋 Resultados del Análisis")
    show_instruction_text("Cuando termines, podrás guardar los resultados en PDF, PNG o Excel.")
    
    # Contenedor principal de resultados
    st.markdown('<div class="main-container fade-in">', unsafe_allow_html=True)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'datos_originales' in resultados:
            filas_originales = len(resultados['datos_originales'])
            st.markdown(f"""
            <div class="metric-card fade-in">
                <div class="metric-value">{filas_originales:,}</div>
                <div class="metric-label">Registros Originales</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'datos_limpios' in resultados:
            filas_limpias = len(resultados['datos_limpios'])
            st.markdown(f"""
            <div class="metric-card fade-in">
                <div class="metric-value">{filas_limpias:,}</div>
                <div class="metric-label">Registros Procesados</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'datos_originales' in resultados and 'datos_limpios' in resultados:
            columnas = len(resultados['datos_limpios'].columns)
            st.markdown(f"""
            <div class="metric-card fade-in">
                <div class="metric-value">{columnas}</div>
                <div class="metric-label">Variables</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'datos_originales' in resultados and 'datos_limpios' in resultados:
            valores_faltantes_originales = resultados['datos_originales'].isnull().sum().sum()
            valores_faltantes_limpias = resultados['datos_limpios'].isnull().sum().sum()
            valores_completados = valores_faltantes_originales - valores_faltantes_limpias
            st.markdown(f"""
            <div class="metric-card fade-in">
                <div class="metric-value">{valores_completados:,}</div>
                <div class="metric-label">Datos Completados</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Archivos generados
    if 'ruta_exportacion' in resultados or 'ruta_reporte' in resultados:
        show_subsection_title("📁 Archivos Generados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'ruta_exportacion' in resultados:
                st.markdown(f"""
                <div class="info-card fade-in">
                    <strong>📊 Datos Procesados</strong><br>
                    <em>Archivo:</em> {resultados['ruta_exportacion']}<br>
                    <em>Formato:</em> CSV limpio y procesado
                </div>
                """, unsafe_allow_html=True)
                
                # Botón de descarga
                with open(resultados['ruta_exportacion'], 'r', encoding='utf-8') as f:
                    st.download_button(
                        label="💾 Descargar Datos Procesados",
                        data=f.read(),
                        file_name=resultados['ruta_exportacion'],
                        mime="text/csv",
                        help="Descarga los datos limpios y procesados"
                    )
        
        with col2:
            if 'ruta_reporte' in resultados and resultados['ruta_reporte']:
                st.markdown(f"""
                <div class="info-card fade-in">
                    <strong>📄 Reporte PDF</strong><br>
                    <em>Archivo:</em> {resultados['ruta_reporte']}<br>
                    <em>Contenido:</em> Análisis completo y visualizaciones
                </div>
                """, unsafe_allow_html=True)
                
                # Botón de descarga del PDF
                if os.path.exists(resultados['ruta_reporte']):
                    with open(resultados['ruta_reporte'], 'rb') as f:
                        st.download_button(
                            label="📄 Descargar Reporte PDF",
                            data=f.read(),
                            file_name=resultados['ruta_reporte'],
                            mime="application/pdf",
                            help="Descarga el reporte completo en PDF"
                        )
    
    # Resumen de procesamiento
    show_subsection_title("📊 Resumen del Procesamiento")
    
    st.markdown("""
    <div class="instruction-text fade-in">
        <strong>✅ Procesamiento completado exitosamente</strong><br>
        • Los datos han sido limpiados y procesados<br>
        • Se han completado los valores faltantes<br>
        • Se han generado visualizaciones y reportes<br>
        • Los archivos están listos para descargar
    </div>
    """, unsafe_allow_html=True)
    
    # Próximos pasos
    show_subsection_title("🎯 Próximos Pasos Sugeridos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card fade-in">
            <strong>🔍 Explorar Datos</strong><br>
            Usa las herramientas individuales para explorar variables específicas y generar gráficos personalizados.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card fade-in">
            <strong>🇨🇱 Validar Chile</strong><br>
            Si tienes datos geográficos de Chile, valida y enriquece con información oficial.
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-card fade-in">
            <strong>📈 Análisis Avanzado</strong><br>
            Realiza análisis ponderado o muestreo estratificado según tus necesidades.
        </div>
        """, unsafe_allow_html=True)
    
    # Microtexto final
    show_micro_text("Los resultados se mantienen en memoria durante esta sesión. Descarga los archivos para guardarlos permanentemente.")

def mostrar_exploracion_avanzada(df: pd.DataFrame):
    """Muestra exploración avanzada de datos con mejor UX."""
    st.subheader("📊 Exploración Avanzada de Datos")
    
    # Información básica
    info = explorar_datos(df)
    
    # Distribución de variables numéricas
    if info['numeric_columns']:
        st.write("**📈 Distribución de variables numéricas:**")
        
        # Seleccionar variable para histograma
        var_seleccionada = st.selectbox(
            "Selecciona una variable para visualizar:",
            info['numeric_columns'],
            key="hist_var"
        )
        
        if var_seleccionada:
            fig = px.histogram(df, x=var_seleccionada, 
                             title=f"Distribución de {var_seleccionada}",
                             nbins=30)
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlaciones
    if len(info['numeric_columns']) > 1:
        st.write("**🔗 Matriz de correlaciones:**")
        corr_matrix = df[info['numeric_columns']].corr()
        fig = px.imshow(corr_matrix, 
                       title="Matriz de Correlaciones",
                       color_continuous_scale='RdBu',
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

def mostrar_comparacion_datos(df_original: pd.DataFrame, df_limpio: pd.DataFrame):
    """Muestra comparación entre datos originales y limpios con mejor UX."""
    st.subheader("🔄 Comparación: Datos Originales vs Limpios")
    
    # Tabs para organizar la información
    tab1, tab2, tab3 = st.tabs(["📋 Vista Previa", "📊 Estadísticas", "📈 Visualización"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📄 Datos Originales:**")
            st.dataframe(df_original.head(), use_container_width=True)
            
        with col2:
            st.write("**🧹 Datos Limpios:**")
            st.dataframe(df_limpio.head(), use_container_width=True)
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Valores faltantes originales", df_original.isnull().sum().sum())
            st.metric("Valores faltantes limpios", df_limpio.isnull().sum().sum())
        
        with col2:
            st.metric("Filas originales", len(df_original))
            st.metric("Filas limpias", len(df_limpio))
        
        with col3:
            st.metric("Columnas originales", len(df_original.columns))
            st.metric("Columnas limpias", len(df_limpio.columns))
    
    with tab3:
        # Comparación de valores faltantes por columna
        null_comparison = pd.DataFrame({
            'Columna': df_original.columns,
            'Faltantes Originales': df_original.isnull().sum(),
            'Faltantes Limpios': df_limpio.isnull().sum()
        })
        
        fig = px.bar(null_comparison, x='Columna', y=['Faltantes Originales', 'Faltantes Limpios'],
                    title="Comparación de Valores Faltantes por Columna",
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)

def mostrar_analisis_ponderado(df: pd.DataFrame):
    """Interfaz para análisis ponderado con mejor UX."""
    st.subheader("📈 Análisis Ponderado")
    
    # Verificar si hay columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        show_warning_message("Se necesitan al menos 2 columnas numéricas para el análisis ponderado.")
        return
    
    # Selección de variables
    col1, col2 = st.columns(2)
    
    with col1:
        variable = st.selectbox("Variable a analizar", numeric_cols, key="var_ponderada")
    
    with col2:
        peso = st.selectbox("Variable de peso", numeric_cols, key="peso_ponderado")
    
    if st.button("📊 Calcular estadísticas ponderadas", 
                disabled=st.session_state.pipeline_running):
        try:
            with st.spinner("Calculando estadísticas ponderadas..."):
                resultados = analisis_ponderado(df, variable, peso)
            show_success_message("Análisis ponderado completado")
        except Exception as e:
            show_error_message(f"Error en análisis ponderado: {str(e)}")

def mostrar_muestreo_estratificado(df: pd.DataFrame):
    """Interfaz para muestreo estratificado con mejor UX."""
    st.subheader("📋 Muestreo Estratificado")
    
    # Seleccionar columna de estratificación
    columna_estratificacion = st.selectbox(
        "Columna para estratificación:",
        df.columns.tolist(),
        key="col_estratificacion"
    )
    
    # Configurar parámetros
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Tamaño del conjunto de prueba",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05
        )
    
    with col2:
        random_state = st.number_input(
            "Semilla aleatoria",
            min_value=0,
            max_value=1000,
            value=42
        )
    
    if st.button("📊 Realizar muestreo estratificado", 
                disabled=st.session_state.pipeline_running):
        try:
            with st.spinner("Realizando muestreo estratificado..."):
                train_df, test_df = muestreo_estratificado(
                    df, columna_estratificacion, test_size, random_state
                )
            show_success_message("Muestreo estratificado completado")
        except Exception as e:
            show_error_message(f"Error en muestreo estratificado: {str(e)}")

if __name__ == "__main__":
    main() 