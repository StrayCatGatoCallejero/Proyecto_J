# -*- coding: utf-8 -*-
"""
Versión simplificada del Asistente de Visualización de Datos
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import io
import base64

# Configuración de la página - DEBE SER LA PRIMERA FUNCIÓN DE STREAMLIT
st.set_page_config(
    page_title="Asistente de Visualización - Proyecto J",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Verificar disponibilidad de kaleido para exportación PNG
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False
    st.warning("⚠️ Kaleido no está instalado. La exportación PNG no estará disponible. Instala con: pip install kaleido")

# Verificar disponibilidad de fpdf2 para exportación PDF
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    st.warning("⚠️ fpdf no está disponible - generación de PDF limitada")

def exportar_grafico_png(fig, nombre_archivo="grafico.png"):
    """Exporta un gráfico de Plotly como imagen PNG"""
    if not KALEIDO_AVAILABLE:
        st.error("❌ Kaleido no está disponible. Instala con: pip install kaleido")
        return False
    
    try:
        # Configurar el motor de renderizado para PNG (usando la nueva API)
        pio.defaults.default_format = "png"
        pio.defaults.default_width = 1200
        pio.defaults.default_height = 800
        
        # Convertir el gráfico a PNG
        img_bytes = fig.to_image(format="png", width=1200, height=800)
        
        # Crear botón de descarga
        st.download_button(
            label="🖼️ Descargar PNG",
            data=img_bytes,
            file_name=nombre_archivo,
            mime="image/png"
        )
        return True
    except Exception as e:
        st.error(f"❌ Error exportando PNG: {e}")
        return False

def mostrar_grafico_con_descarga(fig, titulo="Gráfico", nombre_archivo=None):
    """Muestra un gráfico con opciones de descarga"""
    if fig is not None:
        # Mostrar el gráfico
        st.plotly_chart(fig, use_container_width=True)
        
        # Botones de descarga
        if nombre_archivo is None:
            nombre_archivo = f"{titulo.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        col1, col2 = st.columns(2)
        with col1:
            if KALEIDO_AVAILABLE:
                if st.button(f"🖼️ Exportar {titulo} como PNG"):
                    exportar_grafico_png(fig, nombre_archivo)
            else:
                st.info("🖼️ Exportación PNG no disponible (instala kaleido)")
        
        with col2:
            # Opción para descargar como HTML interactivo
            html_bytes = fig.to_html(include_plotlyjs='cdn')
            st.download_button(
                label="📄 Exportar como HTML",
                data=html_bytes,
                file_name=nombre_archivo.replace('.png', '.html'),
                mime="text/html"
            )

def generar_pdf_mejorado(df, titulo="Reporte de Datos", filename="reporte.pdf"):
    """Genera un PDF con mejor manejo de errores"""
    try:
        from fpdf import FPDF
        import tempfile
        
        # Crear PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt=titulo, ln=True, align='C')
        
        # Información del dataset
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Filas: {len(df)}", ln=True)
        pdf.cell(200, 10, txt=f"Columnas: {len(df.columns)}", ln=True)
        
        # Estadísticas descriptivas
        pdf.cell(200, 10, txt="Estadísticas Descriptivas:", ln=True)
        stats = df.describe()
        for col in stats.columns:
            pdf.cell(200, 10, txt=f"{col}: Media={stats[col]['mean']:.2f}", ln=True)
        
        # Guardar archivo temporal
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf.output(temp_file.name)
        
        return True, temp_file.name
    except Exception as e:
        return False, f"Error generando PDF: {str(e)}"

# Funciones de análisis de datos
def obtener_columnas_numericas(df):
    """Obtiene las columnas numéricas del dataframe"""
    return df.select_dtypes(include=['number']).columns.tolist()

def obtener_columnas_categoricas(df):
    """Obtiene las columnas categóricas del dataframe"""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def detect_types(df):
    """Detecta automáticamente los tipos de datos"""
    types_info = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isnull().sum()
        unique = df[col].nunique()
        
        # Clasificar tipo
        if df[col].dtype in ['int64', 'float64']:
            tipo = "numérico"
        elif df[col].dtype == 'object':
            if df[col].nunique() <= 10:
                tipo = "categórico"
            else:
                tipo = "texto largo"
        elif 'datetime' in str(df[col].dtype):
            tipo = "fecha"
        elif df[col].dtype == 'bool':
            tipo = "booleano"
        else:
            tipo = "otro"
        
        types_info.append({
            "columna": col,
            "tipo": tipo,
            "dtype": dtype,
            "valores_unicos": unique,
            "valores_faltantes": missing
        })
    
    return pd.DataFrame(types_info)

def crear_visualizacion(df, col, tipo_vis, tipo_col):
    """Crea visualizaciones usando Plotly"""
    datos_limpios = df[col].dropna()

    if tipo_vis == "Histograma" and tipo_col == "numérico":
        fig = px.histogram(
            df, x=col, title=f"Distribución de {col}", nbins=30, marginal="box"
        )
        fig.update_layout(xaxis_title=col, yaxis_title="Frecuencia", showlegend=False)
        return fig

    elif tipo_vis == "Boxplot" and tipo_col == "numérico":
        fig = px.box(df, y=col, title=f"Boxplot de {col}")
        fig.update_layout(yaxis_title=col, showlegend=False)
        return fig

    elif tipo_vis == "Gráfico de barras" and tipo_col in ["categórico", "booleano"]:
        frecuencias = df[col].value_counts().head(20)
        fig = px.bar(
            x=frecuencias.index,
            y=frecuencias.values,
            title=f"Frecuencia de {col}",
            labels={"x": col, "y": "Frecuencia"},
        )
        fig.update_layout(showlegend=False)
        return fig

    elif tipo_vis == "Gráfico de torta" and tipo_col in ["categórico", "booleano"]:
        frecuencias = df[col].value_counts().head(10)
        fig = px.pie(
            values=frecuencias.values,
            names=frecuencias.index,
            title=f"Distribución de {col}",
        )
        return fig

    return None

def generar_heatmap_correlacion(df):
    """Genera un heatmap de correlaciones"""
    numeric_cols = obtener_columnas_numericas(df)
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, 
                    title="Matriz de Correlaciones",
                    color_continuous_scale='RdBu',
                    aspect="auto")
    return fig



# Título principal
st.title("📊 Asistente de Visualización de Datos - Proyecto J")

# Sidebar
st.sidebar.markdown("# 🧭 Navegación")
pages = {
    "📁 Cargar archivo": "cargar",
    "📊 Resumen de datos": "resumen", 
    "🔍 Detección de tipos": "tipos",
    "💡 Sugerencias": "sugerencias",
    "📈 Visualización": "visualizacion",
    "📤 Exportar resultados": "exportar"
}

selected_page = st.sidebar.selectbox(
    "Selecciona una página:",
    list(pages.keys())
)

# Página principal
if selected_page == "📁 Cargar archivo":
    st.header("📁 Cargar archivo de datos")
    st.write("¡Bienvenido! Aquí puedes cargar tu archivo de datos para comenzar el análisis.")
    
    uploaded_file = st.file_uploader(
        "Selecciona un archivo:",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos soportados: CSV, Excel"
    )
    
    if uploaded_file is not None:
        try:
            # Cargar datos según el tipo de archivo
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Guardar en session state
            st.session_state['df'] = df
            
            st.success("✅ Archivo cargado exitosamente!")
            st.write(f"**Nombre:** {uploaded_file.name}")
            st.write(f"**Tamaño:** {uploaded_file.size} bytes")
            st.write(f"**Filas:** {len(df)}")
            st.write(f"**Columnas:** {len(df.columns)}")
            
            # Mostrar primeras filas
            st.subheader("📋 Vista previa de los datos")
            st.dataframe(df.head(), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error cargando archivo: {e}")

elif selected_page == "📊 Resumen de datos":
    st.header("📊 Resumen de datos")
    
    df = st.session_state.get('df')
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Estadísticas descriptivas")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.subheader("📋 Información del dataset")
            st.write(f"**Filas:** {len(df)}")
            st.write(f"**Columnas:** {len(df.columns)}")
            st.write(f"**Memoria utilizada:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Valores faltantes
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.subheader("⚠️ Valores faltantes")
                st.dataframe(missing_data[missing_data > 0], use_container_width=True)
            else:
                st.success("✅ No hay valores faltantes")
        
        # Exportar resumen
        st.subheader("📤 Exportar resumen")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Exportar PDF"):
                if FPDF_AVAILABLE:
                    success, result = generar_pdf_mejorado(df, "Resumen de Datos")
                    if success:
                        with open(result, "rb") as f:
                            st.download_button(
                                "⬇️ Descargar PDF",
                                f.read(),
                                file_name="resumen_datos.pdf",
                                mime="application/pdf"
                            )
                else:
                    st.error("❌ FPDF2 no está instalado")
        
        with col2:
            csv_data = df.describe().to_csv()
            st.download_button(
                "📊 Descargar CSV",
                csv_data,
                file_name="resumen_datos.csv",
                mime="text/csv"
            )
        
        with col3:
            # Crear gráfico de valores faltantes
            if df.isnull().sum().sum() > 0:
                missing_fig = px.bar(
                    x=df.columns,
                    y=df.isnull().sum(),
                    title="Valores faltantes por columna"
                )
                mostrar_grafico_con_descarga(missing_fig, "Valores Faltantes", "valores_faltantes.png")
    
    else:
        st.warning("⚠️ Primero debes cargar un archivo en la página 'Cargar archivo'")

elif selected_page == "🔍 Detección de tipos":
    st.header("🔍 Detección automática de tipos")
    
    df = st.session_state.get('df')
    if df is not None:
        types_df = detect_types(df)
        
        st.subheader("📋 Tipos detectados")
        st.dataframe(types_df, use_container_width=True)
        
        # Gráfico de distribución de tipos
        tipo_counts = types_df['tipo'].value_counts()
        fig_tipos = px.pie(
            values=tipo_counts.values,
            names=tipo_counts.index,
            title="Distribución de tipos de datos"
        )
        mostrar_grafico_con_descarga(fig_tipos, "Distribución de Tipos", "distribucion_tipos.png")
        
        # Exportar tipos
        st.subheader("📤 Exportar tipos")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_tipos = types_df.to_csv(index=False)
            st.download_button(
                "📊 Descargar CSV",
                csv_tipos,
                file_name="tipos_datos.csv",
                mime="text/csv"
            )
        
        with col2:
            if st.button("📄 Exportar PDF"):
                if FPDF_AVAILABLE:
                    success, result = generar_pdf_mejorado(types_df, "Tipos de Datos")
                    if success:
                        with open(result, "rb") as f:
                            st.download_button(
                                "⬇️ Descargar PDF",
                                f.read(),
                                file_name="tipos_datos.pdf",
                                mime="application/pdf"
                            )
    
    else:
        st.warning("⚠️ Primero debes cargar un archivo en la página 'Cargar archivo'")

elif selected_page == "💡 Sugerencias":
    st.header("💡 Sugerencias de visualización")
    
    df = st.session_state.get('df')
    if df is not None:
        types_df = detect_types(df)
        
        st.subheader("🎯 Selecciona una variable para analizar")
        col = st.selectbox(
            "Elige la variable que quieres visualizar:",
            df.columns
        )
        
        if col:
            info = types_df.loc[types_df["columna"] == col].iloc[0]
            tipo_col = info["tipo"]
            
            st.subheader(f"📊 Análisis de: {col}")
            st.write(f"**Tipo detectado:** {tipo_col}")
            st.write(f"**Valores únicos:** {info['valores_unicos']}")
            st.write(f"**Valores faltantes:** {info['valores_faltantes']}")
            
            # Sugerencias según el tipo
            if tipo_col == "numérico":
                st.info("💡 **Sugerencias para variables numéricas:**")
                st.write("- Histograma: Para ver la distribución")
                st.write("- Boxplot: Para detectar outliers")
                st.write("- Scatter plot: Para correlaciones")
                
                # Crear visualizaciones sugeridas
                fig_hist = crear_visualizacion(df, col, "Histograma", tipo_col)
                if fig_hist:
                    mostrar_grafico_con_descarga(fig_hist, f"Histograma de {col}", f"histograma_{col}.png")
                
                fig_box = crear_visualizacion(df, col, "Boxplot", tipo_col)
                if fig_box:
                    mostrar_grafico_con_descarga(fig_box, f"Boxplot de {col}", f"boxplot_{col}.png")
            
            elif tipo_col == "categórico":
                st.info("💡 **Sugerencias para variables categóricas:**")
                st.write("- Gráfico de barras: Para frecuencias")
                st.write("- Gráfico de torta: Para proporciones")
                
                fig_bar = crear_visualizacion(df, col, "Gráfico de barras", tipo_col)
                if fig_bar:
                    mostrar_grafico_con_descarga(fig_bar, f"Barras de {col}", f"barras_{col}.png")
                
                fig_pie = crear_visualizacion(df, col, "Gráfico de torta", tipo_col)
                if fig_pie:
                    mostrar_grafico_con_descarga(fig_pie, f"Torta de {col}", f"torta_{col}.png")
    
    else:
        st.warning("⚠️ Primero debes cargar un archivo en la página 'Cargar archivo'")

elif selected_page == "📈 Visualización":
    st.header("📈 Visualización personalizada")
    
    df = st.session_state.get('df')
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎨 Crear visualización")
            col = st.selectbox("Variable:", df.columns)
            
            if col:
                types_df = detect_types(df)
                info = types_df.loc[types_df["columna"] == col].iloc[0]
                tipo_col = info["tipo"]
                
                # Opciones según el tipo
                if tipo_col == "numérico":
                    tipo_vis = st.selectbox(
                        "Tipo de gráfico:",
                        ["Histograma", "Boxplot"]
                    )
                elif tipo_col in ["categórico", "booleano"]:
                    tipo_vis = st.selectbox(
                        "Tipo de gráfico:",
                        ["Gráfico de barras", "Gráfico de torta"]
                    )
                else:
                    st.warning("⚠️ Tipo de variable no soportado para visualización")
                    tipo_vis = None
                
                if tipo_vis and st.button("📊 Generar gráfico"):
                    fig = crear_visualizacion(df, col, tipo_vis, tipo_col)
                    if fig:
                        mostrar_grafico_con_descarga(fig, f"{tipo_vis} de {col}", f"{tipo_vis.lower().replace(' ', '_')}_{col}.png")
        
        with col2:
            st.subheader("🔥 Análisis de correlaciones")
            numeric_cols = obtener_columnas_numericas(df)
            
            if len(numeric_cols) >= 2:
                if st.button("🔥 Generar heatmap"):
                    fig_heatmap = generar_heatmap_correlacion(df)
                    if fig_heatmap:
                        mostrar_grafico_con_descarga(fig_heatmap, "Heatmap de Correlaciones", "heatmap_correlaciones.png")
            else:
                st.info("ℹ️ Se necesitan al menos 2 variables numéricas para correlaciones")
    
    else:
        st.warning("⚠️ Primero debes cargar un archivo en la página 'Cargar archivo'")

elif selected_page == "📤 Exportar resultados":
    st.header("📤 Exportar resultados")
    
    df = st.session_state.get('df')
    if df is not None:
        st.subheader("📊 Exportar datos")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📄 Exportar PDF"):
                if FPDF_AVAILABLE:
                    success, result = generar_pdf_mejorado(df, "Reporte Completo")
                    if success:
                        with open(result, "rb") as f:
                            st.download_button(
                                "⬇️ Descargar PDF",
                                f.read(),
                                file_name="reporte_completo.pdf",
                                mime="application/pdf"
                            )
                else:
                    st.error("❌ FPDF2 no está instalado")
        
        with col2:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📊 Descargar CSV",
                csv_data,
                file_name="datos_completos.csv",
                mime="text/csv"
            )
        
        with col3:
            # Exportar como Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Datos', index=False)
                df.describe().to_excel(writer, sheet_name='Estadísticas')
            buffer.seek(0)
            
            st.download_button(
                "📈 Descargar Excel",
                buffer.getvalue(),
                file_name="datos_completos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col4:
            st.info("🖼️ PNG disponible en cada gráfico")
        
        # Información del sistema
        st.subheader("ℹ️ Información del sistema")
        st.write(f"**Formatos disponibles:**")
        st.write("✅ PDF - Reportes completos")
        st.write("✅ CSV - Datos procesados")
        st.write("✅ Excel - Datos tabulares")
        st.write("✅ PNG - Gráficos de alta calidad")
        st.write("✅ HTML - Gráficos interactivos")
        
        if KALEIDO_AVAILABLE:
            st.success("✅ Exportación PNG disponible")
        else:
            st.warning("⚠️ Exportación PNG no disponible (instala kaleido)")
        
        if FPDF_AVAILABLE:
            st.success("✅ Exportación PDF disponible")
        else:
            st.warning("⚠️ Exportación PDF no disponible (instala fpdf2)")
    
    else:
        st.warning("⚠️ Primero debes cargar un archivo en la página 'Cargar archivo'")

# Footer
st.markdown("---")
st.caption("© 2025 Proyecto J - Pipeline Modular + Streamlit Wizard Unificado") 