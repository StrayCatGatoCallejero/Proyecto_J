import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
import io
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de estilo
plt.style.use('default')
sns.set_palette("husl")

# =====================
# Función de detección de tipos de columna
# =====================
def detectar_tipos_columnas(df: pd.DataFrame, umbral_cardinalidad=20, umbral_texto_largo=50):
    """
    Detecta el tipo de dato de cada columna en un DataFrame de pandas.
    """
    resumen = []
    for col in df.columns:
        serie = df[col]
        tipo = None
        detalles = ""
        if serie.isnull().all():
            tipo = "vacía"
            detalles = "Todos los valores son NaN"
        elif pd.api.types.is_bool_dtype(serie):
            tipo = "booleano"
        elif pd.api.types.is_datetime64_any_dtype(serie):
            tipo = "fecha/tiempo"
        elif pd.api.types.is_numeric_dtype(serie):
            tipo = "numérico"
        elif pd.api.types.is_object_dtype(serie) or pd.api.types.is_categorical_dtype(serie):
            n_unicos = serie.nunique(dropna=True)
            muestra = serie.dropna().astype(str).sample(min(10, len(serie.dropna())), random_state=1) if len(serie.dropna()) > 0 else []
            longitudes = muestra.map(len) if len(muestra) > 0 else []
            if n_unicos <= umbral_cardinalidad:
                tipo = "categórico"
                detalles = f"{n_unicos} valores únicos"
            elif len(longitudes) > 0 and np.mean(longitudes) > umbral_texto_largo:
                tipo = "texto libre"
                detalles = f"Longitud promedio texto: {np.mean(longitudes):.1f}"
            elif serie.apply(lambda x: isinstance(x, (int, float, np.number))).any():
                tipo = "mixto"
                detalles = "Contiene mezcla de tipos (numérico y texto)"
            else:
                tipo = "texto"
        else:
            tipo = "requiere revisión"
            detalles = f"Tipo detectado: {serie.dtype}"
        resumen.append({
            "columna": col,
            "tipo_detectado": tipo,
            "detalles": detalles
        })
    return pd.DataFrame(resumen)

# =====================
# Función para sugerir visualizaciones (simplificada)
# =====================
def sugerir_visualizaciones(tipo, df=None, col=None):
    """
    Sugiere visualizaciones según el tipo de variable.
    Estructura preparada para futuras visualizaciones bivariadas.
    """
    sugerencias = []
    
    # Visualizaciones univariadas básicas
    if tipo == "numérico":
        sugerencias = ["Histograma", "Boxplot", "Estadísticas descriptivas"]
    elif tipo == "categórico":
        sugerencias = ["Gráfico de barras", "Gráfico de torta", "Tabla de frecuencias"]
    elif tipo == "booleano":
        sugerencias = ["Gráfico de barras", "Tabla de frecuencias"]
    elif tipo == "fecha/tiempo":
        sugerencias = ["Serie temporal", "Distribución temporal"]
    elif tipo == "texto libre":
        sugerencias = ["Tabla de frecuencias", "Longitud de texto"]
    else:
        sugerencias = ["Tabla de frecuencias"]
    
    # Nota: Aquí se pueden agregar sugerencias bivariadas en el futuro
    # Por ejemplo: "Dispersión", "Boxplot agrupado", "Correlación", etc.
    
    return sugerencias

# =====================
# Función para cargar archivos de distintos formatos
# =====================
def cargar_archivo(uploaded_file):
    """Carga archivos .csv, .xlsx, .sav, .dta en un DataFrame."""
    nombre = uploaded_file.name.lower()
    if nombre.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif nombre.endswith('.xlsx') or nombre.endswith('.xls'):
        return pd.read_excel(uploaded_file)
    elif nombre.endswith('.sav'):
        df, meta = pyreadstat.read_sav(uploaded_file)
        return df
    elif nombre.endswith('.dta'):
        df, meta = pyreadstat.read_dta(uploaded_file)
        return df
    else:
        raise ValueError("Formato de archivo no soportado")

# =====================
# Funciones de visualización (estructura preparada para expansión)
# =====================
def crear_visualizacion(df, col, tipo_vis, tipo_col):
    """
    Crea visualizaciones usando Plotly para mejor interactividad.
    Estructura preparada para futuras visualizaciones bivariadas.
    """
    
    # Limpiar datos
    datos_limpios = df[col].dropna()
    
    if tipo_vis == "Histograma" and tipo_col == "numérico":
        fig = px.histogram(
            df, x=col, 
            title=f"Distribución de {col}",
            nbins=30,
            marginal="box"
        )
        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Frecuencia",
            showlegend=False
        )
        return fig
    
    elif tipo_vis == "Boxplot" and tipo_col == "numérico":
        fig = px.box(
            df, y=col,
            title=f"Boxplot de {col}"
        )
        fig.update_layout(
            yaxis_title=col,
            showlegend=False
        )
        return fig
    
    elif tipo_vis == "Gráfico de barras" and tipo_col in ["categórico", "booleano"]:
        # Limitar a top 20 categorías para mejor visualización
        frecuencias = df[col].value_counts().head(20)
        fig = px.bar(
            x=frecuencias.index, 
            y=frecuencias.values,
            title=f"Frecuencia de {col}",
            labels={'x': col, 'y': 'Frecuencia'}
        )
        fig.update_layout(showlegend=False)
        return fig
    
    elif tipo_vis == "Gráfico de torta" and tipo_col in ["categórico", "booleano"]:
        # Limitar a top 10 categorías para mejor visualización
        frecuencias = df[col].value_counts().head(10)
        fig = px.pie(
            values=frecuencias.values,
            names=frecuencias.index,
            title=f"Distribución de {col}"
        )
        return fig
    
    elif tipo_vis == "Serie temporal" and tipo_col == "fecha/tiempo":
        # Agrupar por fecha y contar
        df_temp = df.copy()
        df_temp[col] = pd.to_datetime(df_temp[col])
        serie_temporal = df_temp[col].value_counts().sort_index()
        
        fig = px.line(
            x=serie_temporal.index,
            y=serie_temporal.values,
            title=f"Serie temporal de {col}",
            labels={'x': 'Fecha', 'y': 'Frecuencia'}
        )
        return fig
    
    elif tipo_vis == "Estadísticas descriptivas" and tipo_col == "numérico":
        # Crear tabla de estadísticas
        stats = df[col].describe()
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Estadística', 'Valor']),
            cells=dict(values=[
                ['Conteo', 'Media', 'Desv. Est.', 'Mínimo', '25%', 'Mediana', '75%', 'Máximo'],
                [f"{stats['count']:.0f}", f"{stats['mean']:.2f}", f"{stats['std']:.2f}", 
                 f"{stats['min']:.2f}", f"{stats['25%']:.2f}", f"{stats['50%']:.2f}", 
                 f"{stats['75%']:.2f}", f"{stats['max']:.2f}"]
            ])
        )])
        fig.update_layout(title=f"Estadísticas descriptivas de {col}")
        return fig
    
    elif tipo_vis == "Tabla de frecuencias":
        frecuencias = df[col].value_counts()
        fig = go.Figure(data=[go.Table(
            header=dict(values=[col, 'Frecuencia', 'Porcentaje']),
            cells=dict(values=[
                frecuencias.index.astype(str),
                frecuencias.values,
                [f"{(freq/len(df)*100):.1f}%" for freq in frecuencias.values]
            ])
        )])
        fig.update_layout(title=f"Tabla de frecuencias de {col}")
        return fig
    
    elif tipo_vis == "Longitud de texto" and tipo_col == "texto libre":
        longitudes = df[col].astype(str).str.len()
        fig = px.histogram(
            x=longitudes,
            title=f"Distribución de longitudes de texto en {col}",
            nbins=30
        )
        fig.update_layout(
            xaxis_title="Longitud del texto",
            yaxis_title="Frecuencia"
        )
        return fig
    
    else:
        # Fallback: tabla de frecuencias
        frecuencias = df[col].value_counts().head(20)
        fig = go.Figure(data=[go.Table(
            header=dict(values=[col, 'Frecuencia']),
            cells=dict(values=[
                frecuencias.index.astype(str),
                frecuencias.values
            ])
        )])
        fig.update_layout(title=f"Frecuencias de {col}")
        return fig

# =====================
# Wizard de visualización de datos
# =====================
st.set_page_config(
    page_title="Asistente de Visualización de Datos", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Asistente de Visualización de Datos")
st.markdown("**Guía paso a paso para crear visualizaciones efectivas de tus datos**")

# Inicialización del estado de la sesión
if 'paso' not in st.session_state:
    st.session_state.paso = 1
if 'df' not in st.session_state:
    st.session_state.df = None
if 'resumen' not in st.session_state:
    st.session_state.resumen = None
if 'columna_seleccionada' not in st.session_state:
    st.session_state.columna_seleccionada = None
if 'tipo_columna' not in st.session_state:
    st.session_state.tipo_columna = None
if 'visualizacion' not in st.session_state:
    st.session_state.visualizacion = None

# Sidebar: navegación y ayuda
with st.sidebar:
    st.title("🗺️ Navegación")
    st.progress(st.session_state.paso / 7)
    
    st.markdown("""
    **Pasos a seguir:**
    1. 📁 Cargar archivo
    2. 📊 Resumen de datos
    3. 🔍 Detección de tipos
    4. 💡 Sugerencias
    5. 🎨 Selección de gráfico
    6. 📈 Visualización
    7. 💾 Exportar resultados
    """)
    
    st.markdown("---")
    st.markdown("**💡 Tip:** En el futuro podrás crear visualizaciones que relacionen dos variables.")
    
    if st.button("🔄 Reiniciar Asistente"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Paso 1: Cargar archivo
def paso_1():
    st.header("📁 Paso 1: Cargar archivo de datos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        archivo = st.file_uploader(
            "Carga tu archivo de datos", 
            type=["csv", "xlsx", "xls", "sav", "dta"],
            help="Formatos soportados: CSV, Excel, SPSS (.sav), Stata (.dta)"
        )
        
        if archivo is not None:
            try:
                with st.spinner("Cargando archivo..."):
                    df = cargar_archivo(archivo)
                    st.session_state.df = df
                
                st.success(f"✅ Archivo cargado correctamente")
                st.info(f"📊 **Datos:** {df.shape[0]:,} filas × {df.shape[1]} columnas")
                
                if st.button("➡️ Continuar al siguiente paso", type="primary"):
                    st.session_state.paso = 2
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error al cargar archivo: {str(e)}")
                st.info("💡 Asegúrate de que el archivo no esté corrupto y sea del formato correcto.")
        else:
            st.info("📤 Por favor, sube un archivo de datos para comenzar.")
    
    with col2:
        st.markdown("""
        **📋 Formatos soportados:**
        - **CSV** (.csv)
        - **Excel** (.xlsx, .xls)
        - **SPSS** (.sav)
        - **Stata** (.dta)
        
        **💡 Consejo:** Para mejores resultados, asegúrate de que tu archivo tenga encabezados en la primera fila.
        """)

# Paso 2: Resumen automático
def paso_2():
    st.header("📊 Paso 2: Resumen automático de los datos")
    
    df = st.session_state.df
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Información general")
        st.metric("Filas", f"{df.shape[0]:,}")
        st.metric("Columnas", df.shape[1])
        st.metric("Memoria utilizada", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        st.subheader("🔍 Primeras filas")
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.subheader("📈 Tipos de datos")
        tipos_df = pd.DataFrame(df.dtypes, columns=["Tipo"])
        tipos_df["No nulos"] = df.count()
        tipos_df["% Completitud"] = (df.count() / len(df) * 100).round(1)
        st.dataframe(tipos_df, use_container_width=True)
        
        st.subheader("⚠️ Valores faltantes")
        missing_df = pd.DataFrame({
            "Columna": df.columns,
            "Valores faltantes": df.isnull().sum(),
            "% Faltantes": (df.isnull().sum() / len(df) * 100).round(1)
        })
        st.dataframe(missing_df[missing_df["Valores faltantes"] > 0], use_container_width=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("➡️ Continuar", type="primary"):
            st.session_state.paso = 3
            st.rerun()
    with col1:
        if st.button("⬅️ Volver"):
            st.session_state.paso = 1
            st.rerun()

# Paso 3: Detección automática de tipo de variable
def paso_3():
    st.header("🔍 Paso 3: Detección automática de tipos de variables")
    
    df = st.session_state.df
    
    with st.spinner("Analizando tipos de variables..."):
        resumen = detectar_tipos_columnas(df)
        st.session_state.resumen = resumen
    
    st.subheader("📋 Resultados del análisis")
    st.dataframe(resumen, use_container_width=True)
    
    # Resumen visual de tipos
    st.subheader("📊 Distribución de tipos de variables")
    tipo_counts = resumen["tipo_detectado"].value_counts()
    fig = px.pie(
        values=tipo_counts.values,
        names=tipo_counts.index,
        title="Distribución de tipos de variables detectados"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 Los tipos detectados automáticamente te ayudarán a elegir las mejores visualizaciones.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("➡️ Continuar", type="primary"):
            st.session_state.paso = 4
            st.rerun()
    with col1:
        if st.button("⬅️ Volver"):
            st.session_state.paso = 2
            st.rerun()

# Paso 4: Sugerencia de visualizaciones
def paso_4():
    st.header("💡 Paso 4: Sugerencias de visualización")
    
    df = st.session_state.df
    resumen = st.session_state.resumen
    
    col = st.selectbox(
        "🎯 Selecciona la variable que quieres visualizar:",
        resumen["columna"],
        help="Elige la variable que te interesa analizar"
    )
    
    tipo = resumen[resumen["columna"] == col]["tipo_detectado"].values[0]
    st.session_state.columna_seleccionada = col
    st.session_state.tipo_columna = tipo
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"📊 Variable: {col}")
        st.info(f"**Tipo detectado:** {tipo}")
        
        # Mostrar estadísticas rápidas
        if tipo == "numérico":
            stats = df[col].describe()
            st.metric("Media", f"{stats['mean']:.2f}")
            st.metric("Mediana", f"{stats['50%']:.2f}")
            st.metric("Desv. Est.", f"{stats['std']:.2f}")
        elif tipo in ["categórico", "booleano"]:
            st.metric("Valores únicos", df[col].nunique())
            st.metric("Valor más frecuente", df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A")
    
    with col2:
        st.subheader("🎨 Visualizaciones sugeridas")
        sugerencias = sugerir_visualizaciones(tipo, df, col)
        
        for i, sugerencia in enumerate(sugerencias, 1):
            st.write(f"{i}. **{sugerencia}**")
        
        st.info("💡 Estas sugerencias están basadas en el tipo de variable detectado.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("➡️ Continuar", type="primary"):
            st.session_state.paso = 5
            st.rerun()
    with col1:
        if st.button("⬅️ Volver"):
            st.session_state.paso = 3
            st.rerun()

# Paso 5: Selección de visualización
def paso_5():
    st.header("🎨 Paso 5: Selección de visualización")
    
    df = st.session_state.df
    col = st.session_state.columna_seleccionada
    tipo = st.session_state.tipo_columna
    
    st.subheader(f"📊 Variable seleccionada: {col}")
    st.info(f"**Tipo:** {tipo}")
    
    sugerencias = sugerir_visualizaciones(tipo, df, col)
    
    vis = st.selectbox(
        "🎨 Elige el tipo de visualización:",
        sugerencias,
        help="Selecciona la visualización que mejor represente tus datos"
    )
    
    st.session_state.visualizacion = vis
    
    # Mostrar descripción de la visualización
    descripciones = {
        "Histograma": "Muestra la distribución de frecuencias de una variable numérica",
        "Boxplot": "Visualiza la distribución y detecta valores atípicos",
        "Gráfico de barras": "Compara frecuencias entre categorías",
        "Gráfico de torta": "Muestra proporciones de una variable categórica",
        "Serie temporal": "Visualiza cambios a lo largo del tiempo",
        "Estadísticas descriptivas": "Tabla con medidas estadísticas resumidas",
        "Tabla de frecuencias": "Lista detallada de frecuencias y porcentajes",
        "Longitud de texto": "Distribución de longitudes de texto"
    }
    
    st.info(f"**Descripción:** {descripciones.get(vis, 'Visualización de datos')}")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("➡️ Crear visualización", type="primary"):
            st.session_state.paso = 6
            st.rerun()
    with col1:
        if st.button("⬅️ Volver"):
            st.session_state.paso = 4
            st.rerun()

# Paso 6: Mostrar gráfico o indicador
def paso_6():
    st.header("📈 Paso 6: Visualización de los datos")
    
    df = st.session_state.df
    col = st.session_state.columna_seleccionada
    vis = st.session_state.visualizacion
    tipo = st.session_state.tipo_columna
    
    st.subheader(f"📊 Resultado: {vis} de '{col}'")
    
    try:
        fig = crear_visualizacion(df, col, vis, tipo)
        st.plotly_chart(fig, use_container_width=True)
        
        # Información adicional
        with st.expander("📋 Información adicional"):
            st.write(f"**Variable:** {col}")
            st.write(f"**Tipo:** {tipo}")
            st.write(f"**Visualización:** {vis}")
            st.write(f"**Total de registros:** {len(df):,}")
            st.write(f"**Registros válidos:** {len(df[col].dropna()):,}")
            st.write(f"**Registros faltantes:** {df[col].isnull().sum():,}")
            
            if tipo == "numérico":
                stats = df[col].describe()
                st.write("**Estadísticas descriptivas:**")
                st.dataframe(pd.DataFrame(stats).T)
    
    except Exception as e:
        st.error(f"❌ Error al crear la visualización: {str(e)}")
        st.info("💡 Intenta con otra visualización o revisa los datos.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("➡️ Continuar", type="primary"):
            st.session_state.paso = 7
            st.rerun()
    with col1:
        if st.button("⬅️ Volver"):
            st.session_state.paso = 5
            st.rerun()

# Paso 7: Exportar resultados
def paso_7():
    st.header("💾 Paso 7: Exportar resultados")
    
    df = st.session_state.df
    resumen = st.session_state.resumen
    
    st.success("🎉 ¡Has completado el asistente de visualización!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Descargar datos")
        st.download_button(
            "📊 Datos originales (CSV)",
            df.to_csv(index=False),
            file_name="datos_originales.csv",
            mime="text/csv"
        )
        
        st.download_button(
            "📋 Resumen de tipos (CSV)",
            resumen.to_csv(index=False),
            file_name="resumen_tipos_variables.csv",
            mime="text/csv"
        )
    
    with col2:
        st.subheader("📈 Próximas mejoras")
        st.markdown("""
        **🚀 En futuras versiones podrás:**
        - 📊 Crear visualizaciones con dos variables
        - 🎨 Personalizar colores y estilos
        - 📱 Exportar gráficos como imágenes
        - 🔄 Guardar y cargar configuraciones
        - 📈 Análisis estadísticos avanzados
        """)
    
    st.markdown("---")
    st.info("💡 **Consejo:** Para exportar la visualización como imagen, usa el botón de descarga de Plotly en la esquina superior derecha del gráfico.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Crear nueva visualización", type="primary"):
            st.session_state.paso = 4
            st.rerun()
    with col1:
        if st.button("⬅️ Volver"):
            st.session_state.paso = 6
            st.rerun()

# =====================
# Controlador de pasos
# =====================
if st.session_state.paso == 1:
    paso_1()
elif st.session_state.paso == 2:
    paso_2()
elif st.session_state.paso == 3:
    paso_3()
elif st.session_state.paso == 4:
    paso_4()
elif st.session_state.paso == 5:
    paso_5()
elif st.session_state.paso == 6:
    paso_6()
elif st.session_state.paso == 7:
    paso_7() 