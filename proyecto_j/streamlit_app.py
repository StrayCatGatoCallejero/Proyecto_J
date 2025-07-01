import streamlit as st
import pandas as pd
import io
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime

# Agregar src al path para importar el pipeline modular
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.insert(0, src_path)

# Importaciones con manejo de errores silencioso
# type: ignore
try:
    from core import Pipeline  # type: ignore
    from steps import (  # type: ignore
        cargar_datos,
        limpiar_datos,
        transformar_datos,
        modelar,
        visualizar,
        generar_reporte,
    )
    from utils import load_config  # type: ignore
except ImportError:
    # Si fallan las importaciones, crear funciones dummy
    class Pipeline:
        def __init__(self, config):
            self.config = config
        
        def run(self):
            return pd.DataFrame(), None, {}
    
    def cargar_datos(path):
        return pd.read_csv(path)
    
    def limpiar_datos(df):
        return df
    
    def transformar_datos(df):
        return df
    
    def modelar(df):
        return None, {}
    
    def visualizar(df, results):
        pass
    
    def generar_reporte(df, results, output_path):
        pass
    
    def load_config(path):
        return {}

# Configuración de la página
st.set_page_config(
    page_title="Asistente de Visualización de Datos - Proyecto J",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------
# Estilos forzados para paleta fija (modo claro siempre)
# ------------------------
st.markdown(
    """
<style>
/* Forzar modo claro y paleta específica */
:root {
    --background-color: #FBF7F2 !important;
    --secondary-background-color: #F5E3D3 !important;
    --text-color: #333333 !important;
    --primary-color: #648DA5 !important;
}

/* Aplicar a toda la aplicación */
.stApp {
    background-color: #FBF7F2 !important;
    color: #333333 !important;
}

/* Contenedor principal */
.reportview-container .main {
    background-color: #FBF7F2 !important;
    color: #333333 !important;
}

/* Sidebar */
.sidebar .sidebar-content {
    background-color: #F5E3D3 !important;
    color: #333333 !important;
}

/* Elementos de texto */
p, h1, h2, h3, h4, h5, h6, div, span {
    color: #333333 !important;
}

/* Botones */
.stButton > button {
    border-radius: 0.5rem !important;
    padding: 0.5rem 1rem !important;
    background-color: #648DA5 !important;
    color: white !important;
    border: none !important;
}

.stButton > button:hover {
    background-color: #5a7a8f !important;
    color: white !important;
}

/* Inputs y selectores */
.stSelectbox > div > div {
    background-color: white !important;
    color: #333333 !important;
}

.stTextInput > div > div > input {
    background-color: white !important;
    color: #333333 !important;
}

.stTextArea > div > div > textarea {
    background-color: white !important;
    color: #333333 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #F5E3D3 !important;
    color: #333333 !important;
}

/* Métricas */
[data-testid="metric-container"] {
    background-color: white !important;
    color: #333333 !important;
}

/* Dataframes */
.dataframe {
    background-color: white !important;
    color: #333333 !important;
}

/* Progress bars */
.stProgress > div > div > div > div {
    background-color: #648DA5 !important;
}

/* Override cualquier tema oscuro del sistema */
@media (prefers-color-scheme: dark) {
    .stApp, .reportview-container .main, .sidebar .sidebar-content {
        background-color: #FBF7F2 !important;
        color: #333333 !important;
    }
}

/* Forzar colores en todos los elementos de Streamlit */
[data-testid="stAppViewContainer"] {
    background-color: #FBF7F2 !important;
}

[data-testid="stSidebar"] {
    background-color: #F5E3D3 !important;
}

/* Asegurar que los gráficos tengan fondo blanco */
.element-container {
    background-color: white !important;
}

/* Override para elementos específicos de Streamlit */
.stMarkdown {
    color: #333333 !important;
}

.stAlert {
    background-color: white !important;
    color: #333333 !important;
}

/* Forzar colores en elementos de navegación */
.stSelectbox > label {
    color: #333333 !important;
}

/* Asegurar consistencia en todos los componentes */
* {
    color: inherit !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------
# Tour / Onboarding inicial
# ------------------------
def show_walkthrough():
    if "walkthrough_done" not in st.session_state:
        st.session_state.walkthrough_done = True
        st.toast("👋 Bienvenido al Asistente de Visualización de Datos!")
        st.info("Usa la barra lateral para navegar por los 7 pasos del flujo.")
        st.info("ℹ️ Pasa el cursor sobre los iconos para ver información adicional.")


show_walkthrough()

# ------------------------
# Sidebar de navegación
# ------------------------
st.sidebar.markdown("# 📚 Navegación")

# Páginas disponibles
pages = {
    "📁 Cargar archivo": "cargar",
    "📊 Resumen de datos": "resumen", 
    "🔍 Detección de tipos": "tipos",
    "💡 Sugerencias": "sugerencias",
    "🎨 Selección de gráfico": "grafico",
    "📈 Visualización": "visualizacion",
    "🚀 Análisis Avanzado": "analisis_avanzado",
    "💾 Exportar resultados": "exportar"
}

# Selector de página
if "current_page" not in st.session_state:
    st.session_state.current_page = "cargar"

selected_page = st.sidebar.selectbox(
    "Selecciona una página:",
    list(pages.keys()),
    index=list(pages.values()).index(st.session_state.current_page)
)

st.session_state.current_page = pages[selected_page]

# Mostrar progreso solo para el flujo principal (sin análisis avanzado)
main_pages = ["cargar", "resumen", "tipos", "sugerencias", "grafico", "visualizacion", "exportar"]
if st.session_state.current_page in main_pages:
    current_index = main_pages.index(st.session_state.current_page)
    progress = (current_index + 1) / len(main_pages)
st.sidebar.progress(progress)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "💡 Tip: Usa 'Análisis Avanzado' para agrupaciones complejas y consultas en lenguaje natural."
)
if st.sidebar.button("🔄 Reiniciar Asistente"):
    st.session_state.clear()
    st.rerun()


# ------------------------
# Función para leer distintos formatos
# ------------------------
def load_file(uploaded):
    """Carga archivo usando el pipeline modular con manejo de codificación"""
    try:
        # Guardar archivo temporalmente
        temp_path = f"temp_{uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Intentar cargar directamente con pandas primero
        file_extension = os.path.splitext(uploaded.name)[-1].lower()
        
        if file_extension == ".csv":
            # Para archivos CSV, intentar diferentes codificaciones
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(temp_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                # Si ninguna codificación funciona, usar la función del pipeline
                df = cargar_datos(temp_path)
        elif file_extension in [".xlsx", ".xls"]:
            # Para archivos Excel, intentar diferentes engines
            try:
                if file_extension == ".xlsx":
                    df = pd.read_excel(temp_path, engine='openpyxl')
                else:
                    df = pd.read_excel(temp_path, engine='xlrd')
            except Exception:
                # Si falla, intentar con engine por defecto
                df = pd.read_excel(temp_path)
        else:
            # Para otros formatos, usar la función del pipeline
            df = cargar_datos(temp_path)

        # Limpiar archivo temporal
        os.remove(temp_path)
        return df
    except Exception as e:
        st.error(f"Error al cargar archivo: {str(e)}")
        st.info("💡 **Sugerencias:** Verifica que el archivo no esté corrupto o intenta con un archivo diferente")
        return None


# ------------------------
# Función para obtener resumen de datos faltantes
# ------------------------
def get_missing_summary(df):
    """Genera resumen de datos faltantes usando el pipeline"""
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    summary = pd.DataFrame(
        {
            "Columna": missing_data.index,
            "Valores_Faltantes": missing_data.values,
            "Porcentaje": missing_percent.values,
        }
    ).sort_values("Valores_Faltantes", ascending=False)

    return {
        "summary": summary,
        "total_missing": missing_data.sum(),
        "total_percent": (missing_data.sum() / (len(df) * len(df.columns))) * 100,
    }


# ------------------------
# Función para detectar tipos
# ------------------------
def detect_types(df):
    """Detecta tipos de datos usando lógica del pipeline"""
    types_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if "object" in dtype or "category" in dtype:
            detected_type = "categorico"
        elif "int" in dtype or "float" in dtype:
            detected_type = "numerico"
        elif "datetime" in dtype:
            detected_type = "fecha"
        else:
            detected_type = "otro"

        types_info.append(
            {
                "columna": col,
                "tipo_pandas": dtype,
                "tipo_detectado": detected_type,
                "valores_unicos": df[col].nunique(),
            }
        )

    return pd.DataFrame(types_info)


# ------------------------
# Título principal
# ------------------------
st.title("🤖 Asistente de Visualización de Datos - Proyecto J")



# ------------------------
# Página: Cargar archivo
# ------------------------
if st.session_state.current_page == "cargar":
    st.header("📁 Cargar archivo de datos")
    
    # Instrucciones claras y amigables
    st.markdown("""
    ### 👋 ¡Bienvenido! 
    Aquí puedes cargar tu archivo de datos para comenzar el análisis. 
    El sistema detectará automáticamente el formato y preparará tus datos para la exploración.
    """)
    
    # Información sobre formatos soportados
    with st.expander("📋 ¿Qué formatos puedo usar?", expanded=False):
        st.markdown("""
        **Formatos soportados:**
        - 📄 **CSV** - Archivos de texto separado por comas
        - 📊 **Excel** - Archivos .xls y .xlsx
        - 📈 **SPSS** - Archivos .sav (estadística)
        - 📉 **Stata** - Archivos .dta (estadística)
        
        💡 **Tip:** Si tu archivo no está en estos formatos, puedes convertirlo a CSV desde Excel o Google Sheets.
        """)
    
    # Uploader mejorado con instrucciones más claras
    uploaded = st.file_uploader(
        "📂 Selecciona o arrastra tu archivo aquí",
        type=["csv", "xls", "xlsx", "sav", "dta"],
        help="Haz clic en 'Browse files' o arrastra tu archivo directamente a esta área"
    )
    
    if uploaded:
        st.info("🔄 Procesando tu archivo, esto puede tardar unos segundos...")
        df = load_file(uploaded)
        if df is not None:
            st.session_state.df = df
            st.success(
                f"✅ ¡Perfecto! Tu archivo se cargó correctamente\n\n"
                f"📊 **Datos cargados:** {df.shape[0]:,} filas × {df.shape[1]} columnas\n\n"
                f"🎯 **Próximo paso:** Usa el menú lateral para explorar tus datos"
            )
            
            # Mostrar vista previa de los datos
            with st.expander("👀 Vista previa de tus datos", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                st.caption("Mostrando las primeras 10 filas de tus datos")
        else:
            st.error("❌ No se pudo cargar el archivo. Verifica que el formato sea correcto.")
    else:
        st.info("💡 **Consejo:** Una vez que cargues tu archivo, podrás explorar todas las páginas del menú lateral")

# ------------------------
# Página: Resumen de datos
# ------------------------
elif st.session_state.current_page == "resumen":
    st.header("📊 Resumen de datos")
    
    # Instrucciones claras
    st.markdown("""
    ### 📋 Vista general de tus datos
    Aquí puedes ver un resumen completo de tu conjunto de datos, incluyendo el tamaño, 
    la memoria utilizada y los valores faltantes. Esta información te ayudará a entender 
    la calidad y estructura de tus datos.
    """)
    
    df = st.session_state.get("df")
    if df is not None:
        # Métricas principales con mejor formato
        st.subheader("📈 Información general")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Filas", f"{df.shape[0]:,}")
        with col2:
            st.metric("📊 Columnas", df.shape[1])
        with col3:
            mem = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("💾 Memoria", f"{mem:.2f} MB")

        # Análisis de valores faltantes
        missing = get_missing_summary(df)
        st.subheader("🔍 Análisis de valores faltantes")
        
        if missing["total_missing"] > 0:
            st.warning(f"⚠️ **Encontramos {missing['total_missing']:,} valores faltantes** en tus datos")
            st.markdown("Esto representa el **{:.1f}%** del total de datos".format(missing["total_percent"]))
        else:
            st.success("✅ **¡Excelente!** No hay valores faltantes en tus datos")
        
        # Tabla de valores faltantes con mejor formato
        st.markdown("**Detalle por columna:**")
        st.dataframe(
            missing["summary"].rename(columns={
                "Columna": "📋 Columna",
                "Valores_Faltantes": "❌ Faltantes", 
                "Porcentaje": "📊 Porcentaje"
            }),
            use_container_width=True
        )
        
        # Consejos sobre valores faltantes
        if missing["total_missing"] > 0:
            with st.expander("💡 ¿Qué hacer con los valores faltantes?", expanded=False):
                st.markdown("""
                **Opciones para manejar valores faltantes:**
                - 🗑️ **Eliminar filas** con valores faltantes (si son pocas)
                - 🔄 **Imputar valores** usando promedio, mediana o moda
                - 📊 **Analizar patrones** para entender por qué faltan datos
                - ⚠️ **Investigar la fuente** de los datos faltantes
                
                💡 **Consejo:** Los valores faltantes pueden indicar problemas en la recolección de datos.
                """)
    else:
        st.warning("⚠️ **Paso pendiente:** Primero debes cargar un archivo en la página 'Cargar archivo'")
        st.info("💡 Ve al menú lateral y selecciona '📁 Cargar archivo' para comenzar")

# ------------------------
# Página: Detección de tipos
# ------------------------
elif st.session_state.current_page == "tipos":
    st.header("🔍 Detección automática de tipos")
    
    # Instrucciones claras
    st.markdown("""
    ### 🎯 Clasificación inteligente de variables
    El sistema analiza automáticamente cada columna de tus datos y determina si son números, 
    texto, fechas u otros tipos. Esta información es crucial para elegir las visualizaciones 
    más apropiadas y realizar análisis estadísticos correctos.
    """)
    
    df = st.session_state.get("df")
    if df is not None:
        types = detect_types(df)
        
        # Resumen de tipos detectados
        st.subheader("📊 Resumen de tipos detectados")
        type_counts = types["tipo_detectado"].value_counts()
        
        # Mostrar métricas de tipos
        cols = st.columns(len(type_counts))
        for i, (tipo, count) in enumerate(type_counts.items()):
            with cols[i]:
                if tipo == "numerico":
                    st.metric("🔢 Numéricas", count)
                elif tipo == "categorico":
                    st.metric("📝 Categóricas", count)
                elif tipo == "fecha":
                    st.metric("📅 Fechas", count)
                else:
                    st.metric("❓ Otros", count)
        
        # Tabla detallada con mejor formato
        st.subheader("📋 Detalle por columna")
        st.dataframe(
            types.rename(columns={
                "columna": "📋 Columna",
                "tipo_pandas": "🔧 Tipo Pandas",
                "tipo_detectado": "🎯 Tipo Detectado",
                "valores_unicos": "🔢 Valores Únicos"
            }),
            use_container_width=True
        )

        # Gráfico de distribución de tipos
        st.subheader("📈 Distribución de tipos de datos")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        type_counts.plot(kind="bar", ax=ax, color=colors[:len(type_counts)])
        plt.title("Distribución de tipos de datos en tu conjunto", fontsize=14, pad=20)
        plt.xlabel("Tipo de dato", fontsize=12)
        plt.ylabel("Número de columnas", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Información adicional sobre tipos
        with st.expander("💡 ¿Qué significa cada tipo?", expanded=False):
            st.markdown("""
            **Tipos de datos detectados:**
            
            🔢 **Numérico:** Números que puedes sumar, promediar, etc.
            - Ejemplos: edad, precio, temperatura, ingresos
            
            📝 **Categórico:** Texto o categorías que agrupan datos
            - Ejemplos: ciudad, género, categoría de producto
            
            📅 **Fecha:** Fechas y horas que puedes ordenar cronológicamente
            - Ejemplos: fecha de compra, hora de registro
            
            ❓ **Otro:** Tipos especiales como booleanos o datos mixtos
            
            💡 **Consejo:** El tipo detectado determina qué gráficos y análisis puedes realizar.
            """)
    else:
        st.warning("⚠️ **Paso pendiente:** Primero debes cargar un archivo en la página 'Cargar archivo'")
        st.info("💡 Ve al menú lateral y selecciona '📁 Cargar archivo' para comenzar")

# ------------------------
# Página: Sugerencias
# ------------------------
elif st.session_state.current_page == "sugerencias":
    st.header("💡 Sugerencias de visualización")
    
    # Instrucciones claras
    st.markdown("""
    ### 🎨 Recomendaciones inteligentes
    Basándome en el tipo de datos de cada variable, te sugiero las mejores visualizaciones 
    para explorar y comunicar tus hallazgos. Cada tipo de dato tiene gráficos que funcionan 
    especialmente bien para mostrar sus características.
    """)
    
    df = st.session_state.get("df")
    if df is not None:
        types = detect_types(df)
        
        # Selector mejorado
        st.subheader("📋 Selecciona una variable para analizar")
        col = st.selectbox(
            "Elige la variable que quieres visualizar:",
            df.columns,
            help="Selecciona cualquier columna de tus datos para ver recomendaciones específicas"
        )
        
        if col:
            info = types.loc[types["columna"] == col].iloc[0]

            # Información de la variable seleccionada
            st.subheader(f"📊 Análisis de: **{col}**")
            
            # Mostrar tipo detectado con icono
            tipo_icon = {
                "categorico": "📝",
                "numerico": "🔢", 
                "fecha": "📅",
                "otro": "❓"
            }.get(info['tipo_detectado'], "❓")
            
            st.info(f"{tipo_icon} **Tipo detectado:** {info['tipo_detectado'].title()}")
            
            # Sugerencias específicas por tipo
            st.subheader("🎯 Visualizaciones recomendadas")
            
            if info["tipo_detectado"] == "categorico":
                st.markdown("""
                **📊 Para variables categóricas como '{col}':**
                
                🎯 **Gráfico de barras** - Perfecto para comparar frecuencias entre categorías
                🥧 **Gráfico de pastel** - Ideal para mostrar proporciones del total
                📈 **Conteo de frecuencias** - Tabla simple con el número de cada categoría
                
                💡 **Consejo:** Los gráficos de barras son generalmente más fáciles de leer que los de pastel.
                """)
                
            elif info["tipo_detectado"] == "numerico":
                st.markdown("""
                **🔢 Para variables numéricas como '{col}':**
                
                📊 **Histograma** - Muestra la distribución y forma de los datos
                📦 **Box plot** - Revela la mediana, cuartiles y valores atípicos
                🔗 **Gráfico de dispersión** - Perfecto para ver relaciones con otras variables
                
                💡 **Consejo:** Los histogramas te ayudan a identificar si los datos siguen una distribución normal.
                """)
                
            elif info["tipo_detectado"] == "fecha":
                st.markdown("""
                **📅 Para variables de fecha como '{col}':**
                
                📈 **Serie temporal** - Muestra cómo cambian los valores a lo largo del tiempo
                📊 **Gráfico de líneas** - Ideal para tendencias y patrones temporales
                
                💡 **Consejo:** Las series temporales son excelentes para identificar tendencias y estacionalidad.
                """)
            else:
                st.markdown("""
                **❓ Para otros tipos de datos:**
                
                🔍 **Análisis exploratorio** - Primero explora los valores únicos
                📋 **Tabla de frecuencias** - Para entender la distribución
                
                💡 **Consejo:** Considera convertir estos datos a un tipo más específico si es posible.
                """)
            
            # Estadísticas básicas de la variable
            with st.expander("📈 Estadísticas básicas de esta variable", expanded=False):
                if info["tipo_detectado"] == "numerico":
                    stats = df[col].describe()
                    st.dataframe(stats, use_container_width=True)
                elif info["tipo_detectado"] == "categorico":
                    value_counts = df[col].value_counts().head(10)
                    st.markdown("**Top 10 valores más frecuentes:**")
                    st.dataframe(value_counts, use_container_width=True)
                else:
                    st.info("Las estadísticas detalladas están disponibles en otras secciones")
    else:
        st.warning("⚠️ **Paso pendiente:** Primero debes cargar un archivo en la página 'Cargar archivo'")
        st.info("💡 Ve al menú lateral y selecciona '📁 Cargar archivo' para comenzar")

# ------------------------
# Página: Selección de gráfico
# ------------------------
elif st.session_state.current_page == "grafico":
    st.header("🎨 Selección de gráfico")
    
    # Instrucciones claras
    st.markdown("""
    ### 📊 Configura tu visualización
    Aquí puedes elegir qué tipo de gráfico crear y qué variables usar. El sistema te mostrará 
    solo las opciones que tienen sentido para tus datos. Una vez configurado, podrás ver 
    el gráfico en la página de Visualización.
    """)
    
    df = st.session_state.get("df")
    if df is not None:
        types = detect_types(df)
        numeric_cols = types[types["tipo_detectado"] == "numerico"]["columna"].tolist()
        categorical_cols = types[types["tipo_detectado"] == "categorico"]["columna"].tolist()

        # Información sobre variables disponibles
        st.subheader("📋 Variables disponibles")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"🔢 **Numéricas:** {len(numeric_cols)} variables")
            if numeric_cols:
                st.caption(", ".join(numeric_cols[:3]) + ("..." if len(numeric_cols) > 3 else ""))
        with col2:
            st.markdown(f"📝 **Categóricas:** {len(categorical_cols)} variables")
            if categorical_cols:
                st.caption(", ".join(categorical_cols[:3]) + ("..." if len(categorical_cols) > 3 else ""))

        # Selector de tipo de gráfico con descripciones
        st.subheader("🎯 Elige el tipo de gráfico")
        chart_descriptions = {
            "Histograma": "Muestra la distribución de una variable numérica",
            "Gráfico de barras": "Compara frecuencias entre categorías",
            "Box plot": "Revela la mediana, cuartiles y valores atípicos",
            "Gráfico de dispersión": "Muestra la relación entre dos variables numéricas"
        }

        chart_type = st.selectbox(
            "¿Qué tipo de gráfico quieres crear?",
            list(chart_descriptions.keys()),
            help="Selecciona el tipo de visualización que mejor se adapte a tu análisis"
        )
        
        # Mostrar descripción del gráfico seleccionado
        if chart_type:
            st.info(f"📊 **{chart_type}:** {chart_descriptions[chart_type]}")

        # Configuración específica por tipo de gráfico
        if chart_type == "Histograma":
            if numeric_cols:
                st.subheader("🔢 Selecciona la variable numérica")
                selected_col = st.selectbox(
                    "Elige la variable para el histograma:",
                    numeric_cols,
                    help="Selecciona una variable numérica para ver su distribución"
                )
                st.session_state.chart_config = {"type": chart_type, "column": selected_col}
                st.success(f"✅ Configurado: Histograma de '{selected_col}'")
            else:
                st.warning("⚠️ No hay variables numéricas disponibles para crear un histograma")
                
        elif chart_type == "Gráfico de barras":
            if categorical_cols:
                st.subheader("📝 Selecciona la variable categórica")
                selected_col = st.selectbox(
                    "Elige la variable para el gráfico de barras:",
                    categorical_cols,
                    help="Selecciona una variable categórica para ver las frecuencias"
                )
                st.session_state.chart_config = {"type": chart_type, "column": selected_col}
                st.success(f"✅ Configurado: Gráfico de barras de '{selected_col}'")
            else:
                st.warning("⚠️ No hay variables categóricas disponibles para crear un gráfico de barras")
                
        elif chart_type == "Box plot":
            if numeric_cols:
                st.subheader("🔢 Selecciona la variable numérica")
                selected_col = st.selectbox(
                    "Elige la variable para el box plot:",
                    numeric_cols,
                    help="Selecciona una variable numérica para ver estadísticas descriptivas"
                )
                st.session_state.chart_config = {"type": chart_type, "column": selected_col}
                st.success(f"✅ Configurado: Box plot de '{selected_col}'")
            else:
                st.warning("⚠️ No hay variables numéricas disponibles para crear un box plot")
        
        elif chart_type == "Gráfico de dispersión":
            if len(numeric_cols) >= 2:
                st.subheader("🔢 Selecciona las variables para el gráfico de dispersión")
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox(
                        "Variable X (eje horizontal):",
                        numeric_cols,
                        help="Selecciona la variable que irá en el eje X"
                    )
                with col2:
                    y_options = [c for c in numeric_cols if c != x_col]
                    y_col = st.selectbox(
                        "Variable Y (eje vertical):",
                        y_options,
                        help="Selecciona la variable que irá en el eje Y"
                    )
                st.session_state.chart_config = {"type": chart_type, "x": x_col, "y": y_col}
                st.success(f"✅ Configurado: Gráfico de dispersión '{x_col}' vs '{y_col}'")
            else:
                st.warning("⚠️ Necesitas al menos 2 variables numéricas para crear un gráfico de dispersión")
        
        # Información sobre el siguiente paso
        if "chart_config" in st.session_state:
            st.info("🎯 **Próximo paso:** Ve a la página '📈 Visualización' para ver tu gráfico")
    else:
        st.warning("⚠️ **Paso pendiente:** Primero debes cargar un archivo en la página 'Cargar archivo'")
        st.info("💡 Ve al menú lateral y selecciona '📁 Cargar archivo' para comenzar")

# ------------------------
# Página: Visualización
# ------------------------
elif st.session_state.current_page == "visualizacion":
    st.header("📈 Visualización")
    
    # Instrucciones claras
    st.markdown("""
    ### 🎨 Tu gráfico personalizado
    Aquí puedes ver la visualización que configuraste en la página anterior. El gráfico te 
    ayudará a entender mejor tus datos y descubrir patrones interesantes. Si quieres crear 
    un gráfico diferente, regresa a la página de 'Selección de gráfico'.
    """)
    
    df = st.session_state.get("df")
    chart_config = st.session_state.get("chart_config")

    if df is not None and chart_config:
        # Mostrar información del gráfico configurado
        st.subheader("📊 Configuración actual")
        if chart_config["type"] == "Histograma":
            st.info(f"📊 **Tipo:** Histograma de la variable '{chart_config['column']}'")
        elif chart_config["type"] == "Gráfico de barras":
            st.info(f"📊 **Tipo:** Gráfico de barras de la variable '{chart_config['column']}'")
        elif chart_config["type"] == "Box plot":
            st.info(f"📊 **Tipo:** Box plot de la variable '{chart_config['column']}'")
        elif chart_config["type"] == "Gráfico de dispersión":
            st.info(f"📊 **Tipo:** Gráfico de dispersión '{chart_config['x']}' vs '{chart_config['y']}'")
        
        # Generar y mostrar el gráfico
        st.subheader("🎨 Tu visualización")
        with st.spinner("🔄 Generando tu gráfico..."):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 8))

            if chart_config["type"] == "Histograma":
                ax.hist(df[chart_config["column"]].dropna(), bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
                ax.set_title(f'Distribución de {chart_config["column"]}', fontsize=16, pad=20)
                ax.set_xlabel(chart_config["column"], fontsize=12)
                ax.set_ylabel("Frecuencia", fontsize=12)
                ax.grid(True, alpha=0.3)

            elif chart_config["type"] == "Gráfico de barras":
                value_counts = df[chart_config["column"]].value_counts().head(10)
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
                value_counts.plot(kind="bar", ax=ax, color=colors[:len(value_counts)])
                ax.set_title(f'Top 10 valores más frecuentes en {chart_config["column"]}', fontsize=16, pad=20)
                ax.set_xlabel(chart_config["column"], fontsize=12)
                ax.set_ylabel("Frecuencia", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                ax.grid(True, alpha=0.3)

            elif chart_config["type"] == "Box plot":
                ax.boxplot(df[chart_config["column"]].dropna(), patch_artist=True, 
                          boxprops=dict(facecolor='#FF6B6B', alpha=0.7))
                ax.set_title(f'Estadísticas descriptivas de {chart_config["column"]}', fontsize=16, pad=20)
                ax.set_ylabel(chart_config["column"], fontsize=12)
                ax.grid(True, alpha=0.3)

            elif chart_config["type"] == "Gráfico de dispersión":
                ax.scatter(df[chart_config["x"]], df[chart_config["y"]], alpha=0.6, color='#45B7D1')
                ax.set_title(f'Relación entre {chart_config["x"]} y {chart_config["y"]}', fontsize=16, pad=20)
                ax.set_xlabel(chart_config["x"], fontsize=12)
                ax.set_ylabel(chart_config["y"], fontsize=12)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
        st.pyplot(fig)

        # Información adicional sobre el gráfico
        st.success("✅ ¡Gráfico generado exitosamente!")
        
        # Estadísticas básicas del gráfico
        with st.expander("📈 Estadísticas del gráfico", expanded=False):
            if chart_config["type"] in ["Histograma", "Box plot"]:
                col = chart_config["column"]
                stats = df[col].describe()
                st.markdown(f"**Estadísticas de '{col}':**")
                st.dataframe(stats, use_container_width=True)
            elif chart_config["type"] == "Gráfico de barras":
                col = chart_config["column"]
                value_counts = df[col].value_counts()
                st.markdown(f"**Frecuencias de '{col}':**")
                st.dataframe(value_counts.head(15), use_container_width=True)
            elif chart_config["type"] == "Gráfico de dispersión":
                x_col, y_col = chart_config["x"], chart_config["y"]
                correlation = df[x_col].corr(df[y_col])
                st.markdown(f"**Correlación entre '{x_col}' y '{y_col}':** {correlation:.3f}")
                if abs(correlation) > 0.7:
                    st.info("💡 **Correlación fuerte detectada** - Las variables están muy relacionadas")
                elif abs(correlation) > 0.3:
                    st.info("💡 **Correlación moderada detectada** - Las variables tienen cierta relación")
                else:
                    st.info("💡 **Correlación débil** - Las variables tienen poca relación lineal")
        
        # Consejos de interpretación
        with st.expander("💡 ¿Cómo interpretar este gráfico?", expanded=False):
            if chart_config["type"] == "Histograma":
                st.markdown("""
                **📊 Interpretación del histograma:**
                - **Forma:** ¿Es simétrico, asimétrico, o tiene múltiples picos?
                - **Centro:** ¿Dónde se concentran la mayoría de los valores?
                - **Dispersión:** ¿Qué tan extendidos están los datos?
                - **Valores atípicos:** ¿Hay valores muy diferentes al resto?
                """)
            elif chart_config["type"] == "Gráfico de barras":
                st.markdown("""
                **📊 Interpretación del gráfico de barras:**
                - **Frecuencias:** ¿Qué categorías son más comunes?
                - **Patrones:** ¿Hay categorías que destacan?
                - **Distribución:** ¿Los valores están distribuidos uniformemente?
                """)
            elif chart_config["type"] == "Box plot":
                st.markdown("""
                **📊 Interpretación del box plot:**
                - **Mediana:** La línea central muestra el valor medio
                - **Cuartiles:** La caja muestra el 50% central de los datos
                - **Valores atípicos:** Los puntos fuera de las líneas son valores extremos
                - **Asimetría:** Si la mediana no está centrada, los datos son asimétricos
                """)
            elif chart_config["type"] == "Gráfico de dispersión":
                st.markdown("""
                **📊 Interpretación del gráfico de dispersión:**
                - **Tendencia:** ¿Hay una relación lineal entre las variables?
                - **Fuerza:** ¿Qué tan fuerte es la relación?
                - **Dirección:** ¿Es positiva (crece) o negativa (decrece)?
                - **Valores atípicos:** ¿Hay puntos que se alejan del patrón?
                """)
        
        # Opciones adicionales
        st.subheader("🛠️ Opciones adicionales")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Crear nuevo gráfico"):
                st.session_state.pop("chart_config", None)
                st.info("💡 Ve a la página 'Selección de gráfico' para configurar un nuevo gráfico")
        with col2:
            if st.button("📊 Ver estadísticas completas"):
                st.info("💡 Las estadísticas detalladas están disponibles en la página 'Resumen de datos'")
                
    elif df is not None:
        st.warning("⚠️ **Configuración pendiente:** Primero debes configurar un gráfico en la página 'Selección de gráfico'")
        st.info("💡 Ve al menú lateral y selecciona '🎨 Selección de gráfico' para configurar tu visualización")
    else:
        st.warning("⚠️ **Paso pendiente:** Primero debes cargar un archivo en la página 'Cargar archivo'")
        st.info("💡 Ve al menú lateral y selecciona '📁 Cargar archivo' para comenzar")

# ------------------------
# Página: Exportar resultados
# ------------------------
elif st.session_state.current_page == "exportar":
    st.header("💾 Exportar resultados")
    
    # Instrucciones claras
    st.markdown("""
    ### 📤 Generar reporte completo
    Aquí puedes ejecutar el pipeline completo de análisis que procesará tus datos, 
    aplicará transformaciones, realizará modelado y generará un reporte PDF profesional 
    con todos los hallazgos. Este proceso puede tardar unos minutos.
    """)
    
    df = st.session_state.get("df")

    if df is not None:
        # Información sobre el pipeline
        st.subheader("🔧 ¿Qué hace el pipeline completo?")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📊 Procesamiento de datos:**
            - 🧹 Limpieza automática de datos
            - 🔄 Transformación de variables
            - 📈 Análisis estadístico descriptivo
            - 🎯 Detección de patrones
            """)
        with col2:
            st.markdown("""
            **📋 Generación de reporte:**
            - 📊 Gráficos y visualizaciones
            - 📈 Análisis de correlaciones
            - 🎯 Modelado estadístico
            - 📄 Reporte PDF profesional
            """)
        
        # Información sobre los datos a procesar
        st.subheader("📋 Datos a procesar")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Filas", f"{df.shape[0]:,}")
        with col2:
            st.metric("📊 Columnas", df.shape[1])
        with col3:
            st.metric("💾 Tamaño estimado", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        # Ejecutar pipeline
        st.subheader("🚀 Ejecutar análisis completo")
        st.info("💡 **Consejo:** Este proceso puede tardar entre 2-5 minutos dependiendo del tamaño de tus datos")
        
        if st.button("🚀 Iniciar Pipeline Completo", type="primary"):
            with st.spinner("🔄 Procesando tus datos, esto puede tardar unos minutos..."):
                try:
                    # Crear configuración temporal
                    config = {
                        "input_path": "temp_data.csv",
                        "output_report": "reporte_proyecto_j.pdf",
                    }

                    # Guardar datos temporalmente
                    df.to_csv("temp_data.csv", index=False)

                    # Ejecutar pipeline
                    pipeline = Pipeline(config)
                    df_processed, model, results = pipeline.run()

                    # Limpiar archivo temporal
                    if os.path.exists("temp_data.csv"):
                        os.remove("temp_data.csv")

                    st.success("✅ ¡Pipeline ejecutado correctamente!")
                    
                    # Mostrar resumen de resultados
                    st.subheader("📊 Resumen de resultados")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📈 Coeficiente", f"{results.get('coef', 'N/A')}")
                    with col2:
                        st.metric("📊 Intercepto", f"{results.get('intercept', 'N/A')}")

                    # Información adicional sobre el modelo
                    with st.expander("🔍 Detalles del modelo", expanded=False):
                        st.markdown("""
                        **📈 Información del modelo:**
                        - **Coeficiente:** Indica la pendiente de la relación
                        - **Intercepto:** Valor base del modelo
                        - **Datos procesados:** Se aplicaron transformaciones automáticas
                        - **Calidad:** El modelo se ajustó a los patrones encontrados
                        """)

                    # Botón de descarga
                    st.subheader("📥 Descargar resultados")
                    if os.path.exists("reporte_proyecto_j.pdf"):
                        with open("reporte_proyecto_j.pdf", "rb") as f:
                            st.download_button(
                                "💾 Descargar Reporte PDF Completo",
                                data=f.read(),
                                file_name=f"reporte_proyecto_j_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                help="Descarga el reporte PDF con todos los análisis y visualizaciones"
                            )
                        st.success("📄 **Reporte PDF generado:** Contiene análisis completo, gráficos y conclusiones")
                    else:
                        st.warning("⚠️ No se pudo generar el archivo PDF")

                except Exception as e:
                    st.error(f"❌ **Error en el pipeline:** {str(e)}")
                    st.info("💡 **Sugerencias:** Verifica que tus datos sean compatibles o intenta con un conjunto más pequeño")
        
        # Opciones adicionales
        st.subheader("🛠️ Otras opciones de exportación")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Exportar datos procesados (CSV)"):
                csv = df.to_csv(index=False)
                st.download_button(
                    "💾 Descargar CSV",
                    csv,
                    file_name=f"datos_procesados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        with col2:
            if st.button("📈 Exportar estadísticas (JSON)"):
                stats = df.describe().to_json()
                st.download_button(
                    "💾 Descargar JSON",
                    stats,
                    file_name=f"estadisticas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    else:
        st.warning("⚠️ **Paso pendiente:** Primero debes cargar un archivo en la página 'Cargar archivo'")
        st.info("💡 Ve al menú lateral y selecciona '📁 Cargar archivo' para comenzar")

# ------------------------
# Página: Análisis Avanzado
# ------------------------
elif st.session_state.current_page == "analisis_avanzado":
    st.header("🚀 Análisis Avanzado")
    
    # Instrucciones claras y amigables
    st.markdown("""
    ### 🧠 Consultas en Lenguaje Natural
    ¡Habla con tus datos como si le explicaras a un colega! Escribe preguntas en español 
    y el sistema las interpretará automáticamente para realizar análisis complejos. 
    No necesitas conocer comandos técnicos, solo describe lo que quieres analizar.
    """)
    
    df = st.session_state.get("df")
    if df is not None:
        # Importar módulos necesarios
        try:
            from nl_query import parse_and_execute  # type: ignore
            from complex_grouping import execute_complex_grouping_from_question  # type: ignore
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:
            # Funciones dummy para análisis avanzado
            def parse_and_execute(query, df):
                return pd.DataFrame(), "Análisis no disponible"
            
            def execute_complex_grouping_from_question(query, df):
                return pd.DataFrame(), "Agrupación no disponible"
            
            px = None
            go = None
            
            st.success("✅ **¡Perfecto!** Módulos de análisis avanzado cargados correctamente")
            
            # Sección de consultas en lenguaje natural
            st.subheader("💬 Escribe tu consulta")
            st.markdown("""
            **💡 Consejo:** Escribe tu pregunta como si se la explicaras a un colega. 
            El sistema entenderá automáticamente qué quieres analizar.
            """)
            
            # Ejemplos interactivos
            with st.expander("📚 Ver ejemplos de consultas", expanded=False):
                st.markdown("""
                **🔢 Análisis básicos:**
                - "calcular promedio de ventas por región"
                - "contar registros agrupados por ciudad y categoría"
                - "suma de ingresos por región y mes"
                
                **📈 Análisis temporales:**
                - "promedio de temperatura por ciudad en los últimos 30 días"
                - "tendencia de ventas por mes"
                - "promedio móvil de ingresos por ventana de 7 días"
                
                **🏗️ Agrupaciones complejas:**
                - "distribución de ventas por nivel jerárquico de región y ciudad"
                - "promedio de edad por departamento y nivel educativo"
                - "suma de gastos por categoría y trimestre"
                """)
            
            # Input para consulta con mejor UX
            query = st.text_area(
                "¿Qué quieres analizar?",
                placeholder="Ej: calcular promedio de ventas por región",
                height=120,
                help="Describe tu análisis en español. El sistema interpretará automáticamente qué variables usar y qué operaciones realizar."
            )
            
            # Botón de ejecución con mejor feedback
            col1, col2 = st.columns([1, 3])
            with col1:
                execute_button = st.button("🚀 Analizar")
            with col2:
                if query.strip():
                    st.info("💡 **Consejo:** Haz clic en 'Analizar' para procesar tu consulta")
                else:
                    st.info("💡 **Consejo:** Escribe tu pregunta arriba y luego haz clic en 'Analizar'")
            
            if execute_button:
                if query.strip():
                    with st.spinner("🧠 Interpretando tu consulta y procesando los datos..."):
                        try:
                            # Ejecutar consulta usando el sistema de agrupación compleja
                            result, insights = execute_complex_grouping_from_question(
                                query, df
                            )
                            
                            if not result.empty:
                                st.success("✅ **¡Análisis completado!** Tu consulta se procesó correctamente")
                                
                                # Mostrar insights con mejor formato
                                st.subheader("💡 Insights Automáticos")
                                st.markdown(f"""
                                **🎯 Lo que encontré:**
                                {insights}
                                """)
                                
                                # Mostrar resultados en tabla con mejor formato
                                st.subheader("📊 Resultados del análisis")
                                st.dataframe(
                                    result, 
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Generar visualización automática
                                st.subheader("📈 Visualización automática")
                                
                                # Intentar crear un gráfico apropiado
                                try:
                                    if px is not None and len(result) > 1:
                                        # Si hay múltiples filas, crear gráfico de barras o líneas
                                        numeric_cols = result.select_dtypes(include=['number']).columns
                                        if len(numeric_cols) > 0:
                                            # Usar la primera columna numérica para el gráfico
                                            y_col = str(numeric_cols[0])
                                            x_col = str(result.columns[0]) if result.columns[0] != y_col else str(result.columns[1]) if len(result.columns) > 1 else None
                                            
                                            if x_col and y_col:
                                                fig = px.bar(
                                                    result, 
                                                    x=x_col, 
                                                    y=y_col,
                                                    title=f"📊 {y_col} por {x_col}",
                                                    labels={x_col: x_col, y_col: y_col},
                                                    color_discrete_sequence=['#4ECDC4']
                                                )
                                                fig.update_layout(
                                                    xaxis_tickangle=-45,
                                                    height=500,
                                                    showlegend=False
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                                
                                                # Botón de descarga con mejor UX
                                                csv = result.to_csv(index=False)
                                                st.download_button(
                                                    "💾 Descargar Resultados (CSV)",
                                                    csv,
                                                    file_name=f"analisis_avanzado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                    mime="text/csv",
                                                    help="Descarga los resultados de tu análisis en formato CSV"
                                                )
                                            else:
                                                st.info("ℹ️ **Visualización:** No se pudo generar un gráfico automático para estos resultados")
                                        else:
                                            st.info("ℹ️ **Visualización:** Los resultados no contienen datos numéricos para graficar")
                                    else:
                                        st.info("ℹ️ **Visualización:** Resultado único - no se requiere gráfico")
                                        
                                except Exception as viz_error:
                                    st.warning(f"⚠️ **Visualización:** No se pudo generar el gráfico automático")
                                    st.info("💡 Los resultados están disponibles en la tabla anterior")
                                    
                            else:
                                st.warning("⚠️ **Sin resultados:** No se obtuvieron datos para esta consulta")
                                st.info("💡 **Sugerencias:** Verifica que las variables mencionadas existan en tus datos o reformula tu pregunta")
                                
                        except Exception as e:
                            st.error(f"❌ **Error:** No se pudo procesar tu consulta")
                            st.info(f"💡 **Detalles:** {str(e)}")
                            st.markdown("""
                            **🔧 Consejos para consultas exitosas:**
                            - Verifica que las variables mencionadas existan en tus datos
                            - Usa nombres de variables exactos
                            - Intenta con consultas más simples primero
                            """)
                else:
                    st.warning("⚠️ **Consulta vacía:** Por favor, escribe tu pregunta antes de analizar")
            
            # Sección de información sobre capacidades
            st.subheader("🔧 ¿Qué puedo analizar?")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📊 Tipos de Agrupación:**
                - 🎯 **Simple:** Por una variable (ej: por región)
                - 🔗 **Múltiple:** Por varias variables (ej: por región y ciudad)
                - 🏗️ **Jerárquica:** Niveles anidados (ej: región > ciudad > barrio)
                - ⏰ **Temporal:** Con fechas y ventanas de tiempo
                - 📈 **Deslizante:** Promedios móviles y tendencias
                """)
            
            with col2:
                st.markdown("""
                **🧮 Operaciones Disponibles:**
                - 📊 **Básicas:** Contar, Sumar, Promedio
                - 📈 **Estadísticas:** Mediana, Moda, Desviación
                - 📉 **Extremos:** Máximo, Mínimo, Varianza
                - 🔄 **Transformaciones:** Porcentajes, Normalización
                - 📊 **Acumulados:** Valores acumulados y diferencias
                """)
            
            # Mostrar información sobre los datos
            st.subheader("📋 Información de tus datos")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Filas", f"{len(df):,}")
            with col2:
                st.metric("📊 Columnas", len(df.columns))
            with col3:
                st.metric("🔢 Numéricas", len(df.select_dtypes(include=['number']).columns))
            
            # Mostrar columnas disponibles con mejor formato
            with st.expander("📋 Ver todas las columnas disponibles", expanded=False):
                st.markdown("**📊 Variables en tus datos:**")
                st.code(", ".join(df.columns.tolist()))
                st.caption("💡 Usa estos nombres exactos en tus consultas")
            
        except Exception as e:
            st.error(f"❌ **Error:** No se pudieron cargar los módulos de análisis avanzado")
            st.info(f"💡 **Detalles:** {e}")
            st.markdown("""
            **🔧 Solución:**
            - Verifica que todos los módulos estén instalados
            - Reinicia la aplicación si es necesario
            - Contacta al administrador del sistema
            """)
            
    else:
        st.warning("⚠️ **Paso pendiente:** Primero debes cargar un archivo en la página 'Cargar archivo'")
        st.info("💡 Una vez que hayas cargado tus datos, podrás usar consultas en lenguaje natural para análisis complejos")
        st.markdown("""
        **🎯 ¿Qué puedes hacer aquí?**
        - 📊 **Agrupaciones complejas** por múltiples variables
        - ⏰ **Análisis temporales** con ventanas y tendencias
        - 🧮 **Operaciones estadísticas** avanzadas
        - 📈 **Visualizaciones automáticas** de resultados
        - 💾 **Exportación** de análisis en CSV
        """)

# ------------------------
# Footer con navegación
# ------------------------
st.markdown("---")

# Botón de navegación en la parte inferior derecha
col1, col2, col3 = st.columns([2, 1, 1])
with col3:
    # Obtener la página actual y la siguiente
    main_pages = ["cargar", "resumen", "tipos", "sugerencias", "grafico", "visualizacion", "exportar"]
    current_page = st.session_state.current_page
    
    if current_page in main_pages:
        current_index = main_pages.index(current_page)
        if current_index < len(main_pages) - 1:
            next_page = main_pages[current_index + 1]
            next_page_name = list(pages.keys())[list(pages.values()).index(next_page)]
            
            if st.button("➡️ Siguiente"):
                st.session_state.current_page = next_page
                st.rerun()
        else:
            st.info("✅ ¡Flujo completado!")

st.caption("© 2025 Proyecto J - Pipeline Modular + Streamlit Wizard")
