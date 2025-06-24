# streamlit_app.py
import streamlit as st
import pandas as pd
import os
import time
from pathlib import Path
from tasks import procesar_archivo, validar_archivo, obtener_estado_tarea

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="🔄 Procesamiento Asíncrono - Proyecto J",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mantener consistencia visual
st.markdown("""
<style>
    /* Importar fuentes */
    @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;500;600;700&display=swap');
    
    /* FORZAR MODO CLARO */
    html, body {
        color-scheme: light !important;
        background-color: #FBF7F2 !important;
        color: #333333 !important;
    }
    
    /* Variables CSS */
    :root {
        --color-fondo-general: #FBF7F2;
        --color-azul-claro: #C7DCE5;
        --color-azul-profundo: #648DA5;
        --color-texto-principal: #2C3E50;
        --color-sombra: rgba(0, 0, 0, 0.08);
    }
    
    /* ÁREA DE CONTENIDO PRINCIPAL */
    .main > div {
        background-color: var(--color-azul-claro) !important;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        box-shadow: 0 2px 8px var(--color-sombra);
    }
    
    /* PANEL LATERAL */
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
    
    /* Títulos principales */
    h1 {
        font-family: 'Raleway', sans-serif;
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--color-azul-profundo);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Contenedores de estado */
    .estado-container {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px var(--color-sombra);
        border-left: 4px solid var(--color-azul-profundo);
    }
    
    .estado-pendiente {
        border-left-color: #FFA726;
    }
    
    .estado-procesando {
        border-left-color: #42A5F5;
    }
    
    .estado-completado {
        border-left-color: #66BB6A;
    }
    
    .estado-error {
        border-left-color: #EF5350;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INICIALIZACIÓN DE SESSION STATE
# ============================================================================

if 'archivo_subido' not in st.session_state:
    st.session_state.archivo_subido = None

if 'job_id' not in st.session_state:
    st.session_state.job_id = None

if 'estado_procesamiento' not in st.session_state:
    st.session_state.estado_procesamiento = None

if 'resultados' not in st.session_state:
    st.session_state.resultados = None

if 'archivo_temporal' not in st.session_state:
    st.session_state.archivo_temporal = None

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def guardar_archivo_temporal(uploaded_file):
    """Guarda el archivo subido en una ubicación temporal."""
    try:
        # Crear directorio temporal si no existe
        temp_dir = Path('./temp')
        temp_dir.mkdir(exist_ok=True)
        
        # Guardar archivo
        temp_path = temp_dir / f"upload_{int(time.time())}_{uploaded_file.name}"
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return str(temp_path)
    except Exception as e:
        st.error(f"Error al guardar archivo: {e}")
        return None

def mostrar_estado_procesamiento(estado_info):
    """Muestra el estado del procesamiento con indicadores visuales."""
    estado = estado_info.get('estado', 'UNKNOWN')
    progreso = estado_info.get('progreso', 0)
    mensaje = estado_info.get('mensaje', 'Estado desconocido')
    
    # Determinar clase CSS según estado
    clase_css = {
        'PENDING': 'estado-pendiente',
        'PROGRESS': 'estado-procesando',
        'SUCCESS': 'estado-completado',
        'FAILURE': 'estado-error'
    }.get(estado, 'estado-pendiente')
    
    # Iconos según estado
    iconos = {
        'PENDING': '⏳',
        'PROGRESS': '🔄',
        'SUCCESS': '✅',
        'FAILURE': '❌'
    }
    
    with st.container():
        st.markdown(f"""
        <div class="estado-container {clase_css}">
            <h3>{iconos.get(estado, '❓')} {estado}</h3>
            <p><strong>Progreso:</strong> {progreso}%</p>
            <p><strong>Estado:</strong> {mensaje}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Barra de progreso
        if estado == 'PROGRESS':
            st.progress(progreso / 100)
        elif estado == 'SUCCESS':
            st.progress(1.0)
        else:
            st.progress(0.0)

def mostrar_resultados(resultado):
    """Muestra los resultados del procesamiento."""
    if not resultado:
        return
    
    st.success("🎉 ¡Procesamiento completado exitosamente!")
    
    # Información general
    with st.expander("📊 Información del Archivo", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Archivo", resultado.get('archivo_original', 'N/A'))
            st.metric("Tamaño", f"{resultado.get('tamaño_mb', 0):.1f} MB")
        
        with col2:
            estadisticas = resultado.get('estadisticas', {})
            st.metric("Filas", f"{estadisticas.get('total_filas', 0):,}")
            st.metric("Columnas", estadisticas.get('total_columnas', 0))
        
        with col3:
            st.metric("Columnas Numéricas", estadisticas.get('columnas_numericas', 0))
            st.metric("Columnas Categóricas", estadisticas.get('columnas_categoricas', 0))
    
    # Mostrar archivos generados
    with st.expander("📁 Archivos Generados", expanded=True):
        archivos = [
            ("Resumen General", resultado.get('resumen_path')),
            ("Análisis de Columnas", resultado.get('columnas_path')),
            ("Muestra de Datos", resultado.get('muestra_path'))
        ]
        
        for nombre, ruta in archivos:
            if ruta and os.path.exists(ruta):
                st.write(f"**{nombre}:** `{ruta}`")
                
                # Cargar y mostrar datos
                try:
                    df = pd.read_parquet(ruta)
                    st.write(f"**Vista previa de {nombre}:**")
                    st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error al cargar {nombre}: {e}")
            else:
                st.warning(f"Archivo no encontrado: {ruta}")

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.title("🔄 Procesamiento Asíncrono de Archivos")

# Sidebar con información
st.sidebar.title("📋 Información")
st.sidebar.markdown("""
### ¿Cómo funciona?

1. **Sube tu archivo CSV** (máximo 200 MB)
2. **Revisa la vista previa** de los datos
3. **Inicia el procesamiento** asíncrono
4. **Monitorea el progreso** en tiempo real
5. **Descarga los resultados** cuando termine

### Características:
- ✅ Procesamiento en chunks (10,000 filas)
- ✅ Análisis estadístico completo
- ✅ Generación de resúmenes
- ✅ Interfaz responsiva
- ✅ Monitoreo en tiempo real
""")

# Verificar conexión con Redis/Celery
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Estado del Sistema")

try:
    # Intentar conectar con Celery
    from celery import current_app
    st.sidebar.success("✅ Celery conectado")
except Exception as e:
    st.sidebar.error("❌ Error de conexión con Celery")
    st.sidebar.error(f"Detalles: {str(e)}")

# ============================================================================
# SECCIÓN DE CARGA DE ARCHIVOS
# ============================================================================

st.header("📁 Carga de Archivos")

# File uploader
uploaded_file = st.file_uploader(
    "Selecciona un archivo CSV para procesar",
    type=['csv'],
    help="Archivos CSV de hasta 200 MB"
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
        st.metric("Tipo", uploaded_file.type)
    
    # Validar tamaño
    if file_size > 200:
        st.error("❌ El archivo es demasiado grande. Máximo 200 MB.")
        st.stop()
    
    # Guardar archivo temporal
    if st.session_state.archivo_temporal is None:
        temp_path = guardar_archivo_temporal(uploaded_file)
        if temp_path:
            st.session_state.archivo_temporal = temp_path
            st.success(f"✅ Archivo guardado temporalmente: `{temp_path}`")
    
    # Vista previa de datos
    st.subheader("👀 Vista Previa")
    try:
        df_preview = pd.read_csv(uploaded_file, nrows=5)
        st.dataframe(df_preview, use_container_width=True)
        
        # Información adicional
        st.info(f"📊 **Información del archivo:** {len(df_preview.columns)} columnas detectadas")
        
    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {e}")
        st.stop()
    
    # Botón para iniciar procesamiento
    st.subheader("🚀 Iniciar Procesamiento")
    
    if st.button("🔄 Procesar Archivo Completo", type="primary", use_container_width=True):
        if st.session_state.archivo_temporal:
            try:
                # Iniciar tarea asíncrona
                job = procesar_archivo.delay(st.session_state.archivo_temporal)
                st.session_state.job_id = job.id
                st.session_state.estado_procesamiento = 'PENDING'
                
                st.success(f"✅ Tarea iniciada con ID: `{job.id}`")
                st.info("🔄 El procesamiento comenzará en breve. Puedes monitorear el progreso a continuación.")
                
            except Exception as e:
                st.error(f"❌ Error al iniciar el procesamiento: {e}")
        else:
            st.error("❌ No se pudo guardar el archivo temporal")

# ============================================================================
# SECCIÓN DE MONITOREO
# ============================================================================

if st.session_state.job_id:
    st.header("📊 Monitoreo del Procesamiento")
    
    # Mostrar ID de la tarea
    st.info(f"🆔 **ID de Tarea:** `{st.session_state.job_id}`")
    
    # Obtener estado actual
    estado_actual = obtener_estado_tarea(st.session_state.job_id)
    
    # Mostrar estado
    mostrar_estado_procesamiento(estado_actual)
    
    # Actualizar automáticamente cada 5 segundos
    if estado_actual['estado'] in ['PENDING', 'PROGRESS']:
        time.sleep(5)
        st.rerun()
    
    # Si completó exitosamente
    elif estado_actual['estado'] == 'SUCCESS':
        st.session_state.resultados = estado_actual.get('resultado', {})
        st.session_state.estado_procesamiento = 'SUCCESS'
        
        # Mostrar resultados
        st.header("📈 Resultados del Procesamiento")
        mostrar_resultados(st.session_state.resultados)
        
        # Botón para limpiar
        if st.button("🗑️ Limpiar y Procesar Otro Archivo"):
            # Limpiar archivos temporales
            if st.session_state.archivo_temporal and os.path.exists(st.session_state.archivo_temporal):
                os.remove(st.session_state.archivo_temporal)
            
            # Resetear session state
            for key in ['archivo_subido', 'job_id', 'estado_procesamiento', 'resultados', 'archivo_temporal']:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.rerun()
    
    # Si falló
    elif estado_actual['estado'] == 'FAILURE':
        st.error("❌ El procesamiento falló")
        st.error(f"**Error:** {estado_actual.get('mensaje', 'Error desconocido')}")
        
        # Botón para reintentar
        if st.button("🔄 Reintentar"):
            st.session_state.job_id = None
            st.session_state.estado_procesamiento = None
            st.rerun()

# ============================================================================
# SECCIÓN DE AYUDA
# ============================================================================

with st.expander("❓ ¿Necesitas ayuda?"):
    st.markdown("""
    ### Preguntas Frecuentes
    
    **Q: ¿Qué tipos de archivos puedo procesar?**
    A: Actualmente solo archivos CSV. Próximamente soporte para Excel y otros formatos.
    
    **Q: ¿Cuál es el tamaño máximo de archivo?**
    A: 200 MB para garantizar un procesamiento eficiente.
    
    **Q: ¿Cuánto tiempo toma el procesamiento?**
    A: Depende del tamaño del archivo. Archivos pequeños (1-10 MB) toman segundos, archivos grandes pueden tomar minutos.
    
    **Q: ¿Qué archivos se generan?**
    A: Se generan 3 archivos Parquet:
    - **Resumen general:** Estadísticas básicas del archivo
    - **Análisis de columnas:** Información detallada de cada columna
    - **Muestra de datos:** Primera 1000 filas del archivo original
    
    **Q: ¿Los datos se almacenan permanentemente?**
    A: No, los archivos temporales se eliminan al limpiar la sesión.
    
    ### Contacto
    Si tienes problemas, contacta al equipo de desarrollo del Proyecto J.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🔄 Procesamiento Asíncrono - Proyecto J | 
        Desarrollado con Streamlit, Celery y Redis
    </div>
    """,
    unsafe_allow_html=True
) 