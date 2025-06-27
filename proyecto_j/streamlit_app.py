import streamlit as st
import pandas as pd
import io
import sys
import os
from pathlib import Path

# Agregar src al path para importar el pipeline modular
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core import Pipeline
from steps import cargar_datos, limpiar_datos, transformar_datos, modelar, visualizar, generar_reporte
from utils import load_config

# Configuración de la página
st.set_page_config(
    page_title='Asistente de Visualización de Datos - Proyecto J',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ------------------------
# Estilos y paleta
# ------------------------
st.markdown("""
<style>
body { color: #333333; background-color: #fafafa; }
.sidebar .sidebar-content { background-color: #fffbf0; }
.reportview-container .main { background-color: #ffffff; }
.stButton>button { border-radius: 0.5rem; padding: 0.5rem 1rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------
# Tour / Onboarding inicial
# ------------------------
def show_walkthrough():
    if 'walkthrough_done' not in st.session_state:
        st.session_state.walkthrough_done = True
        st.toast('👋 Bienvenido al Asistente de Visualización de Datos!')
        st.info('Usa la barra lateral para navegar por los 7 pasos del flujo.')
        st.info('ℹ️ Pasa el cursor sobre los iconos para ver información adicional.')

show_walkthrough()

# ------------------------
# Sidebar de navegación
# ------------------------
st.sidebar.markdown('# 📚 Navegación')
steps = [
    '📁 Cargar archivo',
    '📊 Resumen de datos',
    '🔍 Detección de tipos',
    '💡 Sugerencias',
    '🎨 Selección de gráfico',
    '📈 Visualización',
    '💾 Exportar resultados'
]
if 'step' not in st.session_state:
    st.session_state.step = 0

for i, label in enumerate(steps):
    st.sidebar.write(f"{i+1}. {label}")
progress = (st.session_state.step + 1) / len(steps)
st.sidebar.progress(progress)
st.sidebar.markdown('---')
st.sidebar.markdown('💡 Tip: en el futuro podrás crear visualizaciones que relacionen múltiples variables.')
if st.sidebar.button('🔄 Reiniciar Asistente'):
    st.session_state.clear()
    st.experimental_rerun()

# ------------------------
# Función para leer distintos formatos
# ------------------------
def load_file(uploaded):
    """Carga archivo usando el pipeline modular"""
    try:
        # Guardar archivo temporalmente
        temp_path = f"temp_{uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        
        # Usar función del pipeline
        df = cargar_datos(temp_path)
        
        # Limpiar archivo temporal
        os.remove(temp_path)
        return df
    except Exception as e:
        st.error(f'Error al cargar archivo: {str(e)}')
        return None

# ------------------------
# Función para obtener resumen de datos faltantes
# ------------------------
def get_missing_summary(df):
    """Genera resumen de datos faltantes usando el pipeline"""
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    summary = pd.DataFrame({
        'Columna': missing_data.index,
        'Valores_Faltantes': missing_data.values,
        'Porcentaje': missing_percent.values
    }).sort_values('Valores_Faltantes', ascending=False)
    
    return {
        'summary': summary,
        'total_missing': missing_data.sum(),
        'total_percent': (missing_data.sum() / (len(df) * len(df.columns))) * 100
    }

# ------------------------
# Función para detectar tipos
# ------------------------
def detect_types(df):
    """Detecta tipos de datos usando lógica del pipeline"""
    types_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if 'object' in dtype or 'category' in dtype:
            detected_type = 'categorico'
        elif 'int' in dtype or 'float' in dtype:
            detected_type = 'numerico'
        elif 'datetime' in dtype:
            detected_type = 'fecha'
        else:
            detected_type = 'otro'
        
        types_info.append({
            'columna': col,
            'tipo_pandas': dtype,
            'tipo_detectado': detected_type,
            'valores_unicos': df[col].nunique()
        })
    
    return pd.DataFrame(types_info)

# ------------------------
# Carga de datos
# ------------------------
st.title('🤖 Asistente de Visualización de Datos - Proyecto J')
if st.session_state.step == 0:
    st.header('Paso 1: Cargar archivo de datos')
    st.markdown('Formatos soportados: CSV, Excel (.xls/.xlsx), SPSS (.sav), Stata (.dta)')
    uploaded = st.file_uploader('Drag and drop o selecciona un archivo', type=['csv','xls','xlsx','sav','dta'])
    if uploaded:
        df = load_file(uploaded)
        if df is not None:
            st.session_state.df = df
            st.success('Archivo cargado correctamente: ' + str(df.shape[0]) + ' filas × ' + str(df.shape[1]) + ' columnas')
            if st.button('➡️ Continuar al Paso 2'):
                st.session_state.step = 1
                st.experimental_rerun()

# ------------------------
# Paso 2: Resumen
# ------------------------
if st.session_state.step == 1:
    st.header('Paso 2: Resumen de datos')
    df = st.session_state.get('df')
    if df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Filas', df.shape[0])
        with col2:
            st.metric('Columnas', df.shape[1])
        with col3:
            mem = df.memory_usage(deep=True).sum() / 1024**2
            st.metric('Memoria', f"{mem:.2f} MB")
        
        missing = get_missing_summary(df)
        st.subheader('Valores faltantes')
        st.dataframe(missing['summary'])
        
        if st.button('➡️ Continuar al Paso 3'):
            st.session_state.step = 2
            st.experimental_rerun()

# ------------------------
# Paso 3: Detección de tipos
# ------------------------
if st.session_state.step == 2:
    st.header('Paso 3: Detección automática de tipos')
    df = st.session_state.get('df')
    if df is not None:
        types = detect_types(df)
        st.subheader('Tipos detectados')
        st.dataframe(types)
        
        # Gráfico de distribución de tipos
        import matplotlib.pyplot as plt
        type_counts = types['tipo_detectado'].value_counts()
        fig, ax = plt.subplots()
        type_counts.plot(kind='bar', ax=ax)
        plt.title('Distribución de tipos de datos')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        if st.button('➡️ Continuar al Paso 4'):
            st.session_state.step = 3
            st.experimental_rerun()

# ------------------------
# Paso 4: Sugerencias
# ------------------------
if st.session_state.step == 3:
    st.header('Paso 4: Sugerencias de visualización')
    df = st.session_state.get('df')
    if df is not None:
        types = detect_types(df)
        col = st.selectbox('Selecciona variable', df.columns)
        info = types.loc[types['columna']==col].iloc[0]
        
        st.info(f"Tipo detectado: {info['tipo_detectado']}")
        st.markdown('**Visualizaciones sugeridas:**')
        
        if info['tipo_detectado'] == 'categorico':
            st.write('- Gráfico de barras')
            st.write('- Gráfico de pastel')
            st.write('- Conteo de frecuencias')
        elif info['tipo_detectado'] == 'numerico':
            st.write('- Histograma')
            st.write('- Box plot')
            st.write('- Gráfico de dispersión')
        elif info['tipo_detectado'] == 'fecha':
            st.write('- Serie temporal')
            st.write('- Gráfico de líneas')
        
        if st.button('➡️ Continuar al Paso 5'):
            st.session_state.step = 4
            st.experimental_rerun()

# ------------------------
# Paso 5: Selección de gráfico
# ------------------------
if st.session_state.step == 4:
    st.header('Paso 5: Selección de gráfico')
    df = st.session_state.get('df')
    if df is not None:
        types = detect_types(df)
        numeric_cols = types[types['tipo_detectado'] == 'numerico']['columna'].tolist()
        categorical_cols = types[types['tipo_detectado'] == 'categorico']['columna'].tolist()
        
        chart_type = st.selectbox(
            'Tipo de gráfico',
            ['Histograma', 'Gráfico de barras', 'Box plot', 'Gráfico de dispersión']
        )
        
        if chart_type == 'Histograma' and numeric_cols:
            selected_col = st.selectbox('Variable numérica', numeric_cols)
            st.session_state.chart_config = {'type': chart_type, 'column': selected_col}
        elif chart_type == 'Gráfico de barras' and categorical_cols:
            selected_col = st.selectbox('Variable categórica', categorical_cols)
            st.session_state.chart_config = {'type': chart_type, 'column': selected_col}
        elif chart_type == 'Box plot' and numeric_cols:
            selected_col = st.selectbox('Variable numérica', numeric_cols)
            st.session_state.chart_config = {'type': chart_type, 'column': selected_col}
        elif chart_type == 'Gráfico de dispersión' and len(numeric_cols) >= 2:
            col1 = st.selectbox('Variable X', numeric_cols)
            col2 = st.selectbox('Variable Y', [c for c in numeric_cols if c != col1])
            st.session_state.chart_config = {'type': chart_type, 'x': col1, 'y': col2}
        
        if st.button('➡️ Continuar al Paso 6'):
            st.session_state.step = 5
            st.experimental_rerun()

# ------------------------
# Paso 6: Visualización
# ------------------------
if st.session_state.step == 5:
    st.header('Paso 6: Visualización')
    df = st.session_state.get('df')
    chart_config = st.session_state.get('chart_config')
    
    if df is not None and chart_config:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_config['type'] == 'Histograma':
            ax.hist(df[chart_config['column']].dropna(), bins=30, alpha=0.7)
            ax.set_title(f'Histograma de {chart_config["column"]}')
            ax.set_xlabel(chart_config['column'])
            ax.set_ylabel('Frecuencia')
            
        elif chart_config['type'] == 'Gráfico de barras':
            value_counts = df[chart_config['column']].value_counts().head(10)
            value_counts.plot(kind='bar', ax=ax)
            ax.set_title(f'Top 10 valores de {chart_config["column"]}')
            ax.set_xlabel(chart_config['column'])
            ax.set_ylabel('Frecuencia')
            plt.xticks(rotation=45)
            
        elif chart_config['type'] == 'Box plot':
            ax.boxplot(df[chart_config['column']].dropna())
            ax.set_title(f'Box plot de {chart_config["column"]}')
            ax.set_ylabel(chart_config['column'])
            
        elif chart_config['type'] == 'Gráfico de dispersión':
            ax.scatter(df[chart_config['x']], df[chart_config['y']], alpha=0.6)
            ax.set_title(f'{chart_config["x"]} vs {chart_config["y"]}')
            ax.set_xlabel(chart_config['x'])
            ax.set_ylabel(chart_config['y'])
        
        st.pyplot(fig)
        
        if st.button('➡️ Continuar al Paso 7'):
            st.session_state.step = 6
            st.experimental_rerun()

# ------------------------
# Paso 7: Exportar resultados
# ------------------------
if st.session_state.step == 6:
    st.header('Paso 7: Exportar resultados')
    df = st.session_state.get('df')
    
    if df is not None:
        st.subheader('Ejecutar pipeline completo')
        st.info('El pipeline realizará: limpieza, transformación, modelado y generará un reporte PDF.')
        
        if st.button('🚀 Ejecutar Pipeline Completo'):
            with st.spinner('Ejecutando pipeline...'):
                try:
                    # Crear configuración temporal
                    config = {
                        'input_path': 'temp_data.csv',
                        'output_report': 'reporte_proyecto_j.pdf'
                    }
                    
                    # Guardar datos temporalmente
                    df.to_csv('temp_data.csv', index=False)
                    
                    # Ejecutar pipeline
                    pipeline = Pipeline(config)
                    df_processed, model, results = pipeline.run()
                    
                    # Limpiar archivo temporal
                    if os.path.exists('temp_data.csv'):
                        os.remove('temp_data.csv')
                    
                    st.success('Pipeline ejecutado correctamente!')
                    
                    # Mostrar resultados
                    st.subheader('Resultados del modelo')
                    st.write(f"Coeficiente: {results.get('coef', 'N/A')}")
                    st.write(f"Intercepto: {results.get('intercept', 'N/A')}")
                    
                    # Botón de descarga
                    if os.path.exists('reporte_proyecto_j.pdf'):
                        with open('reporte_proyecto_j.pdf', 'rb') as f:
                            st.download_button(
                                '💾 Descargar Reporte PDF',
                                data=f.read(),
                                file_name='reporte_proyecto_j.pdf',
                                mime='application/pdf'
                            )
                    
                except Exception as e:
                    st.error(f'Error en el pipeline: {str(e)}')
        
        if st.button('🔄 Reiniciar Asistente'):
            st.session_state.clear()
            st.experimental_rerun()

# ------------------------
# Footer
# ------------------------
st.markdown('---')
st.caption('© 2025 Proyecto J - Pipeline Modular + Streamlit Wizard') 