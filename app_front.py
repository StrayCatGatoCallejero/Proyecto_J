# app_front.py
import streamlit as st
from estadistica.estadistica import (
    cargar_archivo,
    calcular_media,
    calcular_moda,
    calcular_percentiles,
    generar_histograma
)

st.set_page_config(page_title="🔢 Estadísticas Ninja", layout="wide")
st.title("🔢 Procesamiento Estadístico + Frontend")

archivo = st.file_uploader("📂 Sube tu archivo .sav o .dta", type=["sav", "dta"])
if archivo:
    with open("data/temp_file", "wb") as f:
        f.write(archivo.getbuffer())
    try:
        df = cargar_archivo("data/temp_file")
        st.success("Archivo cargado correctamente 🎉")
        
        cols_num = df.select_dtypes(include=["number"]).columns.tolist()
        columna = st.selectbox("🔍 Selecciona columna numérica", cols_num)
        
        if columna:
            st.subheader("📊 Estadísticas básicas")
            st.write(f"• Media: **{calcular_media(df, columna):.2f}**")
            st.write(f"• Moda: **{', '.join(map(str, calcular_moda(df, columna)))}**")
            pct = calcular_percentiles(df, columna)
            st.write("• Percentiles:")
            st.write(pct)
            
            st.subheader("📈 Histograma")
            fig = generar_histograma(df, columna)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error: {e}")
