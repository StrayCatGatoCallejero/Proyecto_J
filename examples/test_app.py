import streamlit as st

st.title("Test App - Máquina de Relojería")
st.write("¡Hola! Esta es una aplicación de prueba.")

# Test de componentes básicos
st.header("Componentes de Prueba")

# Métricas
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Filas", "1,000")
with col2:
    st.metric("Columnas", "15")
with col3:
    st.metric("Memoria", "2.5 MB")

# Gráfico simple
import pandas as pd
import numpy as np

data = pd.DataFrame({"x": np.random.randn(100), "y": np.random.randn(100)})

st.scatter_chart(data)

st.success("✅ ¡La aplicación funciona correctamente!")
