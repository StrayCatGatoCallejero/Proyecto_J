# app_front.py
import streamlit as st
from estadistica.estadistica import (
    cargar_archivo,
    calcular_media,
    calcular_moda,
    calcular_percentiles,
    generar_histograma,
    calcular_correlacion_pearson,
    calcular_correlacion_spearman,
    generar_heatmap_correlacion,
    obtener_columnas_numericas,
    obtener_columnas_categoricas,
    crear_tabla_contingencia,
    calcular_chi_cuadrado,
    generar_grafico_tabla_contingencia,
    calcular_porcentajes_tabla_contingencia,
    interpretar_chi_cuadrado,
    crear_filtros_dinamicos,
    aplicar_filtros,
    obtener_estadisticas_filtradas,
    generar_estadisticas_descriptivas_completas,
    generar_resumen_correlaciones,
    generar_resumen_tablas_contingencia,
    generar_csv_datos_filtrados,
    generar_excel_completo,
    generar_html_reporte,
    generar_boxplot,
    generar_scatter_plot,
    generar_diagrama_densidad,
    generar_grafico_barras,
    generar_histograma_densidad,
    generar_violin_plot,
    generar_heatmap_correlacion_avanzado,
    generar_panel_visualizaciones,
    generar_scatter_matrix
)

st.set_page_config(page_title="🔢 Estadísticas Ninja", layout="wide")
st.title("🔢 Procesamiento Estadístico + Frontend")

# Sidebar para navegación
st.sidebar.title("📊 Navegación")
pagina = st.sidebar.selectbox(
    "Selecciona la sección:",
    ["🔍 Filtros", "📈 Estadísticas Básicas", "🔗 Análisis de Correlaciones", "📊 Tablas de Contingencia", "📊 Visualizaciones Avanzadas", "📤 Exportar Resultados"]
)

archivo = st.file_uploader("📂 Sube tu archivo .sav o .dta", type=["sav", "dta"])

if archivo:
    with open("data/temp_file", "wb") as f:
        f.write(archivo.getbuffer())
    try:
        df = cargar_archivo("data/temp_file")
        st.success("Archivo cargado correctamente 🎉")
        
        # Inicializar filtros en session_state si no existen
        if 'filtros_aplicados' not in st.session_state:
            st.session_state.filtros_aplicados = {}
        
        # Inicializar datos de análisis en session_state
        if 'datos_analisis' not in st.session_state:
            st.session_state.datos_analisis = {}
        
        if pagina == "🔍 Filtros":
            st.header("🔍 Filtros Dinámicos")
            st.write("Configura filtros para personalizar tu análisis. Los filtros se aplicarán a todas las secciones.")
            
            # Crear información de filtros
            filtros_info = crear_filtros_dinamicos(df)
            
            if filtros_info:
                st.subheader("📋 Configuración de Filtros")
                
                # Separar variables numéricas y categóricas
                variables_numericas = [col for col, info in filtros_info.items() if info['tipo'] == 'numerico']
                variables_categoricas = [col for col, info in filtros_info.items() if info['tipo'] == 'categorico']
                
                # Filtros para variables numéricas
                if variables_numericas:
                    st.write("**🎯 Filtros por Rango (Variables Numéricas):**")
                    
                    for col in variables_numericas:
                        info = filtros_info[col]
                        min_val, max_val = info['min'], info['max']
                        
                        # Crear slider para rango
                        rango = st.slider(
                            f"📊 {col}",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=(float(min_val), float(max_val)),
                            step=(max_val - min_val) / 100,
                            help=f"Selecciona el rango para {col}"
                        )
                        
                        # Guardar filtro en session_state
                        st.session_state.filtros_aplicados[col] = {
                            'min': rango[0],
                            'max': rango[1]
                        }
                
                # Filtros para variables categóricas
                if variables_categoricas:
                    st.write("**🏷️ Filtros por Categoría (Variables Categóricas):**")
                    
                    for col in variables_categoricas:
                        info = filtros_info[col]
                        categorias = info['categorias']
                        
                        # Crear multiselect para categorías
                        categorias_seleccionadas = st.multiselect(
                            f"📋 {col}",
                            options=categorias,
                            default=categorias,
                            help=f"Selecciona las categorías de {col} que quieres incluir"
                        )
                        
                        # Guardar filtro en session_state
                        st.session_state.filtros_aplicados[col] = categorias_seleccionadas
                
                # Aplicar filtros y mostrar estadísticas
                df_filtrado = aplicar_filtros(df, st.session_state.filtros_aplicados)
                stats_filtradas = obtener_estadisticas_filtradas(df, st.session_state.filtros_aplicados)
                
                # Mostrar resumen de filtros aplicados
                st.subheader("📊 Resumen de Filtros")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📈 Total Original", stats_filtradas['n_original'])
                
                with col2:
                    st.metric("✅ Datos Filtrados", stats_filtradas['n_filtrado'])
                
                with col3:
                    st.metric("📊 % de Muestra", f"{stats_filtradas['porcentaje_muestra']:.1f}%")
                
                # Mostrar filtros activos
                if st.session_state.filtros_aplicados:
                    st.subheader("🔧 Filtros Activos")
                    for col, filtro in st.session_state.filtros_aplicados.items():
                        if isinstance(filtro, dict):
                            st.write(f"• **{col}**: {filtro['min']:.2f} - {filtro['max']:.2f}")
                        elif isinstance(filtro, list):
                            st.write(f"• **{col}**: {', '.join(filtro)}")
                
                # Botón para limpiar filtros
                if st.button("🗑️ Limpiar Todos los Filtros"):
                    st.session_state.filtros_aplicados = {}
                    st.rerun()
                
                # Vista previa de datos filtrados
                st.subheader("👀 Vista Previa de Datos Filtrados")
                st.dataframe(df_filtrado.head(10))
                
                # Botones de exportación para datos filtrados
                st.subheader("📤 Exportar Datos Filtrados")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = generar_csv_datos_filtrados(df, st.session_state.filtros_aplicados)
                    st.download_button(
                        label="📄 Descargar CSV",
                        data=csv_data,
                        file_name="datos_filtrados.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    excel_data = generar_excel_completo(df, st.session_state.filtros_aplicados)
                    st.download_button(
                        label="📊 Descargar Excel",
                        data=excel_data,
                        file_name="datos_filtrados.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
            else:
                st.warning("⚠️ No se encontraron variables para filtrar.")
        
        elif pagina == "📈 Estadísticas Básicas":
            st.header("📈 Estadísticas Básicas")
            
            # Aplicar filtros si existen
            if st.session_state.filtros_aplicados:
                df_analisis = aplicar_filtros(df, st.session_state.filtros_aplicados)
                stats_filtradas = obtener_estadisticas_filtradas(df, st.session_state.filtros_aplicados)
                
                st.info(f"📊 Analizando {stats_filtradas['n_filtrado']} de {stats_filtradas['n_original']} observaciones ({stats_filtradas['porcentaje_muestra']:.1f}% de la muestra)")
            else:
                df_analisis = df
                st.info("📊 Analizando todos los datos (sin filtros aplicados)")
            
            cols_num = obtener_columnas_numericas(df_analisis)
            
            if not cols_num:
                st.warning("⚠️ No hay variables numéricas disponibles para análisis.")
            else:
                columna = st.selectbox("🔍 Selecciona columna numérica", cols_num)
                
                if columna:
                    st.subheader("📊 Estadísticas básicas")
                    st.write(f"• Media: **{calcular_media(df_analisis, columna):.2f}**")
                    st.write(f"• Moda: **{', '.join(map(str, calcular_moda(df_analisis, columna)))}**")
                    pct = calcular_percentiles(df_analisis, columna)
                    st.write("• Percentiles:")
                    st.write(pct)
                    
                    st.subheader("📈 Histograma")
                    fig = generar_histograma(df_analisis, columna)
                    st.pyplot(fig)
                    
                    # Generar estadísticas descriptivas completas para exportación
                    estadisticas_completas = generar_estadisticas_descriptivas_completas(df_analisis)
                    
                    # Guardar en session_state para exportación
                    st.session_state.datos_analisis['estadisticas_descriptivas'] = estadisticas_completas
                    
                    # Botones de exportación
                    st.subheader("📤 Exportar Estadísticas")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_stats = estadisticas_completas.to_csv(index=False)
                        st.download_button(
                            label="📄 Descargar CSV",
                            data=csv_stats,
                            file_name="estadisticas_descriptivas.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        excel_stats = generar_excel_completo(df, st.session_state.filtros_aplicados, estadisticas_completas)
                        st.download_button(
                            label="📊 Descargar Excel",
                            data=excel_stats,
                            file_name="estadisticas_descriptivas.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        
        elif pagina == "🔗 Análisis de Correlaciones":
            st.header("🔗 Análisis de Correlaciones")
            
            # Aplicar filtros si existen
            if st.session_state.filtros_aplicados:
                df_analisis = aplicar_filtros(df, st.session_state.filtros_aplicados)
                stats_filtradas = obtener_estadisticas_filtradas(df, st.session_state.filtros_aplicados)
                
                st.info(f"📊 Analizando {stats_filtradas['n_filtrado']} de {stats_filtradas['n_original']} observaciones ({stats_filtradas['porcentaje_muestra']:.1f}% de la muestra)")
            else:
                df_analisis = df
                st.info("📊 Analizando todos los datos (sin filtros aplicados)")
            
            cols_num = obtener_columnas_numericas(df_analisis)
            
            if len(cols_num) < 2:
                st.warning("⚠️ Se necesitan al menos 2 variables numéricas para calcular correlaciones.")
            else:
                st.subheader("📋 Selección de Variables")
                st.write("Selecciona 2 o más variables numéricas para analizar sus correlaciones:")
                
                # Selección múltiple de variables
                variables_seleccionadas = st.multiselect(
                    "🔍 Variables a analizar:",
                    cols_num,
                    default=cols_num[:min(5, len(cols_num))],  # Por defecto selecciona hasta 5 variables
                    help="Selecciona al menos 2 variables para calcular correlaciones"
                )
                
                if len(variables_seleccionadas) >= 2:
                    st.subheader("📊 Matriz de Correlaciones")
                    
                    # Selector de tipo de correlación
                    tipo_correlacion = st.radio(
                        "🎯 Tipo de correlación:",
                        ["Pearson", "Spearman"],
                        horizontal=True,
                        help="Pearson: para relaciones lineales, Spearman: para relaciones monótonas"
                    )
                    
                    # Calcular correlación según el tipo seleccionado
                    if tipo_correlacion == "Pearson":
                        matriz_corr = calcular_correlacion_pearson(df_analisis, variables_seleccionadas)
                        titulo_heatmap = "Matriz de Correlación de Pearson"
                    else:
                        matriz_corr = calcular_correlacion_spearman(df_analisis, variables_seleccionadas)
                        titulo_heatmap = "Matriz de Correlación de Spearman"
                    
                    # Mostrar matriz de correlación como tabla
                    st.write("**Matriz de Correlación:**")
                    st.dataframe(matriz_corr.style.background_gradient(cmap='coolwarm', center=0))
                    
                    # Mostrar heatmap
                    st.subheader("🔥 Heatmap de Correlación")
                    fig_heatmap = generar_heatmap_correlacion(matriz_corr, titulo_heatmap)
                    st.pyplot(fig_heatmap)
                    
                    # Generar resumen de correlaciones para exportación
                    resumen_correlaciones = generar_resumen_correlaciones(df_analisis, variables_seleccionadas, tipo_correlacion.lower())
                    
                    # Guardar en session_state para exportación
                    st.session_state.datos_analisis['correlaciones'] = resumen_correlaciones
                    
                    # Información adicional sobre las correlaciones
                    st.subheader("📝 Interpretación")
                    st.write("""
                    **Guía de interpretación:**
                    - **1.0 a 0.7**: Correlación muy fuerte positiva
                    - **0.7 a 0.5**: Correlación fuerte positiva  
                    - **0.5 a 0.3**: Correlación moderada positiva
                    - **0.3 a 0.1**: Correlación débil positiva
                    - **0.1 a -0.1**: Sin correlación
                    - **-0.1 a -0.3**: Correlación débil negativa
                    - **-0.3 a -0.5**: Correlación moderada negativa
                    - **-0.5 a -0.7**: Correlación fuerte negativa
                    - **-0.7 a -1.0**: Correlación muy fuerte negativa
                    """)
                    
                    # Estadísticas adicionales
                    st.subheader("📈 Estadísticas de la Muestra")
                    st.write(f"• **Número de observaciones:** {len(df_analisis[variables_seleccionadas].dropna())}")
                    st.write(f"• **Variables analizadas:** {len(variables_seleccionadas)}")
                    
                    # Mostrar correlaciones más fuertes
                    st.subheader("🔍 Correlaciones Destacadas")
                    # Obtener pares de correlaciones (sin diagonal)
                    correlaciones = []
                    for i in range(len(matriz_corr.columns)):
                        for j in range(i+1, len(matriz_corr.columns)):
                            var1 = matriz_corr.columns[i]
                            var2 = matriz_corr.columns[j]
                            corr_valor = matriz_corr.iloc[i, j]
                            correlaciones.append((var1, var2, corr_valor))
                    
                    # Ordenar por valor absoluto de correlación
                    correlaciones.sort(key=lambda x: abs(x[2]), reverse=True)
                    
                    # Mostrar las 5 correlaciones más fuertes
                    st.write("**Top 5 correlaciones más fuertes:**")
                    for i, (var1, var2, corr_valor) in enumerate(correlaciones[:5], 1):
                        color = "🟢" if corr_valor > 0 else "🔴"
                        st.write(f"{i}. {color} **{var1}** ↔ **{var2}**: {corr_valor:.3f}")
                    
                    # Botones de exportación
                    st.subheader("📤 Exportar Correlaciones")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_corr = resumen_correlaciones.to_csv(index=False)
                        st.download_button(
                            label="📄 Descargar CSV",
                            data=csv_corr,
                            file_name=f"correlaciones_{tipo_correlacion.lower()}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        excel_corr = generar_excel_completo(df, st.session_state.filtros_aplicados, None, resumen_correlaciones)
                        st.download_button(
                            label="📊 Descargar Excel",
                            data=excel_corr,
                            file_name=f"correlaciones_{tipo_correlacion.lower()}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                else:
                    st.warning("⚠️ Por favor selecciona al menos 2 variables para continuar.")
        
        elif pagina == "📊 Tablas de Contingencia":
            st.header("📊 Tablas de Contingencia y Prueba χ²")
            
            # Aplicar filtros si existen
            if st.session_state.filtros_aplicados:
                df_analisis = aplicar_filtros(df, st.session_state.filtros_aplicados)
                stats_filtradas = obtener_estadisticas_filtradas(df, st.session_state.filtros_aplicados)
                
                st.info(f"📊 Analizando {stats_filtradas['n_filtrado']} de {stats_filtradas['n_original']} observaciones ({stats_filtradas['porcentaje_muestra']:.1f}% de la muestra)")
            else:
                df_analisis = df
                st.info("📊 Analizando todos los datos (sin filtros aplicados)")
            
            cols_cat = obtener_columnas_categoricas(df_analisis)
            
            if len(cols_cat) < 2:
                st.warning("⚠️ Se necesitan al menos 2 variables categóricas para crear tablas de contingencia.")
            else:
                st.subheader("📋 Selección de Variables Categóricas")
                st.write("Selecciona dos variables categóricas para analizar su relación:")
                
                # Selección de variables categóricas
                col1, col2 = st.columns(2)
                with col1:
                    variable1 = st.selectbox(
                        "🔍 Primera variable:",
                        cols_cat,
                        help="Selecciona la primera variable categórica"
                    )
                
                with col2:
                    variable2 = st.selectbox(
                        "🔍 Segunda variable:",
                        [col for col in cols_cat if col != variable1],
                        help="Selecciona la segunda variable categórica"
                    )
                
                if variable1 and variable2:
                    st.subheader("📊 Tabla de Contingencia")
                    
                    # Crear tabla de contingencia
                    tabla_contingencia = crear_tabla_contingencia(df_analisis, variable1, variable2)
                    
                    # Mostrar tabla de contingencia
                    st.write(f"**Tabla de Contingencia: {variable1} vs {variable2}**")
                    st.dataframe(tabla_contingencia)
                    
                    # Calcular y mostrar porcentajes
                    st.subheader("📈 Análisis de Porcentajes")
                    porcentajes = calcular_porcentajes_tabla_contingencia(df_analisis, variable1, variable2)
                    
                    # Tabs para diferentes tipos de porcentajes
                    tab1, tab2, tab3 = st.tabs(["Por Fila", "Por Columna", "Del Total"])
                    
                    with tab1:
                        st.write("**Porcentajes por fila** (porcentaje de cada columna dentro de cada fila):")
                        st.dataframe(porcentajes['porcentajes_fila'].round(2))
                    
                    with tab2:
                        st.write("**Porcentajes por columna** (porcentaje de cada fila dentro de cada columna):")
                        st.dataframe(porcentajes['porcentajes_columna'].round(2))
                    
                    with tab3:
                        st.write("**Porcentajes del total** (porcentaje de cada celda del total):")
                        st.dataframe(porcentajes['porcentajes_total'].round(2))
                    
                    # Prueba de Chi-cuadrado
                    st.subheader("🔬 Prueba de Chi-cuadrado (χ²)")
                    
                    # Calcular chi-cuadrado
                    resultados_chi = calcular_chi_cuadrado(df_analisis, variable1, variable2)
                    
                    # Mostrar resultados
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Estadísticas del test:**")
                        st.write(f"• **χ² = {resultados_chi['chi2_statistic']:.4f}**")
                        st.write(f"• **p-valor = {resultados_chi['p_value']:.4f}**")
                        st.write(f"• **Grados de libertad = {resultados_chi['degrees_of_freedom']}**")
                        st.write(f"• **Tamaño de muestra = {resultados_chi['sample_size']}**")
                    
                    with col2:
                        st.write("**Medidas de asociación:**")
                        st.write(f"• **Cramer's V = {resultados_chi['cramer_v']:.4f}**")
                        st.write(f"• **Coeficiente de contingencia = {resultados_chi['pearson_c']:.4f}**")
                    
                    # Interpretación
                    st.subheader("📝 Interpretación")
                    interpretacion = interpretar_chi_cuadrado(resultados_chi)
                    st.write(interpretacion)
                    
                    # Información adicional sobre interpretación
                    st.write("""
                    **Guía de interpretación:**
                    - **p < 0.05**: Existe una relación significativa entre las variables
                    - **p ≥ 0.05**: No hay evidencia suficiente de relación entre las variables
                    - **Cramer's V < 0.1**: Efecto muy pequeño
                    - **Cramer's V 0.1-0.3**: Efecto pequeño
                    - **Cramer's V 0.3-0.5**: Efecto moderado
                    - **Cramer's V > 0.5**: Efecto grande
                    """)
                    
                    # Visualizaciones
                    st.subheader("📊 Visualizaciones")
                    fig_visualizacion = generar_grafico_tabla_contingencia(df_analisis, variable1, variable2)
                    st.pyplot(fig_visualizacion)
                    
                    # Información sobre frecuencias esperadas
                    st.subheader("📋 Frecuencias Esperadas")
                    st.write("Las frecuencias esperadas bajo la hipótesis de independencia:")
                    frecuencias_esperadas = pd.DataFrame(
                        resultados_chi['expected_frequencies'],
                        index=df_analisis[variable1].unique(),
                        columns=df_analisis[variable2].unique()
                    )
                    st.dataframe(frecuencias_esperadas.round(2))
                    
                    # Generar resumen completo para exportación
                    resumen_tablas = generar_resumen_tablas_contingencia(df_analisis, variable1, variable2)
                    
                    # Guardar en session_state para exportación
                    st.session_state.datos_analisis['tablas_contingencia'] = resumen_tablas
                    
                    # Botones de exportación
                    st.subheader("📤 Exportar Análisis de Contingencia")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Exportar tabla de contingencia como CSV
                        csv_tabla = tabla_contingencia.to_csv()
                        st.download_button(
                            label="📄 Descargar CSV",
                            data=csv_tabla,
                            file_name=f"tabla_contingencia_{variable1}_{variable2}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        excel_tabla = generar_excel_completo(df, st.session_state.filtros_aplicados, None, None, resumen_tablas)
                        st.download_button(
                            label="📊 Descargar Excel",
                            data=excel_tabla,
                            file_name=f"tabla_contingencia_{variable1}_{variable2}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                else:
                    st.warning("⚠️ Por favor selecciona dos variables categóricas diferentes para continuar.")
        
        elif pagina == "📊 Visualizaciones Avanzadas":
            st.header("📊 Visualizaciones Avanzadas")
            
            # Aplicar filtros si existen
            if st.session_state.filtros_aplicados:
                df_analisis = aplicar_filtros(df, st.session_state.filtros_aplicados)
                stats_filtradas = obtener_estadisticas_filtradas(df, st.session_state.filtros_aplicados)
                
                st.info(f"📊 Analizando {stats_filtradas['n_filtrado']} de {stats_filtradas['n_original']} observaciones ({stats_filtradas['porcentaje_muestra']:.1f}% de la muestra)")
            else:
                df_analisis = df
                st.info("📊 Analizando todos los datos (sin filtros aplicados)")
            
            cols_num = obtener_columnas_numericas(df_analisis)
            cols_cat = obtener_columnas_categoricas(df_analisis)
            
            if not cols_num:
                st.warning("⚠️ No hay variables numéricas disponibles para visualizaciones avanzadas.")
            else:
                st.subheader("🎨 Tipos de Visualizaciones")
                
                # Selector de tipo de visualización
                tipo_visualizacion = st.selectbox(
                    "🔍 Selecciona el tipo de visualización:",
                    [
                        "📊 Panel Completo de Visualizaciones",
                        "📦 Boxplot",
                        "🔄 Scatter Plot",
                        "📈 Diagrama de Densidad",
                        "📊 Histograma con Densidad",
                        "🎻 Violin Plot",
                        "📊 Gráfico de Barras",
                        "🔥 Heatmap de Correlación Avanzado",
                        "🔗 Matriz de Scatter Plots"
                    ],
                    help="Elige el tipo de visualización que quieres generar"
                )
                
                if tipo_visualizacion == "📊 Panel Completo de Visualizaciones":
                    st.subheader("📊 Panel Completo de Visualizaciones")
                    st.write("Genera un panel completo con múltiples visualizaciones para una variable.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        variable_principal = st.selectbox("📊 Variable principal:", cols_num)
                    
                    with col2:
                        variable_grupo = st.selectbox(
                            "🏷️ Variable de agrupación (opcional):", 
                            ["Ninguna"] + cols_cat
                        )
                        if variable_grupo == "Ninguna":
                            variable_grupo = None
                    
                    if st.button("🎨 Generar Panel"):
                        fig_panel = generar_panel_visualizaciones(df_analisis, variable_principal, variable_grupo)
                        st.pyplot(fig_panel)
                        
                        st.write("**Panel incluye:**")
                        st.write("• Histograma con densidad")
                        st.write("• Boxplot")
                        st.write("• Diagrama de densidad")
                        st.write("• Violin plot (si hay grupo) o Q-Q plot (sin grupo)")
                
                elif tipo_visualizacion == "📦 Boxplot":
                    st.subheader("📦 Boxplot")
                    st.write("Visualiza la distribución de una variable numérica, opcionalmente agrupada por una variable categórica.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        variable_numerica = st.selectbox("📊 Variable numérica:", cols_num)
                    
                    with col2:
                        variable_categorica = st.selectbox(
                            "🏷️ Variable de agrupación (opcional):", 
                            ["Ninguna"] + cols_cat
                        )
                        if variable_categorica == "Ninguna":
                            variable_categorica = None
                    
                    if st.button("📦 Generar Boxplot"):
                        fig_boxplot = generar_boxplot(df_analisis, variable_numerica, variable_categorica)
                        st.pyplot(fig_boxplot)
                
                elif tipo_visualizacion == "🔄 Scatter Plot":
                    st.subheader("🔄 Scatter Plot")
                    st.write("Visualiza la relación entre dos variables numéricas, opcionalmente coloreado por una variable categórica.")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        variable_x = st.selectbox("📊 Variable X:", cols_num)
                    
                    with col2:
                        variable_y = st.selectbox("📊 Variable Y:", cols_num)
                    
                    with col3:
                        variable_color = st.selectbox(
                            "🎨 Variable de color (opcional):", 
                            ["Ninguna"] + cols_cat
                        )
                        if variable_color == "Ninguna":
                            variable_color = None
                    
                    if st.button("🔄 Generar Scatter Plot"):
                        fig_scatter = generar_scatter_plot(df_analisis, variable_x, variable_y, variable_color)
                        st.pyplot(fig_scatter)
                
                elif tipo_visualizacion == "📈 Diagrama de Densidad":
                    st.subheader("📈 Diagrama de Densidad")
                    st.write("Visualiza la distribución de densidad de una variable numérica, opcionalmente agrupada.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        variable_densidad = st.selectbox("📊 Variable:", cols_num)
                    
                    with col2:
                        variable_grupo_densidad = st.selectbox(
                            "🏷️ Variable de agrupación (opcional):", 
                            ["Ninguna"] + cols_cat
                        )
                        if variable_grupo_densidad == "Ninguna":
                            variable_grupo_densidad = None
                    
                    if st.button("📈 Generar Diagrama de Densidad"):
                        fig_densidad = generar_diagrama_densidad(df_analisis, variable_densidad, variable_grupo_densidad)
                        st.pyplot(fig_densidad)
                
                elif tipo_visualizacion == "📊 Histograma con Densidad":
                    st.subheader("📊 Histograma con Densidad")
                    st.write("Combina histograma y curva de densidad para una visualización completa de la distribución.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        variable_hist_dens = st.selectbox("📊 Variable:", cols_num)
                    
                    with col2:
                        variable_grupo_hist_dens = st.selectbox(
                            "🏷️ Variable de agrupación (opcional):", 
                            ["Ninguna"] + cols_cat
                        )
                        if variable_grupo_hist_dens == "Ninguna":
                            variable_grupo_hist_dens = None
                    
                    if st.button("📊 Generar Histograma con Densidad"):
                        fig_hist_dens = generar_histograma_densidad(df_analisis, variable_hist_dens, variable_grupo_hist_dens)
                        st.pyplot(fig_hist_dens)
                
                elif tipo_visualizacion == "🎻 Violin Plot":
                    st.subheader("🎻 Violin Plot")
                    st.write("Visualiza la distribución completa de una variable numérica por grupos categóricos.")
                    
                    if not cols_cat:
                        st.warning("⚠️ Se necesita al menos una variable categórica para generar violin plots.")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            variable_numerica_violin = st.selectbox("📊 Variable numérica:", cols_num)
                        
                        with col2:
                            variable_categorica_violin = st.selectbox("🏷️ Variable categórica:", cols_cat)
                        
                        if st.button("🎻 Generar Violin Plot"):
                            fig_violin = generar_violin_plot(df_analisis, variable_numerica_violin, variable_categorica_violin)
                            st.pyplot(fig_violin)
                
                elif tipo_visualizacion == "📊 Gráfico de Barras":
                    st.subheader("📊 Gráfico de Barras")
                    st.write("Visualiza frecuencias de variables categóricas o promedios de variables numéricas por grupos.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        variable_categorica_barras = st.selectbox("🏷️ Variable categórica:", cols_cat)
                    
                    with col2:
                        variable_numerica_barras = st.selectbox(
                            "📊 Variable numérica (opcional):", 
                            ["Ninguna"] + cols_num
                        )
                        if variable_numerica_barras == "Ninguna":
                            variable_numerica_barras = None
                    
                    if st.button("📊 Generar Gráfico de Barras"):
                        fig_barras = generar_grafico_barras(df_analisis, variable_categorica_barras, variable_numerica_barras)
                        st.pyplot(fig_barras)
                
                elif tipo_visualizacion == "🔥 Heatmap de Correlación Avanzado":
                    st.subheader("🔥 Heatmap de Correlación Avanzado")
                    st.write("Genera un heatmap de correlación con análisis adicional de las correlaciones más fuertes.")
                    
                    if len(cols_num) < 2:
                        st.warning("⚠️ Se necesitan al menos 2 variables numéricas para generar el heatmap.")
                    else:
                        variables_heatmap = st.multiselect(
                            "🔍 Variables para el heatmap:",
                            cols_num,
                            default=cols_num[:min(6, len(cols_num))],
                            help="Selecciona las variables para el análisis de correlación"
                        )
                        
                        if len(variables_heatmap) >= 2:
                            if st.button("🔥 Generar Heatmap Avanzado"):
                                fig_heatmap_avanzado = generar_heatmap_correlacion_avanzado(df_analisis, variables_heatmap)
                                st.pyplot(fig_heatmap_avanzado)
                        else:
                            st.warning("⚠️ Selecciona al menos 2 variables para continuar.")
                
                elif tipo_visualizacion == "🔗 Matriz de Scatter Plots":
                    st.subheader("🔗 Matriz de Scatter Plots")
                    st.write("Genera una matriz de scatter plots para visualizar todas las relaciones entre variables numéricas.")
                    
                    if len(cols_num) < 2:
                        st.warning("⚠️ Se necesitan al menos 2 variables numéricas para generar la matriz.")
                    else:
                        variables_scatter_matrix = st.multiselect(
                            "🔍 Variables para la matriz:",
                            cols_num,
                            default=cols_num[:min(6, len(cols_num))],
                            help="Selecciona hasta 6 variables para la matriz de scatter plots"
                        )
                        
                        if len(variables_scatter_matrix) >= 2:
                            if st.button("🔗 Generar Matriz de Scatter Plots"):
                                fig_scatter_matrix = generar_scatter_matrix(df_analisis, variables_scatter_matrix)
                                st.pyplot(fig_scatter_matrix)
                        else:
                            st.warning("⚠️ Selecciona al menos 2 variables para continuar.")
        
        elif pagina == "📤 Exportar Resultados":
            st.header("📤 Exportar Resultados Completos")
            st.write("Genera reportes completos con todos los análisis realizados.")
            
            # Verificar si hay datos de análisis disponibles
            if not st.session_state.datos_analisis:
                st.warning("⚠️ No hay análisis disponibles para exportar. Realiza algunos análisis primero.")
            else:
                st.subheader("📋 Resumen de Análisis Disponibles")
                
                analisis_disponibles = []
                if 'estadisticas_descriptivas' in st.session_state.datos_analisis:
                    analisis_disponibles.append("📈 Estadísticas Descriptivas")
                
                if 'correlaciones' in st.session_state.datos_analisis:
                    analisis_disponibles.append("🔗 Análisis de Correlaciones")
                
                if 'tablas_contingencia' in st.session_state.datos_analisis:
                    analisis_disponibles.append("📊 Tablas de Contingencia")
                
                for analisis in analisis_disponibles:
                    st.write(f"✅ {analisis}")
                
                st.subheader("📤 Opciones de Exportación")
                
                # Exportar Excel completo
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📊 Reporte Excel Completo**")
                    st.write("Incluye todas las hojas con datos filtrados, estadísticas, correlaciones y tablas de contingencia.")
                    
                    excel_completo = generar_excel_completo(
                        df, 
                        st.session_state.filtros_aplicados,
                        st.session_state.datos_analisis.get('estadisticas_descriptivas'),
                        st.session_state.datos_analisis.get('correlaciones'),
                        st.session_state.datos_analisis.get('tablas_contingencia')
                    )
                    
                    st.download_button(
                        label="📊 Descargar Excel Completo",
                        data=excel_completo,
                        file_name="reporte_analisis_completo.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    st.write("**📄 Reporte HTML**")
                    st.write("Genera un reporte HTML formateado con todos los análisis y resultados.")
                    
                    html_reporte = generar_html_reporte(
                        df,
                        st.session_state.filtros_aplicados,
                        st.session_state.datos_analisis.get('estadisticas_descriptivas'),
                        st.session_state.datos_analisis.get('correlaciones'),
                        st.session_state.datos_analisis.get('tablas_contingencia')
                    )
                    
                    st.download_button(
                        label="📄 Descargar HTML",
                        data=html_reporte,
                        file_name="reporte_analisis.html",
                        mime="text/html"
                    )
                
                # Información adicional
                st.subheader("ℹ️ Información sobre los Formatos")
                
                with st.expander("📊 Formato Excel"):
                    st.write("""
                    **Ventajas del formato Excel:**
                    - Múltiples hojas organizadas
                    - Fácil de manipular y analizar
                    - Compatible con la mayoría de software estadístico
                    - Incluye todos los datos y resultados
                    """)
                
                with st.expander("📄 Formato HTML"):
                    st.write("""
                    **Ventajas del formato HTML:**
                    - Formato profesional y legible
                    - Fácil de compartir por email
                    - Se puede abrir en cualquier navegador
                    - Incluye interpretaciones y guías
                    """)
                
                # Botón para limpiar datos de análisis
                if st.button("🗑️ Limpiar Datos de Análisis"):
                    st.session_state.datos_analisis = {}
                    st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
