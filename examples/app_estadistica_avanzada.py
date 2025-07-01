import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency
import io
import base64
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(
    page_title="📊 Análisis Estadístico Avanzado",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ecf0f1;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Inicialización del session state
if "df" not in st.session_state:
    st.session_state.df = None
if "df_filtrado" not in st.session_state:
    st.session_state.df_filtrado = None
if "filtros_aplicados" not in st.session_state:
    st.session_state.filtros_aplicados = {}
if "resultados_correlacion" not in st.session_state:
    st.session_state.resultados_correlacion = None
if "resultados_contingencia" not in st.session_state:
    st.session_state.resultados_contingencia = None

# ============================================================================
# FUNCIONES DE ANÁLISIS ESTADÍSTICO
# ============================================================================


def detectar_tipos_columnas(
    df: pd.DataFrame, umbral_cardinalidad=20, umbral_texto_largo=50
):
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
            # Verificar si es entero o flotante
            if pd.api.types.is_integer_dtype(serie):
                tipo = "numérico (entero)"
                detalles = f"Rango: {serie.min()} a {serie.max()}"
            else:
                tipo = "numérico (flotante)"
                detalles = f"Rango: {serie.min():.2f} a {serie.max():.2f}"
        elif pd.api.types.is_object_dtype(serie) or pd.api.types.is_categorical_dtype(
            serie
        ):
            n_unicos = serie.nunique(dropna=True)
            muestra = (
                serie.dropna()
                .astype(str)
                .sample(min(10, len(serie.dropna())), random_state=1)
                if len(serie.dropna()) > 0
                else []
            )
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
        resumen.append({"columna": col, "tipo_detectado": tipo, "detalles": detalles})
    return pd.DataFrame(resumen)


def calcular_correlacion(df, variables, metodo="pearson"):
    """Calcula matriz de correlación para variables numéricas."""
    try:
        # Filtrar solo variables numéricas con detección más robusta
        vars_numericas = []
        for var in variables:
            if var in df.columns:
                # Verificar si es numérico usando pandas
                if pd.api.types.is_numeric_dtype(df[var]):
                    vars_numericas.append(var)
                else:
                    # Intentar convertir a numérico si es posible
                    try:
                        pd.to_numeric(df[var], errors="raise")
                        vars_numericas.append(var)
                    except (ValueError, TypeError):
                        continue

        if len(vars_numericas) < 2:
            return (
                None,
                f"Se necesitan al menos 2 variables numéricas. Variables válidas: {vars_numericas}",
            )

        # Calcular correlación
        if metodo == "pearson":
            corr_matrix = df[vars_numericas].corr(method="pearson")
        else:  # spearman
            corr_matrix = df[vars_numericas].corr(method="spearman")

        return corr_matrix, None
    except Exception as e:
        return None, str(e)


def crear_heatmap_correlacion(corr_matrix, titulo):
    """Crea un heatmap de correlación usando Plotly."""
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr_matrix.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=titulo,
        width=600,
        height=500,
        xaxis_title="Variables",
        yaxis_title="Variables",
    )

    return fig


def crear_tabla_contingencia(df, var1, var2):
    """Crea tabla de contingencia y realiza prueba chi-cuadrado."""
    try:
        # Crear tabla de contingencia
        tabla = pd.crosstab(df[var1], df[var2], margins=True, margins_name="Total")

        # Calcular porcentajes
        tabla_porcentaje = pd.crosstab(df[var1], df[var2], normalize="index") * 100

        # Prueba chi-cuadrado (sin los totales)
        tabla_sin_totales = pd.crosstab(df[var1], df[var2])
        chi2, p_value, dof, expected = chi2_contingency(tabla_sin_totales)

        # Interpretación
        if p_value < 0.001:
            interpretacion = "Muy significativa (p < 0.001)"
        elif p_value < 0.01:
            interpretacion = "Altamente significativa (p < 0.01)"
        elif p_value < 0.05:
            interpretacion = "Significativa (p < 0.05)"
        else:
            interpretacion = "No significativa (p >= 0.05)"

        return {
            "tabla": tabla,
            "tabla_porcentaje": tabla_porcentaje,
            "chi2": chi2,
            "p_value": p_value,
            "dof": dof,
            "interpretacion": interpretacion,
        }
    except Exception as e:
        return None


def crear_visualizacion_contingencia(df, var1, var2, tipo="barras"):
    """Crea visualizaciones para tabla de contingencia."""
    if tipo == "barras":
        fig = px.bar(
            df.groupby([var1, var2]).size().reset_index(name="Frecuencia"),
            x=var1,
            y="Frecuencia",
            color=var2,
            title=f"Distribución de {var1} por {var2}",
            barmode="group",
        )
    elif tipo == "heatmap":
        tabla = pd.crosstab(df[var1], df[var2])
        fig = go.Figure(
            data=go.Heatmap(
                z=tabla.values,
                x=tabla.columns,
                y=tabla.index,
                colorscale="Blues",
                text=tabla.values,
                texttemplate="%{text}",
                textfont={"size": 12},
            )
        )
        fig.update_layout(
            title=f"Tabla de Contingencia: {var1} vs {var2}",
            xaxis_title=var2,
            yaxis_title=var1,
        )

    return fig


def aplicar_filtros(df, filtros):
    """Aplica filtros dinámicos al DataFrame."""
    df_filtrado = df.copy()

    for columna, filtro in filtros.items():
        if filtro["tipo"] == "rango":
            min_val, max_val = filtro["valores"]
            df_filtrado = df_filtrado[
                (df_filtrado[columna] >= min_val) & (df_filtrado[columna] <= max_val)
            ]
        elif filtro["tipo"] == "categorias":
            categorias_seleccionadas = filtro["valores"]
            df_filtrado = df_filtrado[
                df_filtrado[columna].isin(categorias_seleccionadas)
            ]

    return df_filtrado


def crear_visualizacion_avanzada(df, tipo, variables, **kwargs):
    """Crea visualizaciones avanzadas."""
    if tipo == "boxplot":
        if len(variables) == 1:
            fig = px.box(df, y=variables[0], title=f"Boxplot de {variables[0]}")
        else:
            fig = px.box(
                df,
                x=variables[1],
                y=variables[0],
                title=f"Boxplot de {variables[0]} por {variables[1]}",
            )

    elif tipo == "scatter":
        fig = px.scatter(
            df,
            x=variables[0],
            y=variables[1],
            title=f"Dispersión: {variables[0]} vs {variables[1]}",
            trendline="ols" if kwargs.get("linea_tendencia", False) else None,
        )

    elif tipo == "density":
        fig = px.histogram(
            df,
            x=variables[0],
            nbins=30,
            title=f"Distribución de densidad de {variables[0]}",
            marginal="box",
        )

    elif tipo == "violin":
        if len(variables) == 1:
            fig = px.violin(df, y=variables[0], title=f"Violin plot de {variables[0]}")
        else:
            fig = px.violin(
                df,
                x=variables[1],
                y=variables[0],
                title=f"Violin plot de {variables[0]} por {variables[1]}",
            )

    elif tipo == "scatter_matrix":
        fig = px.scatter_matrix(df[variables], title="Matriz de dispersión")

    elif tipo == "heatmap_avanzada":
        corr_matrix = df[variables].corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu_r",
                zmid=0,
            )
        )
        fig.update_layout(title="Matriz de Correlación")

    return fig


def exportar_resultados(df, resultados_corr, resultados_cont, formato="csv"):
    """Exporta resultados en diferentes formatos."""
    if formato == "csv":
        # Crear múltiples archivos CSV
        output = io.StringIO()
        output.write("=== RESUMEN ESTADÍSTICO ===\n\n")
        output.write(
            f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        output.write(f"Total de registros: {len(df):,}\n")
        output.write(f"Total de variables: {len(df.columns)}\n\n")

        # Estadísticas descriptivas
        output.write("=== ESTADÍSTICAS DESCRIPTIVAS ===\n")
        output.write(df.describe().to_csv())
        output.write("\n\n")

        # Correlaciones
        if resultados_corr is not None:
            output.write("=== MATRIZ DE CORRELACIÓN ===\n")
            output.write(resultados_corr.to_csv())
            output.write("\n\n")

        # Tablas de contingencia
        if resultados_cont is not None:
            output.write("=== TABLA DE CONTINGENCIA ===\n")
            output.write(resultados_cont["tabla"].to_csv())
            output.write(f"\nChi-cuadrado: {resultados_cont['chi2']:.4f}\n")
            output.write(f"p-valor: {resultados_cont['p_value']:.4f}\n")
            output.write(f"Interpretación: {resultados_cont['interpretacion']}\n")

        return output.getvalue()

    elif formato == "html":
        # Crear reporte HTML
        html_content = f"""
        <html>
        <head>
            <title>Reporte Estadístico</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Reporte de Análisis Estadístico</h1>
                <p>Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📋 Resumen General</h2>
                <p><strong>Total de registros:</strong> {len(df):,}</p>
                <p><strong>Total de variables:</strong> {len(df.columns)}</p>
                <p><strong>Variables numéricas:</strong> {len(df.select_dtypes(include=[np.number]).columns)}</p>
                <p><strong>Variables categóricas:</strong> {len(df.select_dtypes(include=['object', 'category']).columns)}</p>
            </div>
        """

        # Agregar estadísticas descriptivas
        html_content += f"""
            <div class="section">
                <h2>📈 Estadísticas Descriptivas</h2>
                {df.describe().to_html()}
            </div>
        """

        # Agregar correlaciones si existen
        if resultados_corr is not None:
            html_content += f"""
                <div class="section">
                    <h2>🔗 Matriz de Correlación</h2>
                    {resultados_corr.to_html()}
                </div>
            """

        # Agregar tabla de contingencia si existe
        if resultados_cont is not None:
            html_content += f"""
                <div class="section">
                    <h2>📊 Tabla de Contingencia</h2>
                    {resultados_cont['tabla'].to_html()}
                    <p><strong>Chi-cuadrado:</strong> {resultados_cont['chi2']:.4f}</p>
                    <p><strong>p-valor:</strong> {resultados_cont['p_value']:.4f}</p>
                    <p><strong>Interpretación:</strong> {resultados_cont['interpretacion']}</p>
                </div>
            """

        html_content += """
        </body>
        </html>
        """

        return html_content


# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.markdown(
    '<h1 class="main-header">📊 Análisis Estadístico Avanzado</h1>',
    unsafe_allow_html=True,
)
st.write(
    "Aplicación para análisis estadístico avanzado con filtros dinámicos, correlaciones, tablas de contingencia y visualizaciones."
)

# Sidebar para navegación
with st.sidebar:
    st.title("🗺️ Navegación")

    # Cargar datos
    st.header("📁 Cargar Datos")
    uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV",
        type=["csv"],
        help="Sube tu archivo CSV para comenzar el análisis",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.df_filtrado = df
            st.success(
                f"✅ Datos cargados: {df.shape[0]:,} filas × {df.shape[1]} columnas"
            )
        except Exception as e:
            st.error(f"❌ Error al cargar archivo: {e}")
            st.stop()

    # Información del dataset
    if st.session_state.df is not None:
        st.header("📊 Información del Dataset")
        df = st.session_state.df

        st.metric("Filas", f"{len(df):,}")
        st.metric("Columnas", len(df.columns))
        st.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        # Tipos de variables
        tipos_df = detectar_tipos_columnas(df)
        vars_numericas = tipos_df[
            tipos_df["tipo_detectado"].str.contains("numérico", na=False)
        ]["columna"].tolist()
        vars_categoricas = tipos_df[tipos_df["tipo_detectado"] == "categórico"][
            "columna"
        ].tolist()

        st.metric("Variables Numéricas", len(vars_numericas))
        st.metric("Variables Categóricas", len(vars_categoricas))

# ============================================================================
# SECCIÓN DE FILTROS DINÁMICOS
# ============================================================================

if st.session_state.df is not None:
    st.markdown(
        '<h2 class="section-header">🔍 Filtros Dinámicos</h2>', unsafe_allow_html=True
    )

    df = st.session_state.df
    tipos_df = detectar_tipos_columnas(df)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Variables Numéricas")
        vars_numericas = tipos_df[
            tipos_df["tipo_detectado"].str.contains("numérico", na=False)
        ]["columna"].tolist()

        for var in vars_numericas:
            if var not in st.session_state.filtros_aplicados:
                st.session_state.filtros_aplicados[var] = {
                    "tipo": "rango",
                    "valores": [float(df[var].min()), float(df[var].max())],
                }

            min_val, max_val = st.slider(
                f"Rango de {var}",
                min_value=float(df[var].min()),
                max_value=float(df[var].max()),
                value=st.session_state.filtros_aplicados[var]["valores"],
                key=f"slider_{var}",
            )
            st.session_state.filtros_aplicados[var]["valores"] = [min_val, max_val]

    with col2:
        st.subheader("📋 Variables Categóricas")
        vars_categoricas = tipos_df[tipos_df["tipo_detectado"] == "categórico"][
            "columna"
        ].tolist()

        for var in vars_categoricas:
            categorias = df[var].unique().tolist()

            if var not in st.session_state.filtros_aplicados:
                st.session_state.filtros_aplicados[var] = {
                    "tipo": "categorias",
                    "valores": categorias,
                }

            categorias_seleccionadas = st.multiselect(
                f"Categorías de {var}",
                options=categorias,
                default=st.session_state.filtros_aplicados[var]["valores"],
                key=f"multiselect_{var}",
            )
            st.session_state.filtros_aplicados[var][
                "valores"
            ] = categorias_seleccionadas

    # Aplicar filtros
    if st.button("🔍 Aplicar Filtros", type="primary"):
        df_filtrado = aplicar_filtros(df, st.session_state.filtros_aplicados)
        st.session_state.df_filtrado = df_filtrado
        st.success(f"✅ Filtros aplicados: {len(df_filtrado):,} registros restantes")

    # Mostrar resumen de filtros
    if st.session_state.df_filtrado is not None and len(
        st.session_state.df_filtrado
    ) != len(df):
        st.info(
            f"📊 **Datos filtrados:** {len(st.session_state.df_filtrado):,} de {len(df):,} registros"
        )

        with st.expander("👀 Vista previa de datos filtrados"):
            st.dataframe(st.session_state.df_filtrado.head(), use_container_width=True)

# ============================================================================
# SECCIÓN DE ANÁLISIS DE CORRELACIÓN
# ============================================================================

if st.session_state.df_filtrado is not None:
    st.markdown(
        '<h2 class="section-header">🔗 Análisis de Correlación</h2>',
        unsafe_allow_html=True,
    )

    df_actual = st.session_state.df_filtrado
    tipos_df = detectar_tipos_columnas(df_actual)
    vars_numericas = tipos_df[
        tipos_df["tipo_detectado"].str.contains("numérico", na=False)
    ]["columna"].tolist()

    if len(vars_numericas) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            variables_seleccionadas = st.multiselect(
                "Selecciona variables numéricas:",
                options=vars_numericas,
                default=vars_numericas[: min(5, len(vars_numericas))],
                help="Selecciona al menos 2 variables para el análisis de correlación",
            )

        with col2:
            metodo_correlacion = st.selectbox(
                "Método de correlación:",
                options=["pearson", "spearman"],
                help="Pearson para relaciones lineales, Spearman para relaciones monótonas",
            )

        if len(variables_seleccionadas) >= 2:
            if st.button("📊 Calcular Correlación", type="primary"):
                corr_matrix, error = calcular_correlacion(
                    df_actual, variables_seleccionadas, metodo_correlacion
                )

                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    st.session_state.resultados_correlacion = corr_matrix

                    # Mostrar matriz de correlación
                    st.subheader("📈 Matriz de Correlación")
                    st.dataframe(corr_matrix.round(3), use_container_width=True)

                    # Crear heatmap
                    fig_heatmap = crear_heatmap_correlacion(
                        corr_matrix,
                        f"Matriz de Correlación ({metodo_correlacion.title()})",
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                    # Interpretación
                    st.subheader("💡 Interpretación")
                    st.markdown(
                        """
                    - **Valores cercanos a 1**: Correlación positiva fuerte
                    - **Valores cercanos a -1**: Correlación negativa fuerte  
                    - **Valores cercanos a 0**: Poca o ninguna correlación
                    - **Pearson**: Mide correlación lineal
                    - **Spearman**: Mide correlación monótona (no necesariamente lineal)
                    """
                    )
    else:
        st.warning(
            "⚠️ Se necesitan al menos 2 variables numéricas para el análisis de correlación"
        )

# ============================================================================
# SECCIÓN DE TABLAS DE CONTINGENCIA
# ============================================================================

if st.session_state.df_filtrado is not None:
    st.markdown(
        '<h2 class="section-header">📊 Tablas de Contingencia</h2>',
        unsafe_allow_html=True,
    )

    df_actual = st.session_state.df_filtrado
    tipos_df = detectar_tipos_columnas(df_actual)
    vars_categoricas = tipos_df[tipos_df["tipo_detectado"] == "categórico"][
        "columna"
    ].tolist()

    if len(vars_categoricas) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            var1 = st.selectbox(
                "Variable 1:", options=vars_categoricas, key="cont_var1"
            )

        with col2:
            var2 = st.selectbox(
                "Variable 2:", options=vars_categoricas, key="cont_var2"
            )

        if var1 != var2:
            if st.button("📊 Crear Tabla de Contingencia", type="primary"):
                resultados = crear_tabla_contingencia(df_actual, var1, var2)

                if resultados:
                    st.session_state.resultados_contingencia = resultados

                    # Mostrar tabla de contingencia
                    st.subheader("📋 Tabla de Contingencia")
                    st.dataframe(resultados["tabla"], use_container_width=True)

                    # Mostrar porcentajes
                    st.subheader("📊 Porcentajes por Fila")
                    st.dataframe(
                        resultados["tabla_porcentaje"].round(2),
                        use_container_width=True,
                    )

                    # Resultados de la prueba chi-cuadrado
                    st.subheader("🔬 Prueba Chi-Cuadrado de Independencia")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Chi-cuadrado", f"{resultados['chi2']:.4f}")

                    with col2:
                        st.metric("p-valor", f"{resultados['p_value']:.4f}")

                    with col3:
                        st.metric("Grados de libertad", resultados["dof"])

                    st.info(f"**Interpretación:** {resultados['interpretacion']}")

                    # Visualizaciones
                    st.subheader("📈 Visualizaciones")
                    tipo_vis = st.selectbox(
                        "Tipo de visualización:", ["barras", "heatmap"]
                    )

                    fig_vis = crear_visualizacion_contingencia(
                        df_actual, var1, var2, tipo_vis
                    )
                    st.plotly_chart(fig_vis, use_container_width=True)
                else:
                    st.error("❌ Error al crear la tabla de contingencia")
    else:
        st.warning(
            "⚠️ Se necesitan al menos 2 variables categóricas para las tablas de contingencia"
        )

# ============================================================================
# SECCIÓN DE VISUALIZACIONES AVANZADAS
# ============================================================================

if st.session_state.df_filtrado is not None:
    st.markdown(
        '<h2 class="section-header">🎨 Visualizaciones Avanzadas</h2>',
        unsafe_allow_html=True,
    )

    df_actual = st.session_state.df_filtrado
    tipos_df = detectar_tipos_columnas(df_actual)
    vars_numericas = tipos_df[
        tipos_df["tipo_detectado"].str.contains("numérico", na=False)
    ]["columna"].tolist()
    vars_categoricas = tipos_df[tipos_df["tipo_detectado"] == "categórico"][
        "columna"
    ].tolist()

    col1, col2 = st.columns(2)

    with col1:
        tipo_visualizacion = st.selectbox(
            "Tipo de visualización:",
            options=[
                "boxplot",
                "scatter",
                "density",
                "violin",
                "scatter_matrix",
                "heatmap_avanzada",
            ],
            help="Selecciona el tipo de visualización que deseas crear",
        )

    with col2:
        if tipo_visualizacion in ["scatter", "scatter_matrix", "heatmap_avanzada"]:
            if tipo_visualizacion == "scatter":
                if len(vars_numericas) >= 2:
                    var_x = st.selectbox(
                        "Variable X:", options=vars_numericas, key="vis_x"
                    )
                    var_y = st.selectbox(
                        "Variable Y:", options=vars_numericas, key="vis_y"
                    )
                    variables = [var_x, var_y]
                    linea_tendencia = st.checkbox("Incluir línea de tendencia")
                else:
                    st.warning("Se necesitan al menos 2 variables numéricas")
                    variables = []
                    linea_tendencia = False
            else:
                variables = st.multiselect(
                    "Selecciona variables:",
                    options=vars_numericas,
                    default=vars_numericas[: min(4, len(vars_numericas))],
                    help="Selecciona las variables para la matriz de dispersión o correlación",
                )
                linea_tendencia = False
        elif tipo_visualizacion == "boxplot":
            var_y = st.selectbox(
                "Variable numérica:", options=vars_numericas, key="box_y"
            )
            if vars_categoricas:
                var_x = st.selectbox(
                    "Variable categórica (opcional):",
                    options=["Ninguna"] + vars_categoricas,
                    key="box_x",
                )
                if var_x == "Ninguna":
                    variables = [var_y]
                else:
                    variables = [var_y, var_x]
            else:
                variables = [var_y]
            linea_tendencia = False
        elif tipo_visualizacion == "violin":
            var_y = st.selectbox(
                "Variable numérica:", options=vars_numericas, key="violin_y"
            )
            if vars_categoricas:
                var_x = st.selectbox(
                    "Variable categórica (opcional):",
                    options=["Ninguna"] + vars_categoricas,
                    key="violin_x",
                )
                if var_x == "Ninguna":
                    variables = [var_y]
                else:
                    variables = [var_y, var_x]
            else:
                variables = [var_y]
            linea_tendencia = False
        else:  # density
            variables = [
                st.selectbox(
                    "Variable numérica:", options=vars_numericas, key="density_var"
                )
            ]
            linea_tendencia = False

    if variables and st.button("🎨 Crear Visualización", type="primary"):
        try:
            fig = crear_visualizacion_avanzada(
                df_actual,
                tipo_visualizacion,
                variables,
                linea_tendencia=linea_tendencia,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error al crear la visualización: {e}")

# ============================================================================
# SECCIÓN DE EXPORTACIÓN
# ============================================================================

if st.session_state.df_filtrado is not None:
    st.markdown(
        '<h2 class="section-header">💾 Exportar Resultados</h2>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Exportar Datos")

        # Exportar datos filtrados
        csv_data = st.session_state.df_filtrado.to_csv(index=False)
        st.download_button(
            label="📄 Descargar datos filtrados (CSV)",
            data=csv_data,
            file_name=f"datos_filtrados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        # Exportar estadísticas descriptivas
        if st.session_state.df_filtrado is not None:
            stats_csv = st.session_state.df_filtrado.describe().to_csv()
            st.download_button(
                label="📈 Descargar estadísticas descriptivas (CSV)",
                data=stats_csv,
                file_name=f"estadisticas_descriptivas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    with col2:
        st.subheader("📋 Exportar Análisis Completo")

        # Exportar correlaciones
        if st.session_state.resultados_correlacion is not None:
            corr_csv = st.session_state.resultados_correlacion.to_csv()
            st.download_button(
                label="🔗 Descargar matriz de correlación (CSV)",
                data=corr_csv,
                file_name=f"matriz_correlacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        # Exportar tabla de contingencia
        if st.session_state.resultados_contingencia is not None:
            cont_csv = st.session_state.resultados_contingencia["tabla"].to_csv()
            st.download_button(
                label="📊 Descargar tabla de contingencia (CSV)",
                data=cont_csv,
                file_name=f"tabla_contingencia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    # Reporte completo
    st.subheader("📄 Reporte Completo")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Generar Reporte CSV", type="primary"):
            reporte_csv = exportar_resultados(
                st.session_state.df_filtrado,
                st.session_state.resultados_correlacion,
                st.session_state.resultados_contingencia,
                formato="csv",
            )
            st.download_button(
                label="📄 Descargar reporte completo (CSV)",
                data=reporte_csv,
                file_name=f"reporte_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    with col2:
        if st.button("🌐 Generar Reporte HTML", type="primary"):
            reporte_html = exportar_resultados(
                st.session_state.df_filtrado,
                st.session_state.resultados_correlacion,
                st.session_state.resultados_contingencia,
                formato="html",
            )
            st.download_button(
                label="🌐 Descargar reporte completo (HTML)",
                data=reporte_html,
                file_name=f"reporte_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
            )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        📊 Análisis Estadístico Avanzado - Proyecto J | 
        Desarrollado con Streamlit, Pandas y Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
