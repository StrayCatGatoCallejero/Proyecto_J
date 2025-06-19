# estadistica/estadistica.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, chi2
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import json

def cargar_archivo(path: str) -> pd.DataFrame:
    if path.lower().endswith(".sav"):
        return pd.read_spss(path)
    elif path.lower().endswith(".dta"):
        return pd.read_stata(path)
    else:
        raise ValueError("Formato no soportado: usa .sav o .dta")

def calcular_media(df: pd.DataFrame, columna: str) -> float:
    return float(df[columna].mean())

def calcular_moda(df: pd.DataFrame, columna: str) -> list:
    return df[columna].mode().tolist()

def calcular_percentiles(df: pd.DataFrame, columna: str, percentiles: list = [25, 50, 75]) -> dict:
    qs = df[columna].quantile([p/100 for p in percentiles])
    return {f"P{p}": float(qs.loc[p/100]) for p in percentiles}

def generar_histograma(df: pd.DataFrame, columna: str):
    fig, ax = plt.subplots()
    ax.hist(df[columna].dropna(), bins=20)
    ax.set_title(f"Histograma de {columna}")
    ax.set_xlabel(columna)
    ax.set_ylabel("Frecuencia")
    return fig

def calcular_correlacion_pearson(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """
    Calcula la matriz de correlación de Pearson para las columnas seleccionadas.
    """
    df_seleccionado = df[columnas].dropna()
    return df_seleccionado.corr(method='pearson')

def calcular_correlacion_spearman(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    """
    Calcula la matriz de correlación de Spearman para las columnas seleccionadas.
    """
    df_seleccionado = df[columnas].dropna()
    return df_seleccionado.corr(method='spearman')

def generar_heatmap_correlacion(df_corr: pd.DataFrame, titulo: str = "Matriz de Correlación"):
    """
    Genera un heatmap de la matriz de correlación.
    """
    plt.figure(figsize=(10, 8))
    
    # Crear máscara para mostrar solo la mitad inferior del triángulo
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    
    # Crear el heatmap
    sns.heatmap(df_corr, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8})
    
    plt.title(titulo, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return plt.gcf()

def obtener_columnas_numericas(df: pd.DataFrame) -> list:
    """
    Retorna una lista de las columnas numéricas del DataFrame.
    """
    return df.select_dtypes(include=["number"]).columns.tolist()

def obtener_columnas_categoricas(df: pd.DataFrame) -> list:
    """
    Retorna una lista de las columnas categóricas del DataFrame.
    """
    return df.select_dtypes(include=["object", "category"]).columns.tolist()

def crear_tabla_contingencia(df: pd.DataFrame, columna1: str, columna2: str) -> pd.DataFrame:
    """
    Crea una tabla de contingencia (tabla cruzada) entre dos variables categóricas.
    """
    tabla = pd.crosstab(df[columna1], df[columna2], margins=True, margins_name='Total')
    return tabla

def calcular_chi_cuadrado(df: pd.DataFrame, columna1: str, columna2: str) -> dict:
    """
    Calcula la prueba de chi-cuadrado de independencia entre dos variables categóricas.
    """
    # Crear tabla de contingencia sin totales para el test
    tabla = pd.crosstab(df[columna1], df[columna2])
    
    # Realizar prueba de chi-cuadrado
    chi2_stat, p_value, dof, expected = chi2_contingency(tabla)
    
    # Calcular estadísticas adicionales
    n = tabla.sum().sum()  # Tamaño total de la muestra
    min_dim = min(tabla.shape) - 1  # Dimensión mínima menos 1
    
    # Coeficiente de contingencia de Cramer
    cramer_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0
    
    # Coeficiente de contingencia de Pearson
    pearson_c = np.sqrt(chi2_stat / (chi2_stat + n)) if (chi2_stat + n) > 0 else 0
    
    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'expected_frequencies': expected,
        'cramer_v': cramer_v,
        'pearson_c': pearson_c,
        'sample_size': n
    }

def generar_grafico_tabla_contingencia(df: pd.DataFrame, columna1: str, columna2: str):
    """
    Genera un gráfico de barras apiladas para visualizar la tabla de contingencia.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Barras apiladas
    tabla = pd.crosstab(df[columna1], df[columna2])
    tabla.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_title(f'Distribución de {columna2} por {columna1}')
    ax1.set_xlabel(columna1)
    ax1.set_ylabel('Frecuencia')
    ax1.legend(title=columna2, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    
    # Gráfico 2: Heatmap de la tabla de contingencia
    sns.heatmap(tabla, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title(f'Tabla de Contingencia: {columna1} vs {columna2}')
    
    plt.tight_layout()
    return fig

def calcular_porcentajes_tabla_contingencia(df: pd.DataFrame, columna1: str, columna2: str) -> dict:
    """
    Calcula diferentes tipos de porcentajes para la tabla de contingencia.
    """
    tabla = pd.crosstab(df[columna1], df[columna2])
    
    # Porcentajes por fila (porcentaje de cada columna dentro de cada fila)
    porcentajes_fila = tabla.div(tabla.sum(axis=1), axis=0) * 100
    
    # Porcentajes por columna (porcentaje de cada fila dentro de cada columna)
    porcentajes_columna = tabla.div(tabla.sum(axis=0), axis=1) * 100
    
    # Porcentajes del total
    porcentajes_total = (tabla / tabla.sum().sum()) * 100
    
    return {
        'porcentajes_fila': porcentajes_fila,
        'porcentajes_columna': porcentajes_columna,
        'porcentajes_total': porcentajes_total
    }

def interpretar_chi_cuadrado(resultados: dict) -> str:
    """
    Proporciona una interpretación de los resultados del test de chi-cuadrado.
    """
    p_value = resultados['p_value']
    cramer_v = resultados['cramer_v']
    
    # Interpretación del p-valor
    if p_value < 0.001:
        significancia = "muy altamente significativa (p < 0.001)"
    elif p_value < 0.01:
        significancia = "altamente significativa (p < 0.01)"
    elif p_value < 0.05:
        significancia = "significativa (p < 0.05)"
    else:
        significancia = "no significativa (p ≥ 0.05)"
    
    # Interpretación del tamaño del efecto (Cramer's V)
    if cramer_v < 0.1:
        tamano_efecto = "efecto muy pequeño"
    elif cramer_v < 0.3:
        tamano_efecto = "efecto pequeño"
    elif cramer_v < 0.5:
        tamano_efecto = "efecto moderado"
    else:
        tamano_efecto = "efecto grande"
    
    return f"La relación entre las variables es {significancia} con un {tamano_efecto} (Cramer's V = {cramer_v:.3f})."

def obtener_rango_variable(df: pd.DataFrame, columna: str) -> tuple:
    """
    Obtiene el rango mínimo y máximo de una variable numérica.
    """
    if columna in df.select_dtypes(include=["number"]).columns:
        valores = df[columna].dropna()
        return float(valores.min()), float(valores.max())
    return None, None

def obtener_categorias_variable(df: pd.DataFrame, columna: str) -> list:
    """
    Obtiene las categorías únicas de una variable categórica.
    """
    if columna in df.select_dtypes(include=["object", "category"]).columns:
        return sorted(df[columna].dropna().unique().tolist())
    return []

def aplicar_filtros(df: pd.DataFrame, filtros: dict) -> pd.DataFrame:
    """
    Aplica múltiples filtros a un DataFrame.
    
    Args:
        df: DataFrame original
        filtros: Diccionario con filtros a aplicar
                {
                    'columna_numerica': {'min': valor, 'max': valor},
                    'columna_categorica': ['categoria1', 'categoria2']
                }
    
    Returns:
        DataFrame filtrado
    """
    df_filtrado = df.copy()
    
    for columna, filtro in filtros.items():
        if columna in df_filtrado.columns:
            if isinstance(filtro, dict) and 'min' in filtro and 'max' in filtro:
                # Filtro numérico (rango)
                df_filtrado = df_filtrado[
                    (df_filtrado[columna] >= filtro['min']) & 
                    (df_filtrado[columna] <= filtro['max'])
                ]
            elif isinstance(filtro, list):
                # Filtro categórico (categorías seleccionadas)
                df_filtrado = df_filtrado[df_filtrado[columna].isin(filtro)]
    
    return df_filtrado

def crear_filtros_dinamicos(df: pd.DataFrame, columnas_numericas: list = None, columnas_categoricas: list = None) -> dict:
    """
    Crea un diccionario con información para generar filtros dinámicos.
    
    Returns:
        Diccionario con información de filtros para cada columna
    """
    filtros_info = {}
    
    # Filtros para variables numéricas
    if columnas_numericas is None:
        columnas_numericas = obtener_columnas_numericas(df)
    
    for col in columnas_numericas:
        min_val, max_val = obtener_rango_variable(df, col)
        if min_val is not None and max_val is not None:
            filtros_info[col] = {
                'tipo': 'numerico',
                'min': min_val,
                'max': max_val,
                'actual_min': min_val,
                'actual_max': max_val
            }
    
    # Filtros para variables categóricas
    if columnas_categoricas is None:
        columnas_categoricas = obtener_columnas_categoricas(df)
    
    for col in columnas_categoricas:
        categorias = obtener_categorias_variable(df, col)
        if categorias:
            filtros_info[col] = {
                'tipo': 'categorico',
                'categorias': categorias,
                'categorias_seleccionadas': categorias
            }
    
    return filtros_info

def obtener_estadisticas_filtradas(df: pd.DataFrame, filtros: dict) -> dict:
    """
    Obtiene estadísticas básicas del DataFrame filtrado.
    """
    df_filtrado = aplicar_filtros(df, filtros)
    
    return {
        'n_original': len(df),
        'n_filtrado': len(df_filtrado),
        'porcentaje_muestra': (len(df_filtrado) / len(df)) * 100 if len(df) > 0 else 0,
        'columnas_disponibles': {
            'numericas': obtener_columnas_numericas(df_filtrado),
            'categoricas': obtener_columnas_categoricas(df_filtrado)
        }
    }

# ============================================================================
# FUNCIONES DE EXPORTACIÓN
# ============================================================================

def generar_estadisticas_descriptivas_completas(df: pd.DataFrame, columnas_numericas: list = None) -> pd.DataFrame:
    """
    Genera un DataFrame con estadísticas descriptivas completas para exportar.
    """
    if columnas_numericas is None:
        columnas_numericas = obtener_columnas_numericas(df)
    
    estadisticas = []
    for col in columnas_numericas:
        valores = df[col].dropna()
        if len(valores) > 0:
            stats_dict = {
                'Variable': col,
                'N': len(valores),
                'Media': valores.mean(),
                'Mediana': valores.median(),
                'Desv. Estándar': valores.std(),
                'Mínimo': valores.min(),
                'Máximo': valores.max(),
                'Q1 (25%)': valores.quantile(0.25),
                'Q3 (75%)': valores.quantile(0.75),
                'Rango': valores.max() - valores.min(),
                'Coef. Variación': (valores.std() / valores.mean()) * 100 if valores.mean() != 0 else 0
            }
            estadisticas.append(stats_dict)
    
    return pd.DataFrame(estadisticas)

def generar_resumen_correlaciones(df: pd.DataFrame, variables: list, tipo_correlacion: str = 'pearson') -> pd.DataFrame:
    """
    Genera un DataFrame con resumen de correlaciones para exportar.
    """
    if tipo_correlacion.lower() == 'pearson':
        matriz_corr = calcular_correlacion_pearson(df, variables)
    else:
        matriz_corr = calcular_correlacion_spearman(df, variables)
    
    # Convertir matriz a formato largo para exportación
    correlaciones = []
    for i in range(len(matriz_corr.columns)):
        for j in range(i+1, len(matriz_corr.columns)):
            var1 = matriz_corr.columns[i]
            var2 = matriz_corr.columns[j]
            corr_valor = matriz_corr.iloc[i, j]
            
            # Clasificar la fuerza de la correlación
            if abs(corr_valor) >= 0.7:
                fuerza = "Muy Fuerte"
            elif abs(corr_valor) >= 0.5:
                fuerza = "Fuerte"
            elif abs(corr_valor) >= 0.3:
                fuerza = "Moderada"
            elif abs(corr_valor) >= 0.1:
                fuerza = "Débil"
            else:
                fuerza = "Muy Débil"
            
            correlaciones.append({
                'Variable 1': var1,
                'Variable 2': var2,
                'Correlación': corr_valor,
                'Fuerza': fuerza,
                'Tipo': tipo_correlacion.capitalize()
            })
    
    return pd.DataFrame(correlaciones).sort_values('Correlación', key=abs, ascending=False)

def generar_resumen_tablas_contingencia(df: pd.DataFrame, variable1: str, variable2: str) -> dict:
    """
    Genera un resumen completo de tablas de contingencia para exportar.
    """
    # Tabla de contingencia
    tabla_contingencia = crear_tabla_contingencia(df, variable1, variable2)
    
    # Porcentajes
    porcentajes = calcular_porcentajes_tabla_contingencia(df, variable1, variable2)
    
    # Test de chi-cuadrado
    resultados_chi = calcular_chi_cuadrado(df, variable1, variable2)
    
    # Interpretación
    interpretacion = interpretar_chi_cuadrado(resultados_chi)
    
    return {
        'tabla_contingencia': tabla_contingencia,
        'porcentajes_fila': porcentajes['porcentajes_fila'],
        'porcentajes_columna': porcentajes['porcentajes_columna'],
        'porcentajes_total': porcentajes['porcentajes_total'],
        'resultados_chi': resultados_chi,
        'interpretacion': interpretacion
    }

def convertir_matplotlib_a_base64(fig):
    """
    Convierte una figura de matplotlib a base64 para incluir en HTML.
    """
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode()
    return img_str

def generar_html_reporte(df: pd.DataFrame, filtros: dict, estadisticas_descriptivas: pd.DataFrame = None, 
                        correlaciones: pd.DataFrame = None, tablas_contingencia: dict = None) -> str:
    """
    Genera un reporte HTML completo con todos los análisis.
    """
    fecha_actual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    stats_filtradas = obtener_estadisticas_filtradas(df, filtros)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Reporte de Análisis Estadístico</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            h3 {{ color: #7f8c8d; }}
            .header {{ background-color: #ecf0f1; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
            .metric strong {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .filtros {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .interpretacion {{ background-color: #d1ecf1; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Reporte de Análisis Estadístico</h1>
            <p><strong>Fecha de generación:</strong> {fecha_actual}</p>
            <p><strong>Archivo analizado:</strong> {len(df)} observaciones, {len(df.columns)} variables</p>
        </div>
        
        <div class="section">
            <h2>📋 Resumen de Datos</h2>
            <div class="metric">
                <strong>Total Original:</strong> {stats_filtradas['n_original']}
            </div>
            <div class="metric">
                <strong>Datos Filtrados:</strong> {stats_filtradas['n_filtrado']}
            </div>
            <div class="metric">
                <strong>% de Muestra:</strong> {stats_filtradas['porcentaje_muestra']:.1f}%
            </div>
        </div>
    """
    
    # Sección de filtros aplicados
    if filtros:
        html += """
        <div class="section">
            <h2>🔧 Filtros Aplicados</h2>
            <div class="filtros">
        """
        for col, filtro in filtros.items():
            if isinstance(filtro, dict):
                html += f"<p><strong>{col}:</strong> {filtro['min']:.2f} - {filtro['max']:.2f}</p>"
            elif isinstance(filtro, list):
                html += f"<p><strong>{col}:</strong> {', '.join(filtro)}</p>"
        html += "</div></div>"
    
    # Sección de estadísticas descriptivas
    if estadisticas_descriptivas is not None:
        html += """
        <div class="section">
            <h2>📈 Estadísticas Descriptivas</h2>
        """
        html += estadisticas_descriptivas.to_html(index=False, float_format='%.3f')
        html += "</div>"
    
    # Sección de correlaciones
    if correlaciones is not None:
        html += """
        <div class="section">
            <h2>🔗 Análisis de Correlaciones</h2>
        """
        html += correlaciones.to_html(index=False, float_format='%.3f')
        html += "</div>"
    
    # Sección de tablas de contingencia
    if tablas_contingencia is not None:
        html += """
        <div class="section">
            <h2>📊 Tablas de Contingencia</h2>
        """
        
        # Tabla de contingencia
        html += "<h3>Tabla de Contingencia</h3>"
        html += tablas_contingencia['tabla_contingencia'].to_html(float_format='%.0f')
        
        # Porcentajes
        html += "<h3>Porcentajes por Fila</h3>"
        html += tablas_contingencia['porcentajes_fila'].to_html(float_format='%.2f')
        
        html += "<h3>Porcentajes por Columna</h3>"
        html += tablas_contingencia['porcentajes_columna'].to_html(float_format='%.2f')
        
        # Resultados del chi-cuadrado
        resultados = tablas_contingencia['resultados_chi']
        html += f"""
        <h3>🔬 Prueba de Chi-cuadrado</h3>
        <table>
            <tr><th>Estadístico</th><th>Valor</th></tr>
            <tr><td>χ²</td><td>{resultados['chi2_statistic']:.4f}</td></tr>
            <tr><td>p-valor</td><td>{resultados['p_value']:.4f}</td></tr>
            <tr><td>Grados de libertad</td><td>{resultados['degrees_of_freedom']}</td></tr>
            <tr><td>Cramer's V</td><td>{resultados['cramer_v']:.4f}</td></tr>
            <tr><td>Coeficiente de contingencia</td><td>{resultados['pearson_c']:.4f}</td></tr>
        </table>
        """
        
        # Interpretación
        html += f"""
        <h3>📝 Interpretación</h3>
        <div class="interpretacion">
            {tablas_contingencia['interpretacion']}
        </div>
        """
        
        html += "</div>"
    
    html += """
    </body>
    </html>
    """
    
    return html

def generar_csv_datos_filtrados(df: pd.DataFrame, filtros: dict) -> str:
    """
    Genera un CSV con los datos filtrados.
    """
    df_filtrado = aplicar_filtros(df, filtros)
    return df_filtrado.to_csv(index=False)

def generar_excel_completo(df: pd.DataFrame, filtros: dict, estadisticas_descriptivas: pd.DataFrame = None,
                          correlaciones: pd.DataFrame = None, tablas_contingencia: dict = None) -> bytes:
    """
    Genera un archivo Excel completo con múltiples hojas.
    """
    from io import BytesIO
    
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Hoja 1: Datos filtrados
        df_filtrado = aplicar_filtros(df, filtros)
        df_filtrado.to_excel(writer, sheet_name='Datos_Filtrados', index=False)
        
        # Hoja 2: Resumen de filtros
        resumen_filtros = []
        for col, filtro in filtros.items():
            if isinstance(filtro, dict):
                resumen_filtros.append({
                    'Variable': col,
                    'Tipo': 'Rango',
                    'Valor': f"{filtro['min']:.2f} - {filtro['max']:.2f}"
                })
            elif isinstance(filtro, list):
                resumen_filtros.append({
                    'Variable': col,
                    'Tipo': 'Categorías',
                    'Valor': ', '.join(filtro)
                })
        
        if resumen_filtros:
            pd.DataFrame(resumen_filtros).to_excel(writer, sheet_name='Resumen_Filtros', index=False)
        
        # Hoja 3: Estadísticas descriptivas
        if estadisticas_descriptivas is not None:
            estadisticas_descriptivas.to_excel(writer, sheet_name='Estadisticas_Descriptivas', index=False)
        
        # Hoja 4: Correlaciones
        if correlaciones is not None:
            correlaciones.to_excel(writer, sheet_name='Correlaciones', index=False)
        
        # Hoja 5: Tablas de contingencia
        if tablas_contingencia is not None:
            tablas_contingencia['tabla_contingencia'].to_excel(writer, sheet_name='Tabla_Contingencia')
            tablas_contingencia['porcentajes_fila'].to_excel(writer, sheet_name='Porcentajes_Fila')
            tablas_contingencia['porcentajes_columna'].to_excel(writer, sheet_name='Porcentajes_Columna')
            
            # Resultados del chi-cuadrado
            resultados = tablas_contingencia['resultados_chi']
            resultados_df = pd.DataFrame([
                {'Estadístico': 'χ²', 'Valor': resultados['chi2_statistic']},
                {'Estadístico': 'p-valor', 'Valor': resultados['p_value']},
                {'Estadístico': 'Grados de libertad', 'Valor': resultados['degrees_of_freedom']},
                {'Estadístico': "Cramer's V", 'Valor': resultados['cramer_v']},
                {'Estadístico': 'Coeficiente de contingencia', 'Valor': resultados['pearson_c']}
            ])
            resultados_df.to_excel(writer, sheet_name='Resultados_Chi_Cuadrado', index=False)
    
    output.seek(0)
    return output.getvalue()

def crear_grafico_plotly_correlacion(df_corr: pd.DataFrame, titulo: str = "Matriz de Correlación"):
    """
    Crea un gráfico de correlación interactivo con Plotly.
    """
    # Crear máscara para mostrar solo la mitad inferior del triángulo
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    df_corr_masked = df_corr.mask(mask)
    
    fig = go.Figure(data=go.Heatmap(
        z=df_corr_masked.values,
        x=df_corr_masked.columns,
        y=df_corr_masked.columns,
        colorscale='RdBu',
        zmid=0,
        text=df_corr_masked.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=titulo,
        width=800,
        height=600,
        xaxis_title="Variables",
        yaxis_title="Variables"
    )
    
    return fig

# ============================================================================
# FUNCIONES DE VISUALIZACIONES AVANZADAS
# ============================================================================

def generar_boxplot(df: pd.DataFrame, columna_numerica: str, columna_categorica: str = None):
    """
    Genera un boxplot para una variable numérica, opcionalmente agrupado por una variable categórica.
    """
    plt.figure(figsize=(12, 6))
    
    if columna_categorica:
        # Boxplot agrupado
        sns.boxplot(data=df, x=columna_categorica, y=columna_numerica)
        plt.title(f'Boxplot de {columna_numerica} por {columna_categorica}')
        plt.xticks(rotation=45)
    else:
        # Boxplot simple
        sns.boxplot(data=df, y=columna_numerica)
        plt.title(f'Boxplot de {columna_numerica}')
    
    plt.tight_layout()
    return plt.gcf()

def generar_scatter_plot(df: pd.DataFrame, columna_x: str, columna_y: str, columna_color: str = None):
    """
    Genera un scatter plot entre dos variables numéricas, opcionalmente coloreado por una variable categórica.
    """
    plt.figure(figsize=(10, 8))
    
    if columna_color:
        # Scatter plot con color
        sns.scatterplot(data=df, x=columna_x, y=columna_y, hue=columna_color, alpha=0.7)
        plt.title(f'Scatter Plot: {columna_x} vs {columna_y} (coloreado por {columna_color})')
    else:
        # Scatter plot simple
        sns.scatterplot(data=df, x=columna_x, y=columna_y, alpha=0.7)
        plt.title(f'Scatter Plot: {columna_x} vs {columna_y}')
    
    # Agregar línea de regresión
    sns.regplot(data=df, x=columna_x, y=columna_y, scatter=False, color='red', line_kws={'linestyle': '--'})
    
    plt.tight_layout()
    return plt.gcf()

def generar_diagrama_densidad(df: pd.DataFrame, columna: str, columna_grupo: str = None):
    """
    Genera un diagrama de densidad para una variable numérica, opcionalmente agrupado.
    """
    plt.figure(figsize=(12, 6))
    
    if columna_grupo:
        # Densidad agrupada
        for grupo in df[columna_grupo].unique():
            datos_grupo = df[df[columna_grupo] == grupo][columna].dropna()
            if len(datos_grupo) > 0:
                sns.kdeplot(data=datos_grupo, label=grupo, fill=True, alpha=0.3)
        plt.title(f'Distribución de Densidad de {columna} por {columna_grupo}')
        plt.legend()
    else:
        # Densidad simple
        sns.kdeplot(data=df[columna].dropna(), fill=True)
        plt.title(f'Distribución de Densidad de {columna}')
    
    plt.xlabel(columna)
    plt.ylabel('Densidad')
    plt.tight_layout()
    return plt.gcf()

def generar_grafico_barras(df: pd.DataFrame, columna_categorica: str, columna_numerica: str = None):
    """
    Genera un gráfico de barras para una variable categórica, opcionalmente con valores numéricos.
    """
    plt.figure(figsize=(12, 6))
    
    if columna_numerica:
        # Gráfico de barras con valores
        df_agrupado = df.groupby(columna_categorica)[columna_numerica].mean().sort_values(ascending=False)
        df_agrupado.plot(kind='bar')
        plt.title(f'Promedio de {columna_numerica} por {columna_categorica}')
        plt.ylabel(f'Promedio de {columna_numerica}')
    else:
        # Gráfico de barras de frecuencias
        df[columna_categorica].value_counts().plot(kind='bar')
        plt.title(f'Frecuencia de {columna_categorica}')
        plt.ylabel('Frecuencia')
    
    plt.xlabel(columna_categorica)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def generar_histograma_densidad(df: pd.DataFrame, columna: str, columna_grupo: str = None):
    """
    Genera un histograma con curva de densidad superpuesta.
    """
    plt.figure(figsize=(12, 6))
    
    if columna_grupo:
        # Histograma agrupado
        for grupo in df[columna_grupo].unique():
            datos_grupo = df[df[columna_grupo] == grupo][columna].dropna()
            if len(datos_grupo) > 0:
                plt.hist(datos_grupo, alpha=0.5, label=grupo, density=True, bins=20)
                sns.kdeplot(data=datos_grupo, color='black', linewidth=2)
        plt.legend()
        plt.title(f'Histograma y Densidad de {columna} por {columna_grupo}')
    else:
        # Histograma simple
        plt.hist(df[columna].dropna(), bins=20, density=True, alpha=0.7, color='skyblue')
        sns.kdeplot(data=df[columna].dropna(), color='red', linewidth=2)
        plt.title(f'Histograma y Densidad de {columna}')
    
    plt.xlabel(columna)
    plt.ylabel('Densidad')
    plt.tight_layout()
    return plt.gcf()

def generar_violin_plot(df: pd.DataFrame, columna_numerica: str, columna_categorica: str):
    """
    Genera un violin plot para mostrar la distribución de una variable numérica por grupos.
    """
    plt.figure(figsize=(12, 6))
    
    sns.violinplot(data=df, x=columna_categorica, y=columna_numerica)
    plt.title(f'Violin Plot: Distribución de {columna_numerica} por {columna_categorica}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt.gcf()

def generar_heatmap_correlacion_avanzado(df: pd.DataFrame, variables: list):
    """
    Genera un heatmap de correlación más avanzado con anotaciones y estadísticas.
    """
    # Calcular correlaciones
    matriz_corr = df[variables].corr()
    
    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap principal
    mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
    sns.heatmap(matriz_corr, mask=mask, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', ax=ax1, cbar_kws={"shrink": .8})
    ax1.set_title('Matriz de Correlación')
    
    # Gráfico de barras de correlaciones más fuertes
    correlaciones = []
    for i in range(len(matriz_corr.columns)):
        for j in range(i+1, len(matriz_corr.columns)):
            var1 = matriz_corr.columns[i]
            var2 = matriz_corr.columns[j]
            corr_valor = matriz_corr.iloc[i, j]
            correlaciones.append((f'{var1}-{var2}', corr_valor))
    
    # Ordenar por valor absoluto
    correlaciones.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Tomar las 10 correlaciones más fuertes
    top_corr = correlaciones[:10]
    variables_corr = [x[0] for x in top_corr]
    valores_corr = [x[1] for x in top_corr]
    
    # Gráfico de barras
    colors = ['red' if x < 0 else 'blue' for x in valores_corr]
    ax2.barh(range(len(variables_corr)), [abs(x) for x in valores_corr], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(variables_corr)))
    ax2.set_yticklabels(variables_corr)
    ax2.set_xlabel('Valor Absoluto de Correlación')
    ax2.set_title('Top 10 Correlaciones Más Fuertes')
    
    # Agregar valores en las barras
    for i, v in enumerate(valores_corr):
        ax2.text(abs(v) + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    return fig

def generar_panel_visualizaciones(df: pd.DataFrame, columna_principal: str, columna_grupo: str = None):
    """
    Genera un panel completo de visualizaciones para una variable.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histograma con densidad
    if columna_grupo:
        for grupo in df[columna_grupo].unique():
            datos_grupo = df[df[columna_grupo] == grupo][columna_principal].dropna()
            if len(datos_grupo) > 0:
                axes[0,0].hist(datos_grupo, alpha=0.5, label=grupo, density=True, bins=20)
        axes[0,0].legend()
    else:
        axes[0,0].hist(df[columna_principal].dropna(), bins=20, density=True, alpha=0.7)
    axes[0,0].set_title(f'Histograma de {columna_principal}')
    axes[0,0].set_xlabel(columna_principal)
    axes[0,0].set_ylabel('Densidad')
    
    # 2. Boxplot
    if columna_grupo:
        sns.boxplot(data=df, x=columna_grupo, y=columna_principal, ax=axes[0,1])
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
    else:
        sns.boxplot(data=df, y=columna_principal, ax=axes[0,1])
    axes[0,1].set_title(f'Boxplot de {columna_principal}')
    
    # 3. Diagrama de densidad
    if columna_grupo:
        for grupo in df[columna_grupo].unique():
            datos_grupo = df[df[columna_grupo] == grupo][columna_principal].dropna()
            if len(datos_grupo) > 0:
                sns.kdeplot(data=datos_grupo, label=grupo, ax=axes[1,0])
        axes[1,0].legend()
    else:
        sns.kdeplot(data=df[columna_principal].dropna(), ax=axes[1,0])
    axes[1,0].set_title(f'Distribución de Densidad de {columna_principal}')
    axes[1,0].set_xlabel(columna_principal)
    axes[1,0].set_ylabel('Densidad')
    
    # 4. Violin plot (si hay grupo)
    if columna_grupo:
        sns.violinplot(data=df, x=columna_grupo, y=columna_principal, ax=axes[1,1])
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
        axes[1,1].set_title(f'Violin Plot de {columna_principal} por {columna_grupo}')
    else:
        # Q-Q plot para normalidad
        from scipy import stats
        stats.probplot(df[columna_principal].dropna(), dist="norm", plot=axes[1,1])
        axes[1,1].set_title(f'Q-Q Plot de {columna_principal}')
    
    plt.tight_layout()
    return fig

def generar_scatter_matrix(df: pd.DataFrame, variables: list):
    """
    Genera una matriz de scatter plots para múltiples variables.
    """
    # Limitar a 6 variables para evitar gráficos muy grandes
    if len(variables) > 6:
        variables = variables[:6]
    
    fig = sns.pairplot(df[variables], diag_kind='kde')
    fig.fig.suptitle('Matriz de Scatter Plots', y=1.02)
    fig.fig.set_size_inches(12, 10)
    
    return fig.fig
