import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pyreadstat
import openpyxl
import xlrd
import missingno as msno
import subprocess
import os
import sys
from typing import Union, List, Dict, Any
import chardet
from pathlib import Path

# ============================================================================
# 1. Carga de Datos
# ============================================================================


@st.cache_data
def cargar_datos(path: str) -> pd.DataFrame:
    path = Path(path)  # Asegura que path sea un objeto Path
    """
    Carga datos desde archivos .sav, .dta, .csv, .xls, o .xlsx.
    """
    if not path.exists():
        raise FileNotFoundError(f"El archivo no se encuentra en la ruta: {path}")

    filename = path.name.lower()

    try:
        if filename.endswith(".sav"):
            df, _ = pyreadstat.read_sav(path)
        elif filename.endswith(".dta"):
            df, _ = pyreadstat.read_dta(path)
        elif filename.endswith(".csv"):
            # Detectar encoding con chardet
            with open(path, 'rb') as f:
                rawdata = f.read(10000)
                result = chardet.detect(rawdata)
                encoding = result['encoding'] if result['encoding'] else 'utf-8'
            try:
                df = pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                # Fallback a latin1 si falla el encoding detectado
                df = pd.read_csv(path, encoding='latin1')
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            raise ValueError(
                "Formato de archivo no soportado. Use .sav, .dta, .csv, .xls, o .xlsx."
            )
        return df
    except Exception as e:
        raise IOError(f"Error al leer el archivo {path}: {e}")


# ============================================================================
# 2. Limpieza de Datos
# ============================================================================


def visualizar_valores_faltantes(df: pd.DataFrame):
    """
    Genera una visualización de valores faltantes usando missingno.
    Retorna el objeto de la figura de matplotlib.
    """
    fig = msno.matrix(df).get_figure()
    return fig


@st.cache_data
def limpiar_datos(
    df: pd.DataFrame, estrategia_imputacion: str = "knn", llm_imputer=None
) -> pd.DataFrame:
    """
    Limpia el DataFrame, manejando valores faltantes.

    Estrategias de imputación:
    - 'media': Imputa con la media (solo para numéricos).
    - 'mediana': Imputa con la mediana (solo para numéricos).
    - 'moda': Imputa con la moda (para categóricos).
    - 'knn': Usa KNNImputer de scikit-learn.
    - 'llm': Usa un imputador basado en LLM (experimental).
    """
    df_limpio = df.copy()

    if estrategia_imputacion == "llm":
        if llm_imputer:
            # Lógica para usar el imputador LLM.
            # Esto es un placeholder. La implementación real sería más compleja.
            # df_limpio = llm_imputer.impute(df_limpio)
            print("Imputación con LLM (a implementar)")
            pass
        else:
            raise ValueError("Se requiere un `llm_imputer` para la estrategia 'llm'.")

    # Otras estrategias de imputación más simples
    from sklearn.impute import SimpleImputer, KNNImputer

    num_cols = df_limpio.select_dtypes(include=np.number).columns
    cat_cols = df_limpio.select_dtypes(include=["object", "category"]).columns

    if estrategia_imputacion in ["media", "mediana"]:
        imputer = SimpleImputer(strategy=estrategia_imputacion)
        df_limpio[num_cols] = imputer.fit_transform(df_limpio[num_cols])
    elif estrategia_imputacion == "moda":
        imputer = SimpleImputer(strategy="most_frequent")
        df_limpio[cat_cols] = imputer.fit_transform(df_limpio[cat_cols])
    elif estrategia_imputacion == "knn":
        # KNNImputer solo funciona con datos numéricos.
        # Primero, convertimos categóricos a numéricos para el imputer.
        df_encoded = df_limpio.copy()
        for col in cat_cols:
            df_encoded[col] = (
                df_encoded[col].astype("category").cat.codes.replace(-1, np.nan)
            )

        imputer = KNNImputer()
        df_imputed_encoded = pd.DataFrame(
            imputer.fit_transform(df_encoded), columns=df_encoded.columns
        )

        # Revertir la codificación de las columnas categóricas
        for col in cat_cols:
            original_cats = df_limpio[col].astype("category").cat.categories
            df_imputed_encoded[col] = pd.Categorical.from_codes(
                df_imputed_encoded[col].round().astype(int), categories=original_cats
            )

        df_limpio = df_imputed_encoded

    return df_limpio


# ============================================================================
# 3. Transformación de Datos
# ============================================================================


@st.cache_data
def transformar_datos(df: pd.DataFrame, normalizar: bool = False) -> pd.DataFrame:
    """
    Transforma datos categóricos a numéricos y opcionalmente normaliza.
    """
    df_transformado = df.copy()

    # One-hot encode para categóricos
    df_transformado = pd.get_dummies(df_transformado, dummy_na=True)

    if normalizar:
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        df_transformado[df_transformado.columns] = scaler.fit_transform(
            df_transformado[df_transformado.columns]
        )

    return df_transformado


# ============================================================================
# 4. Modelo de Proyección
# ============================================================================


@st.cache_data
def ajustar_modelo(df: pd.DataFrame, features: List[str], target: str) -> Any:
    """
    Ajusta un modelo de regresión lineal.
    Retorna el modelo entrenado.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X = df[features]
    y = df[target]

    # Dividimos los datos para una evaluación simple
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predecir(model: Any, X_nuevos: pd.DataFrame) -> np.ndarray:
    """
    Realiza predicciones usando el modelo entrenado.
    """
    return model.predict(X_nuevos)


# ============================================================================
# 5. Visualización
# ============================================================================


def visualizar_regresion(
    df: pd.DataFrame, model: Any, features: List[str], target: str
) -> go.Figure:
    """
    Crea un gráfico interactivo con Plotly mostrando los datos y la línea de regresión.
    Funciona mejor para 1 o 2 features.
    """
    if len(features) == 1:
        # Regresión simple
        fig = px.scatter(
            df,
            x=features[0],
            y=target,
            title=f"Regresión Lineal: {target} vs {features[0]}",
            trendline="ols",
            trendline_color_override="red",
        )
    elif len(features) == 2:
        # Regresión múltiple (visualización 3D)

        # Crear una malla para el plano de regresión
        x_range = np.linspace(df[features[0]].min(), df[features[0]].max(), 10)
        y_range = np.linspace(df[features[1]].min(), df[features[1]].max(), 10)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        z_mesh = model.predict(
            pd.DataFrame({features[0]: x_mesh.ravel(), features[1]: y_mesh.ravel()})
        )
        z_mesh = z_mesh.reshape(x_mesh.shape)

        # Gráfico de dispersión 3D de los datos
        fig = px.scatter_3d(
            df,
            x=features[0],
            y=features[1],
            z=target,
            title=f"Regresión Múltiple: {target} vs {features[0]} & {features[1]}",
        )

        # Añadir el plano de regresión
        fig.add_trace(
            go.Surface(
                x=x_range, y=y_range, z=z_mesh, name="Plano de Regresión", opacity=0.7
            )
        )
    else:
        # No se puede visualizar fácilmente con más de 2 features
        print("La visualización solo está implementada para 1 o 2 features.")
        return go.Figure()

    return fig


def visualizar_proyeccion(x_data, y_data, x_proyeccion, y_proyeccion) -> go.Figure:
    """
    Crea un gráfico interactivo con Plotly Express mostrando datos y proyecciones.
    """
    df_real = pd.DataFrame({"x": x_data, "y": y_data, "tipo": "Datos Reales"})
    df_proy = pd.DataFrame({"x": x_proyeccion, "y": y_proyeccion, "tipo": "Proyección"})

    df_plot = pd.concat([df_real, df_proy])

    fig = px.line(df_plot, x="x", y="y", color="tipo", title="Proyección Demográfica")
    return fig


# ============================================================================
# 6. Generación de Reportes
# ============================================================================


def generar_reporte_pdf(df: pd.DataFrame, output_path: str = "reporte.pdf"):
    """
    Genera un reporte en PDF a partir de un DataFrame CSV.
    Llama al script externo `csv-to-pdf-report-generator`.
    """
    temp_csv_path = "temp_report_data.csv"
    df.to_csv(temp_csv_path, index=False)

    report_generator_path = os.path.join(
        "tools", "csv-to-pdf-report-generator", "app.py"
    )

    if not os.path.exists(report_generator_path):
        raise FileNotFoundError(
            f"El script generador de reportes no se encuentra en: {report_generator_path}"
        )

    try:
        subprocess.run(
            [sys.executable, report_generator_path, temp_csv_path, output_path],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Reporte PDF generado exitosamente en {output_path}")
    except subprocess.CalledProcessError as e:
        print("Error al generar el reporte PDF:")
        print(e.stderr)
    finally:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)


# ============================================================================
# Pipeline Principal (Ejemplo de uso)
# ============================================================================


def pipeline_completo(ruta_archivo: str, features: List[str], target: str):
    """
    Ejemplo de un pipeline que ejecuta todos los pasos.
    """
    # 1. Cargar datos
    df = cargar_datos(ruta_archivo)
    print("Datos cargados exitosamente.")

    # 2. Limpiar datos (usando KNN como ejemplo)
    # Asegurémonos que las columnas para el modelo existen antes de limpiar
    cols_modelo = features + [target]
    if not all(col in df.columns for col in cols_modelo):
        raise ValueError(
            f"El DataFrame no contiene todas las columnas necesarias para el modelo: {cols_modelo}"
        )

    df_limpio = limpiar_datos(df, estrategia_imputacion="knn")
    print("Datos limpiados.")

    # 3. Transformar datos (puede no ser necesario para regresión lineal si no hay categóricos)
    # Por ahora, nos aseguramos de que los datos del modelo son numéricos.
    df_modelo = df_limpio[cols_modelo].copy()

    # 4. Ajustar y visualizar el modelo de regresión
    print(f"Ajustando modelo para predecir '{target}' a partir de {features}...")
    modelo = ajustar_modelo(df_modelo, features, target)
    print("Modelo ajustado.")

    # Imprimir coeficientes para insight
    print(f"Coeficientes del modelo: {modelo.coef_}")
    print(f"Intercepto del modelo: {modelo.intercept_:.4f}")

    # 5. Visualizar el resultado de la regresión
    fig_regresion = visualizar_regresion(df_modelo, modelo, features, target)
    if fig_regresion:
        # fig_regresion.show() # .show() abre una ventana, que puede no ser ideal en scripts
        # Guardamos la figura como un archivo HTML para facilitar la inspección.
        fig_regresion.write_html("visualizacion_regresion.html")
        print(
            "Visualización de la regresión guardada en 'visualizacion_regresion.html'"
        )

    # 6. Generar reporte con los datos limpios
    generar_reporte_pdf(df_limpio, "reporte_demografico.pdf")


if __name__ == "__main__":
    # Este bloque es para pruebas.
    ruta_datos_ejemplo = os.path.join("data", "datos_ejemplo.sav")

    # Definimos las variables para nuestro modelo de regresión
    features_a_usar = ["edad", "anios_educacion"]
    target_a_predecir = "ingresos_anuales"

    try:
        print("--- Iniciando pipeline completo ---")
        pipeline_completo(ruta_datos_ejemplo, features_a_usar, target_a_predecir)
        print("\n--- Pipeline finalizado exitosamente ---")

    except FileNotFoundError:
        print(f"Archivo de ejemplo no encontrado en la ruta: {ruta_datos_ejemplo}")
    except Exception as e:
        print(f"Ocurrió un error en el pipeline: {e}")
