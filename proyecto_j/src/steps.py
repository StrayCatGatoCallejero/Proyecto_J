import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from fpdf import FPDF

def cargar_datos(path):
    """Carga datos desde CSV, Excel, SAV o DTA."""
    ext = os.path.splitext(path)[-1].lower()
    if ext in ['.csv']:
        return pd.read_csv(path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(path)
    elif ext in ['.sav', '.zsav']:
        import pyreadstat
        df, _ = pyreadstat.read_sav(path)
        return df
    elif ext in ['.dta']:
        return pd.read_stata(path)
    else:
        raise ValueError(f"Formato de archivo no soportado: {ext}")

def limpiar_datos(df):
    """Visualiza y realiza imputación simple de valores faltantes."""
    msno.matrix(df)
    plt.savefig('missing_data_matrix.png')
    df = df.fillna(df.median(numeric_only=True))
    return df

def transformar_datos(df):
    """Codifica variables categóricas y normaliza numéricas."""
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes
    return df

def modelar(df):
    """Ejemplo: Regresión lineal simple sobre las dos primeras columnas numéricas."""
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) < 2:
        raise ValueError("Se requieren al menos dos columnas numéricas para el modelo.")
    X = df[[num_cols[0]]]
    y = df[num_cols[1]]
    model = LinearRegression().fit(X, y)
    results = {'coef': model.coef_[0], 'intercept': model.intercept_}
    return model, results

def visualizar(df, results):
    """Genera un scatter plot y la recta de regresión."""
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) < 2:
        return
    X = df[num_cols[0]]
    y = df[num_cols[1]]
    plt.figure()
    plt.scatter(X, y, label='Datos')
    plt.plot(X, results['coef']*X + results['intercept'], color='red', label='Regresión')
    plt.xlabel(num_cols[0])
    plt.ylabel(num_cols[1])
    plt.legend()
    plt.savefig('regresion.png')
    plt.close()

def generar_reporte(df, results, output_path):
    """Genera un PDF simple con resultados y gráficos."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Reporte de Análisis', ln=1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Coeficiente: {results.get('coef', 'N/A')}", ln=1)
    pdf.cell(0, 10, f"Intercepto: {results.get('intercept', 'N/A')}", ln=1)
    if os.path.exists('missing_data_matrix.png'):
        pdf.image('missing_data_matrix.png', w=100)
    if os.path.exists('regresion.png'):
        pdf.image('regresion.png', w=100)
    pdf.output(output_path) 