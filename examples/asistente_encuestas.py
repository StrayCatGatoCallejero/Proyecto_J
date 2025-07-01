import pandas as pd
import pyreadstat
import missingno as msno
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from dataprep.eda import create_report
import statsmodels.api as sm
from fpdf import FPDF
import os


# =============================
# 1. Carga universal de datos
# =============================
def cargar_datos(path):
    """Carga archivos en formato CSV, Excel, SAV, DTA."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif ext == ".sav":
        df, _ = pyreadstat.read_sav(path)
        return df
    elif ext == ".dta":
        df, _ = pyreadstat.read_dta(path)
        return df
    else:
        raise ValueError(f"Formato de archivo no soportado: {ext}")


# =============================
# 2. Limpieza e imputación
# =============================
def imputar_datos(df, strategy="mean"):
    """Imputa valores faltantes en columnas numéricas usando la estrategia indicada."""
    df_num = df.select_dtypes(include="number")
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_num), columns=df_num.columns)
    df.update(df_imputed)
    return df


# =============================
# 3. Visualización de valores faltantes
# =============================
def visualizar_faltantes(df, show=True, save_path=None):
    """Muestra y/o guarda la matriz de valores faltantes usando missingno."""
    ax = msno.matrix(df)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


# =============================
# 4. EDA interactivo
# =============================
def reporte_eda(df, output_html="reporte_eda.html"):
    """Genera y guarda un reporte EDA interactivo con dataprep."""
    report = create_report(df)
    report.save(output_html)
    print(f"Reporte EDA guardado en {output_html}")


# =============================
# 5. Análisis con diseño de encuesta (ejemplo OLS)
# =============================
def analisis_ols(df, y, X, weights=None):
    """Ajusta un modelo OLS simple, con soporte para ponderaciones si se proveen."""
    X_ = sm.add_constant(df[X])
    if weights is not None:
        model = sm.WLS(df[y], X_, weights=weights).fit()
    else:
        model = sm.OLS(df[y], X_).fit()
    print(model.summary())
    return model


# =============================
# 6. Exportar resultados a PDF (tabla simple)
# =============================
def exportar_pdf(df, output_pdf="reporte.pdf", max_rows=20):
    """Exporta las primeras filas del DataFrame a un PDF simple."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    # Encabezados
    for col in df.columns:
        pdf.cell(40, 10, str(col), 1)
    pdf.ln()
    # Primeras filas
    for i, row in df.head(max_rows).iterrows():
        for val in row:
            pdf.cell(40, 10, str(val), 1)
        pdf.ln()
    pdf.output(output_pdf)
    print(f"PDF exportado a {output_pdf}")


# =============================
# 7. Flujo completo de ejemplo
# =============================
if __name__ == "__main__":
    # Cambia el path por tu archivo real
    archivo = "data/datos_ejemplo_chile.csv"
    df = cargar_datos(archivo)
    print("Datos cargados:", df.shape)

    # Visualizar valores faltantes
    visualizar_faltantes(df, show=True, save_path="faltantes.png")

    # Imputar valores faltantes (solo columnas numéricas)
    df = imputar_datos(df, strategy="mean")
    print("Imputación completada.")

    # EDA interactivo
    reporte_eda(df, output_html="reporte_eda.html")

    # Análisis OLS (modifica según tus variables)
    # analisis_ols(df, y="ingresos", X=["edad", "nivel_educativo"])

    # Exportar a PDF
    exportar_pdf(df, output_pdf="reporte_tabla.pdf")
