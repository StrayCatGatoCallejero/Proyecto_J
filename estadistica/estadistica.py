# estadistica/estadistica.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
