import pandas as pd
import numpy as np
import pyreadstat
from pandas.api import types as pdt


def read_data(filepath):
    """Lee datos desde CSV, Excel, JSON, SAV, DTA."""
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
        df = pd.read_excel(filepath)
    elif filepath.endswith(".json"):
        df = pd.read_json(filepath)
    elif filepath.endswith(".sav"):
        df, _ = pyreadstat.read_sav(filepath)
    elif filepath.endswith(".dta"):
        df, _ = pyreadstat.read_dta(filepath)
    else:
        raise ValueError("Formato no soportado.")
    return df


def detect_type(series):
    """Detecta tipo de dato de una columna: numeric, categorical, text, date, boolean, unknown."""
    s = series.dropna()
    if s.empty:
        return "unknown"
    if pdt.is_bool_dtype(s):
        return "boolean"
    if pdt.is_categorical_dtype(s):
        return "categorical"
    if pdt.is_numeric_dtype(s):
        return "numeric"
    if pdt.is_datetime64_any_dtype(s):
        return "date"
    # Heurística para texto largo
    if s.apply(lambda x: isinstance(x, str) and len(x.split()) > 3).mean() > 0.2:
        return "text"
    # Heurística para categorizar
    if s.nunique() < max(10, len(s) * 0.05):
        return "categorical"
    return "unknown"


def suggest_analysis(dtype):
    suggestions = {
        "numeric": [
            {
                "desc": "Histograma",
                "snippet": "px.histogram(df, x=col)",
                "type": "plot",
            },
            {"desc": "Boxplot", "snippet": "px.box(df, y=col)", "type": "plot"},
            {"desc": "Estadísticas", "snippet": "df[col].describe()", "type": "text"},
        ],
        "categorical": [
            {
                "desc": "Conteo de categorías",
                "snippet": "df[col].value_counts()",
                "type": "text",
            },
            {
                "desc": "Gráfico de barras",
                "snippet": "px.bar(df[col].value_counts())",
                "type": "plot",
            },
        ],
        "text": [
            {
                "desc": "Wordcloud",
                "snippet": "# Usa wordcloud.WordCloud().generate(' '.join(df[col].dropna()))",
                "type": "plot",
            },
            {
                "desc": "Frecuencias",
                "snippet": "df[col].value_counts()",
                "type": "text",
            },
        ],
        "date": [
            {
                "desc": "Serie temporal",
                "snippet": "px.line(df, x=col, y=df.columns[1])",
                "type": "plot",
            }
        ],
        "boolean": [
            {
                "desc": "Conteo True/False",
                "snippet": "df[col].value_counts()",
                "type": "text",
            },
            {
                "desc": "Gráfico de barras",
                "snippet": "px.bar(df[col].value_counts())",
                "type": "plot",
            },
        ],
        "unknown": [
            {
                "desc": "Revisar manualmente",
                "snippet": "# Revisa los datos",
                "type": "text",
            }
        ],
    }
    return suggestions.get(dtype, [])


def get_column_report(df):
    """Devuelve un DataFrame con el reporte de columnas y sugerencias."""
    rows = []
    for col in df.columns:
        dtype = detect_type(df[col])
        suggestions = suggest_analysis(dtype)
        for sug in suggestions:
            rows.append(
                {
                    "column": col,
                    "detected_type": dtype,
                    "suggested_analysis": sug["desc"],
                    "code_snippet": sug["snippet"],
                    "output_type": sug["type"],
                }
            )
    return pd.DataFrame(rows)


def detect_column_types_simple(df):
    """Devuelve un dict con el tipo detectado para cada columna."""
    return {col: detect_type(df[col]) for col in df.columns}

# Alias para compatibilidad con tests

detect_column_types = get_column_report

# Alias para compatibilidad con tests
suggest_visualizations = suggest_analysis


# Para uso directo en scripts/tests
def main():
    import sys

    df = read_data(sys.argv[1])
    print(get_column_report(df))


if __name__ == "__main__":
    main()
