import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

class VariableClassifier:
    def __init__(self, cat_threshold: float = 0.1, cat_unique_limit: int = 20):
        """
        :param cat_threshold: fracción de valores únicos respecto al total para considerar categórica
        :param cat_unique_limit: tope absoluto de valores únicos para considerar categórica
        """
        self.cat_threshold = cat_threshold
        self.cat_unique_limit = cat_unique_limit

    def classify_dataset(self, df: pd.DataFrame) -> dict:
        """Clasifica todas las variables de un DataFrame y genera metadatos."""
        classification = {}
        n_rows = len(df)
        for col in df.columns:
            series = df[col]
            col_type = self._classify_column(series, n_rows)
            metadata = {
                'type': col_type,
                'n_missing': int(series.isna().sum()),
                'pct_missing': float(series.isna().mean()),
                'n_unique': int(series.nunique(dropna=True)),
                'pct_unique': float(series.nunique(dropna=True) / n_rows) if n_rows > 0 else 0.0,
                'sample_values': [str(val) for val in series.dropna().unique()[:5]]
            }
            classification[col] = metadata
        return classification

    def _classify_column(self, series: pd.Series, n_rows: int) -> str:
        # Detección temporal
        if self._detect_temporal(series):
            return 'temporal'
        # Numérica
        if is_numeric_dtype(series):
            return 'numerica'
        # Categórica
        if self._detect_categorica(series, n_rows):
            return 'categorica'
        # Texto libre (fallback)
        return 'texto_libre'

    def _detect_temporal(self, series: pd.Series) -> bool:
        """Devuelve True si la columna es principalmente fechas."""
        # Primero, si ya es dtype datetime
        if is_datetime64_any_dtype(series):
            return True
        # Intentar parseo: porcentaje de valores convertibles
        parsed = pd.to_datetime(series, errors='coerce')
        pct_parsed = parsed.notna().mean()
        return pct_parsed >= 0.8  # 80% de valores parseables como fecha

    def _detect_categorica(self, series: pd.Series, n_rows: int) -> bool:
        """Devuelve True si la columna parece categórica."""
        n_unique = series.nunique(dropna=True)
        # No categórica si nadie o todos los valores son únicos
        if n_unique == 0 or n_unique >= n_rows:
            return False
        # Umbral relativo y límite absoluto
        if (n_unique / n_rows) <= self.cat_threshold or n_unique <= self.cat_unique_limit:
            return True
        return False

    def _detect_numerica(self, series: pd.Series) -> bool:
        """No se usa explícitamente porque numeric_dtype lo captura."""
        return is_numeric_dtype(series)

    def _detect_texto_libre(self, series: pd.Series) -> bool:
        """Todo lo que no sea temporal, numérico o categórico se considera texto libre."""
        return True  # Fallback

# Ejemplo de uso:
if __name__ == '__main__':
    df = pd.DataFrame({
        'fecha': ['2012-06-01', '2012-06-15', None, 'not a date'],
        'edad': [23, 35, 29, np.nan],
        'categoria': ['A', 'B', 'A', 'C'],
        'comentario': ['hola', 'mundo', 'prueba', 'texto libre']
    })
    clf = VariableClassifier()
    result = clf.classify_dataset(df)
    import pprint; pprint.pprint(result) 