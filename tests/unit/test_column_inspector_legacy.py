"""
Tests para el módulo column_inspector
"""

import pandas as pd
import pytest
import sys
import os

# Agregar el directorio proyecto_j/src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "proyecto_j", "src"))

from column_inspector import detect_column_types, suggest_visualizations


class TestColumnInspector:
    """Tests para el inspector de columnas"""

    def test_detect_column_types(self):
        """Test para detectar tipos de columnas"""
        # Crear DataFrame de prueba
        df = pd.DataFrame(
            {
                "numeric": [1, 2, 3, 4, 5],
                "categorical": ["A", "B", "A", "C", "B"],
                "text": [
                    "texto largo",
                    "otro texto",
                    "más texto",
                    "texto",
                    "texto largo",
                ],
                "boolean": [True, False, True, True, False],
                "date": pd.date_range("2023-01-01", periods=5),
            }
        )

        # Detectar tipos
        types = detect_column_types(df)

        # Verificar que se detectaron correctamente
        assert types["numeric"] == "numeric"
        assert types["categorical"] == "categorical"
        assert types["text"] == "text"
        assert types["boolean"] == "boolean"
        assert types["date"] == "datetime"

    def test_suggest_visualizations(self):
        """Test para sugerir visualizaciones"""
        # Test para columna numérica
        suggestions = suggest_visualizations("numeric", "numeric")
        assert "histogram" in suggestions
        assert "boxplot" in suggestions

        # Test para columna categórica
        suggestions = suggest_visualizations("categorical", "categorical")
        assert "bar" in suggestions
        assert "pie" in suggestions

        # Test para columna de texto
        suggestions = suggest_visualizations("text", "text")
        assert "wordcloud" in suggestions
