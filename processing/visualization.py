"""
Módulo de Visualización Avanzada
===============================

Responsabilidades:
- Sugerir tipos de gráficos según semántica y tipos de datos
- Generar visualizaciones avanzadas (boxplot, scatter, heatmap, wordcloud, etc.)
- Integrar metadata y anotaciones para tooltips/UI
- Logging de todas las operaciones de visualización
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Importar logging
from .logging import log_action

def suggest_charts(
    df: pd.DataFrame, 
    semantic_classification: Dict[str, Any] = None
) -> Dict[str, List[str]]:
    """
    Sugiere tipos de gráficos para cada columna según su semántica y tipo.
    
    Args:
        df: DataFrame a analizar
        semantic_classification: Clasificación semántica previa (opcional)
        
    Returns:
        Diccionario con sugerencias de gráficos por columna
    """
    # Stub: delega a types.suggest_visualizations si está disponible
    from .types import suggest_visualizations
    return suggest_visualizations(df, semantic_classification)

def generate_plot(
    df: pd.DataFrame,
    chart_type: str,
    columns: List[str],
    metadata: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Genera un gráfico avanzado según el tipo solicitado.
    
    Args:
        df: DataFrame de entrada
        chart_type: Tipo de gráfico ('box_plot', 'scatter_plot', 'heatmap', etc.)
        columns: Columnas a graficar
        metadata: Metadata para tooltips/anotaciones (opcional)
        options: Opciones adicionales de visualización (opcional)
        
    Returns:
        Objeto de gráfico (Plotly Figure, imagen base64, etc.)
    """
    # Stub: solo contratos, lógica a implementar
    # Ejemplo de logging
    log_action(
        function='generate_plot',
        step='visualization',
        parameters={'chart_type': chart_type, 'columns': columns},
        before_metrics={'rows': len(df)},
        after_metrics={},
        status='success',
        message=f"Stub de generación de gráfico: {chart_type}"
    )
    return None

def generate_wordcloud(
    text_data: List[str],
    options: Optional[Dict[str, Any]] = None
) -> str:
    """
    Genera una nube de palabras a partir de una lista de textos.
    
    Args:
        text_data: Lista de textos
        options: Opciones de WordCloud (opcional)
        
    Returns:
        Imagen en base64 para mostrar en UI
    """
    # Stub: solo contratos
    log_action(
        function='generate_wordcloud',
        step='visualization',
        parameters={'n_texts': len(text_data)},
        before_metrics={},
        after_metrics={},
        status='success',
        message="Stub de generación de wordcloud"
    )
    return ""

def get_visualization_metadata(
    df: pd.DataFrame,
    semantic_classification: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Genera metadata enriquecida para tooltips y anotaciones de UI.
    
    Args:
        df: DataFrame de entrada
        semantic_classification: Clasificación semántica previa (opcional)
        
    Returns:
        Diccionario de metadata para visualización
    """
    # Stub: solo contratos
    return {col: {'tooltip': f"Columna: {col}"} for col in df.columns}

class VisualizationGenerator:
    """
    Clase principal para generar visualizaciones de ciencias sociales.
    Encapsula todas las funciones de visualización y sugerencias de gráficos.
    """
    
    def __init__(self):
        """Inicializar el VisualizationGenerator."""
        self.suggestions_cache = {}
        self.generated_plots = {}
    
    def suggest_charts(self, df: pd.DataFrame, semantic_classification: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """
        Sugerir tipos de gráficos para el DataFrame.
        
        Args:
            df: DataFrame a analizar
            semantic_classification: Clasificación semántica previa
            
        Returns:
            Diccionario con sugerencias de gráficos por columna
        """
        return suggest_charts(df, semantic_classification)
    
    def generate_plot(self, df: pd.DataFrame, chart_type: str, columns: List[str], 
                     metadata: Optional[Dict[str, Any]] = None, 
                     options: Optional[Dict[str, Any]] = None) -> Any:
        """
        Generar un gráfico específico.
        
        Args:
            df: DataFrame de entrada
            chart_type: Tipo de gráfico
            columns: Columnas a graficar
            metadata: Metadata para tooltips
            options: Opciones adicionales
            
        Returns:
            Objeto de gráfico
        """
        return generate_plot(df, chart_type, columns, metadata, options)
    
    def generate_wordcloud(self, text_data: List[str], options: Optional[Dict[str, Any]] = None) -> str:
        """
        Generar nube de palabras.
        
        Args:
            text_data: Lista de textos
            options: Opciones de WordCloud
            
        Returns:
            Imagen en base64
        """
        return generate_wordcloud(text_data, options)
    
    def get_metadata(self, df: pd.DataFrame, semantic_classification: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Obtener metadata para visualización.
        
        Args:
            df: DataFrame de entrada
            semantic_classification: Clasificación semántica
            
        Returns:
            Metadata para tooltips y anotaciones
        """
        return get_visualization_metadata(df, semantic_classification)
    
    def can_generate_visualizations(self, df: pd.DataFrame) -> bool:
        """
        Verificar si se pueden generar visualizaciones para el DataFrame.
        
        Args:
            df: DataFrame a verificar
            
        Returns:
            True si se pueden generar visualizaciones
        """
        # Verificar si hay columnas numéricas o categóricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        return len(numeric_cols) > 0 or len(categorical_cols) > 0 