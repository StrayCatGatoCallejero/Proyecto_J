"""
Módulo de Visualización - Patrón "Reloj Suizo"
=============================================

Responsabilidades:
- Visualizaciones estáticas con matplotlib
- Visualizaciones interactivas con plotly
- Generación de gráficos según tipo de datos
- Logging sistemático de operaciones
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

warnings.filterwarnings("ignore")

# Importar logging
from .logging import log_action

logger = logging.getLogger(__name__)

def plot_histogram(df: pd.DataFrame, column: str, bins: int = 20, 
                  title: Optional[str] = None) -> Figure:
    """
    Genera histograma de una columna numérica usando matplotlib.
    
    Args:
        df: DataFrame con los datos
        column: Columna numérica a graficar
        bins: Número de bins para el histograma
        title: Título del gráfico (opcional)
        
    Returns:
        Figura de matplotlib
    """
    start_time = datetime.now()
    
    # Validar entrada
    if column not in df.columns:
        raise ValueError(f"La columna {column} no existe en el DataFrame")
    
    series = df[column].dropna()
    if len(series) == 0:
        raise ValueError("No hay datos válidos para graficar")
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(series, bins=bins, color='skyblue', edgecolor='gray', alpha=0.7)
    
    # Configurar título y etiquetas
    if title is None:
        title = f"Histograma de {column}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Añadir estadísticas
    mean_val = series.mean()
    median_val = series.median()
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Media: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', label=f'Mediana: {median_val:.2f}')
    ax.legend()
    
    plt.tight_layout()
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="plot_histogram",
        step="visualization",
        parameters={"column": column, "bins": bins},
        before_metrics={"n_observations": len(series)},
        after_metrics={"figure_created": True},
        status="success",
        message=f"Histograma generado para {column}",
        execution_time=execution_time
    )
    
    return fig

def plot_boxplot(df: pd.DataFrame, column: str, group_column: Optional[str] = None,
                 title: Optional[str] = None) -> Figure:
    """
    Genera boxplot de una columna numérica usando matplotlib.
    
    Args:
        df: DataFrame con los datos
        column: Columna numérica a graficar
        group_column: Columna categórica para agrupar (opcional)
        title: Título del gráfico (opcional)
        
    Returns:
        Figura de matplotlib
    """
    start_time = datetime.now()
    
    # Validar entrada
    if column not in df.columns:
        raise ValueError(f"La columna {column} no existe en el DataFrame")
    
    if group_column and group_column not in df.columns:
        raise ValueError(f"La columna {group_column} no existe en el DataFrame")
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if group_column:
        # Boxplot por grupos
        groups = df[group_column].unique()
        data = [df[df[group_column] == group][column].dropna() for group in groups]
        ax.boxplot(data, label=groups)
        ax.set_xlabel(group_column, fontsize=12)
    else:
        # Boxplot simple
        series = df[column].dropna()
        ax.boxplot(series)
        ax.set_xlabel(column, fontsize=12)
    
    # Configurar título y etiquetas
    if title is None:
        title = f"Boxplot de {column}"
        if group_column:
            title += f" por {group_column}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(column, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="plot_boxplot",
        step="visualization",
        parameters={"column": column, "group_column": group_column},
        before_metrics={"n_observations": len(df[column].dropna())},
        after_metrics={"figure_created": True},
        status="success",
        message=f"Boxplot generado para {column}",
        execution_time=execution_time
    )
    
    return fig

def plot_scatter(df: pd.DataFrame, x_column: str, y_column: str,
                 color_column: Optional[str] = None, title: Optional[str] = None) -> Figure:
    """
    Genera gráfico de dispersión usando matplotlib.
    
    Args:
        df: DataFrame con los datos
        x_column: Columna para el eje X
        y_column: Columna para el eje Y
        color_column: Columna para colorear puntos (opcional)
        title: Título del gráfico (opcional)
        
    Returns:
        Figura de matplotlib
    """
    start_time = datetime.now()
    
    # Validar entrada
    for col in [x_column, y_column]:
        if col not in df.columns:
            raise ValueError(f"La columna {col} no existe en el DataFrame")
    
    if color_column and color_column not in df.columns:
        raise ValueError(f"La columna {color_column} no existe en el DataFrame")
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Preparar datos
    data = df[[x_column, y_column]].dropna()
    if color_column:
        data = df[[x_column, y_column, color_column]].dropna()
    
    if len(data) == 0:
        raise ValueError("No hay datos válidos para graficar")
    
    # Crear scatter plot
    if color_column:
        scatter = ax.scatter(data[x_column], data[y_column], 
                           c=data[color_column], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label=color_column)
    else:
        ax.scatter(data[x_column], data[y_column], alpha=0.6, color='blue')
    
    # Configurar título y etiquetas
    if title is None:
        title = f"Dispersión: {y_column} vs {x_column}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_column, fontsize=12)
    ax.set_ylabel(y_column, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Añadir línea de tendencia
    z = np.polyfit(data[x_column], data[y_column], 1)
    p = np.poly1d(z)
    ax.plot(data[x_column], p(data[x_column]), "r--", alpha=0.8)
    
    plt.tight_layout()
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="plot_scatter",
        step="visualization",
        parameters={"x_column": x_column, "y_column": y_column, "color_column": color_column},
        before_metrics={"n_observations": len(data)},
        after_metrics={"figure_created": True},
        status="success",
        message=f"Gráfico de dispersión generado",
        execution_time=execution_time
    )
    
    return fig

def plot_heatmap(corr: pd.DataFrame, title: Optional[str] = None) -> Figure:
    """
    Genera mapa de calor de correlaciones usando matplotlib.
    
    Args:
        corr: DataFrame de correlaciones
        title: Título del gráfico (opcional)
        
    Returns:
        Figura de matplotlib
    """
    start_time = datetime.now()
    
    # Validar entrada
    if corr.empty:
        raise ValueError("El DataFrame de correlaciones está vacío")
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Crear mapa de calor
    im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Configurar etiquetas
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    
    # Añadir valores de correlación
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # Configurar título
    if title is None:
        title = "Matriz de Correlaciones"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Añadir barra de color
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlación', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="plot_heatmap",
        step="visualization",
        parameters={"n_variables": len(corr.columns)},
        before_metrics={"correlation_matrix_size": corr.shape},
        after_metrics={"figure_created": True},
        status="success",
        message=f"Mapa de calor de correlaciones generado",
        execution_time=execution_time
    )
    
    return fig

def plot_bar_chart(df: pd.DataFrame, column: str, top_n: Optional[int] = None,
                   title: Optional[str] = None) -> Figure:
    """
    Genera gráfico de barras para una columna categórica usando matplotlib.
    
    Args:
        df: DataFrame con los datos
        column: Columna categórica a graficar
        top_n: Número de categorías principales a mostrar (opcional)
        title: Título del gráfico (opcional)
        
    Returns:
        Figura de matplotlib
    """
    start_time = datetime.now()
    
    # Validar entrada
    if column not in df.columns:
        raise ValueError(f"La columna {column} no existe en el DataFrame")
    
    # Calcular frecuencias
    value_counts = df[column].value_counts()
    
    if len(value_counts) == 0:
        raise ValueError("No hay datos válidos para graficar")
    
    # Limitar a top_n si se especifica
    if top_n is not None:
        value_counts = value_counts.head(top_n)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Crear gráfico de barras
    bars = ax.bar(range(len(value_counts)), value_counts.values, 
                  color='skyblue', edgecolor='gray', alpha=0.7)
    
    # Configurar etiquetas
    ax.set_xticks(range(len(value_counts)))
    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Frecuencia', fontsize=12)
    
    # Configurar título
    if title is None:
        title = f"Distribución de {column}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Añadir valores en las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    log_action(
        function="plot_bar_chart",
        step="visualization",
        parameters={"column": column, "top_n": top_n},
        before_metrics={"n_categories": len(value_counts)},
        after_metrics={"figure_created": True},
        status="success",
        message=f"Gráfico de barras generado para {column}",
        execution_time=execution_time
    )
    
    return fig

# Funciones Plotly interactivas (aisladas)
def plotly_histogram(df: pd.DataFrame, column: str, bins: int = 20) -> Any:
    """
    Genera histograma interactivo usando plotly.
    
    Args:
        df: DataFrame con los datos
        column: Columna numérica a graficar
        bins: Número de bins para el histograma
        
    Returns:
        Figura de plotly
    """
    start_time = datetime.now()
    
    # Validar entrada
    if column not in df.columns:
        raise ValueError(f"La columna {column} no existe en el DataFrame")
    
    series = df[column].dropna()
    if len(series) == 0:
        raise ValueError("No hay datos válidos para graficar")
    
    try:
        # Crear histograma
        fig = go.Figure(data=[go.Histogram(x=series, nbinsx=bins)])
        
        # Configurar layout
        fig.update_layout(
                          title=f"Histograma de {column}",
            xaxis_title=column,
            yaxis_title="Frecuencia",
            showlegend=False,
            template="plotly_white"
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        log_action(
            function="plotly_histogram",
            step="visualization",
            parameters={"column": column, "bins": bins},
            before_metrics={"n_observations": len(series)},
            after_metrics={"figure_created": True},
            status="success",
            message=f"Histograma interactivo generado para {column}",
            execution_time=execution_time
        )
        
        return fig
        
    except ImportError:
        raise ImportError("plotly no está instalado. Instale con: pip install plotly")

def plotly_heatmap(corr: pd.DataFrame) -> Any:
    """
    Genera mapa de calor interactivo de correlaciones usando plotly.
    
    Args:
        corr: DataFrame de correlaciones
        
    Returns:
        Figura de plotly
    """
    start_time = datetime.now()
    
    # Validar entrada
    if corr.empty:
        raise ValueError("El DataFrame de correlaciones está vacío")
    
    try:
        # Crear mapa de calor
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Configurar layout
        fig.update_layout(
            title="Matriz de Correlaciones",
            xaxis_title="Variables",
            yaxis_title="Variables",
            template="plotly_white"
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        log_action(
            function="plotly_heatmap",
            step="visualization",
            parameters={"n_variables": len(corr.columns)},
            before_metrics={"correlation_matrix_size": corr.shape},
            after_metrics={"figure_created": True},
            status="success",
            message=f"Mapa de calor interactivo generado",
            execution_time=execution_time
        )
        
        return fig
        
    except ImportError:
        raise ImportError("plotly no está instalado. Instale con: pip install plotly")

def plotly_scatter(df: pd.DataFrame, x_column: str, y_column: str,
                   color_column: Optional[str] = None, size_column: Optional[str] = None) -> Any:
    """
    Genera gráfico de dispersión interactivo usando plotly.
    
    Args:
        df: DataFrame con los datos
        x_column: Columna para el eje X
        y_column: Columna para el eje Y
        color_column: Columna para colorear puntos (opcional)
        size_column: Columna para el tamaño de puntos (opcional)
        
    Returns:
        Figura de plotly
    """
    start_time = datetime.now()
    
    # Validar entrada
    for col in [x_column, y_column]:
        if col not in df.columns:
            raise ValueError(f"La columna {col} no existe en el DataFrame")
    
    if color_column and color_column not in df.columns:
        raise ValueError(f"La columna {color_column} no existe en el DataFrame")
    
    if size_column and size_column not in df.columns:
        raise ValueError(f"La columna {size_column} no existe en el DataFrame")
    
    # Preparar datos
    data = df[[x_column, y_column]].dropna()
    if color_column:
        data = df[[x_column, y_column, color_column]].dropna()
    if size_column:
        data = df[[x_column, y_column, size_column]].dropna()
    if color_column and size_column:
        data = df[[x_column, y_column, color_column, size_column]].dropna()
    
    if len(data) == 0:
        raise ValueError("No hay datos válidos para graficar")
    
    try:
        # Crear scatter plot
        fig = go.Figure()
        
        scatter_kwargs = {
            'x': data[x_column],
            'y': data[y_column],
            'mode': 'markers',
            'marker': {'opacity': 0.6}
        }
        
        if color_column:
            scatter_kwargs['color'] = data[color_column]
            scatter_kwargs['colorbar'] = {'title': color_column}
        
        if size_column:
            scatter_kwargs['marker']['size'] = data[size_column]
        
        fig.add_trace(go.Scatter(**scatter_kwargs))
        
        # Configurar layout
        fig.update_layout(
                        title=f"Dispersión: {y_column} vs {x_column}",
            xaxis_title=x_column,
            yaxis_title=y_column,
            template="plotly_white"
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        log_action(
            function="plotly_scatter",
            step="visualization",
            parameters={"x_column": x_column, "y_column": y_column},
            before_metrics={"n_observations": len(data)},
            after_metrics={"figure_created": True},
            status="success",
            message=f"Gráfico de dispersión interactivo generado",
            execution_time=execution_time
        )
        
        return fig
        
    except ImportError:
        raise ImportError("plotly no está instalado. Instale con: pip install plotly")

class VisualizationGenerator:
    """
    Clase para generación automática de visualizaciones.
    """
    
    def __init__(self):
        """Inicializa el VisualizationGenerator."""
        self.generated_plots = []
        self.plot_config = {
            'template': 'plotly_white',
            'width': 800,
            'height': 600
        }
    
    def generate_visualizations(self, df: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Genera visualizaciones automáticas para un DataFrame.
        
        Args:
            df: DataFrame para visualizar
            config: Configuración de visualizaciones
            
        Returns:
            Diccionario con visualizaciones generadas
        """
        if config:
            self.plot_config.update(config)
        
        visualizations = {}
        
        # Detectar tipos de columnas
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Generar visualizaciones para columnas numéricas
        if numeric_columns:
            visualizations['numeric'] = self._generate_numeric_plots(df, numeric_columns)
        
        # Generar visualizaciones para columnas categóricas
        if categorical_columns:
            visualizations['categorical'] = self._generate_categorical_plots(df, categorical_columns)
        
        # Generar visualizaciones de correlación
        if len(numeric_columns) > 1:
            visualizations['correlation'] = self._generate_correlation_plot(df, numeric_columns)
        
        # Generar dashboard resumen
        visualizations['dashboard'] = self._generate_dashboard(df, numeric_columns, categorical_columns)
        
        self.generated_plots = list(visualizations.keys())
        
        return visualizations
    
    def _generate_numeric_plots(self, df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        """
        Genera visualizaciones para columnas numéricas.
        
        Args:
            df: DataFrame
            numeric_columns: Lista de columnas numéricas
            
        Returns:
            Diccionario con plots numéricos
        """
        plots = {}
        
        for column in numeric_columns[:5]:  # Limitar a 5 columnas
            try:
                # Histograma
                fig_hist = px.histogram(
                    df, 
                    x=column,
                    title=f"Distribución de {column}",
                    template=self.plot_config['template'],
                    width=self.plot_config['width'],
                    height=self.plot_config['height']
                )
                plots[f"{column}_histogram"] = fig_hist
                
                # Box plot
                fig_box = px.box(
                    df,
                    y=column,
                    title=f"Box Plot de {column}",
                    template=self.plot_config['template'],
                    width=self.plot_config['width'],
                    height=self.plot_config['height']
                )
                plots[f"{column}_boxplot"] = fig_box
                
            except Exception as e:
                logger.warning(f"Error generating plots for {column}: {e}")
        
        return plots
    
    def _generate_categorical_plots(self, df: pd.DataFrame, categorical_columns: List[str]) -> Dict[str, Any]:
        """
        Genera visualizaciones para columnas categóricas.
        
        Args:
            df: DataFrame
            categorical_columns: Lista de columnas categóricas
            
        Returns:
            Diccionario con plots categóricos
        """
        plots = {}
        
        for column in categorical_columns[:5]:  # Limitar a 5 columnas
            try:
                # Gráfico de barras
                value_counts = df[column].value_counts()
                
                fig_bar = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Distribución de {column}",
                    labels={'x': column, 'y': 'Frecuencia'},
                    template=self.plot_config['template'],
                    width=self.plot_config['width'],
                    height=self.plot_config['height']
                )
                plots[f"{column}_bar"] = fig_bar
                
                # Gráfico de pastel
                fig_pie = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Proporción de {column}",
                    template=self.plot_config['template'],
                    width=self.plot_config['width'],
                    height=self.plot_config['height']
                )
                plots[f"{column}_pie"] = fig_pie
                
            except Exception as e:
                logger.warning(f"Error generating plots for {column}: {e}")
        
        return plots
    
    def _generate_correlation_plot(self, df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        """
        Genera gráfico de correlación.
        
        Args:
            df: DataFrame
            numeric_columns: Lista de columnas numéricas
            
        Returns:
            Gráfico de correlación
        """
        try:
            # Calcular correlación
            corr_matrix = df[numeric_columns].corr()
            
            # Crear heatmap
            fig_heatmap = px.imshow(
                corr_matrix,
                title="Matriz de Correlación",
                template=self.plot_config['template'],
                width=self.plot_config['width'],
                height=self.plot_config['height'],
                color_continuous_scale='RdBu'
            )
            
            return {"correlation_heatmap": fig_heatmap}
            
        except Exception as e:
            logger.warning(f"Error generating correlation plot: {e}")
            return {}
    
    def _generate_dashboard(self, df: pd.DataFrame, numeric_columns: List[str], categorical_columns: List[str]) -> Dict[str, Any]:
        """
        Genera dashboard resumen.
        
        Args:
            df: DataFrame
            numeric_columns: Columnas numéricas
            categorical_columns: Columnas categóricas
            
        Returns:
            Dashboard resumen
        """
        try:
            # Crear subplots
            n_plots = min(4, len(numeric_columns) + len(categorical_columns))
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[f"Plot {i+1}" for i in range(n_plots)],
                specs=[[{"type": "histogram"}, {"type": "bar"}],
                       [{"type": "box"}, {"type": "scatter"}]]
            )
            
            # Agregar plots al dashboard
            plot_idx = 0
            
            # Histograma de primera columna numérica
            if numeric_columns:
                fig.add_trace(
                    go.Histogram(x=df[numeric_columns[0]], name=numeric_columns[0]),
                    row=1, col=1
                )
                plot_idx += 1
            
            # Gráfico de barras de primera columna categórica
            if categorical_columns:
                value_counts = df[categorical_columns[0]].value_counts()
                fig.add_trace(
                    go.Bar(x=value_counts.index, y=value_counts.values, name=categorical_columns[0]),
                    row=1, col=2
                )
                plot_idx += 1
            
            # Box plot de segunda columna numérica
            if len(numeric_columns) > 1:
                fig.add_trace(
                    go.Box(y=df[numeric_columns[1]], name=numeric_columns[1]),
                    row=2, col=1
                )
                plot_idx += 1
            
            # Scatter plot de dos columnas numéricas
            if len(numeric_columns) > 1:
                fig.add_trace(
                    go.Scatter(x=df[numeric_columns[0]], y=df[numeric_columns[1]], mode='markers', name='Scatter'),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Dashboard Resumen",
                template=self.plot_config['template'],
                width=self.plot_config['width'],
                height=self.plot_config['height']
            )
            
            return {"dashboard": fig}
            
        except Exception as e:
            logger.warning(f"Error generating dashboard: {e}")
            return {}
    
    def save_visualizations(self, visualizations: Dict[str, Any], output_dir: str = "visualizations") -> List[str]:
        """
        Guarda las visualizaciones en archivos HTML.
        
        Args:
            visualizations: Diccionario con visualizaciones
            output_dir: Directorio de salida
            
        Returns:
            Lista de archivos guardados
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        for plot_type, plots in visualizations.items():
            if isinstance(plots, dict):
                for plot_name, fig in plots.items():
                    try:
                        filename = f"{output_dir}/{plot_name}.html"
                        fig.write_html(filename)
                        saved_files.append(filename)
                    except Exception as e:
                        logger.warning(f"Error saving {plot_name}: {e}")
            elif hasattr(plots, 'write_html'):
                try:
                    filename = f"{output_dir}/{plot_type}.html"
                    plots.write_html(filename)
                    saved_files.append(filename)
                except Exception as e:
                    logger.warning(f"Error saving {plot_type}: {e}")
        
        return saved_files
    
    def get_visualization_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de las visualizaciones generadas.
        
        Returns:
            Diccionario con resumen
        """
        return {
            "total_plots": len(self.generated_plots),
            "plot_types": self.generated_plots,
            "config": self.plot_config
        }
