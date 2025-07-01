"""
Módulo de Agrupación Compleja - Versión Avanzada
================================================

Sistema robusto para manejar agrupaciones complejas en análisis de datos:
- Agrupación por múltiples variables
- Agrupación jerárquica
- Agrupación con filtros complejos
- Agrupación temporal con ventanas deslizantes
- Agrupación con transformaciones personalizadas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import re
from collections import defaultdict
import warnings

class ComplexGrouping:
    """
    Clase para manejar agrupaciones complejas en análisis de datos.
    """
    
    def __init__(self, df: pd.DataFrame, variable_metadata: Optional[Dict] = None):
        """
        Inicializa el sistema de agrupación compleja.
        
        Args:
            df: DataFrame con los datos
            variable_metadata: Metadatos de variables (opcional)
        """
        self.df = df.copy()
        self.variable_metadata = variable_metadata or {}
        self.grouping_cache = {}
        
        # Palabras clave para detectar agrupaciones
        self.grouping_keywords = {
            'por': ['por', 'según', 'agrupado por', 'agrupados por', 'por cada', 'por grupo'],
            'cada': ['cada', 'por cada', 'para cada', 'en cada'],
            'grupo': ['grupo', 'grupos', 'categoría', 'categorías', 'clase', 'clases'],
            'nivel': ['nivel', 'niveles', 'jerarquía', 'jerárquico'],
            'ventana': ['ventana', 'ventanas', 'período', 'períodos', 'intervalo', 'intervalos'],
            'rolling': ['rolling', 'móvil', 'móvil', 'deslizante', 'deslizantes']
        }
        
        # Operaciones de agrupación soportadas
        self.grouping_operations = {
            'count': ['contar', 'cuenta', 'número', 'cantidad'],
            'sum': ['suma', 'sumar', 'total'],
            'mean': ['promedio', 'media', 'valor promedio'],
            'median': ['mediana', 'valor central'],
            'std': ['desviación', 'desviacion', 'std'],
            'var': ['varianza', 'variabilidad'],
            'min': ['mínimo', 'minimo', 'menor'],
            'max': ['máximo', 'maximo', 'mayor'],
            'first': ['primero', 'inicial'],
            'last': ['último', 'ultimo', 'final'],
            'nunique': ['únicos', 'unicos', 'distintos', 'diferentes']
        }
    
    def detect_complex_grouping(self, question: str) -> Dict[str, Any]:
        """
        Detecta agrupaciones complejas en una pregunta en lenguaje natural.
        
        Args:
            question: Pregunta en lenguaje natural
            
        Returns:
            Diccionario con parámetros de agrupación detectados
        """
        q = question.lower()
        
        # Detectar tipo de agrupación
        grouping_type = self._detect_grouping_type(q)
        
        # Detectar variables de agrupación
        grouping_vars = self._detect_grouping_variables(q)
        
        # Detectar operaciones de agrupación
        operations = self._detect_grouping_operations(q)
        
        # Detectar filtros de agrupación
        filters = self._detect_grouping_filters(q)
        
        # Detectar parámetros de ventana (si aplica)
        window_params = self._detect_window_parameters(q)
        
        # Detectar transformaciones personalizadas
        transformations = self._detect_custom_transformations(q)
        
        return {
            'type': grouping_type,
            'variables': grouping_vars,
            'operations': operations,
            'filters': filters,
            'window_params': window_params,
            'transformations': transformations,
            'hierarchical': self._detect_hierarchical_grouping(q),
            'rolling': self._detect_rolling_grouping(q)
        }
    
    def _detect_grouping_type(self, question: str) -> str:
        """Detecta el tipo de agrupación."""
        q = question.lower()
        
        if any(word in q for word in self.grouping_keywords['ventana'] + self.grouping_keywords['rolling']):
            return 'temporal_window'
        elif any(word in q for word in self.grouping_keywords['nivel']):
            return 'hierarchical'
        elif any(word in q for word in self.grouping_keywords['por'] + self.grouping_keywords['cada']):
            return 'categorical'
        else:
            return 'simple'
    
    def _detect_grouping_variables(self, question: str) -> List[str]:
        """Detecta variables de agrupación."""
        q = question.lower()
        variables = []
        
        # Buscar variables mencionadas explícitamente
        for col in self.df.columns:
            col_lower = col.lower()
            if col_lower in q or col in q:
                variables.append(col)
        
        # Buscar patrones como "por [variable]"
        por_pattern = r'por\s+(\w+)'
        matches = re.findall(por_pattern, q)
        for match in matches:
            # Buscar variable similar en el DataFrame
            for col in self.df.columns:
                if match in col.lower() or col.lower() in match:
                    if col not in variables:
                        variables.append(col)
        
        # Buscar patrones como "agrupado por [variable]"
        agrupado_pattern = r'agrupado\s+por\s+(\w+)'
        matches = re.findall(agrupado_pattern, q)
        for match in matches:
            for col in self.df.columns:
                if match in col.lower() or col.lower() in match:
                    if col not in variables:
                        variables.append(col)
        
        return variables
    
    def _detect_grouping_operations(self, question: str) -> List[str]:
        """Detecta operaciones de agrupación."""
        q = question.lower()
        operations = []
        
        for op, keywords in self.grouping_operations.items():
            if any(keyword in q for keyword in keywords):
                operations.append(op)
        
        # Si no se detecta ninguna operación, usar operaciones por defecto
        if not operations:
            operations = ['count', 'mean']
        
        return operations
    
    def _detect_grouping_filters(self, question: str) -> Dict[str, Any]:
        """Detecta filtros específicos para agrupación."""
        q = question.lower()
        filters = {}
        
        # Detectar filtros de categorías
        categoria_pattern = r'(\w+)\s+(?:es|son|está|están)\s+(\w+(?:\s+\w+)*)'
        matches = re.findall(categoria_pattern, q)
        for var, value in matches:
            for col in self.df.columns:
                if var.lower() in col.lower():
                    filters[col] = {'type': 'categorical', 'values': [value.strip()]}
        
        # Detectar filtros numéricos
        numerico_pattern = r'(\w+)\s+(?:mayor|menor|igual)\s+(?:a|que)\s+(\d+(?:\.\d+)?)'
        matches = re.findall(numerico_pattern, q)
        for var, value in matches:
            for col in self.df.columns:
                if var.lower() in col.lower():
                    filters[col] = {'type': 'numeric', 'operator': '>=', 'value': float(value)}
        
        return filters
    
    def _detect_window_parameters(self, question: str) -> Dict[str, Any]:
        """Detecta parámetros de ventana temporal."""
        q = question.lower()
        params = {}
        
        # Detectar frecuencia
        if 'diario' in q or 'día' in q or 'dias' in q:
            params['freq'] = 'D'
        elif 'semanal' in q or 'semana' in q or 'semanas' in q:
            params['freq'] = 'W'
        elif 'mensual' in q or 'mes' in q or 'meses' in q:
            params['freq'] = 'M'
        elif 'trimestral' in q or 'trimestre' in q:
            params['freq'] = 'Q'
        elif 'anual' in q or 'año' in q or 'años' in q:
            params['freq'] = 'Y'
        
        # Detectar tamaño de ventana
        window_pattern = r'(\d+)\s*(?:días?|semanas?|meses?|años?)'
        match = re.search(window_pattern, q)
        if match:
            params['window_size'] = int(match.group(1))
        
        return params
    
    def _detect_custom_transformations(self, question: str) -> List[str]:
        """Detecta transformaciones personalizadas."""
        q = question.lower()
        transformations = []
        
        if 'porcentaje' in q or '%' in q:
            transformations.append('percentage')
        if 'normalizado' in q or 'normalizar' in q:
            transformations.append('normalize')
        if 'acumulado' in q or 'acumular' in q:
            transformations.append('cumulative')
        if 'cambio' in q or 'diferencia' in q:
            transformations.append('diff')
        
        return transformations
    
    def _detect_hierarchical_grouping(self, question: str) -> bool:
        """Detecta si es agrupación jerárquica."""
        q = question.lower()
        return any(word in q for word in self.grouping_keywords['nivel'])
    
    def _detect_rolling_grouping(self, question: str) -> bool:
        """Detecta si es agrupación con ventana deslizante."""
        q = question.lower()
        return any(word in q for word in self.grouping_keywords['rolling'])
    
    def execute_complex_grouping(self, grouping_params: Dict[str, Any], 
                               target_variables: List[str] = None) -> pd.DataFrame:
        """
        Ejecuta la agrupación compleja según los parámetros detectados.
        
        Args:
            grouping_params: Parámetros de agrupación detectados
            target_variables: Variables objetivo para las operaciones
            
        Returns:
            DataFrame con los resultados de la agrupación
        """
        if not target_variables:
            # Si no se especifican variables objetivo, usar todas las numéricas
            target_variables = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Aplicar filtros de agrupación
        df_filtered = self._apply_grouping_filters(grouping_params['filters'])
        
        # Ejecutar agrupación según el tipo
        if grouping_params['type'] == 'temporal_window':
            result = self._execute_temporal_window_grouping(df_filtered, grouping_params, target_variables)
        elif grouping_params['type'] == 'hierarchical':
            result = self._execute_hierarchical_grouping(df_filtered, grouping_params, target_variables)
        elif grouping_params['rolling']:
            result = self._execute_rolling_grouping(df_filtered, grouping_params, target_variables)
        else:
            result = self._execute_categorical_grouping(df_filtered, grouping_params, target_variables)
        
        # Aplicar transformaciones personalizadas
        if grouping_params['transformations']:
            result = self._apply_custom_transformations(result, grouping_params['transformations'])
        
        return result
    
    def _apply_grouping_filters(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Aplica filtros específicos para agrupación."""
        df_filtered = self.df.copy()
        
        for var, filter_config in filters.items():
            if var in df_filtered.columns:
                if filter_config['type'] == 'categorical':
                    df_filtered = df_filtered[df_filtered[var].isin(filter_config['values'])]
                elif filter_config['type'] == 'numeric':
                    operator = filter_config['operator']
                    value = filter_config['value']
                    if operator == '>=':
                        df_filtered = df_filtered[df_filtered[var] >= value]
                    elif operator == '<=':
                        df_filtered = df_filtered[df_filtered[var] <= value]
                    elif operator == '>':
                        df_filtered = df_filtered[df_filtered[var] > value]
                    elif operator == '<':
                        df_filtered = df_filtered[df_filtered[var] < value]
                    elif operator == '==':
                        df_filtered = df_filtered[df_filtered[var] == value]
        
        return df_filtered
    
    def _execute_categorical_grouping(self, df: pd.DataFrame, params: Dict[str, Any], 
                                    target_vars: List[str]) -> pd.DataFrame:
        """Ejecuta agrupación categórica simple."""
        if not params['variables']:
            return pd.DataFrame()
        
        # Crear grupo
        group = df.groupby(params['variables'])
        
        # Aplicar operaciones
        result_data = {}
        for var in target_vars:
            if var in df.columns:
                for op in params['operations']:
                    if hasattr(group[var], op):
                        result_data[f"{var}_{op}"] = getattr(group[var], op)()
        
        if result_data:
            result = pd.concat(result_data.values(), axis=1)
            result.columns = result_data.keys()
            return result.reset_index()
        else:
            return pd.DataFrame()
    
    def _execute_hierarchical_grouping(self, df: pd.DataFrame, params: Dict[str, Any], 
                                     target_vars: List[str]) -> pd.DataFrame:
        """Ejecuta agrupación jerárquica."""
        if len(params['variables']) < 2:
            return self._execute_categorical_grouping(df, params, target_vars)
        
        # Filtrar variables que existen en el DataFrame
        valid_vars = [var for var in params['variables'] if var in df.columns]
        if len(valid_vars) < 2:
            return self._execute_categorical_grouping(df, params, target_vars)
        
        # Crear agrupación jerárquica
        group = df.groupby(valid_vars)
        
        # Aplicar operaciones
        result_data = {}
        for var in target_vars:
            if var in df.columns:
                for op in params['operations']:
                    if hasattr(group[var], op):
                        result_data[f"{var}_{op}"] = getattr(group[var], op)()
        
        if result_data:
            result = pd.concat(result_data.values(), axis=1)
            result.columns = result_data.keys()
            
            # Agregar información de nivel jerárquico
            result['nivel_jerarquico'] = result[valid_vars].apply(
                lambda x: ' > '.join(x.astype(str)), axis=1
            )
            
            return result.reset_index()
        else:
            return pd.DataFrame()
    
    def _execute_temporal_window_grouping(self, df: pd.DataFrame, params: Dict[str, Any], 
                                        target_vars: List[str]) -> pd.DataFrame:
        """Ejecuta agrupación con ventana temporal."""
        # Buscar columna de fecha
        date_col = None
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'fecha' in col.lower() or 'date' in col.lower():
                date_col = col
                break
        
        if not date_col:
            # Si no hay columna de fecha, intentar convertir alguna columna
            for col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    date_col = col
                    break
                except:
                    continue
        
        if not date_col:
            return pd.DataFrame({'error': ['No se encontró columna de fecha válida']})
        
        # Configurar frecuencia
        freq = params['window_params'].get('freq', 'D')
        
        # Agrupar por ventana temporal
        df_temp = df.copy()
        df_temp['ventana_temporal'] = pd.to_datetime(df_temp[date_col]).dt.to_period(freq)
        
        group = df_temp.groupby('ventana_temporal')
        
        # Aplicar operaciones
        result_data = {}
        for var in target_vars:
            if var in df.columns:
                for op in params['operations']:
                    if hasattr(group[var], op):
                        result_data[f"{var}_{op}"] = getattr(group[var], op)()
        
        if result_data:
            result = pd.concat(result_data.values(), axis=1)
            result.columns = result_data.keys()
            result = result.reset_index()
            result['ventana_temporal'] = result['ventana_temporal'].astype(str)
            return result
        else:
            return pd.DataFrame()
    
    def _execute_rolling_grouping(self, df: pd.DataFrame, params: Dict[str, Any], 
                                target_vars: List[str]) -> pd.DataFrame:
        """Ejecuta agrupación con ventana deslizante."""
        # Buscar columna de fecha
        date_col = None
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'fecha' in col.lower() or 'date' in col.lower():
                date_col = col
                break
        
        if not date_col:
            return pd.DataFrame({'error': ['No se encontró columna de fecha válida']})
        
        # Ordenar por fecha
        df_sorted = df.sort_values(date_col)
        
        # Configurar ventana
        window_size = params['window_params'].get('window_size', 7)
        
        # Aplicar ventana deslizante
        result_data = {}
        for var in target_vars:
            if var in df.columns:
                for op in params['operations']:
                    if op in ['mean', 'sum', 'std', 'var', 'min', 'max']:
                        result_data[f"{var}_{op}_rolling"] = getattr(df_sorted[var].rolling(window=window_size), op)()
        
        if result_data:
            result = pd.concat(result_data.values(), axis=1)
            result.columns = result_data.keys()
            result[date_col] = df_sorted[date_col]
            return result
        else:
            return pd.DataFrame()
    
    def _apply_custom_transformations(self, df: pd.DataFrame, transformations: List[str]) -> pd.DataFrame:
        """Aplica transformaciones personalizadas."""
        df_transformed = df.copy()
        
        for transform in transformations:
            if transform == 'percentage':
                # Calcular porcentajes
                numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ['index', 'level_0']:  # Evitar columnas de índice
                        total = df_transformed[col].sum()
                        if total != 0:
                            df_transformed[f"{col}_porcentaje"] = (df_transformed[col] / total) * 100
            
            elif transform == 'normalize':
                # Normalizar valores
                numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ['index', 'level_0']:
                        min_val = df_transformed[col].min()
                        max_val = df_transformed[col].max()
                        if max_val != min_val:
                            df_transformed[f"{col}_normalizado"] = (df_transformed[col] - min_val) / (max_val - min_val)
            
            elif transform == 'cumulative':
                # Calcular valores acumulados
                numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ['index', 'level_0']:
                        df_transformed[f"{col}_acumulado"] = df_transformed[col].cumsum()
            
            elif transform == 'diff':
                # Calcular diferencias
                numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ['index', 'level_0']:
                        df_transformed[f"{col}_diferencia"] = df_transformed[col].diff()
        
        return df_transformed
    
    def get_grouping_insights(self, result: pd.DataFrame, grouping_params: Dict[str, Any]) -> str:
        """
        Genera insights automáticos sobre los resultados de agrupación.
        
        Args:
            result: DataFrame con resultados de agrupación
            grouping_params: Parámetros de agrupación utilizados
            
        Returns:
            String con insights generados
        """
        if result.empty:
            return "No se encontraron datos para la agrupación especificada."
        
        insights = []
        
        # Insights básicos
        insights.append(f"**Agrupación por**: {', '.join(grouping_params['variables'])}")
        insights.append(f"**Operaciones aplicadas**: {', '.join(grouping_params['operations'])}")
        insights.append(f"**Total de grupos**: {len(result)}")
        
        # Insights específicos según el tipo de agrupación
        if grouping_params['type'] == 'hierarchical':
            insights.append("**Agrupación jerárquica detectada** - Los datos se organizan en múltiples niveles.")
        
        if grouping_params['rolling']:
            insights.append("**Ventana deslizante aplicada** - Los resultados muestran tendencias suavizadas.")
        
        if grouping_params['type'] == 'temporal_window':
            insights.append("**Agrupación temporal aplicada** - Los datos se organizan por períodos de tiempo.")
        
        # Insights sobre los datos
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:  # Solo las primeras 3 columnas numéricas
                if col in result.columns:
                    max_val = result[col].max()
                    min_val = result[col].max()
                    mean_val = result[col].mean()
                    insights.append(f"**{col}**: Rango {min_val:.2f} - {max_val:.2f}, Promedio {mean_val:.2f}")
        
        return "\n".join(insights)

def detect_complex_grouping_in_question(question: str, df: pd.DataFrame, 
                                      variable_metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para detectar agrupación compleja en una pregunta.
    
    Args:
        question: Pregunta en lenguaje natural
        df: DataFrame con los datos
        variable_metadata: Metadatos de variables (opcional)
        
    Returns:
        Diccionario con parámetros de agrupación detectados
    """
    grouping_system = ComplexGrouping(df, variable_metadata)
    return grouping_system.detect_complex_grouping(question)

def execute_complex_grouping_from_question(question: str, df: pd.DataFrame, 
                                         target_variables: List[str] = None,
                                         variable_metadata: Optional[Dict] = None) -> Tuple[pd.DataFrame, str]:
    """
    Función de conveniencia para ejecutar agrupación compleja desde una pregunta.
    
    Args:
        question: Pregunta en lenguaje natural
        df: DataFrame con los datos
        target_variables: Variables objetivo (opcional)
        variable_metadata: Metadatos de variables (opcional)
        
    Returns:
        Tuple con (DataFrame de resultados, insights)
    """
    grouping_system = ComplexGrouping(df, variable_metadata)
    grouping_params = grouping_system.detect_complex_grouping(question)
    
    if not grouping_params['variables']:
        return pd.DataFrame(), "No se detectaron variables de agrupación válidas."
    
    result = grouping_system.execute_complex_grouping(grouping_params, target_variables)
    insights = grouping_system.get_grouping_insights(result, grouping_params)
    
    return result, insights
