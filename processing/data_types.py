"""
Módulo de Detección de Tipos y Clasificación Semántica
======================================================

Responsabilidades:
- Clasificación semántica de columnas usando NLP y diccionarios
- Detección automática de escalas ordinales y Likert
- Normalización y unificación de niveles categóricos
- Reconocimiento de unidades de medida contextuales
- Análisis de texto libre con tokenización y sentimientos
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
from difflib import SequenceMatcher
import warnings

warnings.filterwarnings("ignore")

# Importar logging
from .logging import log_action


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detecta los tipos de datos de las columnas del DataFrame.

    Args:
        df: DataFrame a analizar

    Returns:
        Diccionario con el tipo detectado para cada columna
    """
    type_mapping = {}

    for column in df.columns:
        col_data = df[column].dropna()

        if len(col_data) == 0:
            type_mapping[column] = "empty"
            continue

        # Detectar tipo numérico
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].dtype in ["int64", "int32"]:
                type_mapping[column] = "numeric_integer"
            else:
                type_mapping[column] = "numeric_float"

        # Detectar tipo datetime
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            type_mapping[column] = "datetime"

        # Detectar tipo booleano
        elif pd.api.types.is_bool_dtype(df[column]):
            type_mapping[column] = "boolean"

        # Detectar tipo categórico
        elif df[column].dtype == "object" or df[column].dtype == "category":
            unique_values = col_data.nunique()
            total_values = len(col_data)

            # Si tiene pocos valores únicos, es categórico
            if unique_values <= min(20, total_values * 0.1):
                type_mapping[column] = "categorical"
            else:
                # Verificar si es texto libre
                avg_length = col_data.astype(str).str.len().mean()
                if avg_length > 50:
                    type_mapping[column] = "text_free"
                else:
                    type_mapping[column] = "text_short"

    return type_mapping


def semantic_classification(
    df: pd.DataFrame, custom_keywords: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Clasifica semánticamente las columnas usando diccionarios de palabras clave.

    Args:
        df: DataFrame a clasificar
        custom_keywords: Diccionario personalizado de palabras clave

    Returns:
        Diccionario con clasificación semántica de cada columna
    """
    # Diccionario base de palabras clave
    base_keywords = {
        "demographic": [
            "edad",
            "age",
            "genero",
            "sexo",
            "gender",
            "sex",
            "nacionalidad",
            "nacionality",
            "estado_civil",
            "marital",
            "civil",
            "educacion",
            "education",
            "ocupacion",
            "occupation",
            "profesion",
            "profession",
        ],
        "socioeconomic": [
            "ingresos",
            "income",
            "salario",
            "salary",
            "renta",
            "rent",
            "pobreza",
            "poverty",
            "vivienda",
            "housing",
            "servicios",
            "services",
            "empleo",
            "employment",
            "trabajo",
            "work",
            "desempleo",
            "unemployment",
        ],
        "opinion": [
            "satisfaccion",
            "satisfaction",
            "confianza",
            "confidence",
            "percepcion",
            "perception",
            "actitud",
            "attitude",
            "opinion",
            "preferencia",
            "preference",
            "gusto",
            "like",
            "disgusto",
            "dislike",
        ],
        "likert": [
            "acuerdo",
            "agreement",
            "frecuencia",
            "frequency",
            "importancia",
            "importance",
            "satisfaccion_escala",
            "satisfaction_scale",
            "muy",
            "very",
            "bastante",
            "quite",
            "poco",
            "little",
            "nada",
            "nothing",
        ],
        "temporal": [
            "fecha",
            "date",
            "tiempo",
            "time",
            "periodo",
            "period",
            "año",
            "year",
            "mes",
            "month",
            "dia",
            "day",
            "hora",
            "hour",
            "minuto",
            "minute",
        ],
        "geographic": [
            "region",
            "region",
            "comuna",
            "commune",
            "provincia",
            "province",
            "pais",
            "country",
            "ciudad",
            "city",
            "direccion",
            "address",
        ],
        "health": [
            "salud",
            "health",
            "enfermedad",
            "disease",
            "sintoma",
            "symptom",
            "medicamento",
            "medication",
            "tratamiento",
            "treatment",
        ],
    }

    # Combinar con palabras clave personalizadas
    if custom_keywords:
        for category, keywords in custom_keywords.items():
            if category in base_keywords:
                base_keywords[category].extend(keywords)
            else:
                base_keywords[category] = keywords

    classifications = {}

    for column in df.columns:
        column_lower = column.lower()
        best_match = None
        best_score = 0.0

        for category, keywords in base_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in column_lower:
                    score += 1.0

            if score > 0:
                score = score / len(keywords)
                if score > best_score:
                    best_score = score
                    best_match = category

        if best_match and best_score > 0.1:
            classifications[column] = {
                "type": best_match,
                "confidence": best_score,
                "keywords_found": [
                    kw for kw in base_keywords[best_match] if kw in column_lower
                ],
            }

    return classifications


def detect_likert_scales(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detecta escalas de Likert en el DataFrame.

    Args:
        df: DataFrame a analizar

    Returns:
        Diccionario con información de escalas de Likert detectadas
    """
    likert_scales = {}
    
    # Palabras clave típicas de escalas Likert
    likert_keywords = [
        'muy', 'bastante', 'poco', 'nada',
        'totalmente', 'parcialmente', 'ligeramente', 'nunca',
        'siempre', 'frecuentemente', 'ocasionalmente', 'raramente',
        'completamente', 'moderadamente', 'levemente', 'absolutamente',
        'muy_de_acuerdo', 'de_acuerdo', 'neutral', 'en_desacuerdo', 'muy_en_desacuerdo'
    ]
    
    for column in df.columns:
        if df[column].dtype == 'object':
            unique_values = df[column].dropna().unique()
            
            # Verificar si los valores únicos contienen palabras clave de Likert
            likert_score = 0
            for value in unique_values:
                value_str = str(value).lower()
                for keyword in likert_keywords:
                    if keyword in value_str:
                        likert_score += 1
                        break
            
            # Si al menos 3 valores únicos contienen palabras clave, considerar Likert
            if likert_score >= 3 and len(unique_values) <= 7:
                likert_scales[column] = {
                    'type': 'likert_scale',
                    'n_levels': len(unique_values),
                    'unique_values': unique_values.tolist(),
                    'confidence': likert_score / len(unique_values)
                }
    
    return likert_scales


def normalize_categorical_levels(
    df: pd.DataFrame, similarity_threshold: float = 0.8
) -> Dict[str, Dict[str, Any]]:
    """
    Normaliza niveles categóricos similares usando similitud de strings.

    Args:
        df: DataFrame a procesar
        similarity_threshold: Umbral de similitud para considerar valores iguales

    Returns:
        Diccionario con información de normalización
    """
    normalization_results = {}
    
    for column in df.columns:
        if df[column].dtype == 'object':
            unique_values = df[column].dropna().unique()
            
            if len(unique_values) <= 1:
                continue
            
            # Encontrar grupos de valores similares
            groups = []
            used_indices = set()
            
            for i, value1 in enumerate(unique_values):
                if i in used_indices:
                    continue
                
                group = [value1]
                used_indices.add(i)
                
                for j, value2 in enumerate(unique_values[i+1:], i+1):
                    if j in used_indices:
                        continue
                    
                    similarity = SequenceMatcher(None, str(value1).lower(), str(value2).lower()).ratio()
                    if similarity >= similarity_threshold:
                        group.append(value2)
                        used_indices.add(j)
                
                if len(group) > 1:
                    groups.append(group)
            
            if groups:
                normalization_results[column] = {
                    'n_groups': len(groups),
                    'groups': groups,
                    'original_unique': len(unique_values),
                    'normalized_unique': len(unique_values) - sum(len(g) - 1 for g in groups)
                }
    
    return normalization_results


def detect_measurement_units(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detecta unidades de medida en columnas numéricas basándose en nombres de columnas.

    Args:
        df: DataFrame a analizar

    Returns:
        Diccionario con unidades de medida detectadas
    """
    units_detected = {}
    
    # Diccionario de unidades de medida comunes
    unit_patterns = {
        'peso': ['kg', 'g', 'lb', 'ton', 'peso', 'weight', 'masa'],
        'longitud': ['m', 'cm', 'km', 'mm', 'altura', 'height', 'largo', 'ancho'],
        'tiempo': ['seg', 'min', 'hora', 'dia', 'mes', 'año', 'time', 'duration'],
        'moneda': ['$', 'peso', 'dolar', 'euro', 'money', 'precio', 'costo'],
        'porcentaje': ['%', 'porcentaje', 'percentage', 'ratio'],
        'temperatura': ['°c', '°f', 'celsius', 'fahrenheit', 'temp'],
        'velocidad': ['km/h', 'm/s', 'mph', 'speed', 'velocidad'],
        'area': ['m2', 'km2', 'ha', 'hectarea', 'area', 'superficie'],
        'volumen': ['l', 'ml', 'm3', 'litro', 'volume', 'capacidad']
    }
    
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            column_lower = column.lower()
            
            for unit_type, patterns in unit_patterns.items():
                for pattern in patterns:
                    if pattern in column_lower:
                        units_detected[column] = {
                            'unit_type': unit_type,
                            'detected_pattern': pattern,
                            'confidence': 'high' if pattern in ['kg', 'm', '$', '%'] else 'medium'
                        }
                        break
                else:
                    continue
                break
    
    return units_detected


def analyze_free_text(
    df: pd.DataFrame, text_columns: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analiza columnas de texto libre.

    Args:
        df: DataFrame a analizar
        text_columns: Lista de columnas de texto (opcional)

    Returns:
        Diccionario con análisis de texto
    """
    if text_columns is None:
        # Detectar columnas de texto automáticamente
        text_columns = []
        for column in df.columns:
            if df[column].dtype == 'object':
                # Verificar si es texto libre (promedio de longitud > 50 caracteres)
                avg_length = df[column].astype(str).str.len().mean()
                if avg_length > 50:
                    text_columns.append(column)
    
    text_analysis = {}
    
    for column in text_columns:
        if column not in df.columns:
            continue
        
        text_data = df[column].dropna().astype(str)
        if len(text_data) == 0:
            continue
        
        # Estadísticas básicas
        lengths = text_data.str.len()
        word_counts = text_data.str.split().str.len()
        
        # Detectar idioma (aproximación simple)
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 'pero', 'sus', 'me', 'hasta', 'hay', 'donde', 'han', 'quien', 'están', 'estado', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros']
        english_words = ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us']
        
        spanish_count = sum(1 for text in text_data for word in text.lower().split() if word in spanish_words)
        english_count = sum(1 for text in text_data for word in text.lower().split() if word in english_words)
        
        detected_language = 'spanish' if spanish_count > english_count else 'english' if english_count > spanish_count else 'unknown'
        
        text_analysis[column] = {
            'n_texts': len(text_data),
            'avg_length': lengths.mean(),
            'max_length': lengths.max(),
            'min_length': lengths.min(),
            'avg_words': word_counts.mean(),
            'max_words': word_counts.max(),
            'min_words': word_counts.min(),
            'detected_language': detected_language,
            'language_confidence': max(spanish_count, english_count) / max(1, spanish_count + english_count)
        }
    
    return text_analysis


def validate_internal_consistency(
    df: pd.DataFrame, rules: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Valida consistencia interna del DataFrame usando reglas personalizadas.

    Args:
        df: DataFrame a validar
        rules: Lista de reglas de validación (opcional)

    Returns:
        Diccionario con resultados de validación
    """
    if rules is None:
        # Reglas por defecto
        rules = [
            {'type': 'range', 'column': 'edad', 'min': 0, 'max': 120},
            {'type': 'range', 'column': 'ingresos', 'min': 0},
            {'type': 'categorical', 'column': 'genero', 'values': ['M', 'F', 'Masculino', 'Femenino']}
        ]
    
    errors: List[str] = []
    warnings: List[str] = []
    
    for rule in rules:
        rule_type = rule.get('type')
        column = rule.get('column')
        
        if column not in df.columns:
            warnings.append(f"Columna {column} no encontrada para regla {rule_type}")
            continue
        
        if rule_type == 'range':
            min_val = rule.get('min')
            max_val = rule.get('max')
            
            if min_val is not None:
                invalid_min = df[df[column] < min_val]
                if len(invalid_min) > 0:
                    errors.append(f"{column}: {len(invalid_min)} valores < {min_val}")
            
            if max_val is not None:
                invalid_max = df[df[column] > max_val]
                if len(invalid_max) > 0:
                    errors.append(f"{column}: {len(invalid_max)} valores > {max_val}")
        
        elif rule_type == 'categorical':
            allowed_values = rule.get('values', [])
            invalid_values = df[~df[column].isin(allowed_values)]
            if len(invalid_values) > 0:
                errors.append(f"{column}: {len(invalid_values)} valores no permitidos")
        
        elif rule_type == 'unique':
            duplicates = df[column].duplicated().sum()
            if duplicates > 0:
                errors.append(f"{column}: {duplicates} valores duplicados")
    
    validation_results: Dict[str, Any] = {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'total_checks': len(rules)
    }
    
    return validation_results


def suggest_visualizations(
    df: pd.DataFrame, semantic_classification: Optional[Dict[str, Any]] = None
) -> Dict[str, List[str]]:
    """
    Sugiere visualizaciones apropiadas basándose en tipos de datos y clasificación semántica.

    Args:
        df: DataFrame a analizar
        semantic_classification: Clasificación semántica de columnas (opcional)

    Returns:
        Diccionario con sugerencias de visualización por columna
    """
    suggestions: Dict[str, List[str]] = {}
    
    for column in df.columns:
        column_suggestions: List[str] = []
        
        # Detectar tipo de datos
        if df[column].dtype in ['int64', 'float64']:
            column_suggestions.extend(['histogram', 'boxplot', 'scatter'])
            
            # Sugerencias específicas basadas en clasificación semántica
            if semantic_classification and column in semantic_classification:
                semantic_type = semantic_classification[column].get('type', '')
                
                if semantic_type == 'demographic':
                    column_suggestions.extend(['bar_chart', 'pie_chart'])
                elif semantic_type == 'temporal':
                    column_suggestions.extend(['line_plot', 'time_series'])
                elif semantic_type == 'geographic':
                    column_suggestions.extend(['map', 'choropleth'])
        
        elif df[column].dtype == 'object':
            unique_count = df[column].nunique()
            
            if unique_count <= 10:
                column_suggestions.extend(['bar_chart', 'pie_chart'])
            else:
                column_suggestions.extend(['word_cloud', 'frequency_plot'])
        
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            column_suggestions.extend(['line_plot', 'time_series', 'calendar_heatmap'])
        
        suggestions[column] = column_suggestions
    
    return suggestions


class SchemaValidator:
    """
    Clase para validación de esquemas de datos.
    """
    
    def __init__(self):
        """Inicializa el SchemaValidator."""
        self.validation_results = {}
    
    def validate(self, df: pd.DataFrame, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Valida un DataFrame contra un esquema.
        
        Args:
            df: DataFrame a validar
            schema: Esquema de validación (opcional)
            
        Returns:
            Diccionario con resultados de validación
        """
        if schema is None:
            # Validación básica sin esquema específico
            return self._validate_basic(df)
        
        return self._validate_with_schema(df, schema)
    
    def _validate_basic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validación básica del DataFrame.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            Resultados de validación
        """
        results = {}
        
        for column in df.columns:
            column_results = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'details': {}
            }
            
            # Verificar tipo de datos
            dtype = str(df[column].dtype)
            column_results['details']['dtype'] = dtype
            
            # Verificar valores únicos
            unique_count = df[column].nunique()
            column_results['details']['unique_count'] = unique_count
            
            # Verificar valores faltantes
            missing_count = df[column].isnull().sum()
            missing_ratio = missing_count / len(df)
            column_results['details']['missing_count'] = missing_count
            column_results['details']['missing_ratio'] = missing_ratio
            
            # Advertencias
            if missing_ratio > 0.5:
                column_results['warnings'].append(f"High missing ratio: {missing_ratio:.2%}")
            
            if unique_count == 1:
                column_results['warnings'].append("Column has only one unique value")
            
            if unique_count == len(df):
                column_results['warnings'].append("Column has all unique values")
            
            results[column] = column_results
        
        return results
    
    def _validate_with_schema(self, df: pd.DataFrame, schema: Dict) -> Dict[str, Any]:
        """
        Validación con esquema específico.
        
        Args:
            df: DataFrame a validar
            schema: Esquema de validación
            
        Returns:
            Resultados de validación
        """
        results = {}
        
        # Obtener validación básica
        basic_results = self._validate_basic(df)
        
        # Aplicar validaciones específicas del esquema
        for column, column_schema in schema.get('columns', {}).items():
            if column not in df.columns:
                results[column] = {
                    'is_valid': False,
                    'errors': [f"Column {column} not found in DataFrame"],
                    'warnings': [],
                    'details': {}
                }
                continue
            
            # Combinar validación básica con específica
            column_results = basic_results.get(column, {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'details': {}
            })
            
            # Validar tipo de datos
            expected_type = column_schema.get('type')
            if expected_type:
                actual_type = str(df[column].dtype)
                if not self._validate_type(actual_type, expected_type):
                    column_results['errors'].append(f"Expected type {expected_type}, got {actual_type}")
                    column_results['is_valid'] = False
            
            # Validar rango
            if 'range' in column_schema:
                range_errors = self._validate_range(df[column], column_schema['range'])
                column_results['errors'].extend(range_errors)
                if range_errors:
                    column_results['is_valid'] = False
            
            # Validar valores permitidos
            if 'allowed_values' in column_schema:
                allowed_errors = self._validate_allowed_values(df[column], column_schema['allowed_values'])
                column_results['errors'].extend(allowed_errors)
                if allowed_errors:
                    column_results['is_valid'] = False
            
            results[column] = column_results
        
        # Agregar columnas no especificadas en el esquema
        for column in df.columns:
            if column not in results:
                results[column] = basic_results.get(column, {
                    'is_valid': True,
                    'errors': [],
                    'warnings': [],
                    'details': {}
                })
        
        return results
    
    def _validate_type(self, actual_type: str, expected_type: str) -> bool:
        """
        Valida el tipo de datos.
        
        Args:
            actual_type: Tipo actual
            expected_type: Tipo esperado
            
        Returns:
            True si el tipo es válido
        """
        # Mapeo de tipos
        type_mapping = {
            'int': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32'],
            'string': ['object', 'string'],
            'bool': ['bool'],
            'datetime': ['datetime64[ns]', 'datetime64[us]', 'datetime64[ms]']
        }
        
        expected_types = type_mapping.get(expected_type, [expected_type])
        return actual_type in expected_types
    
    def _validate_range(self, series: pd.Series, range_config: Dict) -> List[str]:
        """
        Valida el rango de valores.
        
        Args:
            series: Serie a validar
            range_config: Configuración del rango
            
        Returns:
            Lista de errores
        """
        errors = []
        
        min_val = range_config.get('min')
        max_val = range_config.get('max')
        
        if min_val is not None:
            invalid_min = series < min_val
            if invalid_min.any():
                errors.append(f"Values below minimum {min_val}: {invalid_min.sum()} rows")
        
        if max_val is not None:
            invalid_max = series > max_val
            if invalid_max.any():
                errors.append(f"Values above maximum {max_val}: {invalid_max.sum()} rows")
        
        return errors
    
    def _validate_allowed_values(self, series: pd.Series, allowed_values: List) -> List[str]:
        """
        Valida valores permitidos.
        
        Args:
            series: Serie a validar
            allowed_values: Lista de valores permitidos
            
        Returns:
            Lista de errores
        """
        errors = []
        
        invalid_values = ~series.isin(allowed_values)
        if invalid_values.any():
            errors.append(f"Invalid values found: {invalid_values.sum()} rows")
        
        return errors
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de la validación.
        
        Returns:
            Diccionario con resumen de validación
        """
        if not self.validation_results:
            return {"total_columns": 0, "valid_columns": 0, "invalid_columns": 0}
        
        total_columns = len(self.validation_results)
        valid_columns = sum(1 for result in self.validation_results.values() if result.get('is_valid', False))
        invalid_columns = total_columns - valid_columns
        
        return {
            "total_columns": total_columns,
            "valid_columns": valid_columns,
            "invalid_columns": invalid_columns,
            "success_rate": valid_columns / total_columns if total_columns > 0 else 0
        }
