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
warnings.filterwarnings('ignore')

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
            type_mapping[column] = 'empty'
            continue
        
        # Detectar tipo numérico
        if pd.api.types.is_numeric_dtype(df[column]):
            if df[column].dtype in ['int64', 'int32']:
                type_mapping[column] = 'numeric_integer'
            else:
                type_mapping[column] = 'numeric_float'
        
        # Detectar tipo datetime
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            type_mapping[column] = 'datetime'
        
        # Detectar tipo booleano
        elif pd.api.types.is_bool_dtype(df[column]):
            type_mapping[column] = 'boolean'
        
        # Detectar tipo categórico
        elif df[column].dtype == 'object' or df[column].dtype == 'category':
            unique_values = col_data.nunique()
            total_values = len(col_data)
            
            # Si tiene pocos valores únicos, es categórico
            if unique_values <= min(20, total_values * 0.1):
                type_mapping[column] = 'categorical'
            else:
                # Verificar si es texto libre
                avg_length = col_data.astype(str).str.len().mean()
                if avg_length > 50:
                    type_mapping[column] = 'text_free'
                else:
                    type_mapping[column] = 'text_short'
    
    return type_mapping

def semantic_classification(
    df: pd.DataFrame, 
    custom_keywords: Dict[str, List[str]] = None
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
        'demographic': [
            'edad', 'age', 'genero', 'sexo', 'gender', 'sex', 'nacionalidad', 
            'nacionality', 'estado_civil', 'marital', 'civil', 'educacion', 
            'education', 'ocupacion', 'occupation', 'profesion', 'profession'
        ],
        'socioeconomic': [
            'ingresos', 'income', 'salario', 'salary', 'renta', 'rent', 
            'pobreza', 'poverty', 'vivienda', 'housing', 'servicios', 'services',
            'empleo', 'employment', 'trabajo', 'work', 'desempleo', 'unemployment'
        ],
        'opinion': [
            'satisfaccion', 'satisfaction', 'confianza', 'confidence', 
            'percepcion', 'perception', 'actitud', 'attitude', 'opinion',
            'preferencia', 'preference', 'gusto', 'like', 'disgusto', 'dislike'
        ],
        'likert': [
            'acuerdo', 'agreement', 'frecuencia', 'frequency', 'importancia',
            'importance', 'satisfaccion_escala', 'satisfaction_scale',
            'muy', 'very', 'bastante', 'quite', 'poco', 'little', 'nada', 'nothing'
        ],
        'temporal': [
            'fecha', 'date', 'tiempo', 'time', 'periodo', 'period', 'año', 'year',
            'mes', 'month', 'dia', 'day', 'hora', 'hour', 'minuto', 'minute'
        ],
        'geographic': [
            'region', 'region', 'comuna', 'commune', 'provincia', 'province',
            'pais', 'country', 'ciudad', 'city', 'direccion', 'address'
        ],
        'health': [
            'salud', 'health', 'enfermedad', 'disease', 'sintoma', 'symptom',
            'medicamento', 'medication', 'tratamiento', 'treatment'
        ]
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
        scores = {}
        
        # Calcular puntuación para cada categoría
        for category, keywords in base_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in column_lower:
                    score += 1
                # También verificar en los valores únicos
                unique_values = df[column].dropna().astype(str).str.lower().unique()
                for value in unique_values[:10]:  # Solo primeros 10 valores
                    if keyword in value:
                        score += 0.5
            
            scores[category] = score
        
        # Determinar la categoría principal
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])
            confidence = best_category[1] / max(len(base_keywords[best_category[0]]), 1)
            
            classifications[column] = {
                'category': best_category[0] if best_category[1] > 0 else 'unknown',
                'confidence': confidence,
                'scores': scores
            }
        else:
            classifications[column] = {
                'category': 'unknown',
                'confidence': 0.0,
                'scores': {}
            }
    
    return classifications

def detect_likert_scales(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detecta automáticamente escalas de Likert en el DataFrame.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Diccionario con información de escalas Likert detectadas
    """
    likert_patterns = {
        'agreement': [
            r'muy\s+de\s+acuerdo', r'de\s+acuerdo', r'neutral', r'en\s+desacuerdo', 
            r'muy\s+en\s+desacuerdo', r'strongly\s+agree', r'agree', r'neutral', 
            r'disagree', r'strongly\s+disagree'
        ],
        'frequency': [
            r'siempre', r'frecuentemente', r'a\s+veces', r'rara\s+vez', r'nunca',
            r'always', r'frequently', r'sometimes', r'rarely', r'never'
        ],
        'importance': [
            r'muy\s+importante', r'importante', r'neutral', r'poco\s+importante',
            r'nada\s+importante', r'very\s+important', r'important', r'neutral',
            r'not\s+important', r'not\s+important\s+at\s+all'
        ],
        'satisfaction': [
            r'muy\s+satisfecho', r'satisfecho', r'neutral', r'insatisfecho',
            r'muy\s+insatisfecho', r'very\s+satisfied', r'satisfied', r'neutral',
            r'dissatisfied', r'very\s+dissatisfied'
        ]
    }
    
    likert_detections = {}
    
    for column in df.columns:
        if df[column].dtype == 'object':
            unique_values = df[column].dropna().astype(str).str.lower().unique()
            
            for scale_type, patterns in likert_patterns.items():
                matches = 0
                for pattern in patterns:
                    for value in unique_values:
                        if re.search(pattern, value, re.IGNORECASE):
                            matches += 1
                
                # Si encuentra al menos 3 patrones, es probable una escala Likert
                if matches >= 3:
                    likert_detections[column] = {
                        'scale_type': scale_type,
                        'confidence': matches / len(patterns),
                        'unique_values': unique_values.tolist(),
                        'pattern_matches': matches
                    }
                    break
    
    return likert_detections

def normalize_categorical_levels(
    df: pd.DataFrame, 
    similarity_threshold: float = 0.8
) -> Dict[str, Dict[str, Any]]:
    """
    Normaliza y unifica niveles categóricos usando coincidencia difusa.
    
    Args:
        df: DataFrame a normalizar
        similarity_threshold: Umbral de similitud para agrupar valores
        
    Returns:
        Diccionario con sugerencias de normalización
    """
    normalization_suggestions = {}
    
    for column in df.select_dtypes(include=['object']).columns:
        unique_values = df[column].dropna().unique()
        
        if len(unique_values) <= 5:  # No normalizar si hay pocos valores
            continue
        
        # Agrupar valores similares
        groups = []
        used_values = set()
        
        for value in unique_values:
            if value in used_values:
                continue
            
            group = [value]
            used_values.add(value)
            
            for other_value in unique_values:
                if other_value not in used_values:
                    similarity = SequenceMatcher(None, str(value).lower(), 
                                               str(other_value).lower()).ratio()
                    if similarity >= similarity_threshold:
                        group.append(other_value)
                        used_values.add(other_value)
            
            if len(group) > 1:  # Solo reportar grupos con múltiples valores
                groups.append(group)
        
        if groups:
            normalization_suggestions[column] = {
                'groups': groups,
                'suggested_standard': [group[0] for group in groups],
                'total_groups': len(groups),
                'values_to_review': sum(len(group) for group in groups)
            }
    
    return normalization_suggestions

def detect_measurement_units(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detecta unidades de medida contextuales en las columnas.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Diccionario con unidades detectadas
    """
    unit_patterns = {
        'time': {
            'years': r'\b(años?|years?)\b',
            'months': r'\b(meses?|months?)\b',
            'weeks': r'\b(semanas?|weeks?)\b',
            'days': r'\b(días?|days?)\b',
            'hours': r'\b(horas?|hours?)\b',
            'minutes': r'\b(minutos?|minutes?)\b'
        },
        'distance': {
            'kilometers': r'\b(km|kilómetros?|kilometers?)\b',
            'meters': r'\b(m|metros?|meters?)\b',
            'miles': r'\b(millas?|miles?)\b'
        },
        'currency': {
            'clp': r'\b(clp|pesos?|chilean\s+pesos?)\b',
            'usd': r'\b(usd|\$|dollars?|dólares?)\b',
            'eur': r'\b(eur|euros?)\b'
        },
        'weight': {
            'kg': r'\b(kg|kilogramos?|kilograms?)\b',
            'g': r'\b(g|gramos?|grams?)\b',
            'lbs': r'\b(lbs|pounds?|libras?)\b'
        }
    }
    
    unit_detections = {}
    
    for column in df.columns:
        column_lower = column.lower()
        detected_units = {}
        
        for unit_category, patterns in unit_patterns.items():
            for unit_name, pattern in patterns.items():
                if re.search(pattern, column_lower, re.IGNORECASE):
                    detected_units[unit_category] = unit_name
                    break
        
        # También verificar en los valores únicos
        if df[column].dtype == 'object':
            unique_values = df[column].dropna().astype(str).str.lower().unique()
            for value in unique_values[:5]:  # Solo primeros 5 valores
                for unit_category, patterns in unit_patterns.items():
                    for unit_name, pattern in patterns.items():
                        if re.search(pattern, value, re.IGNORECASE):
                            detected_units[unit_category] = unit_name
                            break
        
        if detected_units:
            unit_detections[column] = {
                'detected_units': detected_units,
                'confidence': 'high' if len(detected_units) > 1 else 'medium'
            }
    
    return unit_detections

def analyze_free_text(
    df: pd.DataFrame, 
    text_columns: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analiza texto libre: tokenización, conteo de palabras, sentimientos.
    
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
                avg_length = df[column].dropna().astype(str).str.len().mean()
                if avg_length > 20:  # Considerar texto libre si promedio > 20 caracteres
                    text_columns.append(column)
    
    text_analysis = {}
    
    for column in text_columns:
        if column not in df.columns:
            continue
        
        col_data = df[column].dropna()
        if len(col_data) == 0:
            continue
        
        # Análisis básico
        analysis = {
            'total_entries': len(col_data),
            'avg_length': col_data.astype(str).str.len().mean(),
            'max_length': col_data.astype(str).str.len().max(),
            'min_length': col_data.astype(str).str.len().min(),
            'unique_entries': col_data.nunique(),
            'duplicate_entries': len(col_data) - col_data.nunique()
        }
        
        # Tokenización básica
        all_text = ' '.join(col_data.astype(str))
        words = re.findall(r'\b\w+\b', all_text.lower())
        
        if words:
            analysis['total_words'] = len(words)
            analysis['unique_words'] = len(set(words))
            analysis['avg_words_per_entry'] = len(words) / len(col_data)
            
            # Palabras más frecuentes
            from collections import Counter
            word_counts = Counter(words)
            analysis['most_common_words'] = word_counts.most_common(10)
        
        # Detección de sentimientos básica (palabras positivas/negativas)
        positive_words = ['bueno', 'excelente', 'mejor', 'positivo', 'satisfecho', 
                         'good', 'excellent', 'better', 'positive', 'satisfied']
        negative_words = ['malo', 'terrible', 'peor', 'negativo', 'insatisfecho',
                         'bad', 'terrible', 'worse', 'negative', 'dissatisfied']
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        analysis['sentiment'] = {
            'positive_words': positive_count,
            'negative_words': negative_count,
            'sentiment_ratio': positive_count / (negative_count + 1)  # Evitar división por cero
        }
        
        text_analysis[column] = analysis
    
    return text_analysis

def validate_internal_consistency(
    df: pd.DataFrame, 
    rules: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Valida consistencia interna usando reglas lógicas entre variables.
    
    Args:
        df: DataFrame a validar
        rules: Lista de reglas de consistencia (opcional)
        
    Returns:
        Diccionario con resultados de validación
    """
    if rules is None:
        # Reglas por defecto
        rules = [
            {
                'name': 'age_marital_consistency',
                'condition': 'edad < 18 and estado_civil in ["casado", "divorciado"]',
                'description': 'Personas menores de 18 no pueden estar casadas o divorciadas'
            },
            {
                'name': 'age_education_consistency',
                'condition': 'edad < 6 and educacion > 12',
                'description': 'Personas menores de 6 años no pueden tener más de 12 años de educación'
            },
            {
                'name': 'income_education_consistency',
                'condition': 'educacion < 8 and ingresos > 5000000',
                'description': 'Personas con menos de 8 años de educación no deberían tener ingresos muy altos'
            }
        ]
    
    validation_results = {
        'total_rules': len(rules),
        'passed_rules': 0,
        'failed_rules': 0,
        'rule_details': []
    }
    
    for rule in rules:
        try:
            # Evaluar condición
            condition = rule['condition']
            mask = df.eval(condition)
            violations = mask.sum()
            
            rule_result = {
                'rule_name': rule['name'],
                'description': rule['description'],
                'violations': int(violations),
                'violation_rate': float(violations / len(df)) if len(df) > 0 else 0,
                'status': 'passed' if violations == 0 else 'failed'
            }
            
            validation_results['rule_details'].append(rule_result)
            
            if violations == 0:
                validation_results['passed_rules'] += 1
            else:
                validation_results['failed_rules'] += 1
                
        except Exception as e:
            rule_result = {
                'rule_name': rule['name'],
                'description': rule['description'],
                'error': str(e),
                'status': 'error'
            }
            validation_results['rule_details'].append(rule_result)
    
    return validation_results

def suggest_visualizations(
    df: pd.DataFrame, 
    semantic_classification: Dict[str, Any] = None
) -> Dict[str, List[str]]:
    """
    Sugiere visualizaciones basadas en tipos semánticos de columnas.
    
    Args:
        df: DataFrame a analizar
        semantic_classification: Clasificación semántica previa
        
    Returns:
        Diccionario con sugerencias de visualización por columna
    """
    suggestions = {}
    
    for column in df.columns:
        column_suggestions = []
        
        # Obtener tipo de datos
        dtype = df[column].dtype
        
        # Sugerencias basadas en tipo de datos
        if pd.api.types.is_numeric_dtype(dtype):
            column_suggestions.extend(['histogram', 'box_plot', 'scatter_plot'])
        elif dtype == 'object' or dtype == 'category':
            unique_count = df[column].nunique()
            if unique_count <= 10:
                column_suggestions.extend(['bar_chart', 'pie_chart'])
            else:
                column_suggestions.extend(['horizontal_bar', 'word_cloud'])
        
        # Sugerencias basadas en clasificación semántica
        if semantic_classification and column in semantic_classification:
            category = semantic_classification[column]['category']
            
            if category == 'demographic':
                column_suggestions.extend(['bar_chart', 'pie_chart', 'horizontal_bar'])
            elif category == 'socioeconomic':
                column_suggestions.extend(['histogram', 'box_plot', 'scatter_plot'])
            elif category == 'opinion':
                column_suggestions.extend(['bar_chart', 'horizontal_bar', 'diverging_bar'])
            elif category == 'likert':
                column_suggestions.extend(['stacked_bar', 'horizontal_bar', 'diverging_bar'])
            elif category == 'temporal':
                column_suggestions.extend(['line_chart', 'area_chart', 'heatmap'])
            elif category == 'geographic':
                column_suggestions.extend(['choropleth_map', 'bar_chart'])
            elif category == 'text_free':
                column_suggestions.extend(['word_cloud', 'bar_chart', 'treemap'])
        
        # Eliminar duplicados y ordenar
        suggestions[column] = list(dict.fromkeys(column_suggestions))
    
    return suggestions

class SchemaValidator:
    """
    Validador de esquema para DataFrames de ciencias sociales.
    Permite definir tipos esperados, rangos, patrones, formatos de fecha y reglas de validación.
    Soporta autocorrección y reporte detallado de errores.
    """
    def __init__(self):
        pass

    def validate_schema(self, df: pd.DataFrame, schema: dict) -> dict:
        """
        Valida el DataFrame contra un esquema definido.
        Args:
            df: DataFrame a validar
            schema: Diccionario con especificaciones de columnas
        Returns:
            dict con claves: is_valid (bool), errors (list), autocorrected_df (DataFrame)
        """
        errors = []
        autocorrected_df = df.copy()
        for col, rules in schema.items():
            if col not in df.columns:
                errors.append({'column': col, 'error': 'missing_column'})
                continue
            series = df[col]
            # Tipo
            expected_type = rules.get('type')
            if expected_type:
                if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(series):
                    try:
                        autocorrected_df[col] = pd.to_numeric(series, errors='coerce')
                    except Exception:
                        errors.append({'column': col, 'error': 'type_conversion_failed'})
                elif expected_type == 'datetime' and not pd.api.types.is_datetime64_any_dtype(series):
                    try:
                        autocorrected_df[col] = pd.to_datetime(series, errors='coerce')
                    except Exception:
                        errors.append({'column': col, 'error': 'datetime_conversion_failed'})
                elif expected_type == 'categorical' and not pd.api.types.is_categorical_dtype(series):
                    autocorrected_df[col] = series.astype('category')
            # Rango
            if 'min' in rules or 'max' in rules:
                min_val = rules.get('min', -np.inf)
                max_val = rules.get('max', np.inf)
                invalid = ~series.between(min_val, max_val)
                if invalid.any():
                    errors.append({'column': col, 'error': 'out_of_range', 'count': invalid.sum()})
            # Regex
            if 'regex' in rules:
                pattern = re.compile(rules['regex'])
                invalid = ~series.astype(str).str.match(pattern)
                if invalid.any():
                    errors.append({'column': col, 'error': 'regex_mismatch', 'count': invalid.sum()})
            # Opciones
            if 'options' in rules:
                valid_options = set(rules['options'])
                invalid = ~series.isin(valid_options)
                if invalid.any():
                    errors.append({'column': col, 'error': 'invalid_option', 'count': invalid.sum()})
        is_valid = len(errors) == 0
        return {'is_valid': is_valid, 'errors': errors, 'autocorrected_df': autocorrected_df} 