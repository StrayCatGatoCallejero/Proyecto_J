"""
Test script for survey analysis features - Simple version.
Tests the new survey structure analysis, frequency tables, and textual summaries.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import re
from collections import Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_survey_data(n_samples=100):
    """Generate synthetic social sciences survey data."""
    print("🔬 Generando datos sintéticos de encuesta...")
    
    np.random.seed(42)
    random.seed(42)
    
    # Demographic variables
    ages = np.random.normal(35, 12, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)
    
    genders = np.random.choice(['Masculino', 'Femenino', 'No binario', 'Prefiero no decir'], 
                              n_samples, p=[0.45, 0.45, 0.05, 0.05])
    
    education_levels = np.random.choice([
        'Sin estudios', 'Primaria', 'Secundaria', 'Técnico', 'Universitario', 'Postgrado'
    ], n_samples, p=[0.05, 0.15, 0.30, 0.20, 0.25, 0.05])
    
    # Likert scale variables
    satisfaction_work = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.10, 0.15, 0.25, 0.30, 0.20])
    satisfaction_life = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.10, 0.20, 0.35, 0.30])
    
    # Categorical variables
    political_orientation = np.random.choice([
        'Izquierda', 'Centro-izquierda', 'Centro', 'Centro-derecha', 'Derecha', 'No identificado'
    ], n_samples, p=[0.20, 0.25, 0.20, 0.15, 0.10, 0.10])
    
    # Text variables
    open_opinion = []
    for i in range(n_samples):
        templates = [
            "Creo que el gobierno debería enfocarse más en {topic}.",
            "Mi experiencia con {topic} ha sido {sentiment}.",
            "Es importante que se mejore {topic} para el bienestar de todos.",
            "No estoy satisfecho con el estado actual de {topic}.",
            "Me gustaría ver más inversión en {topic}."
        ]
        
        topics = ['la educación pública', 'el sistema de salud', 'la seguridad ciudadana', 
                 'el transporte público', 'la vivienda', 'el empleo']
        sentiments = ['muy positiva', 'positiva', 'regular', 'negativa', 'muy negativa']
        
        template = random.choice(templates)
        topic = random.choice(topics)
        sentiment = random.choice(sentiments)
        
        response = template.format(topic=topic, sentiment=sentiment)
        open_opinion.append(response)
    
    # Create DataFrame
    data = {
        'id_encuesta': range(1, n_samples + 1),
        'edad': ages,
        'genero': genders,
        'nivel_educacion': education_levels,
        'satisfaccion_trabajo': satisfaction_work,
        'satisfaccion_vida': satisfaction_life,
        'orientacion_politica': political_orientation,
        'opinion_abierta': open_opinion
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    missing_cols = ['orientacion_politica', 'opinion_abierta']
    for col in missing_cols:
        missing_indices = np.random.choice(df.index, size=int(n_samples * 0.05), replace=False)
        df.loc[missing_indices, col] = np.nan
    
    print(f"✅ Datos generados: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df

def summarize_survey_structure(df, metadata):
    """Analyze survey structure and provide comprehensive summary."""
    logger.info("Analyzing survey structure")
    
    semantic_types = metadata.get('semantic_types', {})
    
    # Analyze each column
    columns_analysis = []
    type_counts = {}
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        semantic_type = semantic_types.get(col, 'unknown')
        
        columns_analysis.append({
            'column': col,
            'semantic_type': semantic_type,
            'dtype': dtype,
            'missing_pct': round(missing_pct, 1),
            'unique_values': df[col].nunique(),
            'sample_values': list(df[col].dropna().unique()[:3])
        })
        
        # Count by semantic type
        if semantic_type not in type_counts:
            type_counts[semantic_type] = 0
        type_counts[semantic_type] += 1
    
    # Generate narrative summary
    total_questions = len(df.columns)
    total_missing = sum(1 for col in columns_analysis if col['missing_pct'] > 0)
    avg_missing = sum(col['missing_pct'] for col in columns_analysis) / total_questions
    
    # Build narrative text
    narrative_parts = [f"Esta encuesta tiene {total_questions} preguntas"]
    
    if type_counts:
        type_descriptions = []
        for semantic_type, count in type_counts.items():
            if semantic_type == 'demographic':
                type_descriptions.append(f"{count} demográfica{'s' if count > 1 else ''}")
            elif semantic_type == 'likert':
                type_descriptions.append(f"{count} de escala Likert")
            elif semantic_type == 'categorical':
                type_descriptions.append(f"{count} de opción múltiple")
            elif semantic_type == 'text':
                type_descriptions.append(f"{count} abierta{'s' if count > 1 else ''}")
            elif semantic_type == 'numeric':
                type_descriptions.append(f"{count} numérica{'s' if count > 1 else ''}")
            else:
                type_descriptions.append(f"{count} {semantic_type}")
        
        narrative_parts.append(": " + ", ".join(type_descriptions) + ".")
    
    if total_missing > 0:
        narrative_parts.append(f" El {avg_missing:.1f}% de los campos están vacíos en promedio.")
    
    narrative = "".join(narrative_parts)
    
    return {
        'narrative': narrative,
        'total_questions': total_questions,
        'type_counts': type_counts,
        'columns_analysis': columns_analysis,
        'avg_missing_pct': round(avg_missing, 1),
        'total_missing_columns': total_missing
    }

def frequency_table(df, col):
    """Generate frequency table for categorical or Likert variables."""
    logger.info(f"Generating frequency table for column: {col}")
    
    if col not in df.columns:
        logger.warning(f"Column {col} not found in DataFrame")
        return pd.DataFrame()
    
    # Get value counts
    value_counts = df[col].value_counts(dropna=False)
    total = len(df)
    
    # Calculate percentages
    percentages = (value_counts / total) * 100
    
    # Create frequency table
    freq_table = pd.DataFrame({
        'Valor': value_counts.index,
        'Frecuencia': value_counts.values,
        'Porcentaje': percentages.round(1),
        'Porcentaje_Acumulado': percentages.cumsum().round(1)
    })
    
    # Handle missing values
    if freq_table['Valor'].isna().any():
        freq_table.loc[freq_table['Valor'].isna(), 'Valor'] = 'Valor faltante'
    
    return freq_table

def crosstab_summary(df, col):
    """Generate textual summary for categorical variable."""
    logger.info(f"Generating crosstab summary for column: {col}")
    
    if col not in df.columns:
        return "Columna no encontrada en el dataset."
    
    freq_table = frequency_table(df, col)
    
    if freq_table.empty:
        return "No hay datos disponibles para generar resumen."
    
    # Get top 3 most frequent values
    top_values = freq_table.head(3)
    
    summary_parts = []
    
    # Most frequent value
    most_freq = top_values.iloc[0]
    summary_parts.append(f"La opción '{most_freq['Valor']}' fue la más frecuente ({most_freq['Porcentaje']}%)")
    
    # Second most frequent
    if len(top_values) > 1:
        second_freq = top_values.iloc[1]
        summary_parts.append(f", seguida de '{second_freq['Valor']}' ({second_freq['Porcentaje']}%)")
    
    # Third most frequent
    if len(top_values) > 2:
        third_freq = top_values.iloc[2]
        summary_parts.append(f" y '{third_freq['Valor']}' ({third_freq['Porcentaje']}%)")
    
    summary_parts.append(".")
    
    # Add missing data info if any
    missing_row = freq_table[freq_table['Valor'] == 'Valor faltante']
    if not missing_row.empty:
        missing_pct = missing_row.iloc[0]['Porcentaje']
        summary_parts.append(f" El {missing_pct}% de los datos están faltantes.")
    
    return "".join(summary_parts)

def textual_summary(df, col):
    """Generate comprehensive textual analysis for text columns."""
    logger.info(f"Generating textual summary for column: {col}")
    
    if col not in df.columns:
        return {"error": "Columna no encontrada"}
    
    # Get text data
    text_data = df[col].dropna().astype(str)
    
    if text_data.empty:
        return {"error": "No hay datos de texto disponibles"}
    
    # Word frequency analysis
    all_words = []
    for text in text_data:
        # Simple tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    top_words = word_freq.most_common(20)
    
    # Simple sentiment analysis
    positive_words = ['bueno', 'excelente', 'mejor', 'positivo', 'satisfecho', 'contento', 'feliz', 'agradable']
    negative_words = ['malo', 'terrible', 'peor', 'negativo', 'insatisfecho', 'triste', 'molesto', 'desagradable']
    
    positive_count = sum(1 for word in all_words if word in positive_words)
    negative_count = sum(1 for word in all_words if word in negative_words)
    total_words = len(all_words)
    
    if total_words > 0:
        positive_pct = (positive_count / total_words) * 100
        negative_pct = (negative_count / total_words) * 100
        neutral_pct = 100 - positive_pct - negative_pct
    else:
        positive_pct = negative_pct = neutral_pct = 0
    
    # Find representative sentences
    representative_sentences = []
    if len(text_data) > 0:
        sentences = []
        for text in text_data:
            if len(text) > 10:
                sentences.append(text)
        
        sentences.sort(key=len, reverse=True)
        representative_sentences = sentences[:3]
    
    return {
        'word_cloud': dict(top_words),
        'sentiment': {
            'positive_pct': round(positive_pct, 1),
            'negative_pct': round(negative_pct, 1),
            'neutral_pct': round(neutral_pct, 1),
            'total_words': total_words
        },
        'representative_sentences': representative_sentences,
        'total_responses': len(text_data),
        'avg_response_length': round(np.mean([len(str(text)) for text in text_data]), 1)
    }

def generate_data_dictionary(df, metadata):
    """Generate comprehensive data dictionary for the survey."""
    logger.info("Generating data dictionary")
    
    semantic_types = metadata.get('semantic_types', {})
    
    dictionary_rows = []
    
    for col in df.columns:
        semantic_type = semantic_types.get(col, 'unknown')
        dtype = str(df[col].dtype)
        
        # Get sample values
        sample_values = df[col].dropna().unique()[:5]
        sample_str = ", ".join([str(val) for val in sample_values])
        
        # Generate description based on semantic type
        if semantic_type == 'demographic':
            description = "Variable demográfica que describe características básicas de la población"
        elif semantic_type == 'likert':
            description = "Escala de Likert para medir actitudes o percepciones"
        elif semantic_type == 'categorical':
            description = "Variable categórica con opciones de respuesta predefinidas"
        elif semantic_type == 'text':
            description = "Respuesta de texto libre o abierta"
        elif semantic_type == 'numeric':
            description = "Variable numérica para análisis cuantitativo"
        else:
            description = "Variable de tipo no especificado"
        
        # Generate treatment suggestions
        if semantic_type == 'demographic':
            treatment = "Usar para segmentación y análisis demográfico"
        elif semantic_type == 'likert':
            treatment = "Analizar con tablas de frecuencia y correlaciones"
        elif semantic_type == 'categorical':
            treatment = "Crear tablas de contingencia y análisis de asociación"
        elif semantic_type == 'text':
            treatment = "Análisis de contenido y minería de texto"
        elif semantic_type == 'numeric':
            treatment = "Análisis estadístico descriptivo e inferencial"
        else:
            treatment = "Revisar y clasificar según el contenido"
        
        dictionary_rows.append({
            'Variable': col,
            'Tipo_Semantico': semantic_type,
            'Tipo_Datos': dtype,
            'Descripcion': description,
            'Valores_Ejemplo': sample_str,
            'Valores_Unicos': df[col].nunique(),
            'Valores_Faltantes': df[col].isnull().sum(),
            'Sugerencia_Tratamiento': treatment
        })
    
    return pd.DataFrame(dictionary_rows)

def test_survey_features():
    """Test all survey analysis features."""
    print("🚀 Iniciando pruebas de funcionalidades de encuesta...")
    
    # Generate data
    df = generate_synthetic_survey_data(100)
    
    # Define metadata
    metadata = {
        'semantic_types': {
            'edad': 'demographic',
            'genero': 'categorical',
            'nivel_educacion': 'categorical',
            'satisfaccion_trabajo': 'likert',
            'satisfaccion_vida': 'likert',
            'orientacion_politica': 'categorical',
            'opinion_abierta': 'text'
        }
    }
    
    print("\n" + "="*60)
    print("📋 PRUEBA DE ANÁLISIS DE ESTRUCTURA DE ENCUESTA")
    print("="*60)
    
    # Test survey structure analysis
    structure = summarize_survey_structure(df, metadata)
    print(f"✅ Estructura analizada: {structure['narrative']}")
    print(f"   Total preguntas: {structure['total_questions']}")
    print(f"   Promedio datos faltantes: {structure['avg_missing_pct']}%")
    
    if 'type_counts' in structure:
        print("   Distribución por tipo:")
        for sem_type, count in structure['type_counts'].items():
            print(f"     - {sem_type}: {count}")
    
    print("\n" + "="*60)
    print("📊 PRUEBA DE TABLAS DE FRECUENCIA")
    print("="*60)
    
    # Test frequency tables
    categorical_cols = ['genero', 'nivel_educacion', 'orientacion_politica']
    for col in categorical_cols:
        freq_table = frequency_table(df, col)
        print(f"✅ Tabla de frecuencia para '{col}': {len(freq_table)} filas")
        
        summary = crosstab_summary(df, col)
        print(f"   Resumen: {summary}")
    
    print("\n" + "="*60)
    print("📝 PRUEBA DE ANÁLISIS DE TEXTO")
    print("="*60)
    
    # Test textual analysis
    text_analysis = textual_summary(df, 'opinion_abierta')
    if 'error' not in text_analysis:
        sentiment = text_analysis['sentiment']
        print(f"✅ Análisis de sentimiento:")
        print(f"   Positivo: {sentiment['positive_pct']}%")
        print(f"   Negativo: {sentiment['negative_pct']}%")
        print(f"   Neutral: {sentiment['neutral_pct']}%")
        print(f"   Total palabras: {sentiment['total_words']}")
        print(f"   Respuestas analizadas: {text_analysis['total_responses']}")
        
        if text_analysis['representative_sentences']:
            print("   Frases representativas:")
            for i, sentence in enumerate(text_analysis['representative_sentences'], 1):
                print(f"     {i}. {sentence[:100]}...")
    else:
        print(f"❌ Error en análisis de texto: {text_analysis['error']}")
    
    print("\n" + "="*60)
    print("📚 PRUEBA DE DICCIONARIO DE DATOS")
    print("="*60)
    
    # Test data dictionary
    data_dict = generate_data_dictionary(df, metadata)
    print(f"✅ Diccionario generado: {len(data_dict)} variables")
    print("\nPrimeras 3 variables del diccionario:")
    print(data_dict.head(3).to_string(index=False))
    
    print("\n" + "="*60)
    print("🎉 TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_survey_features() 