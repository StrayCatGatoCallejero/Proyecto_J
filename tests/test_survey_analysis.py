"""
Test script for comprehensive survey analysis features.
Generates synthetic social sciences data and demonstrates all analysis capabilities.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
import sys
import pytest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from orchestrator.pipeline_orchestrator import PipelineOrchestrator
    from processing.logging import setup_logging
    from processing.stats import (
        summarize_survey_structure,
        frequency_table,
        crosstab_summary,
        textual_summary,
        generate_data_dictionary,
        can_generate_visualizations,
    )
    import yaml
    import chardet
except ImportError as e:
    pytest.skip(f"Falta dependencia para test_survey_analysis: {e}", allow_module_level=True)

# Setup logging
setup_logging()


def generate_synthetic_survey_data(n_samples=100):
    """
    Generate comprehensive synthetic social sciences survey data.

    Args:
        n_samples: Number of survey responses to generate

    Returns:
        DataFrame with synthetic survey data
    """
    print("🔬 Generando datos sintéticos de encuesta de ciencias sociales...")

    np.random.seed(42)
    random.seed(42)

    # Demographic variables
    ages = np.random.normal(35, 12, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)

    genders = np.random.choice(
        ["Masculino", "Femenino", "No binario", "Prefiero no decir"],
        n_samples,
        p=[0.45, 0.45, 0.05, 0.05],
    )

    education_levels = np.random.choice(
        [
            "Sin estudios",
            "Primaria",
            "Secundaria",
            "Técnico",
            "Universitario",
            "Postgrado",
        ],
        n_samples,
        p=[0.05, 0.15, 0.30, 0.20, 0.25, 0.05],
    )

    income_levels = np.random.choice(
        [
            "Menos de $300,000",
            "$300,000-$500,000",
            "$500,000-$800,000",
            "$800,000-$1,200,000",
            "Más de $1,200,000",
        ],
        n_samples,
        p=[0.20, 0.30, 0.25, 0.15, 0.10],
    )

    # Likert scale variables
    satisfaction_work = np.random.choice(
        [1, 2, 3, 4, 5], n_samples, p=[0.10, 0.15, 0.25, 0.30, 0.20]
    )
    satisfaction_life = np.random.choice(
        [1, 2, 3, 4, 5], n_samples, p=[0.05, 0.10, 0.20, 0.35, 0.30]
    )
    trust_government = np.random.choice(
        [1, 2, 3, 4, 5], n_samples, p=[0.25, 0.30, 0.25, 0.15, 0.05]
    )

    # Categorical variables
    political_orientation = np.random.choice(
        [
            "Izquierda",
            "Centro-izquierda",
            "Centro",
            "Centro-derecha",
            "Derecha",
            "No identificado",
        ],
        n_samples,
        p=[0.20, 0.25, 0.20, 0.15, 0.10, 0.10],
    )

    main_concern = np.random.choice(
        ["Economía", "Salud", "Educación", "Seguridad", "Medio ambiente", "Otros"],
        n_samples,
        p=[0.30, 0.20, 0.15, 0.15, 0.10, 0.10],
    )

    # Text variables
    open_opinion = []
    for i in range(n_samples):
        templates = [
            "Creo que el gobierno debería enfocarse más en {topic}.",
            "Mi experiencia con {topic} ha sido {sentiment}.",
            "Es importante que se mejore {topic} para el bienestar de todos.",
            "No estoy satisfecho con el estado actual de {topic}.",
            "Me gustaría ver más inversión en {topic}.",
        ]

        topics = [
            "la educación pública",
            "el sistema de salud",
            "la seguridad ciudadana",
            "el transporte público",
            "la vivienda",
            "el empleo",
        ]
        sentiments = ["muy positiva", "positiva", "regular", "negativa", "muy negativa"]

        template = random.choice(templates)
        topic = random.choice(topics)
        sentiment = random.choice(sentiments)

        response = template.format(topic=topic, sentiment=sentiment)
        open_opinion.append(response)

    # Temporal variables
    survey_date = datetime.now() - timedelta(days=random.randint(1, 30))
    survey_dates = [survey_date + timedelta(days=i) for i in range(n_samples)]

    # Geographic variables
    regions = np.random.choice(
        ["Metropolitana", "Valparaíso", "O'Higgins", "Maule", "Biobío", "Araucanía"],
        n_samples,
        p=[0.40, 0.15, 0.10, 0.10, 0.15, 0.10],
    )

    # Create DataFrame
    data = {
        "id_encuesta": range(1, n_samples + 1),
        "edad": ages,
        "genero": genders,
        "nivel_educacion": education_levels,
        "nivel_ingresos": income_levels,
        "satisfaccion_trabajo": satisfaction_work,
        "satisfaccion_vida": satisfaction_life,
        "confianza_gobierno": trust_government,
        "orientacion_politica": political_orientation,
        "preocupacion_principal": main_concern,
        "opinion_abierta": open_opinion,
        "fecha_encuesta": survey_dates,
        "region": regions,
    }

    df = pd.DataFrame(data)

    # Add some missing values
    missing_cols = ["nivel_ingresos", "orientacion_politica", "opinion_abierta"]
    for col in missing_cols:
        missing_indices = np.random.choice(
            df.index, size=int(n_samples * 0.05), replace=False
        )
        df.loc[missing_indices, col] = np.nan

    print(f"✅ Datos generados: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def test_individual_functions():
    """Test individual analysis functions."""
    print("\n" + "=" * 60)
    print("🔬 PRUEBA DE FUNCIONES INDIVIDUALES")
    print("=" * 60)

    # Generate data
    df = generate_synthetic_survey_data(50)

    # Test survey structure analysis
    print("\n📋 Probando análisis de estructura de encuesta...")

    metadata = {
        "semantic_types": {
            "edad": "demographic",
            "genero": "categorical",
            "nivel_educacion": "categorical",
            "satisfaccion_trabajo": "likert",
            "opinion_abierta": "text",
        }
    }

    structure = summarize_survey_structure(df, metadata)
    print(f"   ✅ Estructura analizada: {structure['narrative']}")

    # Test frequency tables
    print("\n📊 Probando tablas de frecuencia...")

    freq_table = frequency_table(df, "genero")
    print(f"   ✅ Tabla de frecuencia generada: {len(freq_table)} filas")

    summary = crosstab_summary(df, "genero")
    print(f"   ✅ Resumen categórico: {summary[:100]}...")

    # Test textual analysis
    print("\n📝 Probando análisis de texto...")

    text_analysis = textual_summary(df, "opinion_abierta")
    if "error" not in text_analysis:
        sentiment = text_analysis["sentiment"]
        print(f"   ✅ Análisis de sentimiento: {sentiment['positive_pct']}% positivo")
        print(
            f"   ✅ Palabras más frecuentes: {len(text_analysis['word_cloud'])} palabras"
        )
    else:
        print(f"   ❌ Error en análisis de texto: {text_analysis['error']}")

    # Test data dictionary
    print("\n📚 Probando generación de diccionario...")

    data_dict = generate_data_dictionary(df, metadata)
    print(f"   ✅ Diccionario generado: {len(data_dict)} variables")

    # Test visualization capability
    print("\n📊 Probando capacidad de visualización...")

    can_viz = can_generate_visualizations(df, metadata)
    print(f"   ✅ Puede generar visualizaciones: {can_viz}")

    print("\n✅ Todas las funciones individuales probadas exitosamente!")


def test_survey_analysis():
    """Test the complete survey analysis pipeline."""
    print("\n" + "=" * 60)
    print("🧪 PRUEBA COMPLETA DEL SISTEMA DE ANÁLISIS DE ENCUESTAS")
    print("=" * 60)

    # Generate synthetic data
    df = generate_synthetic_survey_data(150)

    # Save data temporarily
    temp_file = "temp_survey_data.csv"
    df.to_csv(temp_file, index=False)
    print(f"💾 Datos guardados en: {temp_file}")

    # Initialize orchestrator
    print("\n🔧 Inicializando orquestador...")
    orchestrator = PipelineOrchestrator()

    # Run full pipeline
    print("\n🚀 Ejecutando pipeline completo...")
    orchestrator.run_full_pipeline(temp_file)

    session_data = orchestrator.get_session_data()

    # Display results
    print("\n" + "=" * 60)
    print("📊 RESULTADOS DEL ANÁLISIS")
    print("=" * 60)

    # Basic info
    print(
        f"📋 Datos: {session_data.df.shape[0]} filas × {session_data.df.shape[1]} columnas"
    )
    print(
        f"🔍 Puede visualizar: {session_data.metadata.get('can_visualize', False)}"
    )

    # Survey structure
    if "survey_structure" in session_data.reports:
        structure = session_data.reports["survey_structure"]
        print(f"\n🏗️ Estructura de la encuesta:")
        print(f"   {structure['narrative']}")
        print(f"   Total preguntas: {structure['total_questions']}")
        print(f"   Promedio datos faltantes: {structure['avg_missing_pct']}%")

        if "type_counts" in structure:
            print(f"   Distribución por tipo:")
            for sem_type, count in structure["type_counts"].items():
                print(f"     - {sem_type}: {count}")

    # Frequency tables
    if "frequency_tables" in session_data.reports:
        freq_tables = session_data.reports["frequency_tables"]
        print(f"\n📊 Tablas de frecuencia generadas: {len(freq_tables)}")
        for col_name in freq_tables.keys():
            print(f"   - {col_name}")

    # Crosstab summaries
    if "crosstab_summaries" in session_data.reports:
        summaries = session_data.reports["crosstab_summaries"]
        print(f"\n📝 Resúmenes categóricos generados: {len(summaries)}")
        for col_name, summary in list(summaries.items())[:3]:  # Show first 3
            print(f"   - {col_name}: {summary[:100]}...")

    # Textual summaries
    if "textual_summaries" in session_data.reports:
        text_summaries = session_data.reports["textual_summaries"]
        print(f"\n📝 Análisis de texto generados: {len(text_summaries)}")
        for col_name, text_data in text_summaries.items():
            sentiment = text_data["sentiment"]
            print(
                f"   - {col_name}: {sentiment['positive_pct']}% positivo, "
                f"{sentiment['negative_pct']}% negativo, {sentiment['neutral_pct']}% neutral"
            )

    # Data dictionary
    if "data_dictionary" in session_data.reports:
        data_dict = session_data.reports["data_dictionary"]
        print(f"\n📚 Diccionario de datos generado: {len(data_dict)} variables")

    # Export results
    print(f"\n💾 Exportando resultados...")
    output_path = (
        f"reporte_encuesta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    orchestrator.export_results(output_path)

    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"🧹 Archivo temporal eliminado: {temp_file}")

    print("\n" + "=" * 60)
    print("🎉 PRUEBA COMPLETADA EXITOSAMENTE")
    print("=" * 60)


if __name__ == "__main__":
    print("🚀 Iniciando pruebas del sistema de análisis de encuestas...")

    # Test individual functions first
    test_individual_functions()

    # Test complete pipeline
    test_survey_analysis()

    print("\n🎯 Todas las pruebas completadas!")
