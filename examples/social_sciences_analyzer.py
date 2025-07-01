"""
Analizador Avanzado de Datos de Ciencias Sociales
================================================

Un sistema completo de análisis de datos especializado para investigaciones
en ciencias sociales, encuestas y estudios demográficos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
from fuzzywuzzy import fuzz
import warnings

warnings.filterwarnings("ignore")


class SocialSciencesAnalyzer:
    """
    Analizador especializado para datos de ciencias sociales.
    """
    
    def __init__(self, custom_dictionary: Optional[Dict] = None):
        """
        Inicializar el analizador.
        
        Args:
            custom_dictionary: Diccionario personalizado para clasificación semántica
        """
        self.custom_dictionary = custom_dictionary or {}
        self.semantic_dictionary = self._load_semantic_dictionary()

    def _load_semantic_dictionary(self) -> Dict:
        """Cargar diccionario semántico por defecto."""
        return {
            "demografico": {
                "edad": ["edad", "age", "años", "years", "anios"],
                "genero": ["genero", "sexo", "gender", "sex"],
                "estado_civil": [
                    "estado_civil",
                    "civil",
                    "marital",
                    "casado",
                    "soltero",
                ],
                "educacion": ["educacion", "education", "estudios", "nivel_educativo"],
                "ocupacion": ["ocupacion", "trabajo", "job", "profesion", "cargo"],
            },
            "socioeconomico": {
                "ingresos": ["ingresos", "income", "salario", "salary", "renta"],
                "nivel_socioeconomico": [
                    "nse",
                    "nivel_socioeconomico",
                    "socioeconomic",
                ],
                "vivienda": ["vivienda", "housing", "casa", "hogar", "residencia"],
            },
            "opinion": {
                "satisfaccion": ["satisfaccion", "satisfaction", "satisfecho"],
                "acuerdo": ["acuerdo", "agreement", "de_acuerdo", "concuerdo"],
                "importancia": ["importancia", "importance", "importante"],
            },
            "escalas_likert": {
                "muy_de_acuerdo": [
                    "muy de acuerdo",
                    "totalmente de acuerdo",
                    "completamente de acuerdo",
                ],
                "de_acuerdo": ["de acuerdo", "acuerdo", "estoy de acuerdo"],
                "neutral": ["neutral", "ni de acuerdo ni en desacuerdo", "indiferente"],
                "en_desacuerdo": ["en desacuerdo", "desacuerdo", "no estoy de acuerdo"],
                "muy_en_desacuerdo": [
                    "muy en desacuerdo",
                    "totalmente en desacuerdo",
                    "completamente en desacuerdo",
                ],
            },
        }

    def analyze_survey_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analizar datos de encuesta automáticamente.
        
        Args:
            df: DataFrame con los datos de la encuesta
            
        Returns:
            Diccionario con resultados del análisis
        """
        results = {
            "semantic_classification": self.classify_columns_semantically(df),
            "likert_scales": self.detect_likert_scales(df),
            "normalized_categories": self.normalize_categories(df),
            "consistency_validation": self.validate_consistency(df),
            "visualization_suggestions": self.suggest_visualizations(df),
        }

        return results
    
    def classify_columns_semantically(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Clasificar columnas semánticamente.

        Args:
            df: DataFrame a analizar

        Returns:
            Diccionario con clasificación de columnas
        """
        classification = {}
        
        for col in df.columns:
            col_lower = col.lower()

            # Buscar en diccionario semántico
            for category, subcategories in self.semantic_dictionary.items():
                for subcategory, keywords in subcategories.items():
                    for keyword in keywords:
                        if keyword in col_lower:
                            classification[col] = f"{category}_{subcategory}"
                            break
                    if col in classification:
                        break
                if col in classification:
                    break

            # Si no se encontró, usar heurísticas
            if col not in classification:
                classification[col] = self._classify_by_heuristics(df, col)

        return classification

    def _classify_by_heuristics(self, df: pd.DataFrame, col: str) -> str:
        """Clasificar columna usando heurísticas."""
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()

        if dtype in ["int64", "float64"]:
            if unique_count <= 10:
                return "categorical"
            else:
                return "numeric"
        elif dtype == "object":
            sample_values = df[col].dropna().astype(str)
            avg_length = sample_values.str.len().mean()

            if avg_length > 20:
                return "text"
            else:
                return "categorical"
                    else:
            return "unknown"

    def detect_likert_scales(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detectar escalas Likert en el DataFrame.

        Args:
            df: DataFrame a analizar

        Returns:
            Diccionario con escalas detectadas
        """
        likert_scales = {}
        
        for col in df.columns:
            if df[col].dtype == "object":
                unique_values = df[col].dropna().unique()
                
                # Verificar si los valores coinciden con escalas Likert
                for scale_name, scale_values in self.semantic_dictionary[
                    "escalas_likert"
                ].items():
                    matches = 0
                    for value in unique_values:
                        value_str = str(value).lower()
                        for scale_value in scale_values:
                            if scale_value in value_str:
                                matches += 1
                                break

                    # Si al menos 3 valores coinciden, es una escala Likert
                    if matches >= 3:
                        if col not in likert_scales:
                            likert_scales[col] = []
                        likert_scales[col].append(scale_name)

        return likert_scales

    def normalize_categories(
        self, df: pd.DataFrame, threshold: float = 0.8
    ) -> Dict[str, Dict[str, str]]:
        """
        Normalizar categorías usando fuzzy matching.

        Args:
            df: DataFrame a analizar
            threshold: Umbral de similitud para agrupar categorías

        Returns:
            Diccionario con mapeos de normalización
        """
        normalization = {}
        
        for col in df.columns:
            if df[col].dtype == "object":
                unique_values = df[col].dropna().unique()
                mapping = {}

                # Agrupar valores similares
                processed = set()
                for value1 in unique_values:
                    if value1 in processed:
                        continue

                    similar_values = [value1]
                    for value2 in unique_values:
                        if value2 not in processed and value1 != value2:
                            similarity = (
                                fuzz.ratio(str(value1).lower(), str(value2).lower())
                                / 100
                            )
                            if similarity >= threshold:
                                similar_values.append(value2)
                                processed.add(value2)

                    # Usar el primer valor como estándar
                    standard_value = similar_values[0]
                    for value in similar_values:
                        mapping[str(value)] = standard_value

                    processed.add(value1)

                if mapping:
                    normalization[col] = mapping

        return normalization

    def validate_consistency(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validar consistencia lógica entre variables.

        Args:
            df: DataFrame a analizar

        Returns:
            Diccionario con inconsistencias encontradas
        """
        inconsistencies = {}

        # Verificar edad y estado civil
        if "edad" in df.columns and "estado_civil" in df.columns:
            age_marital_issues = []
            for idx, row in df.iterrows():
                if pd.notna(row["edad"]) and pd.notna(row["estado_civil"]):
                    age = row["edad"]
                    marital = str(row["estado_civil"]).lower()

                    if age < 18 and "casado" in marital:
                        age_marital_issues.append(
                            f"Fila {idx}: Edad {age} y estado civil '{marital}'"
                        )

            if age_marital_issues:
                inconsistencies["edad_estado_civil"] = age_marital_issues

        return inconsistencies

    def suggest_visualizations(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Sugerir visualizaciones apropiadas para cada columna.

        Args:
            df: DataFrame a analizar

        Returns:
            Diccionario con sugerencias de visualización
        """
        suggestions = {}
        
        for col in df.columns:
            col_suggestions = []
            dtype = str(df[col].dtype)

            if dtype in ["int64", "float64"]:
                col_suggestions.extend(["histogram", "boxplot", "scatter"])
            elif dtype == "object":
                unique_count = df[col].nunique()
                if unique_count <= 10:
                    col_suggestions.extend(["bar", "pie"])
                else:
                    col_suggestions.extend(["wordcloud", "frequency_table"])

            suggestions[col] = col_suggestions
        
        return suggestions
    

def analyze_survey_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Función de conveniencia para análisis rápido de datos de encuesta.
    
    Args:
        df: DataFrame con datos de encuesta
        
    Returns:
        Resultados del análisis
    """
    analyzer = SocialSciencesAnalyzer()
    return analyzer.analyze_survey_data(df)
