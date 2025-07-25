"""
Pipeline Demográfico Modular
============================

Un pipeline completo para análisis demográfico que incluye:
- Carga de datos (soporte para .sav, .dta, .csv, .xlsx, .xls)
- Limpieza de datos (usando pandas, missingno, imputación con LLMs)
- Transformación de datos (manejo categórico, normalización opcional)
- Ajuste y proyección de modelos (usando curve_fit)
- Visualización (con Plotly Express)
- Generación de reportes (usando csv-to-pdf-report-generator)

Compatibilidad con Streamlit y caching incluida.
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import tempfile
import shutil

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import missingno as msno
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings
warnings.filterwarnings("ignore")


class PipelineDemografico:
    """
    Pipeline modular para análisis demográfico completo.
    """

    def __init__(self, cache_dir: str = "temp"):
        """
        Inicializar el pipeline demográfico.

        Args:
            cache_dir: Directorio para archivos temporales
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.data = None
        self.cleaned_data = None
        self.transformed_data = None
        self.models = {}
        self.reports = {}

        # Configurar report generator
        self.setup_report_generator()

    def setup_report_generator(self):
        """Configurar el generador de reportes PDF."""
        try:
            report_generator_path = Path("tools/csv-to-pdf-report-generator")
            if report_generator_path.exists():
                sys.path.append(str(report_generator_path))
                self.report_generator_available = True
            else:
                self.report_generator_available = False
                logger.warning(
                    "Report generator no encontrado en tools/csv-to-pdf-report-generator"
                )
        except Exception as e:
            self.report_generator_available = False
            logger.warning(f"No se pudo configurar el report generator: {e}")

    def load_data(
        self, file_path: str, file_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Cargar datos desde diferentes formatos de archivo.

        Args:
            file_path: Ruta al archivo
            file_type: Tipo de archivo (opcional, se infiere automáticamente)

        Returns:
            DataFrame con los datos cargados
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        if file_type is None:
            file_type = file_path.suffix.lower()

        try:
            if file_type in [".csv", ".CSV"]:
                # Intentar diferentes encodings
                encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
                for encoding in encodings:
                    try:
                        data = pd.read_csv(file_path, encoding=encoding)
                        logger.info(f"Datos cargados con encoding {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("No se pudo decodificar el archivo CSV")

            elif file_type in [".xlsx", ".xls"]:
                data = pd.read_excel(file_path, engine="openpyxl")

            elif file_type in [".sav"]:
                try:
                    import pyreadstat

                    data, meta = pyreadstat.read_sav(file_path)
                except ImportError:
                    raise ImportError(
                        "pyreadstat no está instalado. Instale con: pip install pyreadstat"
                    )

            elif file_type in [".dta"]:
                try:
                    data = pd.read_stata(file_path)
                except Exception as e:
                    # Intentar con pyreadstat como fallback
                    try:
                        import pyreadstat

                        data, meta = pyreadstat.read_dta(file_path)
                    except ImportError:
                        raise ImportError(
                            "pyreadstat no está instalado para archivos .dta"
                        )

            else:
                raise ValueError(f"Tipo de archivo no soportado: {file_type}")

            self.data = data
            logger.info(f"Datos cargados exitosamente: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
