"""
Tests unitarios para importaciones del proyecto
"""

import pytest
import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestImports:
    """Tests para verificar que todas las importaciones funcionan correctamente"""
    
    def test_core_imports(self):
        """Test de importaciones principales"""
        try:
            import pandas as pd
            import numpy as np
            import streamlit as st
            import plotly.express as px
            import plotly.graph_objects as go
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy import stats
            from sklearn.preprocessing import StandardScaler
            import pyreadstat
            
            assert True, "Todas las importaciones principales funcionan"
        except ImportError as e:
            pytest.fail(f"Error de importación: {e}")
    
    def test_project_imports(self):
        """Test de importaciones del proyecto"""
        try:
            # Core del proyecto
            from proyecto_j.src.core import Pipeline
            from proyecto_j.src.steps import cargar_datos, limpiar_datos, transformar_datos
            from proyecto_j.src.utils import load_config
            
            # Módulos principales
            from proyecto_j.src.estadistica import (
                cargar_archivo,
                calcular_media,
                calcular_moda,
                calcular_percentiles,
                generar_histograma,
                calcular_correlacion_pearson,
                calcular_correlacion_spearman,
                generar_heatmap_correlacion,
                obtener_columnas_numericas,
                obtener_columnas_categoricas,
            )
            
            from proyecto_j.src.ciencias_sociales import (
                clasificar_variable,
                analisis_descriptivo_cs,
                analisis_bivariado_cs,
                analisis_regresion_multiple_cs,
                analisis_clusters_cs,
                calcular_indice_gini,
                calcular_indice_gini_simple,
                calcular_indice_calidad_vida,
                calcular_indice_calidad_vida_simple,
                validar_supuestos_regresion,
                analizar_valores_perdidos,
                sugerir_imputacion,
            )
            
            assert True, "Todas las importaciones del proyecto funcionan"
        except ImportError as e:
            pytest.fail(f"Error de importación del proyecto: {e}")
    
    def test_processing_imports(self):
        """Test de importaciones del módulo processing"""
        try:
            from processing.json_logging import JsonLogger
            from processing.business_rules import validate_business_rules, ValidationResult
            from processing.data_validators import DataFrameSchema, ValidationResult as DataValidationResult
            from processing.visualization import VisualizationGenerator
            from processing.stats import summary_statistics
            from processing.config_manager import ConfigManager
            from processing.data_types import SchemaValidator
            
            assert True, "Todas las importaciones de processing funcionan"
        except ImportError as e:
            pytest.fail(f"Error de importación de processing: {e}")
    
    def test_orchestrator_imports(self):
        """Test de importaciones del módulo orchestrator"""
        try:
            from orchestrator.pipeline_orchestrator import PipelineOrchestrator
            
            assert True, "Todas las importaciones de orchestrator funcionan"
        except ImportError as e:
            pytest.fail(f"Error de importación de orchestrator: {e}")
    
    def test_streamlit_app_import(self):
        """Test de importación de la app Streamlit"""
        try:
            # Importar el módulo sin ejecutarlo
            import proyecto_j.streamlit_app
            
            assert True, "La app Streamlit se puede importar"
        except ImportError as e:
            pytest.fail(f"Error de importación de streamlit_app: {e}")
    
    def test_optional_imports(self):
        """Test de importaciones opcionales"""
        optional_imports = []
        
        try:
            import missingno as msno
            optional_imports.append("missingno")
        except ImportError:
            pass
        
        try:
            from fpdf import FPDF
            optional_imports.append("fpdf2")
        except ImportError:
            pass
        
        try:
            import redis
            optional_imports.append("redis")
        except ImportError:
            pass
        
        try:
            from celery import Celery
            optional_imports.append("celery")
        except ImportError:
            pass
        
        # Al menos algunas importaciones opcionales deberían estar disponibles
        assert len(optional_imports) > 0, f"Algunas importaciones opcionales disponibles: {optional_imports}"
    
    def test_version_compatibility(self):
        """Test de compatibilidad de versiones"""
        import pandas as pd
        import numpy as np
        import streamlit as st
        
        # Verificar versiones mínimas
        assert pd.__version__ >= "2.0.0", f"pandas version {pd.__version__} es muy antigua"
        assert np.__version__ >= "1.24.0", f"numpy version {np.__version__} es muy antigua"
        assert st.__version__ >= "1.28.0", f"streamlit version {st.__version__} es muy antigua"
    
    def test_path_configuration(self):
        """Test de configuración de paths"""
        # Verificar que el directorio del proyecto está en el path
        project_root_str = str(project_root)
        assert project_root_str in sys.path, f"El directorio del proyecto no está en sys.path: {project_root_str}"
        
        # Verificar que los módulos principales existen
        assert (project_root / "proyecto_j" / "src").exists(), "El directorio src no existe"
        assert (project_root / "processing").exists(), "El directorio processing no existe"
        assert (project_root / "orchestrator").exists(), "El directorio orchestrator no existe"


class TestModuleStructure:
    """Tests para verificar la estructura de módulos"""
    
    def test_core_module_structure(self):
        """Test de estructura del módulo core"""
        from proyecto_j.src import core
        
        # Verificar que la clase Pipeline existe
        assert hasattr(core, 'Pipeline'), "La clase Pipeline no existe en core"
        
        # Verificar que el método run existe
        pipeline = core.Pipeline({})
        assert hasattr(pipeline, 'run'), "El método run no existe en Pipeline"
    
    def test_steps_module_structure(self):
        """Test de estructura del módulo steps"""
        from proyecto_j.src import steps
        
        # Verificar que las funciones principales existen
        required_functions = [
            'cargar_datos',
            'limpiar_datos', 
            'transformar_datos',
            'modelar',
            'visualizar',
            'generar_reporte'
        ]
        
        for func_name in required_functions:
            assert hasattr(steps, func_name), f"La función {func_name} no existe en steps"
    
    def test_estadistica_module_structure(self):
        """Test de estructura del módulo estadistica"""
        from proyecto_j.src import estadistica
        
        # Verificar que las funciones principales existen
        required_functions = [
            'cargar_archivo',
            'calcular_media',
            'calcular_moda',
            'calcular_percentiles',
            'generar_histograma'
        ]
        
        for func_name in required_functions:
            assert hasattr(estadistica, func_name), f"La función {func_name} no existe en estadistica"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 