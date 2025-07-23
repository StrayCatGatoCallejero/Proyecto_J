#!/usr/bin/env python3
"""
Script para probar las importaciones de la aplicación Streamlit.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Prueba todas las importaciones necesarias."""
    
    # Agregar el directorio raíz al path
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))
    
    print("🔍 Probando importaciones...")
    
    try:
        # Probar importación de pandas
        import pandas as pd
        print("✅ pandas importado correctamente")
        
        # Probar importación de streamlit
        import streamlit as st
        print("✅ streamlit importado correctamente")
        
        # Probar importación de numpy
        import numpy as np
        print("✅ numpy importado correctamente")
        
        # Probar importación de matplotlib
        import matplotlib.pyplot as plt
        print("✅ matplotlib importado correctamente")
        
        # Probar importación de seaborn
        import seaborn as sns
        print("✅ seaborn importado correctamente")
        
        # Probar importación de scipy
        from scipy import stats
        print("✅ scipy importado correctamente")
        
        # Probar importación de sklearn
        from sklearn.preprocessing import StandardScaler
        print("✅ sklearn importado correctamente")
        
        # Probar importación de plotly
        import plotly.graph_objects as go
        print("✅ plotly importado correctamente")
        
        # Probar importación de pyreadstat
        import pyreadstat
        print("✅ pyreadstat importado correctamente")
        
        # Probar importaciones del proyecto
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
        print("✅ Funciones de estadistica importadas correctamente")
        
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
        print("✅ Funciones de ciencias_sociales importadas correctamente")
        
        print("\n🎉 ¡Todas las importaciones funcionan correctamente!")
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if not success:
        sys.exit(1) 