#!/usr/bin/env python3
"""
Script para probar la nueva funcionalidad de clasificación de variables.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

def test_variable_classifier():
    """Prueba la clasificación de variables con datos de ejemplo."""
    
    # Agregar el directorio raíz al path
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))
    
    print("🔍 Probando clasificación de variables...")
    
    try:
        # Importar la nueva funcionalidad
        from proyecto_j.src.ciencias_sociales import clasificar_variables_avanzado, VariableClassifier
        
        # Crear datos de ejemplo
        print("📊 Creando datos de ejemplo...")
        
        # Datos de ejemplo con diferentes tipos de variables
        data = {
            'edad': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            'genero': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
            'nivel_educacion': [1, 2, 3, 2, 1, 3, 2, 1, 3, 2],  # Ordinal
            'satisfaccion': [5, 4, 3, 5, 4, 3, 5, 4, 3, 5],  # Likert
            'ingreso': [50000, 60000, 70000, 55000, 65000, 75000, 45000, 55000, 65000, 75000],
            'fecha_nacimiento': pd.date_range('1980-01-01', periods=10, freq='Y'),
            'comentarios': ['Bueno', 'Excelente', 'Regular', 'Muy bueno', 'Bueno', 'Excelente', 'Regular', 'Muy bueno', 'Bueno', 'Excelente'],
            'activo': [True, False, True, False, True, False, True, False, True, False]
        }
        
        df = pd.DataFrame(data)
        
        # Probar clasificación
        print("🔍 Ejecutando clasificación de variables...")
        metadata = clasificar_variables_avanzado(df)
        
        # Mostrar resultados
        print("\n📋 Resultados de la clasificación:")
        print("=" * 50)
        
        for col, meta in metadata.items():
            print(f"\n🔹 Variable: {col}")
            print(f"   Tipo: {meta['type']}")
            print(f"   Dominio: {meta['dominio']}")
            print(f"   Es ordinal: {meta.get('es_ordinal', False)}")
            print(f"   Es binaria: {meta.get('es_binaria', False)}")
            print(f"   Es Likert: {meta.get('es_likert', False)}")
            print(f"   Valores únicos: {meta['n_unique']}")
            print(f"   Valores faltantes: {meta['n_missing']}")
        
        # Probar clasificador directo
        print("\n🔧 Probando VariableClassifier directamente...")
        clf = VariableClassifier(cat_threshold=0.1, cat_unique_limit=20)
        metadata2 = clf.classify_dataset(df)
        
        print(f"✅ Clasificación exitosa: {len(metadata2)} variables procesadas")
        
        # Verificar que los resultados son consistentes
        if metadata == metadata2:
            print("✅ Resultados consistentes entre funciones")
        else:
            print("❌ Inconsistencia en resultados")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_variable_classifier()
    if success:
        print("\n🎉 ¡Prueba de clasificación de variables exitosa!")
    else:
        print("\n❌ Prueba de clasificación de variables falló")
        sys.exit(1) 