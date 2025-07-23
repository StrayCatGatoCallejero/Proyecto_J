"""
Test de Integración de serialize_for_json en Módulos Clave
========================================================

Verifica que los módulos principales limpian correctamente sus metadatos.
"""

import os
import tempfile
import json
import numpy as np
import pandas as pd
from datetime import datetime
import shutil

# Importar módulos a testear
from processing.io import cargar_archivo
from processing.json_logging import serialize_for_json


def test_io_module_serialization():
    """Test del módulo io con serialización"""
    
    print("🧪 Probando módulo io con serialización...")
    
    # Crear datos de prueba
    test_data = pd.DataFrame({
        'edad': np.random.randint(18, 80, 10),
        'ingresos': np.random.normal(50000, 15000, 10),
        'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universidad'], 10)
    })
    
    # Crear archivo temporal
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, 'test_data.csv')
    
    try:
        # Guardar datos de prueba
        test_data.to_csv(test_file, index=False)
        
        # Cargar archivo usando el módulo io
        df, metadata = cargar_archivo(test_file)
        
        print(f"✅ Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")
        
        # Verificar que los metadatos son serializables
        try:
            json_str = json.dumps(metadata, ensure_ascii=False)
            print("✅ Metadatos son JSON serializables")
            
            # Verificar estructura de metadatos
            required_fields = ['format', 'file_size', 'n_rows', 'n_cols', 'load_time']
            for field in required_fields:
                if field not in metadata:
                    print(f"❌ Falta campo requerido: {field}")
                    return False
            
            print("✅ Todos los campos requeridos están presentes")
            return True
            
        except Exception as e:
            print(f"❌ Error serializando metadatos: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error en test de io: {e}")
        return False
    
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def test_serialize_for_json_comprehensive():
    """Test comprehensivo de serialize_for_json"""
    
    print("\n🧪 Probando serialize_for_json con datos complejos...")
    
    # Crear datos complejos típicos del pipeline
    complex_data = {
        'numpy_stats': {
            'mean': np.float64(45.67),
            'std': np.float64(12.34),
            'count': np.int64(1000),
            'percentiles': np.array([25, 50, 75])
        },
        'pandas_info': {
            'rows': np.int64(500),
            'columns': np.int64(10),
            'memory_mb': np.float64(2.5),
            'dtypes': pd.Series(['int64', 'float64', 'object']).to_dict()
        },
        'mixed_data': {
            'numpy_key': np.int64(100),
            'pandas_value': pd.Series([1, 2, 3]),
            'nested': {
                'array': np.array([[1, 2], [3, 4]]),
                'dtype': np.dtype('float64')
            }
        }
    }
    
    try:
        # Serializar
        serialized = serialize_for_json(complex_data)
        
        # Verificar que se puede convertir a JSON
        json_str = json.dumps(serialized, ensure_ascii=False)
        print("✅ Serialización exitosa!")
        
        # Verificar tipos
        check_json_types_recursive(serialized)
        
        return True
        
    except Exception as e:
        print(f"❌ Error en serialización: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_json_types_recursive(obj, path=""):
    """Verifica recursivamente que todos los valores son tipos JSON válidos"""
    
    if obj is None:
        return True
    elif isinstance(obj, (str, int, float, bool)):
        return True
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if not check_json_types_recursive(item, f"{path}[{i}]"):
                return False
        return True
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                print(f"❌ Clave no válida en {path}: {type(key).__name__} = {key}")
                return False
            if not check_json_types_recursive(value, f"{path}.{key}"):
                return False
        return True
    else:
        print(f"❌ Tipo no válido en {path}: {type(obj).__name__} = {obj}")
        return False


def test_pipeline_integration():
    """Test de integración simulando un pipeline completo"""
    
    print("\n🧪 Probando integración en pipeline simulado...")
    
    # Simular datos del pipeline
    pipeline_data = {
        'input_metrics': {
            'rows': np.int64(100),
            'columns': np.int64(5),
            'memory_mb': np.float64(1.2)
        },
        'processing_results': {
            'filtered_rows': np.int64(95),
            'outliers_removed': np.int64(5),
            'processing_time': np.float64(0.15)
        },
        'analysis_results': {
            'correlation_matrix': np.array([[1.0, 0.5], [0.5, 1.0]]),
            'statistics': pd.Series([1, 2, 3, 4, 5]).describe().to_dict(),
            'p_values': np.array([0.01, 0.05, 0.1])
        }
    }
    
    try:
        # Simular limpieza en cada paso del pipeline
        cleaned_data = serialize_for_json(pipeline_data)
        
        # Verificar que todo es serializable
        json_str = json.dumps(cleaned_data, ensure_ascii=False)
        print("✅ Pipeline data serializada exitosamente!")
        
        # Verificar estructura
        if 'input_metrics' in cleaned_data and 'processing_results' in cleaned_data and 'analysis_results' in cleaned_data:
            print("✅ Estructura del pipeline preservada")
            return True
        else:
            print("❌ Estructura del pipeline no preservada")
            return False
            
    except Exception as e:
        print(f"❌ Error en pipeline: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Test de Integración de serialize_for_json en Módulos Clave")
    print("=" * 70)
    
    # Ejecutar tests
    test1 = test_io_module_serialization()
    test2 = test_serialize_for_json_comprehensive()
    test3 = test_pipeline_integration()
    
    print(f"\n📊 Resumen de tests:")
    print(f"  Módulo io: {'✅ PASÓ' if test1 else '❌ FALLÓ'}")
    print(f"  Serialización comprehensiva: {'✅ PASÓ' if test2 else '❌ FALLÓ'}")
    print(f"  Pipeline simulado: {'✅ PASÓ' if test3 else '❌ FALLÓ'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 ¡Todos los tests de integración pasaron!")
        print("💡 Los módulos están listos para producción")
        print("🚀 Puedes proceder con la integración en el pipeline y UI")
    else:
        print("\n⚠️ Algunos tests fallaron. Revisar antes de continuar.") 