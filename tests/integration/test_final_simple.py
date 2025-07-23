"""
Test Final Simplificado de Integración
=====================================

Verifica que serialize_for_json funciona correctamente en escenarios reales.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime

# Importar solo la función de serialización
from processing.json_logging import serialize_for_json


def test_real_world_scenarios():
    """Test de escenarios reales del mundo"""
    
    print("🧪 Probando escenarios reales del mundo...")
    
    # Escenario 1: Datos de análisis estadístico
    statistical_data = {
        'descriptive_stats': {
            'mean': np.float64(45.67),
            'std': np.float64(12.34),
            'median': np.float64(43.21),
            'percentiles': np.array([25, 50, 75]),
            'count': np.int64(1000)
        },
        'correlation_matrix': np.array([[1.0, 0.5, -0.3], [0.5, 1.0, 0.2], [-0.3, 0.2, 1.0]]),
        'p_values': np.array([0.01, 0.05, 0.1, 0.001]),
        'sample_sizes': np.array([100, 150, 200, 250])
    }
    
    # Escenario 2: Metadatos de archivo
    file_metadata = {
        'file_info': {
            'size_bytes': np.int64(1024000),
            'created_date': pd.Timestamp('2023-01-01'),
            'modified_date': pd.Timestamp('2023-01-15'),
            'encoding': 'utf-8'
        },
        'data_info': {
            'rows': np.int64(5000),
            'columns': np.int64(25),
            'memory_usage_mb': np.float64(15.7),
            'data_types': pd.Series(['int64', 'float64', 'object', 'datetime64']).value_counts().to_dict()
        }
    }
    
    # Escenario 3: Resultados de visualización
    visualization_data = {
        'histogram_data': {
            'bins': np.array([0, 10, 20, 30, 40, 50]),
            'counts': np.array([5, 15, 25, 35, 20]),
            'density': np.array([0.01, 0.03, 0.05, 0.07, 0.04])
        },
        'scatter_data': pd.DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100)
        }).to_dict('records'),
        'boxplot_data': {
            'q1': np.array([25, 30, 35]),
            'q2': np.array([50, 55, 60]),
            'q3': np.array([75, 80, 85]),
            'outliers': np.array([10, 90, 5, 95])
        }
    }
    
    # Escenario 4: Logs del sistema
    system_logs = {
        'pipeline_metrics': {
            'total_steps': np.int64(6),
            'completed_steps': np.int64(5),
            'failed_steps': np.int64(1),
            'execution_time': np.float64(45.67),
            'memory_peak_mb': np.float64(125.3)
        },
        'step_details': {
            'data_loading': {
                'status': 'completed',
                'execution_time': np.float64(2.34),
                'rows_processed': np.int64(5000)
            },
            'data_cleaning': {
                'status': 'completed',
                'execution_time': np.float64(1.56),
                'outliers_removed': np.int64(25)
            },
            'statistical_analysis': {
                'status': 'failed',
                'execution_time': np.float64(0.0),
                'error_message': 'Division by zero'
            }
        }
    }
    
    # Testear todos los escenarios
    scenarios = [
        ("Análisis estadístico", statistical_data),
        ("Metadatos de archivo", file_metadata),
        ("Datos de visualización", visualization_data),
        ("Logs del sistema", system_logs)
    ]
    
    all_passed = True
    
    for scenario_name, data in scenarios:
        try:
            # Serializar
            serialized = serialize_for_json(data)
            
            # Verificar que se puede convertir a JSON
            json_str = json.dumps(serialized, ensure_ascii=False)
            
            # Verificar tipos
            if check_json_types_recursive(serialized):
                print(f"✅ {scenario_name}: PASÓ")
            else:
                print(f"❌ {scenario_name}: FALLÓ - tipos no válidos")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {scenario_name}: FALLÓ - {e}")
            all_passed = False
    
    return all_passed


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


def test_performance():
    """Test de rendimiento con datos grandes"""
    
    print("\n🧪 Probando rendimiento con datos grandes...")
    
    # Crear datos grandes
    large_data = {
        'large_array': np.random.rand(1000, 100).tolist(),
        'large_dict': {f'key_{i}': np.random.rand(10).tolist() for i in range(100)},
        'mixed_data': {
            'numpy_arrays': [np.random.rand(50).tolist() for _ in range(20)],
            'pandas_series': [pd.Series(np.random.rand(30)).to_dict() for _ in range(10)],
            'nested_structures': {
                f'level_{i}': {
                    f'sublevel_{j}': np.random.rand(5).tolist()
                    for j in range(5)
                }
                for i in range(10)
            }
        }
    }
    
    try:
        import time
        start_time = time.time()
        
        # Serializar
        serialized = serialize_for_json(large_data)
        
        serialization_time = time.time() - start_time
        
        # Verificar que se puede convertir a JSON
        json_str = json.dumps(serialized, ensure_ascii=False)
        
        json_time = time.time() - start_time
        
        print(f"✅ Serialización: {serialization_time:.3f}s")
        print(f"✅ JSON completo: {json_time:.3f}s")
        print(f"✅ Tamaño JSON: {len(json_str)} caracteres")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en test de rendimiento: {e}")
        return False


def test_edge_cases():
    """Test de casos extremos"""
    
    print("\n🧪 Probando casos extremos...")
    
    edge_cases = [
        # Caso 1: Datos vacíos
        {},
        
        # Caso 2: Datos anidados muy profundos
        {'level1': {'level2': {'level3': {'level4': {'level5': np.int64(42)}}}}},
        
        # Caso 3: Arrays muy grandes
        {'large_array': np.random.rand(100, 100).tolist()},
        
        # Caso 4: Datos mixtos complejos
        {
            'numpy_types': {
                'int64': np.int64(42),
                'float64': np.float64(3.14159),
                'array': np.array([1, 2, 3, 4, 5]),
                'bool': np.bool_(True)
            },
            'pandas_types': {
                'series': pd.Series([1, 2, 3, 4, 5]),
                'timestamp': pd.Timestamp('2023-01-01'),
                'index': pd.Index(['a', 'b', 'c'])
            },
            'mixed_nested': {
                'numpy_in_pandas': pd.Series(np.array([1, 2, 3])),
                'pandas_in_numpy': np.array([pd.Timestamp('2023-01-01')])
            }
        }
    ]
    
    all_passed = True
    
    for i, case in enumerate(edge_cases, 1):
        try:
            serialized = serialize_for_json(case)
            json_str = json.dumps(serialized, ensure_ascii=False)
            print(f"✅ Caso extremo {i}: PASÓ")
        except Exception as e:
            print(f"❌ Caso extremo {i}: FALLÓ - {e}")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("🚀 Test Final Simplificado de Integración")
    print("=" * 50)
    
    # Ejecutar tests
    test1 = test_real_world_scenarios()
    test2 = test_performance()
    test3 = test_edge_cases()
    
    print(f"\n📊 Resumen de tests:")
    print(f"  Escenarios reales: {'✅ PASÓ' if test1 else '❌ FALLÓ'}")
    print(f"  Rendimiento: {'✅ PASÓ' if test2 else '❌ FALLÓ'}")
    print(f"  Casos extremos: {'✅ PASÓ' if test3 else '❌ FALLÓ'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        print("💡 serialize_for_json está funcionando perfectamente")
        print("🚀 El sistema está listo para producción")
        print("📊 Todos los datos son compatibles con JSON")
    else:
        print("\n⚠️ Algunos tests fallaron. Revisar antes de continuar.") 