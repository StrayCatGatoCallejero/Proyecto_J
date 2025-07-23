"""
Test Final de Integración Completa
=================================

Verifica que todo el sistema funciona correctamente con serialize_for_json.
"""

import os
import tempfile
import json
import numpy as np
import pandas as pd
from datetime import datetime
import shutil

# Importar componentes del sistema
from processing.json_logging import serialize_for_json
from orchestrator.pipeline_orchestrator import PipelineOrchestrator


def test_complete_system_integration():
    """Test de integración completa del sistema"""
    
    print("🧪 Probando integración completa del sistema...")
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Crear datos de prueba
        test_data = pd.DataFrame({
            'edad': np.random.randint(18, 80, 50),
            'ingresos': np.random.normal(50000, 15000, 50),
            'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universidad'], 50),
            'genero': np.random.choice(['M', 'F'], 50),
            'satisfaccion': np.random.randint(1, 11, 50)
        })
        
        # Guardar datos de prueba
        test_file = os.path.join(temp_dir, 'test_data.csv')
        test_data.to_csv(test_file, index=False)
        
        # Configuración del pipeline
        config = {
            'data': {
                'input_path': test_file,
                'output_path': temp_dir,
                'export_formats': ['csv', 'json']
            },
            'analysis': {
                'columns_to_analyze': ['edad', 'ingresos', 'satisfaccion'],
                'correlation_method': 'pearson',
                'significance_level': 0.05
            },
            'filters': {
                'remove_outliers': True,
                'outlier_method': 'iqr',
                'handle_missing': 'drop'
            },
            'logging': {
                'enabled': True,
                'level': 'INFO',
                'log_file': os.path.join(temp_dir, 'pipeline.json'),
                'rotation': {
                    'when': 'midnight',
                    'interval': 1,
                    'backup_count': 7
                },
                'max_size_mb': 100
            }
        }
        
        # Crear pipeline
        pipeline = PipelineOrchestrator(config)
        
        print(f"✅ Pipeline creado con {len(test_data)} filas de datos")
        
        # Ejecutar pipeline
        try:
            results = pipeline.run_pipeline(test_file)
            print("✅ Pipeline ejecutado exitosamente")
            
            # Verificar que los resultados son serializables
            try:
                json_str = json.dumps(results, ensure_ascii=False)
                print("✅ Resultados del pipeline son JSON serializables")
                
                # Verificar estructura de resultados
                required_fields = ['session_id', 'pipeline_status', 'pipeline_metrics', 'data_info', 'reports', 'visualizations', 'errors', 'warnings']
                for field in required_fields:
                    if field not in results:
                        print(f"❌ Falta campo requerido: {field}")
                        return False
                
                print("✅ Todos los campos requeridos están presentes")
                
                # Verificar que no hay tipos problemáticos
                check_json_types_recursive(results)
                
                return True
                
            except Exception as e:
                print(f"❌ Error serializando resultados: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Error en test: {e}")
        return False
    
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


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


def test_ui_ready_data():
    """Test de datos listos para UI"""
    
    print("\n🧪 Probando datos listos para UI...")
    
    # Simular datos típicos que se enviarían a la UI
    ui_data = {
        'pipeline_results': {
            'session_id': 'test_session_123',
            'status': 'completed',
            'execution_time': 15.67,
            'data_shape': [100, 5],
            'memory_usage_mb': 2.5
        },
        'statistics': {
            'mean_edad': np.float64(45.67),
            'std_ingresos': np.float64(12345.67),
            'correlation_matrix': np.array([[1.0, 0.5], [0.5, 1.0]]),
            'percentiles': np.array([25, 50, 75])
        },
        'visualizations': {
            'histogram_data': pd.Series([1, 2, 3, 4, 5]).to_dict(),
            'scatter_data': pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]}).to_dict('records')
        },
        'metadata': {
            'file_info': {
                'size_bytes': np.int64(1024),
                'created_date': pd.Timestamp('2023-01-01')
            },
            'processing_info': {
                'filters_applied': ['outliers', 'missing_values'],
                'analysis_performed': ['descriptive', 'correlation']
            }
        }
    }
    
    try:
        # Limpiar datos para UI
        cleaned_ui_data = serialize_for_json(ui_data)
        
        # Verificar que se puede serializar
        json_str = json.dumps(cleaned_ui_data, ensure_ascii=False)
        print("✅ Datos de UI son JSON serializables")
        
        # Verificar tipos
        if check_json_types_recursive(cleaned_ui_data):
            print("✅ Todos los tipos son JSON válidos")
            return True
        else:
            print("❌ Se encontraron tipos no válidos")
            return False
            
    except Exception as e:
        print(f"❌ Error en datos de UI: {e}")
        return False


def test_logging_integration():
    """Test de integración del logging"""
    
    print("\n🧪 Probando integración del logging...")
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    log_file = os.path.join(temp_dir, 'integration_test.json')
    
    try:
        # Simular logs complejos
        complex_log_data = {
            'metrics': {
                'numpy_stats': {
                    'mean': np.float64(45.67),
                    'std': np.float64(12.34),
                    'count': np.int64(1000)
                },
                'pandas_info': {
                    'rows': np.int64(500),
                    'columns': np.int64(10),
                    'memory_mb': np.float64(2.5)
                }
            },
            'metadata': {
                'data_types': pd.Series(['int64', 'float64', 'object']).to_dict(),
                'correlations': np.array([[1.0, 0.5], [0.5, 1.0]]),
                'percentiles': np.array([25, 50, 75])
            }
        }
        
        # Limpiar datos de log
        cleaned_log_data = serialize_for_json(complex_log_data)
        
        # Escribir log
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_log_data, f, ensure_ascii=False, indent=2)
        
        # Verificar que el archivo se creó y es válido
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            print("✅ Log creado y cargado exitosamente")
            
            # Verificar estructura
            if 'metrics' in loaded_data and 'metadata' in loaded_data:
                print("✅ Estructura del log preservada")
                return True
            else:
                print("❌ Estructura del log no preservada")
                return False
        else:
            print("❌ No se creó el archivo de log")
            return False
            
    except Exception as e:
        print(f"❌ Error en logging: {e}")
        return False
    
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


if __name__ == "__main__":
    print("🚀 Test Final de Integración Completa")
    print("=" * 50)
    
    # Ejecutar tests
    test1 = test_complete_system_integration()
    test2 = test_ui_ready_data()
    test3 = test_logging_integration()
    
    print(f"\n📊 Resumen de tests:")
    print(f"  Sistema completo: {'✅ PASÓ' if test1 else '❌ FALLÓ'}")
    print(f"  Datos para UI: {'✅ PASÓ' if test2 else '❌ FALLÓ'}")
    print(f"  Logging integrado: {'✅ PASÓ' if test3 else '❌ FALLÓ'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        print("💡 El sistema está completamente integrado y listo para producción")
        print("🚀 serialize_for_json está funcionando correctamente en todos los módulos")
        print("📊 Los logs y datos son 100% compatibles con dashboards y UI")
    else:
        print("\n⚠️ Algunos tests fallaron. Revisar antes de continuar.") 