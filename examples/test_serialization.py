"""
Test simple de serialización JSON
================================

Verifica que la función to_serializable funciona correctamente.
"""

import json
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime

# Importar la función de serialización
from processing.json_logging import to_serializable, JsonLogger, LogLevel, LogCategory


def test_serialization():
    """Test de serialización de diferentes tipos de datos"""
    
    print("🧪 Probando serialización de diferentes tipos de datos...")
    
    # Crear datos de prueba
    test_data = {
        'numpy_int': np.int64(42),
        'numpy_float': np.float64(3.14159),
        'numpy_array': np.array([1, 2, 3, 4, 5]),
        'numpy_bool': np.bool_(True),
        'pandas_series': pd.Series([1, 2, 3, 4, 5], name='test_series'),
        'pandas_dataframe': pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        }),
        'pandas_timestamp': pd.Timestamp('2023-01-01'),
        'normal_int': 100,
        'normal_float': 2.718,
        'normal_string': 'Hello World',
        'normal_list': [1, 2, 3, 4, 5],
        'normal_dict': {'key1': 'value1', 'key2': 'value2'}
    }
    
    # Intentar serializar con la función personalizada
    try:
        serialized = json.dumps(test_data, indent=2, ensure_ascii=False, default=to_serializable)
        print("✅ Serialización exitosa!")
        print(f"📄 Longitud del JSON: {len(serialized)} caracteres")
        
        # Verificar que se puede deserializar
        deserialized = json.loads(serialized)
        print("✅ Deserialización exitosa!")
        
        # Mostrar algunos ejemplos
        print("\n📊 Ejemplos de conversión:")
        print(f"  numpy_int64(42) -> {type(deserialized['numpy_int']).__name__}: {deserialized['numpy_int']}")
        print(f"  numpy_float64(3.14159) -> {type(deserialized['numpy_float']).__name__}: {deserialized['numpy_float']}")
        print(f"  pandas_series -> {type(deserialized['pandas_series']).__name__}")
        print(f"  pandas_dataframe -> {type(deserialized['pandas_dataframe']).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en serialización: {e}")
        return False


def test_json_logger():
    """Test del JsonLogger con datos complejos"""
    
    print("\n🧪 Probando JsonLogger con datos complejos...")
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    log_file = os.path.join(temp_dir, 'test_log.json')
    
    try:
        # Crear logger
        logger = JsonLogger(
            file_path=log_file,
            level='INFO',
            session_id='test_session_123',
            console_output=True
        )
        
        # Crear datos complejos para el log
        complex_metrics = {
            'numpy_stats': {
                'mean': np.float64(45.67),
                'std': np.float64(12.34),
                'count': np.int64(1000)
            },
            'pandas_info': {
                'rows': np.int64(500),
                'columns': np.int64(10),
                'memory_mb': np.float64(2.5)
            },
            'arrays': {
                'percentiles': np.array([25, 50, 75]),
                'correlations': np.array([[1.0, 0.5], [0.5, 1.0]])
            }
        }
        
        # Log con datos complejos
        logger.log_event(
            level=LogLevel.INFO,
            message="Test de serialización con datos complejos",
            module="test_serialization",
            function="test_json_logger",
            step="serialization_test",
            category=LogCategory.SYSTEM.value,
            parameters={"test_type": "complex_data"},
            before_metrics=complex_metrics,
            after_metrics=complex_metrics,
            execution_time=0.123,
            tags=["test", "serialization"],
            metadata={
                "numpy_data": np.array([1, 2, 3]),
                "pandas_data": pd.Series([1, 2, 3]),
                "mixed_data": {
                    "int": np.int64(42),
                    "float": np.float64(3.14),
                    "array": np.array([1, 2, 3])
                }
            }
        )
        
        # Verificar que el archivo se creó
        if os.path.exists(log_file):
            print("✅ Archivo de log creado exitosamente!")
            
            # Leer y verificar el contenido
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            print(f"📄 Contenido del log ({len(log_content)} caracteres):")
            print(log_content[:500] + "..." if len(log_content) > 500 else log_content)
            
            # Verificar que es JSON válido
            try:
                json.loads(log_content)
                print("✅ Log es JSON válido!")
            except json.JSONDecodeError as e:
                print(f"❌ Error en JSON del log: {e}")
                return False
                
        else:
            print("❌ No se creó el archivo de log")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error en JsonLogger: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Limpiar
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass


def test_plotly_serialization():
    """Test de serialización de objetos Plotly"""
    
    print("\n🧪 Probando serialización de Plotly...")
    
    try:
        import plotly.graph_objs as go
        
        # Crear figura de Plotly
        fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[4, 5, 6]))
        fig.update_layout(title="Test Plot")
        
        # Intentar serializar
        plotly_data = {
            'figure': fig,
            'layout': fig.layout,
            'data': fig.data
        }
        
        serialized = json.dumps(plotly_data, indent=2, ensure_ascii=False, default=to_serializable)
        print("✅ Serialización de Plotly exitosa!")
        print(f"📄 Longitud del JSON: {len(serialized)} caracteres")
        
        return True
        
    except ImportError:
        print("⚠️ Plotly no está instalado, saltando test de Plotly")
        return True
    except Exception as e:
        print(f"❌ Error en serialización de Plotly: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Iniciando tests de serialización JSON...\n")
    
    # Ejecutar tests
    test1 = test_serialization()
    test2 = test_json_logger()
    test3 = test_plotly_serialization()
    
    print(f"\n📊 Resumen de tests:")
    print(f"  Serialización básica: {'✅ PASÓ' if test1 else '❌ FALLÓ'}")
    print(f"  JsonLogger: {'✅ PASÓ' if test2 else '❌ FALLÓ'}")
    print(f"  Plotly: {'✅ PASÓ' if test3 else '❌ FALLÓ'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 ¡Todos los tests de serialización pasaron!")
    else:
        print("\n⚠️ Algunos tests fallaron. Revisar errores arriba.") 