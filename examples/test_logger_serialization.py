"""
Test específico para el JsonLogger con serialize_for_json
=======================================================

Verifica que el logger limpia correctamente todos los campos antes de emitir logs.
"""

import os
import tempfile
import json
import numpy as np
import pandas as pd
from datetime import datetime
import shutil

# Importar el JsonLogger
from processing.json_logging import JsonLogger, LogLevel, LogCategory, serialize_for_json


def test_logger_with_complex_data():
    """Test del JsonLogger con datos complejos"""
    
    print("🧪 Probando JsonLogger con serialize_for_json...")
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    log_file = os.path.join(temp_dir, 'test_logger.json')
    
    try:
        # Crear logger
        logger = JsonLogger(
            file_path=log_file,
            level='INFO',
            session_id='test_serialization_session',
            console_output=True
        )
        
        # Crear datos complejos problemáticos
        complex_data = {
            'numpy_int64': np.int64(42),
            'numpy_float64': np.float64(3.14159),
            'numpy_array': np.array([1, 2, 3, 4, 5]),
            'numpy_dtype': np.dtype('int64'),
            'pandas_series': pd.Series([1, 2, 3], name='test'),
            'pandas_dataframe': pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
            'pandas_timestamp': pd.Timestamp('2023-01-01'),
            'mixed_dict': {
                'numpy_key': np.int64(100),
                'pandas_value': pd.Series([1, 2, 3]),
                'nested': {
                    'array': np.array([[1, 2], [3, 4]]),
                    'dtype': np.dtype('float64')
                }
            }
        }
        
        # Log con datos complejos
        logger.log_event(
            level=LogLevel.INFO,
            message="Test de serialización con datos complejos",
            module="test_serialization",
            function="test_logger_with_complex_data",
            step="serialization_test",
            category=LogCategory.SYSTEM.value,
            parameters={"test_type": "complex_data"},
            before_metrics=complex_data,
            after_metrics=complex_data,
            execution_time=0.123,
            tags=["test", "serialization"],
            metadata=complex_data
        )
        
        # Verificar que el archivo se creó
        if os.path.exists(log_file):
            print("✅ Archivo de log creado exitosamente!")
            
            # Leer y verificar el contenido
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            print(f"📄 Contenido del log ({len(log_content)} caracteres)")
            
            # Verificar que es JSON válido
            try:
                log_event = json.loads(log_content)
                print("✅ Log es JSON válido!")
                
                # Verificar que no hay tipos problemáticos
                check_json_types(log_event)
                
                return True
                
            except json.JSONDecodeError as e:
                print(f"❌ Error en JSON del log: {e}")
                return False
                
        else:
            print("❌ No se creó el archivo de log")
            return False
            
    except Exception as e:
        print(f"❌ Error en JsonLogger: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Limpiar
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def check_json_types(obj, path=""):
    """Verifica recursivamente que todos los valores son tipos JSON válidos"""
    
    if obj is None:
        return True
    elif isinstance(obj, (str, int, float, bool)):
        return True
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if not check_json_types(item, f"{path}[{i}]"):
                return False
        return True
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                print(f"❌ Clave no válida en {path}: {type(key).__name__} = {key}")
                return False
            if not check_json_types(value, f"{path}.{key}"):
                return False
        return True
    else:
        print(f"❌ Tipo no válido en {path}: {type(obj).__name__} = {obj}")
        return False


def test_serialize_for_json_function():
    """Test directo de la función serialize_for_json"""
    
    print("\n🧪 Probando función serialize_for_json directamente...")
    
    # Datos de prueba
    test_data = {
        'numpy_int64': np.int64(42),
        'numpy_float64': np.float64(3.14159),
        'numpy_array': np.array([1, 2, 3]),
        'numpy_dtype': np.dtype('int64'),
        'pandas_series': pd.Series([1, 2, 3], name='test'),
        'pandas_dataframe': pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
        'pandas_timestamp': pd.Timestamp('2023-01-01'),
        'mixed_dict': {
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
        serialized = serialize_for_json(test_data)
        
        # Verificar que se puede convertir a JSON
        json_str = json.dumps(serialized, ensure_ascii=False)
        print("✅ Serialización exitosa!")
        
        # Verificar tipos
        if check_json_types(serialized):
            print("✅ Todos los tipos son JSON válidos!")
            return True
        else:
            print("❌ Se encontraron tipos no válidos")
            return False
            
    except Exception as e:
        print(f"❌ Error en serialización: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Test de JsonLogger con Serialización Robusta")
    print("=" * 60)
    
    # Ejecutar tests
    test1 = test_serialize_for_json_function()
    test2 = test_logger_with_complex_data()
    
    print(f"\n📊 Resumen de tests:")
    print(f"  Función serialize_for_json: {'✅ PASÓ' if test1 else '❌ FALLÓ'}")
    print(f"  JsonLogger con datos complejos: {'✅ PASÓ' if test2 else '❌ FALLÓ'}")
    
    if all([test1, test2]):
        print("\n🎉 ¡Todos los tests de serialización pasaron!")
        print("💡 El JsonLogger está listo para manejar cualquier tipo de datos")
    else:
        print("\n⚠️ Algunos tests fallaron. Revisar errores arriba.") 