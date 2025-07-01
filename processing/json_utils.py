"""
Utilitarios de Serialización JSON
================================

Sistema centralizado para convertir objetos pandas/numpy a tipos primitivos JSON-friendly.
Garantiza que ningún objeto no serializable termine en logs, metadata o exportaciones JSON.
"""

import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union, Optional
from datetime import datetime, date
import warnings
import traceback


def serialize_for_json(obj: Any) -> Any:
    """
    Convierte recursivamente objetos pandas/numpy a tipos primitivos JSON-friendly.
    
    Args:
        obj: Objeto a serializar (cualquier tipo)
        
    Returns:
        Objeto serializable (str, int, float, bool, list, dict, None)
        
    Raises:
        TypeError: Si no se puede serializar el objeto
    """
    
    # None
    if obj is None:
        return None
    
    # Tipos primitivos ya serializables
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Numpy types
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.datetime64,)):
        return str(obj)
    if isinstance(obj, (np.dtype,)):
        return str(obj)
    
    # Pandas types
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Index):
        return obj.tolist()
    if isinstance(obj, pd.Categorical):
        return obj.tolist()
    if isinstance(obj, pd.CategoricalDtype):
        return str(obj)
    if isinstance(obj, pd.DatetimeTZDtype):
        return str(obj)
    if isinstance(obj, pd.Period):
        return str(obj)
    if isinstance(obj, pd.Interval):
        return str(obj)
    
    # Datetime types
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    
    # Lists - aplicar recursivamente
    if isinstance(obj, (list, tuple)):
        try:
            return [serialize_for_json(item) for item in obj]
        except Exception as e:
            warnings.warn(f"Error serializando lista: {e}")
            return [str(item) for item in obj]
    
    # Dicts - aplicar recursivamente
    if isinstance(obj, dict):
        try:
            result = {}
            for key, value in obj.items():
                # Serializar clave
                if isinstance(key, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    key = int(key)
                elif isinstance(key, (np.floating, np.float64, np.float32, np.float16)):
                    key = float(key)
                elif isinstance(key, (np.bool_,)):
                    key = bool(key)
                elif isinstance(key, (np.dtype, pd.CategoricalDtype)):
                    key = str(key)
                elif not isinstance(key, (str, int, float, bool)):
                    key = str(key)
                
                # Serializar valor
                result[key] = serialize_for_json(value)
            return result
        except Exception as e:
            warnings.warn(f"Error serializando diccionario: {e}")
            return {str(k): str(v) for k, v in obj.items()}
    
    # Sets
    if isinstance(obj, set):
        try:
            return [serialize_for_json(item) for item in obj]
        except Exception as e:
            warnings.warn(f"Error serializando set: {e}")
            return [str(item) for item in obj]
    
    # Objetos con método to_dict
    if hasattr(obj, "to_dict"):
        try:
            return serialize_for_json(obj.to_dict())
        except Exception:
            pass
    
    # Objetos con método to_json
    if hasattr(obj, "to_json"):
        try:
            return obj.to_json()
        except Exception:
            pass
    
    # Objetos con método __dict__
    if hasattr(obj, "__dict__"):
        try:
            # Filtrar atributos privados y métodos
            safe_dict = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_') and not callable(value):
                    safe_dict[key] = serialize_for_json(value)
            return safe_dict
        except Exception:
            pass
    
    # Plotly Figure
    try:
        import plotly.graph_objs as go
        if isinstance(obj, go.Figure):
            return {
                "type": "plotly_figure",
                "layout_title": obj.layout.title.text if hasattr(obj.layout, 'title') and obj.layout.title else None,
                "data_count": len(obj.data),
                "data_types": [trace.type for trace in obj.data] if obj.data else []
            }
    except ImportError:
        pass
    
    # Fallback: convertir a string
    try:
        return str(obj)
    except Exception as e:
        warnings.warn(f"No se pudo serializar objeto de tipo {type(obj)}: {e}")
        return f"<{type(obj).__name__}: {str(obj)[:100]}>"


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Serializa un objeto a JSON de forma segura usando serialize_for_json.
    
    Args:
        obj: Objeto a serializar
        **kwargs: Argumentos adicionales para json.dumps
        
    Returns:
        String JSON válido
        
    Raises:
        TypeError: Si no se puede serializar después de aplicar serialize_for_json
    """
    try:
        serialized_obj = serialize_for_json(obj)
        return json.dumps(serialized_obj, ensure_ascii=False, **kwargs)
    except Exception as e:
        # Último recurso: convertir todo a string
        try:
            return json.dumps(str(obj), ensure_ascii=False, **kwargs)
        except Exception:
            raise TypeError(f"No se pudo serializar objeto: {e}")


def validate_json_serializable(obj: Any) -> bool:
    """
    Valida si un objeto es serializable a JSON después de aplicar serialize_for_json.
    
    Args:
        obj: Objeto a validar
        
    Returns:
        True si es serializable, False en caso contrario
    """
    try:
        serialized = serialize_for_json(obj)
        json.dumps(serialized)
        return True
    except Exception:
        return False


def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Limpia un diccionario para serialización JSON.
    
    Args:
        data: Diccionario a limpiar
        
    Returns:
        Diccionario limpio con solo tipos serializables
    """
    return serialize_for_json(data)


def clean_metadata_for_logging(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Limpia metadatos específicamente para logging.
    
    Args:
        metadata: Metadatos a limpiar
        
    Returns:
        Metadatos limpios para logging
    """
    try:
        return serialize_for_json(metadata)
    except Exception as e:
        warnings.warn(f"Error limpiando metadatos para logging: {e}")
        return {"error": "metadata_serialization_failed", "original_error": str(e)}


def clean_metrics_for_logging(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Limpia métricas específicamente para logging.
    
    Args:
        metrics: Métricas a limpiar
        
    Returns:
        Métricas limpias para logging
    """
    try:
        return serialize_for_json(metrics)
    except Exception as e:
        warnings.warn(f"Error limpiando métricas para logging: {e}")
        return {"error": "metrics_serialization_failed", "original_error": str(e)}


# Función de conveniencia para uso directo en imports
def to_json_safe(obj: Any) -> Any:
    """
    Alias para serialize_for_json para uso directo.
    
    Args:
        obj: Objeto a serializar
        
    Returns:
        Objeto serializable
    """
    return serialize_for_json(obj)


if __name__ == "__main__":
    # Tests básicos
    print("🧪 Probando utilitarios de serialización JSON...")
    
    # Test con tipos complejos
    test_data = {
        'numpy_int': np.int64(42),
        'numpy_float': np.float64(3.14159),
        'numpy_array': np.array([1, 2, 3]),
        'pandas_series': pd.Series([1, 2, 3]),
        'pandas_timestamp': pd.Timestamp('2023-01-01'),
        'pandas_categorical': pd.Categorical(['A', 'B', 'A']),
        'pandas_dtype': pd.CategoricalDtype(['A', 'B']),
        'numpy_dtype': np.dtype('int64'),
        'normal_dict': {'key': 'value'},
        'normal_list': [1, 2, 3],
        'mixed_data': {
            'numpy_in_dict': np.int64(100),
            'pandas_in_dict': pd.Series([1, 2, 3])
        }
    }
    
    try:
        serialized = serialize_for_json(test_data)
        json_str = json.dumps(serialized, indent=2)
        print("✅ Serialización exitosa!")
        print(f"📄 JSON generado ({len(json_str)} caracteres):")
        print(json_str[:500] + "..." if len(json_str) > 500 else json_str)
        
        # Validar que es JSON válido
        json.loads(json_str)
        print("✅ JSON válido!")
        
    except Exception as e:
        print(f"❌ Error en serialización: {e}")
        traceback.print_exc() 