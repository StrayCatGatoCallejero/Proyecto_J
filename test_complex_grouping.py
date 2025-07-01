"""
Script de prueba para el sistema de agrupación compleja
======================================================

Prueba las funcionalidades de agrupación compleja con datos sintéticos.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Agregar el directorio del proyecto al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'proyecto_j', 'src'))

try:
    from complex_grouping import ComplexGrouping, detect_complex_grouping_in_question, execute_complex_grouping_from_question
    print("✅ Módulo de agrupación compleja importado correctamente")
except ImportError as e:
    print(f"❌ Error importando módulo: {e}")
    sys.exit(1)

def crear_datos_sinteticos():
    """Crea datos sintéticos para las pruebas."""
    np.random.seed(42)
    
    # Crear fechas
    fechas = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Crear datos sintéticos
    n_registros = len(fechas)
    
    datos = {
        'fecha': fechas,
        'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], n_registros),
        'ciudad': np.random.choice(['Ciudad A', 'Ciudad B', 'Ciudad C', 'Ciudad D'], n_registros),
        'categoria': np.random.choice(['Alto', 'Medio', 'Bajo'], n_registros),
        'ventas': np.random.randint(100, 1000, n_registros),
        'ingresos': np.random.uniform(1000, 5000, n_registros),
        'clientes': np.random.randint(10, 100, n_registros),
        'temperatura': np.random.uniform(15, 30, n_registros)
    }
    
    df = pd.DataFrame(datos)
    return df

def probar_deteccion_agrupacion():
    """Prueba la detección de agrupaciones complejas."""
    print("\n🔍 Probando detección de agrupaciones complejas...")
    
    df = crear_datos_sinteticos()
    grouping_system = ComplexGrouping(df)
    
    # Casos de prueba
    casos_prueba = [
        "calcular promedio de ventas por región",
        "contar registros agrupados por ciudad y categoría",
        "suma de ingresos por región y mes",
        "promedio de temperatura por ciudad en los últimos 30 días",
        "distribución de ventas por nivel jerárquico de región y ciudad",
        "promedio móvil de ingresos por ventana de 7 días"
    ]
    
    for i, pregunta in enumerate(casos_prueba, 1):
        print(f"\n📝 Caso {i}: {pregunta}")
        
        try:
            params = grouping_system.detect_complex_grouping(pregunta)
            print(f"   Tipo: {params['type']}")
            print(f"   Variables: {params['variables']}")
            print(f"   Operaciones: {params['operations']}")
            print(f"   Jerárquico: {params['hierarchical']}")
            print(f"   Rolling: {params['rolling']}")
            print(f"   Transformaciones: {params['transformations']}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def probar_ejecucion_agrupacion():
    """Prueba la ejecución de agrupaciones complejas."""
    print("\n⚙️ Probando ejecución de agrupaciones complejas...")
    
    df = crear_datos_sinteticos()
    
    # Casos de prueba
    casos_prueba = [
        ("promedio de ventas por región", ['ventas']),
        ("contar registros por ciudad y categoría", None),
        ("suma de ingresos por región", ['ingresos']),
        ("promedio de temperatura por ciudad", ['temperatura'])
    ]
    
    for i, (pregunta, target_vars) in enumerate(casos_prueba, 1):
        print(f"\n📊 Caso {i}: {pregunta}")
        
        try:
            result, insights = execute_complex_grouping_from_question(pregunta, df, target_vars)
            
            if not result.empty:
                print(f"   ✅ Resultado obtenido: {len(result)} filas")
                print(f"   Columnas: {list(result.columns)}")
                print(f"   Primeras filas:")
                print(result.head(3).to_string())
                print(f"   Insights: {insights[:100]}...")
            else:
                print(f"   ⚠️ No se obtuvieron resultados")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def probar_agrupacion_jerarquica():
    """Prueba agrupación jerárquica específica."""
    print("\n🏗️ Probando agrupación jerárquica...")
    
    df = crear_datos_sinteticos()
    grouping_system = ComplexGrouping(df)
    
    pregunta = "calcular promedio de ventas por región y ciudad con análisis jerárquico"
    
    try:
        params = grouping_system.detect_complex_grouping(pregunta)
        print(f"Parámetros detectados: {params}")
        
        result = grouping_system.execute_complex_grouping(params, ['ventas'])
        
        if not result.empty:
            print(f"✅ Agrupación jerárquica exitosa: {len(result)} filas")
            if 'nivel_jerarquico' in result.columns:
                print("✅ Columna de nivel jerárquico creada")
                print(result[['region', 'ciudad', 'nivel_jerarquico', 'ventas_mean']].head())
        else:
            print("⚠️ No se obtuvieron resultados")
            
    except Exception as e:
        print(f"❌ Error en agrupación jerárquica: {e}")

def probar_agrupacion_temporal():
    """Prueba agrupación temporal."""
    print("\n📅 Probando agrupación temporal...")
    
    df = crear_datos_sinteticos()
    grouping_system = ComplexGrouping(df)
    
    pregunta = "calcular promedio de ventas por mes"
    
    try:
        params = grouping_system.detect_complex_grouping(pregunta)
        print(f"Parámetros detectados: {params}")
        
        result = grouping_system.execute_complex_grouping(params, ['ventas'])
        
        if not result.empty:
            print(f"✅ Agrupación temporal exitosa: {len(result)} filas")
            print(result.head())
        else:
            print("⚠️ No se obtuvieron resultados")
            
    except Exception as e:
        print(f"❌ Error en agrupación temporal: {e}")

def probar_transformaciones():
    """Prueba transformaciones personalizadas."""
    print("\n🔄 Probando transformaciones personalizadas...")
    
    df = crear_datos_sinteticos()
    grouping_system = ComplexGrouping(df)
    
    pregunta = "calcular porcentaje de ventas por región"
    
    try:
        params = grouping_system.detect_complex_grouping(pregunta)
        print(f"Parámetros detectados: {params}")
        
        result = grouping_system.execute_complex_grouping(params, ['ventas'])
        
        if not result.empty:
            print(f"✅ Transformaciones aplicadas: {len(result)} filas")
            print("Columnas resultantes:", list(result.columns))
            if any('porcentaje' in col for col in result.columns):
                print("✅ Columna de porcentaje creada")
                print(result.head())
        else:
            print("⚠️ No se obtuvieron resultados")
            
    except Exception as e:
        print(f"❌ Error en transformaciones: {e}")

def main():
    """Función principal de pruebas."""
    print("🚀 Iniciando pruebas del sistema de agrupación compleja")
    print("=" * 60)
    
    # Crear datos de prueba
    print("📊 Creando datos sintéticos...")
    df = crear_datos_sinteticos()
    print(f"✅ Datos creados: {len(df)} registros, {len(df.columns)} columnas")
    print(f"Columnas: {list(df.columns)}")
    
    # Ejecutar pruebas
    probar_deteccion_agrupacion()
    probar_ejecucion_agrupacion()
    probar_agrupacion_jerarquica()
    probar_agrupacion_temporal()
    probar_transformaciones()
    
    print("\n" + "=" * 60)
    print("✅ Pruebas completadas")

if __name__ == "__main__":
    main() 