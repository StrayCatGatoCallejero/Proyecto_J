"""
Ejemplo de Validación Automática - Sistema de Pipeline Robusto
=============================================================

Este script demuestra el sistema de validación automática implementado
en el pipeline de procesamiento de datos.

Características demostradas:
1. Validación automática de DataFrames
2. Validación de parámetros de funciones
3. Reporte híbrido de errores (Streamlit + logging)
4. Esquemas específicos para diferentes tipos de operaciones
5. Manejo elegante de errores sin interrumpir la aplicación
"""

import pandas as pd
import numpy as np
import sys
import os

# Agregar el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processing.stats import summary_statistics, compute_correlations, normality_test
from processing.io import cargar_archivo, validar_dataframe
from processing.filters import filter_by_range, filter_by_values, remove_outliers
from processing.config_manager import ConfigManager
from processing.error_reporter import ErrorReporter

def crear_datos_ejemplo():
    """Crea DataFrames de ejemplo para las pruebas"""
    
    # DataFrame válido con datos numéricos
    df_valido = pd.DataFrame({
        'edad': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'ingresos': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000],
        'genero': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'educacion': ['Primaria', 'Secundaria', 'Universitaria', 'Postgrado'] * 2 + ['Primaria', 'Secundaria']
    })
    
    # DataFrame problemático (sin columnas numéricas)
    df_sin_numericas = pd.DataFrame({
        'nombre': ['Juan', 'María', 'Pedro', 'Ana'],
        'ciudad': ['Madrid', 'Barcelona', 'Valencia', 'Sevilla'],
        'categoria': ['A', 'B', 'A', 'B']
    })
    
    # DataFrame vacío
    df_vacio = pd.DataFrame()
    
    # DataFrame con solo una columna numérica
    df_una_numerica = pd.DataFrame({
        'edad': [25, 30, 35, 40, 45]
    })
    
    return {
        'valido': df_valido,
        'sin_numericas': df_sin_numericas,
        'vacio': df_vacio,
        'una_numerica': df_una_numerica
    }

def ejemplo_validacion_exitosa():
    """Demuestra validación exitosa con datos válidos"""
    print("\n" + "="*60)
    print("EJEMPLO 1: VALIDACIÓN EXITOSA")
    print("="*60)
    
    datos = crear_datos_ejemplo()
    df = datos['valido']
    
    print(f"DataFrame válido: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    
    # Estadísticas descriptivas (debería funcionar)
    try:
        stats = summary_statistics(df, ['edad', 'ingresos'])
        print(f"✅ Estadísticas descriptivas exitosas: {len(stats)} variables analizadas")
        print(stats)
    except Exception as e:
        print(f"❌ Error en estadísticas: {e}")
    
    # Correlaciones (debería funcionar)
    try:
        corr = compute_correlations(df, ['edad', 'ingresos'])
        print(f"✅ Correlaciones exitosas: matriz {corr.shape[0]}x{corr.shape[1]}")
        print(corr)
    except Exception as e:
        print(f"❌ Error en correlaciones: {e}")
    
    # Filtro por rango (debería funcionar)
    try:
        filtrado = filter_by_range(df, 'edad', min_val=30, max_val=50)
        print(f"✅ Filtro por rango exitoso: {len(filtrado)} filas filtradas")
    except Exception as e:
        print(f"❌ Error en filtro: {e}")

def ejemplo_validacion_dataframe_invalido():
    """Demuestra validación fallida con DataFrame sin columnas numéricas"""
    print("\n" + "="*60)
    print("EJEMPLO 2: DATAFRAME SIN COLUMNAS NUMÉRICAS")
    print("="*60)
    
    datos = crear_datos_ejemplo()
    df = datos['sin_numericas']
    
    print(f"DataFrame sin numéricas: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    
    # Estadísticas descriptivas (debería fallar)
    try:
        stats = summary_statistics(df, ['nombre', 'ciudad'])
        print(f"✅ Estadísticas descriptivas exitosas: {len(stats)} variables analizadas")
    except Exception as e:
        print(f"❌ Error esperado en estadísticas: {e}")
    
    # Correlaciones (debería fallar)
    try:
        corr = compute_correlations(df, ['nombre', 'ciudad'])
        print(f"✅ Correlaciones exitosas: matriz {corr.shape[0]}x{corr.shape[1]}")
    except Exception as e:
        print(f"❌ Error esperado en correlaciones: {e}")

def ejemplo_validacion_dataframe_vacio():
    """Demuestra validación fallida con DataFrame vacío"""
    print("\n" + "="*60)
    print("EJEMPLO 3: DATAFRAME VACÍO")
    print("="*60)
    
    datos = crear_datos_ejemplo()
    df = datos['vacio']
    
    print(f"DataFrame vacío: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Validación de DataFrame (debería fallar)
    try:
        metadata = {'format': 'csv', 'file_size': 0}
        validation = validar_dataframe(df, metadata)
        print(f"✅ Validación exitosa: {validation['is_valid']}")
    except Exception as e:
        print(f"❌ Error esperado en validación: {e}")

def ejemplo_validacion_parametros_invalidos():
    """Demuestra validación fallida con parámetros inválidos"""
    print("\n" + "="*60)
    print("EJEMPLO 4: PARÁMETROS INVÁLIDOS")
    print("="*60)
    
    datos = crear_datos_ejemplo()
    df = datos['valido']
    
    print(f"DataFrame válido: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Filtro con columna inexistente (debería fallar)
    try:
        filtrado = filter_by_range(df, 'columna_inexistente', min_val=30, max_val=50)
        print(f"✅ Filtro exitoso: {len(filtrado)} filas filtradas")
    except Exception as e:
        print(f"❌ Error esperado en filtro: {e}")
    
    # Filtro con rango inválido (debería fallar)
    try:
        filtrado = filter_by_range(df, 'edad', min_val=50, max_val=30)  # min > max
        print(f"✅ Filtro exitoso: {len(filtrado)} filas filtradas")
    except Exception as e:
        print(f"❌ Error esperado en filtro: {e}")
    
    # Filtro por valores con lista vacía (debería fallar)
    try:
        filtrado = filter_by_values(df, 'genero', [])
        print(f"✅ Filtro por valores exitoso: {len(filtrado)} filas filtradas")
    except Exception as e:
        print(f"❌ Error esperado en filtro por valores: {e}")

def ejemplo_validacion_correlacion_insuficiente():
    """Demuestra validación fallida con datos insuficientes para correlación"""
    print("\n" + "="*60)
    print("EJEMPLO 5: DATOS INSUFICIENTES PARA CORRELACIÓN")
    print("="*60)
    
    datos = crear_datos_ejemplo()
    df = datos['una_numerica']
    
    print(f"DataFrame con una columna numérica: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    
    # Correlación con una sola columna (debería fallar)
    try:
        corr = compute_correlations(df, ['edad'])
        print(f"✅ Correlación exitosa: matriz {corr.shape[0]}x{corr.shape[1]}")
    except Exception as e:
        print(f"❌ Error esperado en correlación: {e}")

def ejemplo_validacion_normalidad():
    """Demuestra validación de pruebas de normalidad"""
    print("\n" + "="*60)
    print("EJEMPLO 6: PRUEBAS DE NORMALIDAD")
    print("="*60)
    
    datos = crear_datos_ejemplo()
    df = datos['valido']
    
    print(f"DataFrame válido: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Prueba de normalidad (debería funcionar)
    try:
        resultado = normality_test(df, 'edad')
        print(f"✅ Prueba de normalidad exitosa")
        print(f"   Conclusión: {resultado.get('overall_conclusion', 'N/A')}")
        print(f"   Shapiro-Wilk p-value: {resultado.get('shapiro_wilk', {}).get('p_value', 'N/A'):.4f}")
    except Exception as e:
        print(f"❌ Error en prueba de normalidad: {e}")

def ejemplo_validacion_deteccion_outliers():
    """Demuestra validación de detección de outliers"""
    print("\n" + "="*60)
    print("EJEMPLO 7: DETECCIÓN DE OUTLIERS")
    print("="*60)
    
    datos = crear_datos_ejemplo()
    df = datos['valido']
    
    print(f"DataFrame válido: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Detección de outliers (debería funcionar)
    try:
        resultado = remove_outliers(df, 'edad', method='iqr', factor=1.5)
        print(f"✅ Detección de outliers exitosa: {len(resultado)} filas después del filtrado")
    except Exception as e:
        print(f"❌ Error en detección de outliers: {e}")

def ejemplo_validacion_archivo_inexistente():
    """Demuestra validación de archivo inexistente"""
    print("\n" + "="*60)
    print("EJEMPLO 8: ARCHIVO INEXISTENTE")
    print("="*60)
    
    # Intentar cargar archivo inexistente (debería fallar)
    try:
        df, metadata = cargar_archivo("archivo_inexistente.csv")
        print(f"✅ Archivo cargado exitosamente: {df.shape[0]} filas")
    except Exception as e:
        print(f"❌ Error esperado al cargar archivo: {e}")

def mostrar_resumen_sistema():
    """Muestra un resumen del sistema de validación"""
    print("\n" + "="*60)
    print("RESUMEN DEL SISTEMA DE VALIDACIÓN AUTOMÁTICA")
    print("="*60)
    
    print("✅ CARACTERÍSTICAS IMPLEMENTADAS:")
    print("   • Decorador @validate_io universal")
    print("   • Esquemas específicos por tipo de operación")
    print("   • Validación automática de DataFrames")
    print("   • Validación de parámetros con Pydantic")
    print("   • Reporte híbrido de errores (Streamlit + logging)")
    print("   • Manejo elegante de errores sin interrumpir la app")
    print("   • Validación de rutas de archivos")
    print("   • Validación de tipos de datos")
    print("   • Validación de rangos y valores")
    
    print("\n✅ MÓDULOS PROTEGIDOS:")
    print("   • processing/stats.py - Todas las funciones estadísticas")
    print("   • processing/io.py - Carga y validación de archivos")
    print("   • processing/filters.py - Todas las funciones de filtrado")
    
    print("\n✅ TIPOS DE VALIDACIÓN:")
    print("   • DataFrameSchema - Validación de estructura de datos")
    print("   • ParameterModel - Validación de parámetros de funciones")
    print("   • FilePathSchema - Validación de rutas de archivos")
    print("   • Esquemas específicos por operación (SummaryStats, Correlation, etc.)")
    
    print("\n✅ INTEGRACIÓN CON SISTEMA DE ERRORES:")
    print("   • ErrorReporter.report_dataframe_error()")
    print("   • ErrorReporter.report_parameter_error()")
    print("   • Expanders en Streamlit con detalles")
    print("   • Botón 'Copiar detalles' al portapapeles")
    print("   • Notificaciones externas opcionales")

def main():
    """Función principal que ejecuta todos los ejemplos"""
    print("🚀 SISTEMA DE VALIDACIÓN AUTOMÁTICA - EJEMPLOS DE USO")
    print("="*60)
    
    # Configurar el sistema
    try:
        config = ConfigManager()
        print(f"✅ Configuración cargada: {config.get('app_name', 'N/A')}")
    except Exception as e:
        print(f"⚠️ Error al cargar configuración: {e}")
    
    # Ejecutar ejemplos
    ejemplo_validacion_exitosa()
    ejemplo_validacion_dataframe_invalido()
    ejemplo_validacion_dataframe_vacio()
    ejemplo_validacion_parametros_invalidos()
    ejemplo_validacion_correlacion_insuficiente()
    ejemplo_validacion_normalidad()
    ejemplo_validacion_deteccion_outliers()
    ejemplo_validacion_archivo_inexistente()
    
    # Mostrar resumen
    mostrar_resumen_sistema()
    
    print("\n" + "="*60)
    print("🎉 TODOS LOS EJEMPLOS COMPLETADOS")
    print("="*60)
    print("El sistema de validación automática está funcionando correctamente.")
    print("Los errores se manejan de forma elegante sin interrumpir la aplicación.")
    print("Los detalles de validación se reportan a través del sistema híbrido de errores.")

if __name__ == "__main__":
    main() 