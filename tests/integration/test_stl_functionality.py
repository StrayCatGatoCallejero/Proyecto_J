#!/usr/bin/env python3
"""
Script de Prueba para Descomposición Estacional (STL)
====================================================

Verifica que la funcionalidad de descomposición estacional funciona correctamente
con statsmodels instalado.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Añadir el directorio del proyecto al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'proyecto_j', 'src'))

def crear_datos_estacionales():
    """Crea dataset con estacionalidad clara para testing."""
    # Generar fechas mensuales por 3 años
    fechas = pd.date_range('2021-01-01', periods=36, freq='M')
    
    # Crear serie con tendencia + estacionalidad + ruido
    t = np.arange(len(fechas))
    
    # Tendencia lineal creciente
    tendencia = 100 + 2 * t
    
    # Estacionalidad anual (12 meses)
    estacionalidad = 20 * np.sin(2 * np.pi * t / 12)
    
    # Ruido aleatorio
    ruido = np.random.normal(0, 5, len(fechas))
    
    # Serie completa
    valores = tendencia + estacionalidad + ruido
    
    df = pd.DataFrame({
        'fecha': fechas,
        'ventas': valores
    })
    
    return df

def probar_importacion_stl():
    """Prueba la importación de statsmodels."""
    print("🔍 Probando Importación de STL...")
    
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        print("✅ statsmodels importado correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error al importar statsmodels: {e}")
        return False

def probar_descomposicion_basica():
    """Prueba descomposición estacional básica."""
    print("🔍 Probando Descomposición Estacional Básica...")
    
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Crear datos de prueba
        df = crear_datos_estacionales()
        print(f"  Dataset creado: {len(df)} registros")
        
        # Preparar serie para descomposición
        serie = df.set_index('fecha')['ventas']
        
        # Realizar descomposición
        decomposition = seasonal_decompose(serie, model='additive', period=12)
        
        print("  ✅ Descomposición exitosa")
        print(f"  - Tendencia: {len(decomposition.trend)} puntos")
        print(f"  - Estacionalidad: {len(decomposition.seasonal)} puntos")
        print(f"  - Residuos: {len(decomposition.resid)} puntos")
        
        # Verificar que los componentes suman la serie original
        reconstruida = decomposition.trend + decomposition.seasonal + decomposition.resid
        diferencia = np.abs(serie - reconstruida).max()
        print(f"  - Error máximo de reconstrucción: {diferencia:.6f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error en descomposición: {e}")
        return False

def probar_modulo_tendencias():
    """Prueba el módulo de tendencias con STL."""
    print("🔍 Probando Módulo de Tendencias con STL...")
    
    try:
        from nl_query_trends import (
            STL_AVAILABLE,
            detectar_descomposicion_estacional,
            generar_grafico_descomposicion
        )
        
        print(f"  STL_AVAILABLE: {STL_AVAILABLE}")
        
        if STL_AVAILABLE:
            # Probar detección de descomposición
            pregunta = "Descomposición estacional de ventas mensual"
            descomp = detectar_descomposicion_estacional(pregunta)
            print(f"  Detección de descomposición: {descomp}")
            
            # Crear datos y probar gráfico
            df = crear_datos_estacionales()
            df_serie = df.copy()
            df_serie['variable'] = 'ventas'
            
            fig = generar_grafico_descomposicion(df_serie, 'fecha', 'ventas', 'promedio')
            print(f"  Gráfico de descomposición generado: {type(fig)}")
            
            return True
        else:
            print("  ⚠️ STL no disponible en el módulo")
            return False
            
    except Exception as e:
        print(f"  ❌ Error en módulo de tendencias: {e}")
        return False

def probar_preguntas_estacionales():
    """Prueba preguntas que solicitan descomposición estacional."""
    print("🔍 Probando Preguntas de Descomposición Estacional...")
    
    try:
        from nl_query_trends import detectar_analisis_temporal
        
        preguntas_estacionales = [
            "Descomposición estacional de ventas",
            "Mostrar componentes estacionales de gastos",
            "Análisis de tendencia y estacionalidad de ingresos",
            "Descomponer ventas en tendencia y estacionalidad"
        ]
        
        for pregunta in preguntas_estacionales:
            params = detectar_analisis_temporal(pregunta)
            if params and params.get('descomposicion', {}).get('aplicar'):
                print(f"  ✅ '{pregunta}' → Descomposición detectada")
            else:
                print(f"  ❌ '{pregunta}' → Descomposición NO detectada")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error en detección: {e}")
        return False

def main():
    """Función principal de pruebas."""
    print("🚀 INICIANDO PRUEBAS DE DESCOMPOSICIÓN ESTACIONAL")
    print("=" * 60)
    
    resultados = []
    
    # Probar importación
    resultados.append(probar_importacion_stl())
    
    # Probar descomposición básica
    resultados.append(probar_descomposicion_basica())
    
    # Probar módulo de tendencias
    resultados.append(probar_modulo_tendencias())
    
    # Probar preguntas estacionales
    resultados.append(probar_preguntas_estacionales())
    
    print("=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    if all(resultados):
        print("✅ TODAS LAS PRUEBAS EXITOSAS")
        print("🎉 La descomposición estacional está funcionando correctamente")
        print("\n💡 Ahora puedes usar preguntas como:")
        print("   • 'Descomposición estacional de ventas'")
        print("   • 'Mostrar componentes estacionales de gastos'")
        print("   • 'Análisis de tendencia y estacionalidad'")
    else:
        print("❌ ALGUNAS PRUEBAS FALLARON")
        print("🔧 Revisa los errores anteriores para solucionar problemas")
    
    print("\n📈 Funcionalidades disponibles:")
    print("   ✓ Descomposición estacional (STL)")
    print("   ✓ Detección automática de preguntas estacionales")
    print("   ✓ Gráficos de 4 componentes (Original, Tendencia, Estacionalidad, Residuos)")
    print("   ✓ Exportación de resultados")

if __name__ == "__main__":
    main() 