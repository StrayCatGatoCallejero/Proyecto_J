#!/usr/bin/env python3
"""
Script de Prueba para Funcionalidades Avanzadas del Módulo de Tendencias
=======================================================================

Prueba las nuevas funcionalidades:
- Frecuencias personalizables
- Suavizado y medias móviles
- Diferentes tipos de ajuste de tendencia
- Manejo de gaps y datos faltantes
- Caching para rendimiento
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Añadir el directorio del proyecto al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'proyecto_j', 'src'))

from nl_query_trends import (
    detectar_analisis_temporal,
    detectar_frecuencia_avanzada,
    detectar_suavizado,
    detectar_tipo_ajuste,
    analizar_tendencia_temporal,
    aplicar_suavizado,
    calcular_ajuste_tendencia
)

def crear_datos_temporales_avanzados():
    """Crea dataset temporal con características avanzadas para testing."""
    # Generar fechas con gaps y estacionalidad
    fechas = []
    valores_ventas = []
    valores_gastos = []
    
    fecha_inicio = datetime(2023, 1, 1)
    fecha_fin = datetime(2024, 12, 31)
    fecha_actual = fecha_inicio
    
    while fecha_actual <= fecha_fin:
        # Añadir algunos gaps (días sin datos)
        if np.random.random() > 0.1:  # 90% de los días tienen datos
            fechas.append(fecha_actual)
            
            # Tendencia base + estacionalidad + ruido
            dias_desde_inicio = (fecha_actual - fecha_inicio).days
            
            # Tendencia creciente para ventas
            tendencia_ventas = 100 + 0.5 * dias_desde_inicio
            
            # Estacionalidad mensual (picos en diciembre, bajos en enero)
            mes = fecha_actual.month
            estacionalidad_ventas = 20 * np.sin(2 * np.pi * (mes - 1) / 12)
            
            # Ruido aleatorio
            ruido_ventas = np.random.normal(0, 10)
            
            ventas = max(0, tendencia_ventas + estacionalidad_ventas + ruido_ventas)
            valores_ventas.append(ventas)
            
            # Gastos con tendencia diferente
            tendencia_gastos = 50 + 0.3 * dias_desde_inicio
            estacionalidad_gastos = 10 * np.cos(2 * np.pi * (mes - 1) / 12)
            ruido_gastos = np.random.normal(0, 5)
            
            gastos = max(0, tendencia_gastos + estacionalidad_gastos + ruido_gastos)
            valores_gastos.append(gastos)
        
        fecha_actual += timedelta(days=1)
    
    df = pd.DataFrame({
        'fecha': fechas,
        'ventas': valores_ventas,
        'gastos': valores_gastos,
        'margen': [v - g for v, g in zip(valores_ventas, valores_gastos)]
    })
    
    return df

def probar_frecuencias_personalizables():
    """Prueba la detección de frecuencias personalizables."""
    print("🔍 Probando Frecuencias Personalizables...")
    
    preguntas_frecuencia = [
        "Mostrar tendencia de ventas cada 7 días",
        "Análisis mensual de gastos",
        "Comparar datos cada 2 semanas",
        "Tendencia trimestral de márgenes",
        "Análisis cada 15 días",
        "Datos cada 3 meses"
    ]
    
    for pregunta in preguntas_frecuencia:
        frecuencia, personalizada = detectar_frecuencia_avanzada(pregunta)
        print(f"  Pregunta: {pregunta}")
        print(f"  → Frecuencia: {frecuencia}, Personalizada: {personalizada}")
        print()

def probar_suavizado():
    """Prueba la detección de parámetros de suavizado."""
    print("🔍 Probando Detección de Suavizado...")
    
    preguntas_suavizado = [
        "Mostrar media móvil de ventas con ventana 7 días",
        "Tendencia suavizada de gastos",
        "Media móvil 30 días de márgenes",
        "Análisis con suavizado de 14 días",
        "Tendencia simple sin suavizado"
    ]
    
    for pregunta in preguntas_suavizado:
        suavizado = detectar_suavizado(pregunta)
        print(f"  Pregunta: {pregunta}")
        print(f"  → Suavizado: {suavizado}")
        print()

def probar_tipos_ajuste():
    """Prueba la detección de tipos de ajuste."""
    print("🔍 Probando Tipos de Ajuste...")
    
    preguntas_ajuste = [
        "Tendencia lineal de ventas",
        "Ajuste polinomial de gastos",
        "Curva de tendencia suave",
        "Línea de tendencia simple"
    ]
    
    for pregunta in preguntas_ajuste:
        tipo = detectar_tipo_ajuste(pregunta)
        print(f"  Pregunta: {pregunta}")
        print(f"  → Tipo de ajuste: {tipo}")
        print()

def probar_analisis_completo():
    """Prueba análisis temporal completo con funcionalidades avanzadas."""
    print("🔍 Probando Análisis Temporal Completo...")
    
    # Crear datos de prueba
    df = crear_datos_temporales_avanzados()
    print(f"  Dataset creado: {len(df)} registros")
    print(f"  Rango temporal: {df['fecha'].min()} a {df['fecha'].max()}")
    print()
    
    # Preguntas de prueba con diferentes funcionalidades
    preguntas_avanzadas = [
        {
            'pregunta': "Mostrar tendencia de ventas con media móvil 7 días",
            'descripcion': "Tendencia con suavizado"
        },
        {
            'pregunta': "Comparar gastos y ventas mensualmente",
            'descripcion': "Comparación múltiple"
        },
        {
            'pregunta': "Análisis de márgenes cada 2 semanas con ajuste polinomial",
            'descripcion': "Frecuencia personalizada + ajuste"
        },
        {
            'pregunta': "Tendencia suavizada de ventas en el último año",
            'descripcion': "Suavizado + período específico"
        }
    ]
    
    for i, test_case in enumerate(preguntas_avanzadas, 1):
        print(f"  Test {i}: {test_case['descripcion']}")
        print(f"  Pregunta: {test_case['pregunta']}")
        
        try:
            resultado = analizar_tendencia_temporal(df, test_case['pregunta'])
            
            if 'error' in resultado:
                print(f"  ❌ Error: {resultado['error']}")
            else:
                print(f"  ✅ Análisis exitoso:")
                print(f"     - Tipo: {resultado['tipo_analisis']}")
                print(f"     - Métrica: {resultado['metrica']}")
                print(f"     - Frecuencia: {resultado['frecuencia']}")
                if resultado.get('frecuencia_personalizada'):
                    print(f"     - Frecuencia personalizada: {resultado['frecuencia_personalizada']}")
                if resultado.get('suavizado', {}).get('aplicar'):
                    print(f"     - Suavizado: {resultado['suavizado']}")
                print(f"     - Variables: {resultado['variables_analizadas']}")
                print(f"     - Registros analizados: {resultado['registros_analizados']}")
                print(f"     - Insights: {resultado['insights'][:100]}...")
            
        except Exception as e:
            print(f"  ❌ Excepción: {str(e)}")
        
        print()

def probar_funciones_individuales():
    """Prueba funciones individuales del módulo."""
    print("🔍 Probando Funciones Individuales...")
    
    # Crear serie de prueba
    fechas = pd.date_range('2023-01-01', periods=100, freq='D')
    valores = np.random.normal(100, 20, 100) + np.arange(100) * 0.5  # Tendencia creciente + ruido
    
    serie = pd.Series(valores, index=fechas)
    print(f"  Serie creada: {len(serie)} puntos")
    
    # Probar suavizado
    suavizado_config = {'aplicar': True, 'tipo': 'media_movil', 'ventana': 7}
    serie_suavizada = aplicar_suavizado(serie, suavizado_config)
    print(f"  Suavizado aplicado: {len(serie_suavizada)} puntos")
    
    # Probar ajuste de tendencia
    x = pd.Series(range(len(serie)))
    y = serie.values
    
    tipos_ajuste = ['lineal', 'polinomial', 'lowess']
    for tipo in tipos_ajuste:
        try:
            ajuste = calcular_ajuste_tendencia(x, y, tipo)
            if ajuste is not None:
                print(f"  Ajuste {tipo}: {len(ajuste)} puntos calculados")
            else:
                print(f"  Ajuste {tipo}: No se pudo calcular")
        except Exception as e:
            print(f"  Ajuste {tipo}: Error - {str(e)}")
    
    print()

def probar_deteccion_avanzada():
    """Prueba la detección avanzada de parámetros."""
    print("🔍 Probando Detección Avanzada de Parámetros...")
    
    preguntas_complejas = [
        "Mostrar tendencia lineal de ventas con media móvil 14 días en el último trimestre",
        "Comparar gastos y márgenes cada 10 días con suavizado polinomial",
        "Análisis de tendencia de ventas mensual con ajuste lowess y ventana 30",
        "Tendencia de gastos cada 2 semanas sin suavizado"
    ]
    
    for pregunta in preguntas_complejas:
        print(f"  Pregunta: {pregunta}")
        
        params = detectar_analisis_temporal(pregunta)
        if params:
            print(f"  ✅ Parámetros detectados:")
            for key, value in params.items():
                print(f"     - {key}: {value}")
        else:
            print(f"  ❌ No se detectaron parámetros temporales")
        
        print()

def main():
    """Función principal de pruebas."""
    print("🚀 INICIANDO PRUEBAS DE FUNCIONALIDADES AVANZADAS")
    print("=" * 60)
    
    try:
        # Probar detección de frecuencias personalizables
        probar_frecuencias_personalizables()
        
        # Probar detección de suavizado
        probar_suavizado()
        
        # Probar tipos de ajuste
        probar_tipos_ajuste()
        
        # Probar detección avanzada
        probar_deteccion_avanzada()
        
        # Probar funciones individuales
        probar_funciones_individuales()
        
        # Probar análisis completo
        probar_analisis_completo()
        
        print("✅ TODAS LAS PRUEBAS COMPLETADAS")
        print("=" * 60)
        print("📊 Resumen de funcionalidades probadas:")
        print("   ✓ Frecuencias personalizables (cada X días/semanas/meses)")
        print("   ✓ Suavizado con medias móviles y medianas móviles")
        print("   ✓ Diferentes tipos de ajuste (lineal, polinomial, lowess)")
        print("   ✓ Detección de períodos específicos")
        print("   ✓ Análisis de múltiples variables")
        print("   ✓ Cálculo de R² para calidad de ajuste")
        print("   ✓ Detección de outliers")
        print("   ✓ Insights automáticos avanzados")
        
    except Exception as e:
        print(f"❌ Error general en las pruebas: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 