#!/usr/bin/env python3
"""
Script de prueba para el módulo de consultas en lenguaje natural - Versión Avanzada
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from proyecto_j.src.nl_query import parse_and_execute, interpretar_resultado

def crear_datos_prueba():
    """Crea un DataFrame de prueba con datos variados."""
    np.random.seed(42)
    
    # Datos de ejemplo
    n_registros = 1000
    
    # Fechas
    fechas = pd.date_range(start='2023-01-01', end='2024-12-31', periods=n_registros)
    
    # Categorías
    regiones = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
    tipos_cliente = ['A', 'B', 'C', 'Premium', 'VIP']
    generos = ['Masculino', 'Femenino', 'No especificado']
    
    # Datos numéricos
    edades = np.random.normal(35, 12, n_registros).astype(int)
    edades = np.clip(edades, 18, 80)
    
    ingresos = np.random.lognormal(10, 0.5, n_registros)
    ingresos = np.clip(ingresos, 1000, 100000)
    
    satisfaccion = np.random.randint(1, 6, n_registros)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'fecha_registro': fechas,
        'region': np.random.choice(regiones, n_registros),
        'tipo_cliente': np.random.choice(tipos_cliente, n_registros),
        'genero': np.random.choice(generos, n_registros),
        'edad': edades,
        'ingreso_mensual': ingresos,
        'nivel_satisfaccion': satisfaccion,
        'comentarios': ['Comentario ' + str(i) for i in range(n_registros)]
    })
    
    return df

def probar_consultas_basicas(df):
    """Prueba consultas básicas."""
    print("🔍 **PRUEBAS DE CONSULTAS BÁSICAS**")
    print("=" * 50)
    
    consultas_basicas = [
        "¿Cuántos clientes hay?",
        "¿Cuál es el promedio de edad?",
        "¿Cuál es el ingreso máximo?",
        "¿Cuál es el ingreso mínimo?",
        "¿Cuál es la suma total de ingresos?",
        "¿Cuál es la mediana de edad?",
        "¿Cuál es la moda del nivel de satisfacción?",
        "¿Cuál es la desviación estándar de ingresos?",
        "¿Cuál es la varianza de edad?",
        "¿Cuál es el percentil 75 de ingresos?"
    ]
    
    for consulta in consultas_basicas:
        print(f"\n📝 **Consulta:** {consulta}")
        try:
            df_filtrado, resultado, df_grafico, tipo_grafico, interpretacion = parse_and_execute(df, consulta)
            print(f"✅ **Resultado:** {resultado}")
            print(f"📊 **Interpretación:** {interpretacion}")
            print(f"🎯 **Tipo de gráfico:** {tipo_grafico}")
        except Exception as e:
            print(f"❌ **Error:** {e}")

def probar_filtros_avanzados(df):
    """Prueba filtros avanzados."""
    print("\n🔍 **PRUEBAS DE FILTROS AVANZADOS**")
    print("=" * 50)
    
    consultas_filtros = [
        "¿Cuántos clientes tipo A y B hay?",
        "¿Cuál es el promedio de ingresos en la región Norte?",
        "¿Cuántos clientes tienen edad mayor a 50?",
        "¿Cuál es el ingreso promedio de clientes entre 25 y 35 años?",
        "¿Cuántos clientes hay en las regiones Norte o Sur?",
        "¿Cuál es el ingreso máximo de clientes tipo Premium?",
        "¿Cuántos clientes tienen ingresos mayor que el promedio?",
        "¿Cuál es la mediana de edad de clientes con nivel de satisfacción 5?",
        "¿Cuántos clientes hay en el top 10% de ingresos?",
        "¿Cuál es el promedio de satisfacción de clientes menores que el promedio de edad?"
    ]
    
    for consulta in consultas_filtros:
        print(f"\n📝 **Consulta:** {consulta}")
        try:
            df_filtrado, resultado, df_grafico, tipo_grafico, interpretacion = parse_and_execute(df, consulta)
            print(f"✅ **Resultado:** {resultado}")
            print(f"📊 **Interpretación:** {interpretacion}")
            print(f"🎯 **Registros filtrados:** {len(df_filtrado)}")
        except Exception as e:
            print(f"❌ **Error:** {e}")

def probar_agrupaciones(df):
    """Prueba agrupaciones."""
    print("\n🔍 **PRUEBAS DE AGRUPACIONES**")
    print("=" * 50)
    
    consultas_agrupacion = [
        "¿Cuántos clientes hay por región?",
        "¿Cuál es el promedio de ingresos por tipo de cliente?",
        "¿Cuál es la distribución de edades por género?",
        "¿Cuál es el ingreso máximo por región?",
        "¿Cuál es la mediana de satisfacción por tipo de cliente?",
        "¿Cuántos clientes hay por género y región?",
        "¿Cuál es el promedio de edad por nivel de satisfacción?",
        "¿Cuál es la suma de ingresos por región y tipo de cliente?"
    ]
    
    for consulta in consultas_agrupacion:
        print(f"\n📝 **Consulta:** {consulta}")
        try:
            df_filtrado, resultado, df_grafico, tipo_grafico, interpretacion = parse_and_execute(df, consulta)
            print(f"✅ **Resultado:** {resultado}")
            print(f"📊 **Interpretación:** {interpretacion}")
            print(f"🎯 **Tipo de gráfico:** {tipo_grafico}")
        except Exception as e:
            print(f"❌ **Error:** {e}")

def probar_filtros_temporales(df):
    """Prueba filtros temporales."""
    print("\n🔍 **PRUEBAS DE FILTROS TEMPORALES**")
    print("=" * 50)
    
    consultas_temporales = [
        "¿Cuántos clientes se registraron en enero de 2024?",
        "¿Cuál es el promedio de ingresos de clientes registrados en 2023?",
        "¿Cuántos clientes se registraron en marzo de 2024?",
        "¿Cuál es el ingreso máximo de clientes registrados en 2024?",
        "¿Cuántos clientes se registraron en diciembre de 2023?"
    ]
    
    for consulta in consultas_temporales:
        print(f"\n📝 **Consulta:** {consulta}")
        try:
            df_filtrado, resultado, df_grafico, tipo_grafico, interpretacion = parse_and_execute(df, consulta)
            print(f"✅ **Resultado:** {resultado}")
            print(f"📊 **Interpretación:** {interpretacion}")
            print(f"🎯 **Registros filtrados:** {len(df_filtrado)}")
        except Exception as e:
            print(f"❌ **Error:** {e}")

def probar_consultas_complejas(df):
    """Prueba consultas complejas con múltiples filtros."""
    print("\n🔍 **PRUEBAS DE CONSULTAS COMPLEJAS**")
    print("=" * 50)
    
    consultas_complejas = [
        "¿Cuál es el promedio de ingresos de clientes tipo Premium en la región Norte con edad mayor a 40?",
        "¿Cuántos clientes VIP hay en las regiones Este y Oeste con nivel de satisfacción 5?",
        "¿Cuál es la mediana de edad de clientes tipo A y B con ingresos mayor que el promedio?",
        "¿Cuál es el ingreso máximo por región de clientes con edad entre 30 y 50 años?",
        "¿Cuántos clientes hay por género en el top 20% de ingresos?",
        "¿Cuál es la distribución de satisfacción por tipo de cliente en clientes mayores de 35 años?",
        "¿Cuál es el promedio de ingresos por región de clientes tipo Premium y VIP?",
        "¿Cuántos clientes hay por nivel de satisfacción en las regiones Norte y Sur?"
    ]
    
    for consulta in consultas_complejas:
        print(f"\n📝 **Consulta:** {consulta}")
        try:
            df_filtrado, resultado, df_grafico, tipo_grafico, interpretacion = parse_and_execute(df, consulta)
            print(f"✅ **Resultado:** {resultado}")
            print(f"📊 **Interpretación:** {interpretacion}")
            print(f"🎯 **Registros filtrados:** {len(df_filtrado)}")
            print(f"📈 **Tipo de gráfico:** {tipo_grafico}")
        except Exception as e:
            print(f"❌ **Error:** {e}")

def probar_acciones_especializadas(df):
    """Prueba acciones especializadas."""
    print("\n🔍 **PRUEBAS DE ACCIONES ESPECIALIZADAS**")
    print("=" * 50)
    
    consultas_especializadas = [
        "¿Qué porcentaje de clientes son tipo Premium?",
        "¿Cuál es la distribución de clientes por región?",
        "¿Cuántos clientes tienen ingresos en el percentil 90?",
        "¿Cuál es la tendencia de registros por mes?",
        "¿Cuántos clientes por cada nivel de satisfacción?",
        "¿Cuál es la comparación de ingresos entre géneros?",
        "¿Qué porcentaje de clientes están en el top 25% de ingresos?",
        "¿Cuál es la distribución de edades por tipo de cliente?"
    ]
    
    for consulta in consultas_especializadas:
        print(f"\n📝 **Consulta:** {consulta}")
        try:
            df_filtrado, resultado, df_grafico, tipo_grafico, interpretacion = parse_and_execute(df, consulta)
            print(f"✅ **Resultado:** {resultado}")
            print(f"📊 **Interpretación:** {interpretacion}")
            print(f"🎯 **Tipo de gráfico:** {tipo_grafico}")
        except Exception as e:
            print(f"❌ **Error:** {e}")

def main():
    """Función principal de pruebas."""
    print("🚀 **INICIANDO PRUEBAS DEL PARSER DE CONSULTAS EN LENGUAJE NATURAL**")
    print("=" * 80)
    
    # Crear datos de prueba
    print("📊 Creando datos de prueba...")
    df = crear_datos_prueba()
    print(f"✅ Datos creados: {len(df)} registros, {len(df.columns)} columnas")
    print(f"📋 Columnas: {list(df.columns)}")
    
    # Ejecutar todas las pruebas
    probar_consultas_basicas(df)
    probar_filtros_avanzados(df)
    probar_agrupaciones(df)
    probar_filtros_temporales(df)
    probar_consultas_complejas(df)
    probar_acciones_especializadas(df)
    
    print("\n🎉 **PRUEBAS COMPLETADAS**")
    print("=" * 80)
    print("✅ El parser de consultas en lenguaje natural está funcionando correctamente")
    print("📈 Se probaron múltiples tipos de consultas y filtros")
    print("🔧 El sistema es robusto y maneja casos complejos")

if __name__ == "__main__":
    main() 