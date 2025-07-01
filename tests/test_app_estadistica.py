#!/usr/bin/env python3
"""
Test script for the advanced statistical analysis app functions.
This script tests the core functions without running the Streamlit interface.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path to import the app functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from the app
from app_estadistica_avanzada import (
    detectar_tipos_columnas,
    calcular_correlacion,
    crear_tabla_contingencia,
    aplicar_filtros,
)


def crear_datos_ejemplo():
    """Crea un DataFrame de ejemplo para las pruebas."""
    np.random.seed(42)

    # Datos numéricos
    n = 1000
    edad = np.random.normal(35, 10, n)
    ingresos = np.random.lognormal(10, 0.5, n)
    educacion = np.random.poisson(12, n)

    # Datos categóricos
    genero = np.random.choice(
        ["Masculino", "Femenino", "No binario"], n, p=[0.48, 0.48, 0.04]
    )
    region = np.random.choice(["Norte", "Sur", "Este", "Oeste"], n)
    empleo = np.random.choice(
        ["Empleado", "Desempleado", "Estudiante", "Jubilado"], n, p=[0.6, 0.1, 0.2, 0.1]
    )

    # Crear DataFrame
    df = pd.DataFrame(
        {
            "edad": edad,
            "ingresos": ingresos,
            "educacion": educacion,
            "genero": genero,
            "region": region,
            "empleo": empleo,
        }
    )

    return df


def test_deteccion_tipos():
    """Prueba la función de detección de tipos de columnas."""
    print("🧪 Probando detección de tipos de columnas...")

    df = crear_datos_ejemplo()
    tipos_df = detectar_tipos_columnas(df)

    # Convertir a diccionario para facilitar las verificaciones
    tipos = dict(zip(tipos_df["columna"], tipos_df["tipo_detectado"]))

    # Verificar que se detectaron los tipos correctos
    assert (
        "numérico" in tipos["edad"]
    ), f"Edad debería ser numérico, pero es {tipos['edad']}"
    assert (
        tipos["genero"] == "categórico"
    ), f"Género debería ser categórico, pero es {tipos['genero']}"

    print(f"✅ Tipos detectados: {tipos}")


def test_correlacion():
    """Prueba la función de cálculo de correlación."""
    print("🧪 Probando cálculo de correlación...")
    df = crear_datos_ejemplo()
    variables = ["edad", "ingresos", "educacion"]
    # Probar correlación Pearson
    corr_pearson, error = calcular_correlacion(df, variables, "pearson")
    assert error is None, f"Error en correlación Pearson: {error}"
    assert corr_pearson.shape == (
        3,
        3,
    ), f"Matriz de correlación debería ser 3x3, pero es {corr_pearson.shape}"
    # Probar correlación Spearman
    corr_spearman, error = calcular_correlacion(df, variables, "spearman")
    assert error is None, f"Error en correlación Spearman: {error}"
    print(f"✅ Correlación Pearson calculada: {corr_pearson.shape}")
    print(f"✅ Correlación Spearman calculada: {corr_spearman.shape}")


def test_tabla_contingencia():
    """Prueba la función de tabla de contingencia."""
    print("🧪 Probando tabla de contingencia...")
    df = crear_datos_ejemplo()
    resultados = crear_tabla_contingencia(df, "genero", "empleo")
    assert resultados is not None, "La tabla de contingencia no debería ser None"
    assert "tabla" in resultados, "Debería contener la tabla"
    assert "chi2" in resultados, "Debería contener el estadístico chi2"
    assert "p_value" in resultados, "Debería contener el p-valor"
    assert "interpretacion" in resultados, "Debería contener la interpretación"
    print(f"✅ Tabla de contingencia creada")
    print(f"   Chi2: {resultados['chi2']:.4f}")
    print(f"   p-valor: {resultados['p_value']:.4f}")
    print(f"   Interpretación: {resultados['interpretacion']}")


def test_filtros():
    """Prueba la función de aplicación de filtros."""
    print("🧪 Probando aplicación de filtros...")

    df = crear_datos_ejemplo()

    # Definir filtros
    filtros = {
        "edad": {"tipo": "rango", "valores": [25, 45]},
        "genero": {"tipo": "categorias", "valores": ["Masculino", "Femenino"]},
    }

    # Aplicar filtros
    df_filtrado = aplicar_filtros(df, filtros)

    # Verificar que los filtros se aplicaron correctamente
    assert len(df_filtrado) <= len(
        df
    ), "Los datos filtrados no deberían ser más que los originales"
    assert all(df_filtrado["edad"] >= 25), "Todas las edades deberían ser >= 25"
    assert all(df_filtrado["edad"] <= 45), "Todas las edades deberían ser <= 45"
    assert all(
        df_filtrado["genero"].isin(["Masculino", "Femenino"])
    ), "Solo deberían estar los géneros seleccionados"

    print(f"✅ Filtros aplicados: {len(df_filtrado)} registros de {len(df)} originales")


def test_datos_ejemplo():
    """Prueba la creación de datos de ejemplo."""
    print("🧪 Probando creación de datos de ejemplo...")

    df = crear_datos_ejemplo()

    # Verificar estructura
    assert len(df) == 1000, f"Debería tener 1000 filas, pero tiene {len(df)}"
    assert (
        len(df.columns) == 6
    ), f"Debería tener 6 columnas, pero tiene {len(df.columns)}"

    # Verificar tipos de datos
    assert df["edad"].dtype in ["float64", "float32"], "Edad debería ser numérico"
    assert df["genero"].dtype == "object", "Género debería ser objeto (categórico)"

    # Verificar que no hay valores nulos
    assert df.isnull().sum().sum() == 0, "No debería haber valores nulos"

    print(f"✅ Datos de ejemplo creados: {df.shape}")
    print(f"   Columnas: {list(df.columns)}")


def ejecutar_todas_las_pruebas():
    """Ejecuta todas las pruebas y reporta resultados."""
    print("🚀 Iniciando pruebas del módulo de análisis estadístico avanzado...")
    print("=" * 60)

    pruebas = [
        ("Datos de ejemplo", test_datos_ejemplo),
        ("Detección de tipos", test_deteccion_tipos),
        ("Cálculo de correlación", test_correlacion),
        ("Tabla de contingencia", test_tabla_contingencia),
        ("Aplicación de filtros", test_filtros),
    ]

    resultados = []

    for nombre, prueba in pruebas:
        try:
            prueba()
            resultados.append((nombre, True, None))
            print(f"✅ {nombre}: PASÓ")
        except Exception as e:
            resultados.append((nombre, False, str(e)))
            print(f"❌ {nombre}: FALLÓ - {e}")
        print("-" * 40)

    # Resumen final
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)

    exitos = sum(1 for _, paso, _ in resultados if paso)
    total = len(resultados)

    for nombre, paso, error in resultados:
        estado = "✅ PASÓ" if paso else f"❌ FALLÓ ({error})"
        print(f"{nombre}: {estado}")

    print(f"\n🎯 Resultado final: {exitos}/{total} pruebas exitosas")

    if exitos == total:
        print(
            "🎉 ¡Todas las pruebas pasaron! El módulo está funcionando correctamente."
        )
        return True
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa los errores arriba.")
        return False


if __name__ == "__main__":
    try:
        exito = ejecutar_todas_las_pruebas()
        sys.exit(0 if exito else 1)
    except Exception as e:
        print(f"❌ Error general en las pruebas: {e}")
        sys.exit(1)
