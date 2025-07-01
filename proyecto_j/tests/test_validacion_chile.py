"""
Pruebas para el módulo de validación de Chile
=============================================

Este módulo contiene pruebas exhaustivas para validar el funcionamiento
correcto del módulo de validación y enriquecimiento de datos de Chile.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

# Importar el módulo de validación de Chile
try:
    from proyecto_j.src.validacion_chile import (
        cargar_referencias,
        validar_estructura_referencias,
        validar_region,
        validar_comuna,
        validar_geografia_chile,
        enriquecer_con_geografia,
        enriquecer_con_indicadores,
        analisis_por_region,
        comparar_con_promedio_nacional,
        obtener_lista_regiones,
        obtener_lista_comunas,
        buscar_geografia_chile,
    )

    VALIDACION_CHILE_AVAILABLE = True
except ImportError:
    VALIDACION_CHILE_AVAILABLE = False

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def datos_ejemplo():
    """Crea datos de ejemplo para las pruebas."""
    return pd.DataFrame(
        {
            "region_code": ["01", "02", "13", "99"],  # 99 es inválido
            "comuna_code": ["2101", "2301", "13501", "99999"],  # 99999 es inválido
            "variable_numerica": [10, 20, 30, 40],
            "variable_categorica": ["A", "B", "C", "D"],
        }
    )


@pytest.fixture
def datos_validos():
    """Crea datos válidos para las pruebas."""
    return pd.DataFrame(
        {
            "region_code": ["01", "02", "13"],
            "comuna_code": ["2101", "2301", "13501"],
            "variable_numerica": [10, 20, 30],
            "variable_categorica": ["A", "B", "C"],
        }
    )


# ============================================================================
# PRUEBAS DE CARGA DE REFERENCIAS
# ============================================================================


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_cargar_referencias():
    """Prueba que se puedan cargar todas las referencias."""
    referencias = cargar_referencias()

    # Verificar que se cargaron todos los archivos
    assert "regiones" in referencias
    assert "provincias" in referencias
    assert "comunas" in referencias
    assert "indicadores" in referencias

    # Verificar que no están vacíos
    assert len(referencias["regiones"]) > 0
    assert len(referencias["provincias"]) > 0
    assert len(referencias["comunas"]) > 0
    assert len(referencias["indicadores"]) > 0


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_estructura_regiones():
    """Prueba que las regiones tengan la estructura correcta."""
    referencias = cargar_referencias()
    regiones = referencias["regiones"]

    # Verificar columnas requeridas
    columnas_requeridas = [
        "region_code",
        "region_nombre",
        "poblacion_2023",
        "superficie_km2",
    ]
    for col in columnas_requeridas:
        assert col in regiones.columns, f"Columna faltante: {col}"

    # Verificar que no hay valores nulos en códigos
    assert not regiones["region_code"].isnull().any()
    assert not regiones["region_nombre"].isnull().any()

    # Verificar que los códigos son únicos
    assert regiones["region_code"].nunique() == len(regiones)


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_estructura_provincias():
    """Prueba que las provincias tengan la estructura correcta."""
    referencias = cargar_referencias()
    provincias = referencias["provincias"]

    # Verificar columnas requeridas
    columnas_requeridas = ["provincia_code", "provincia_nombre", "region_code"]
    for col in columnas_requeridas:
        assert col in provincias.columns, f"Columna faltante: {col}"

    # Verificar que no hay valores nulos en códigos
    assert not provincias["provincia_code"].isnull().any()
    assert not provincias["provincia_nombre"].isnull().any()
    assert not provincias["region_code"].isnull().any()


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_estructura_comunas():
    """Prueba que las comunas tengan la estructura correcta."""
    referencias = cargar_referencias()
    comunas = referencias["comunas"]

    # Verificar columnas requeridas
    columnas_requeridas = ["comuna_code", "comuna_nombre", "provincia_code"]
    for col in columnas_requeridas:
        assert col in comunas.columns, f"Columna faltante: {col}"

    # Verificar que no hay valores nulos en códigos
    assert not comunas["comuna_code"].isnull().any()
    assert not comunas["comuna_nombre"].isnull().any()
    assert not comunas["provincia_code"].isnull().any()


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_estructura_indicadores():
    """Prueba que los indicadores tengan la estructura correcta."""
    referencias = cargar_referencias()
    indicadores = referencias["indicadores"]

    # Verificar columnas requeridas
    columnas_requeridas = [
        "region_code",
        "idh_2021",
        "pib_per_capita_2021",
        "porcentaje_urbano_2022",
    ]
    for col in columnas_requeridas:
        assert col in indicadores.columns, f"Columna faltante: {col}"

    # Verificar que no hay valores nulos en códigos de región
    assert not indicadores["region_code"].isnull().any()


# ============================================================================
# PRUEBAS DE VALIDACIÓN
# ============================================================================


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_validar_region_datos_validos(datos_validos):
    """Prueba validación de regiones con datos válidos."""
    es_valido, errores = validar_region(datos_validos, "region_code")

    assert es_valido, f"Validación falló con datos válidos: {errores}"
    assert len(errores) == 0


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_validar_region_datos_invalidos(datos_ejemplo):
    """Prueba validación de regiones con datos inválidos."""
    es_valido, errores = validar_region(datos_ejemplo, "region_code")

    assert not es_valido, "Validación debería fallar con datos inválidos"
    assert len(errores) > 0
    assert "99" in str(errores[0]), "Error debería mencionar la región inválida"


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_validar_comuna_datos_validos(datos_validos):
    """Prueba validación de comunas con datos válidos."""
    es_valido, errores = validar_comuna(datos_validos, "comuna_code")

    assert es_valido, f"Validación falló con datos válidos: {errores}"
    assert len(errores) == 0


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_validar_comuna_datos_invalidos(datos_ejemplo):
    """Prueba validación de comunas con datos inválidos."""
    es_valido, errores = validar_comuna(datos_ejemplo, "comuna_code")

    assert not es_valido, "Validación debería fallar con datos inválidos"
    assert len(errores) > 0
    assert "99999" in str(errores[0]), "Error debería mencionar la comuna inválida"


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_validar_geografia_chile_completa(datos_validos):
    """Prueba validación completa de geografía con datos válidos."""
    resultados = validar_geografia_chile(datos_validos, "region_code", "comuna_code")

    assert resultados["valido"], f"Validación falló: {resultados['errores']}"
    assert len(resultados["errores"]) == 0


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_validar_geografia_chile_con_errores(datos_ejemplo):
    """Prueba validación completa de geografía con datos inválidos."""
    resultados = validar_geografia_chile(datos_ejemplo, "region_code", "comuna_code")

    assert not resultados["valido"], "Validación debería fallar con datos inválidos"
    assert len(resultados["errores"]) > 0


# ============================================================================
# PRUEBAS DE ENRIQUECIMIENTO
# ============================================================================


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_enriquecer_con_geografia(datos_validos):
    """Prueba enriquecimiento con información geográfica."""
    df_enriquecido = enriquecer_con_geografia(
        datos_validos, "region_code", "comuna_code"
    )

    # Verificar que se agregaron columnas
    columnas_esperadas = [
        "nombre_region",
        "poblacion_region_2023",
        "superficie_region_km2",
        "nombre_comuna",
        "nombre_provincia",
    ]

    for col in columnas_esperadas:
        assert col in df_enriquecido.columns, f"Columna faltante: {col}"

    # Verificar que no se perdió información original
    assert len(df_enriquecido) == len(datos_validos)


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_enriquecer_con_indicadores(datos_validos):
    """Prueba enriquecimiento con indicadores socioeconómicos."""
    df_enriquecido = enriquecer_con_indicadores(datos_validos, "region_code")

    # Verificar que se agregaron columnas de indicadores
    columnas_esperadas = ["idh_2021", "pib_per_capita_2021", "porcentaje_urbano_2022"]

    for col in columnas_esperadas:
        assert col in df_enriquecido.columns, f"Columna faltante: {col}"

    # Verificar que no se perdió información original
    assert len(df_enriquecido) == len(datos_validos)


# ============================================================================
# PRUEBAS DE ANÁLISIS
# ============================================================================


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_analisis_por_region(datos_validos):
    """Prueba análisis estadístico por región."""
    analisis = analisis_por_region(datos_validos, "variable_numerica", "region_code")

    # Verificar estructura del resultado
    columnas_esperadas = ["region_code", "count", "mean", "std", "min", "max", "median"]
    for col in columnas_esperadas:
        assert col in analisis.columns, f"Columna faltante en análisis: {col}"

    # Verificar que hay resultados para cada región
    assert len(analisis) == datos_validos["region_code"].nunique()


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_comparar_con_promedio_nacional(datos_validos):
    """Prueba comparación con promedio nacional."""
    comparacion = comparar_con_promedio_nacional(
        datos_validos, "variable_numerica", "region_code"
    )

    # Verificar columnas adicionales
    columnas_esperadas = [
        "diferencia_promedio_nacional",
        "porcentaje_diferencia",
        "categoria_diferencia",
    ]
    for col in columnas_esperadas:
        assert col in comparacion.columns, f"Columna faltante en comparación: {col}"

    # Verificar que hay resultados para cada región
    assert len(comparacion) == datos_validos["region_code"].nunique()


# ============================================================================
# PRUEBAS DE FUNCIONES DE UTILIDAD
# ============================================================================


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_obtener_lista_regiones():
    """Prueba obtención de lista de regiones."""
    regiones = obtener_lista_regiones()

    assert isinstance(regiones, pd.DataFrame)
    assert "region_code" in regiones.columns
    assert "region_nombre" in regiones.columns
    assert len(regiones) > 0


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_obtener_lista_comunas():
    """Prueba obtención de lista de comunas."""
    comunas = obtener_lista_comunas()

    assert isinstance(comunas, pd.DataFrame)
    assert "comuna_code" in comunas.columns
    assert "comuna_nombre" in comunas.columns
    assert len(comunas) > 0


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_obtener_lista_comunas_filtrada():
    """Prueba obtención de lista de comunas filtrada por región."""
    comunas = obtener_lista_comunas("13")  # Región Metropolitana

    assert isinstance(comunas, pd.DataFrame)
    assert len(comunas) > 0
    assert all(comunas["region_code"] == "13")


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_buscar_geografia_chile():
    """Prueba búsqueda de términos geográficos."""
    resultados = buscar_geografia_chile("Santiago")

    assert isinstance(resultados, pd.DataFrame)
    assert len(resultados) > 0
    assert "Santiago" in resultados["nombre"].values


# ============================================================================
# PRUEBAS DE INTEGRIDAD DE DATOS
# ============================================================================


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_integridad_referencial_regiones():
    """Prueba que todas las regiones referenciadas en provincias existan."""
    referencias = cargar_referencias()

    regiones_validas = set(referencias["regiones"]["region_code"].astype(str))
    provincias_regiones = set(referencias["provincias"]["region_code"].astype(str))

    regiones_faltantes = provincias_regiones - regiones_validas
    assert len(regiones_faltantes) == 0, f"Regiones faltantes: {regiones_faltantes}"


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_integridad_referencial_provincias():
    """Prueba que todas las provincias referenciadas en comunas existan."""
    referencias = cargar_referencias()

    provincias_validas = set(referencias["provincias"]["provincia_code"].astype(str))
    comunas_provincias = set(referencias["comunas"]["provincia_code"].astype(str))

    provincias_faltantes = comunas_provincias - provincias_validas
    assert (
        len(provincias_faltantes) == 0
    ), f"Provincias faltantes: {provincias_faltantes}"


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_integridad_referencial_indicadores():
    """Prueba que todos los indicadores tengan regiones válidas."""
    referencias = cargar_referencias()

    regiones_validas = set(referencias["regiones"]["region_code"].astype(str))
    indicadores_regiones = set(referencias["indicadores"]["region_code"].astype(str))

    regiones_faltantes = indicadores_regiones - regiones_validas
    assert (
        len(regiones_faltantes) == 0
    ), f"Regiones faltantes en indicadores: {regiones_faltantes}"


# ============================================================================
# PRUEBAS DE CASOS ESPECIALES
# ============================================================================


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_validar_region_columna_inexistente(datos_validos):
    """Prueba validación con columna que no existe."""
    es_valido, errores = validar_region(datos_validos, "columna_inexistente")

    assert not es_valido
    assert len(errores) > 0
    assert "no encontrada" in errores[0]


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_validar_comuna_columna_inexistente(datos_validos):
    """Prueba validación con columna que no existe."""
    es_valido, errores = validar_comuna(datos_validos, "columna_inexistente")

    assert not es_valido
    assert len(errores) > 0
    assert "no encontrada" in errores[0]


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_enriquecer_sin_columnas_geograficas(datos_validos):
    """Prueba enriquecimiento sin especificar columnas geográficas."""
    df_enriquecido = enriquecer_con_geografia(datos_validos)

    # Debería devolver el DataFrame original sin cambios
    assert len(df_enriquecido) == len(datos_validos)
    assert list(df_enriquecido.columns) == list(datos_validos.columns)


# ============================================================================
# PRUEBAS DE RENDIMIENTO
# ============================================================================


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_rendimiento_carga_referencias():
    """Prueba que la carga de referencias sea eficiente."""
    import time

    start_time = time.time()
    referencias = cargar_referencias()
    end_time = time.time()

    # Debería cargar en menos de 5 segundos
    assert end_time - start_time < 5.0, "Carga de referencias demasiado lenta"


@pytest.mark.skipif(
    not VALIDACION_CHILE_AVAILABLE, reason="Módulo de validación de Chile no disponible"
)
def test_rendimiento_validacion_grandes_datos():
    """Prueba rendimiento con datasets grandes."""
    # Crear dataset grande
    datos_grandes = pd.DataFrame(
        {
            "region_code": ["01"] * 10000 + ["02"] * 10000,
            "comuna_code": ["2101"] * 10000 + ["2301"] * 10000,
            "variable": np.random.randn(20000),
        }
    )

    import time

    start_time = time.time()
    resultados = validar_geografia_chile(datos_grandes, "region_code", "comuna_code")
    end_time = time.time()

    # Debería validar en menos de 10 segundos
    assert end_time - start_time < 10.0, "Validación demasiado lenta"
    assert resultados["valido"]


if __name__ == "__main__":
    # Ejecutar pruebas básicas
    print("Ejecutando pruebas de validación de Chile...")

    if VALIDACION_CHILE_AVAILABLE:
        # Prueba básica de carga
        try:
            referencias = cargar_referencias()
            print(
                f"✅ Carga exitosa: {len(referencias['regiones'])} regiones, {len(referencias['comunas'])} comunas"
            )
        except Exception as e:
            print(f"❌ Error en carga: {e}")
    else:
        print("❌ Módulo de validación de Chile no disponible")

    print("Pruebas completadas.")
