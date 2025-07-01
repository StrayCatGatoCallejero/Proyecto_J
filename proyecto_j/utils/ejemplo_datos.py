# data/ejemplo_datos.py
import pandas as pd
import numpy as np
from pyreadstat import write_sav


def crear_datos_ejemplo():
    """
    Crea un conjunto de datos de ejemplo con variables correlacionadas
    para probar la funcionalidad de correlaciones y tablas de contingencia.
    """
    np.random.seed(42)  # Para reproducibilidad

    n = 200  # Número de observaciones

    # Crear variables numéricas con diferentes tipos de correlaciones
    edad = np.random.normal(35, 10, n)
    altura = (
        160 + 0.5 * edad + np.random.normal(0, 5, n)
    )  # Correlación positiva con edad
    peso = (
        60 + 0.3 * altura + np.random.normal(0, 8, n)
    )  # Correlación positiva con altura
    ingresos = (
        30000 + 500 * edad - 200 * np.random.normal(0, 1, n)
    )  # Correlación moderada con edad
    satisfaccion = (
        7 - 0.05 * edad + 0.1 * ingresos / 1000 + np.random.normal(0, 1, n)
    )  # Correlación negativa con edad

    # Crear variables categóricas con relaciones conocidas
    # Género (relacionado con altura y peso)
    genero = np.random.choice(["Masculino", "Femenino"], n, p=[0.5, 0.5]).tolist()

    # Nivel educativo (relacionado con ingresos)
    nivel_educativo = np.random.choice(
        ["Primaria", "Secundaria", "Universidad", "Postgrado"],
        n,
        p=[0.2, 0.3, 0.35, 0.15],
    ).tolist()

    # Estado civil (relacionado con edad y satisfacción)
    estado_civil = []
    for i in range(n):
        if edad[i] < 25:
            estado_civil.append(np.random.choice(["Soltero", "Casado"], p=[0.8, 0.2]))
        elif edad[i] < 40:
            estado_civil.append(
                np.random.choice(["Soltero", "Casado", "Divorciado"], p=[0.3, 0.6, 0.1])
            )
        else:
            estado_civil.append(
                np.random.choice(["Casado", "Divorciado", "Viudo"], p=[0.5, 0.3, 0.2])
            )

    # Tipo de trabajo (relacionado con nivel educativo e ingresos)
    tipo_trabajo = []
    for i in range(n):
        if nivel_educativo[i] in ["Primaria", "Secundaria"]:
            tipo_trabajo.append(
                np.random.choice(
                    ["Manual", "Servicios", "Administrativo"], p=[0.6, 0.3, 0.1]
                )
            )
        else:
            tipo_trabajo.append(
                np.random.choice(
                    ["Administrativo", "Profesional", "Directivo"], p=[0.3, 0.5, 0.2]
                )
            )

    # Zona de residencia (variable independiente)
    zona_residencia = np.random.choice(
        ["Urbana", "Suburbana", "Rural"], n, p=[0.6, 0.3, 0.1]
    ).tolist()

    # Crear DataFrame
    df = pd.DataFrame(
        {
            # Variables numéricas
            "edad": np.round(edad, 1),
            "altura_cm": np.round(altura, 1),
            "peso_kg": np.round(peso, 1),
            "ingresos_anuales": np.round(ingresos, 0),
            "satisfaccion_vida": np.round(satisfaccion, 1),
            "horas_trabajo": np.random.normal(40, 8, n),
            "anios_educacion": np.random.normal(12, 3, n),
            # Variables categóricas
            "genero": genero,
            "nivel_educativo": nivel_educativo,
            "estado_civil": estado_civil,
            "tipo_trabajo": tipo_trabajo,
            "zona_residencia": zona_residencia,
        }
    )

    # Asegurar que los valores numéricos estén en rangos razonables
    df["edad"] = df["edad"].clip(18, 70)
    df["altura_cm"] = df["altura_cm"].clip(150, 200)
    df["peso_kg"] = df["peso_kg"].clip(40, 120)
    df["ingresos_anuales"] = df["ingresos_anuales"].clip(15000, 80000)
    df["satisfaccion_vida"] = df["satisfaccion_vida"].clip(1, 10)
    df["horas_trabajo"] = df["horas_trabajo"].clip(20, 60)
    df["anios_educacion"] = df["anios_educacion"].clip(8, 20)

    return df


if __name__ == "__main__":
    # Crear datos de ejemplo
    df_ejemplo = crear_datos_ejemplo()

    # Guardar como archivo .sav
    write_sav(df_ejemplo, "data/datos_ejemplo.sav")

    print(" Archivo de datos de ejemplo creado: data/datos_ejemplo.sav")
    print(
        f" Datos generados: {len(df_ejemplo)} observaciones, {len(df_ejemplo.columns)} variables"
    )

    print("\n Variables numéricas:")
    cols_num = df_ejemplo.select_dtypes(include=["number"]).columns.tolist()
    for col in cols_num:
        print(f"  • {col}")

    print("\n Variables categóricas:")
    cols_cat = df_ejemplo.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cols_cat:
        print(f"  • {col}")

    print("\n Correlaciones esperadas (variables numéricas):")
    print("  • edad ↔ altura_cm: Correlación positiva")
    print("  • altura_cm ↔ peso_kg: Correlación positiva")
    print("  • edad ↔ ingresos_anuales: Correlación moderada")
    print("  • edad ↔ satisfaccion_vida: Correlación negativa")

    print("\n Relaciones esperadas (variables categóricas):")
    print("  • genero ↔ altura_cm/peso_kg: Diferencias por género")
    print("  • nivel_educativo ↔ ingresos_anuales: Relación positiva")
    print("  • nivel_educativo ↔ tipo_trabajo: Relación fuerte")
    print("  • edad ↔ estado_civil: Relación por grupos de edad")

    # Mostrar algunas tablas de contingencia de ejemplo
    print("\n Ejemplos de tablas de contingencia:")
    print("  • género vs tipo_trabajo")
    print("  • nivel_educativo vs zona_residencia")
    print("  • estado_civil vs nivel_educativo")
