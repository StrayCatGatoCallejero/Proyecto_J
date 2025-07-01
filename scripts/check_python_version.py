#!/usr/bin/env python3
"""
Script para verificar que Python 3.11 esté instalado correctamente
y que todas las dependencias sean compatibles.
"""

import sys
import subprocess
import importlib


def check_python_version():
    """Verificar que estamos usando Python 3.11 o superior."""
    print("🐍 Verificando versión de Python...")

    version = sys.version_info
    print(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 11:
        print("   ✅ Python 3.11+ detectado correctamente")
        return True
    else:
        print("   ❌ Se requiere Python 3.11 o superior")
        print(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False


def check_dependencies():
    """Verificar que las dependencias principales estén instaladas."""
    print("\n📦 Verificando dependencias principales...")

    required_packages = [
        "pandas",
        "numpy",
        "streamlit",
        "plotly",
        "scipy",
        "sklearn",
        "matplotlib",
        "statsmodels",
        "missingno",
        "openpyxl",
        "pyreadstat",
        "pyyaml",
        "fpdf2",
        "fuzzywuzzy",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NO INSTALADO")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Paquetes faltantes: {', '.join(missing_packages)}")
        print("   Ejecuta: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ Todas las dependencias están instaladas")
        return True


def check_package_versions():
    """Verificar versiones de paquetes críticos."""
    print("\n🔍 Verificando versiones de paquetes...")

    try:
        import pandas as pd
        import numpy as np
        import streamlit as st

        print(f"   pandas: {pd.__version__}")
        print(f"   numpy: {np.__version__}")
        print(f"   streamlit: {st.__version__}")

        # Verificar versiones mínimas
        pd_version = tuple(map(int, pd.__version__.split(".")[:2]))
        np_version = tuple(map(int, np.__version__.split(".")[:2]))

        if pd_version >= (2, 0):
            print("   ✅ pandas >= 2.0.0")
        else:
            print("   ⚠️  pandas < 2.0.0 (recomendado actualizar)")

        if np_version >= (1, 24):
            print("   ✅ numpy >= 1.24.0")
        else:
            print("   ⚠️  numpy < 1.24.0 (recomendado actualizar)")

    except Exception as e:
        print(f"   ❌ Error verificando versiones: {e}")
        return False

    return True


def check_streamlit_apps():
    """Verificar que las aplicaciones Streamlit existan."""
    print("\n🚀 Verificando aplicaciones Streamlit...")

    streamlit_apps = [
        "app_front.py",
        "app_estadistica_avanzada.py",
        "app_encuestas.py",
        "wizard_streamlit.py",
        "social_sciences_streamlit.py",
    ]

    missing_apps = []

    for app in streamlit_apps:
        try:
            with open(app, "r") as f:
                print(f"   ✅ {app}")
        except FileNotFoundError:
            print(f"   ❌ {app} - NO ENCONTRADO")
            missing_apps.append(app)

    if missing_apps:
        print(f"\n⚠️  Aplicaciones faltantes: {', '.join(missing_apps)}")
        return False
    else:
        print("\n✅ Todas las aplicaciones Streamlit están presentes")
        return True


def main():
    """Función principal de verificación."""
    print("🔬 Verificación de Proyecto J - Python 3.11")
    print("=" * 50)

    checks = [
        check_python_version(),
        check_dependencies(),
        check_package_versions(),
        check_streamlit_apps(),
    ]

    print("\n" + "=" * 50)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 50)

    if all(checks):
        print("🎉 ¡TODAS LAS VERIFICACIONES PASARON!")
        print("\n✅ El proyecto está listo para ejecutarse")
        print("🚀 Comando para ejecutar: streamlit run app_front.py")
    else:
        print("❌ ALGUNAS VERIFICACIONES FALLARON")
        print("\n🔧 Pasos para solucionar:")
        print("   1. Instalar Python 3.11 o superior")
        print("   2. Ejecutar: pip install -r requirements.txt")
        print("   3. Verificar que todos los archivos estén presentes")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
