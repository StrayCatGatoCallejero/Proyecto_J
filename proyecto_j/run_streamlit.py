#!/usr/bin/env python3
"""
Script para ejecutar Streamlit con la configuración correcta
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Obtener el directorio actual
    current_dir = Path(__file__).parent.absolute()
    
    # Verificar que estamos en el directorio correcto
    if not (current_dir / "streamlit_app.py").exists():
        print("❌ Error: No se encontró streamlit_app.py en el directorio actual")
        print(f"Directorio actual: {current_dir}")
        return 1
    
    # Verificar que existe la carpeta src
    if not (current_dir / "src").exists():
        print("❌ Error: No se encontró la carpeta src")
        return 1
    
    # Verificar archivos necesarios en src
    required_files = ["core.py", "steps.py", "utils.py"]
    for file in required_files:
        if not (current_dir / "src" / file).exists():
            print(f"❌ Error: No se encontró {file} en la carpeta src")
            return 1
    
    # Verificar configuración de Streamlit
    config_file = Path(".streamlit/config.toml")
    if not config_file.exists():
        print("⚠️ Advertencia: No se encontró .streamlit/config.toml")
        print("Se usará la configuración por defecto de Streamlit")
    
    print("✅ Verificación completada. Iniciando Streamlit...")
    print(f"📁 Directorio de trabajo: {current_dir}")
    print(f"🎨 Configuración: {config_file if config_file.exists() else 'Por defecto'}")
    
    # Cambiar al directorio del proyecto
    os.chdir(current_dir)
    
    # Ejecutar Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar Streamlit: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Streamlit detenido por el usuario")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 