#!/usr/bin/env python3
"""
Script para ejecutar Streamlit con el PYTHONPATH configurado correctamente.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Obtener la ruta del directorio actual (raíz del proyecto)
    project_root = Path(__file__).parent.absolute()
    
    # Configurar PYTHONPATH
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_pythonpath = f"{project_root};{current_pythonpath}"
    os.environ['PYTHONPATH'] = new_pythonpath
    
    print(f"🔧 Configurando PYTHONPATH: {project_root}")
    print(f"📁 Ejecutando Streamlit desde: {project_root}")
    
    # Verificar que el archivo de la app existe
    app_file = project_root / "examples" / "app_front.py"
    if not app_file.exists():
        print(f"❌ Error: No se encuentra el archivo {app_file}")
        sys.exit(1)
    
    # Verificar que los módulos necesarios existen
    src_dir = project_root / "proyecto_j" / "src"
    if not src_dir.exists():
        print(f"❌ Error: No se encuentra el directorio {src_dir}")
        sys.exit(1)
    
    estadistica_file = src_dir / "estadistica.py"
    if not estadistica_file.exists():
        print(f"❌ Error: No se encuentra el archivo {estadistica_file}")
        sys.exit(1)
    
    ciencias_sociales_file = src_dir / "ciencias_sociales.py"
    if not ciencias_sociales_file.exists():
        print(f"❌ Error: No se encuentra el archivo {ciencias_sociales_file}")
        sys.exit(1)
    
    print("✅ Todos los archivos necesarios están presentes")
    
    # Ejecutar Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_file)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Streamlit detenido por el usuario")
        sys.exit(0)

if __name__ == "__main__":
    main() 