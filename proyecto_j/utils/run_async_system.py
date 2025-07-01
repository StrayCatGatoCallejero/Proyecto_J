#!/usr/bin/env python3
"""
🔄 SISTEMA DE PROCESAMIENTO ASÍNCRONO - PROYECTO J
==================================================

Script para configurar y ejecutar el sistema de procesamiento asíncrono
con Celery y Redis.

INSTRUCCIONES DE USO:
1. Instalar dependencias: pip install -r requirements_async.txt
2. Iniciar Redis: redis-server (o usar Docker)
3. Ejecutar worker: python run_async_system.py --worker
4. Ejecutar app: python run_async_system.py --app
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path


def verificar_dependencias():
    """Verifica que todas las dependencias estén instaladas."""
    print("🔍 Verificando dependencias...")

    dependencias = ["celery", "redis", "pandas", "pyarrow", "streamlit"]

    faltantes = []
    for dep in dependencias:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - NO INSTALADO")
            faltantes.append(dep)

    if faltantes:
        print(f"\n❌ Dependencias faltantes: {', '.join(faltantes)}")
        print("💡 Ejecuta: pip install -r requirements_async.txt")
        return False

    print("✅ Todas las dependencias están instaladas")
    return True


def verificar_redis():
    """Verifica que Redis esté ejecutándose."""
    print("\n🔍 Verificando conexión con Redis...")

    try:
        import redis

        r = redis.Redis(host="localhost", port=6379, db=0)
        r.ping()
        print("✅ Redis está ejecutándose")
        return True
    except Exception as e:
        print(f"❌ Error conectando a Redis: {e}")
        print("\n💡 Para iniciar Redis:")
        print("   - Windows: Descarga Redis desde https://redis.io/download")
        print("   - Linux/Mac: brew install redis && redis-server")
        print("   - Docker: docker run -d -p 6379:6379 redis:alpine")
        return False


def crear_directorios():
    """Crea los directorios necesarios."""
    print("\n📁 Creando directorios...")

    directorios = ["./temp", "./resultados"]

    for directorio in directorios:
        Path(directorio).mkdir(exist_ok=True)
        print(f"✅ {directorio}")


def iniciar_worker():
    """Inicia el worker de Celery."""
    print("\n🚀 Iniciando worker de Celery...")
    print("💡 El worker procesará las tareas en segundo plano")
    print("💡 Mantén esta ventana abierta mientras uses la aplicación")

    try:
        # Comando para iniciar el worker
        cmd = [
            sys.executable,
            "-m",
            "celery",
            "-A",
            "tasks",
            "worker",
            "--loglevel=info",
            "--concurrency=1",
        ]

        print(f"📋 Comando: {' '.join(cmd)}")
        print("\n🔄 Iniciando worker...")

        # Ejecutar el worker
        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n⏹️  Worker detenido por el usuario")
    except Exception as e:
        print(f"❌ Error iniciando worker: {e}")


def iniciar_app():
    """Inicia la aplicación Streamlit."""
    print("\n🚀 Iniciando aplicación Streamlit...")
    print("💡 La aplicación estará disponible en http://localhost:8501")

    try:
        # Comando para iniciar Streamlit
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "streamlit_app.py",
            "--server.port=8501",
            "--server.address=localhost",
        ]

        print(f"📋 Comando: {' '.join(cmd)}")
        print("\n🔄 Iniciando aplicación...")

        # Ejecutar Streamlit
        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n⏹️  Aplicación detenida por el usuario")
    except Exception as e:
        print(f"❌ Error iniciando aplicación: {e}")


def mostrar_instrucciones():
    """Muestra las instrucciones de uso."""
    print(
        """
🔄 SISTEMA DE PROCESAMIENTO ASÍNCRONO - PROYECTO J
==================================================

📋 INSTRUCCIONES DE USO:

1. 📦 INSTALAR DEPENDENCIAS:
   pip install -r requirements_async.txt

2. 🗄️  INICIAR REDIS (en una terminal separada):
   - Windows: redis-server
   - Linux/Mac: redis-server
   - Docker: docker run -d -p 6379:6379 redis:alpine

3. 🔄 INICIAR WORKER (en una terminal separada):
   python run_async_system.py --worker

4. 🌐 INICIAR APLICACIÓN (en otra terminal):
   python run_async_system.py --app

5. 📱 USAR LA APLICACIÓN:
   - Abre http://localhost:8501 en tu navegador
   - Sube un archivo CSV (máximo 200 MB)
   - Monitorea el progreso en tiempo real
   - Descarga los resultados cuando termine

📁 ARCHIVOS GENERADOS:
- ./temp/ - Archivos temporales
- ./resultados/ - Archivos de resultados (Parquet)

🔧 COMANDOS ÚTILES:
- Verificar sistema: python run_async_system.py --check
- Solo worker: python run_async_system.py --worker
- Solo app: python run_async_system.py --app
- Ayuda: python run_async_system.py --help

⚠️  NOTAS IMPORTANTES:
- Mantén Redis ejecutándose mientras uses el sistema
- Mantén el worker ejecutándose mientras proceses archivos
- Los archivos temporales se limpian automáticamente
- El sistema maneja archivos de hasta 200 MB
"""
    )


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Sistema de procesamiento asíncrono - Proyecto J",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--worker", action="store_true", help="Iniciar worker de Celery"
    )

    parser.add_argument(
        "--app", action="store_true", help="Iniciar aplicación Streamlit"
    )

    parser.add_argument(
        "--check", action="store_true", help="Verificar configuración del sistema"
    )

    args = parser.parse_args()

    # Si no se especificó ningún argumento, mostrar instrucciones
    if not any([args.worker, args.app, args.check]):
        mostrar_instrucciones()
        return

    # Verificar dependencias
    if not verificar_dependencias():
        sys.exit(1)

    # Crear directorios
    crear_directorios()

    # Verificar Redis si no es solo check
    if not args.check:
        if not verificar_redis():
            print("\n❌ No se puede continuar sin Redis")
            sys.exit(1)

    # Ejecutar según el argumento
    if args.check:
        print("\n✅ Verificación completada")
        print("💡 El sistema está listo para usar")

    elif args.worker:
        iniciar_worker()

    elif args.app:
        iniciar_app()


if __name__ == "__main__":
    main()
