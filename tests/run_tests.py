#!/usr/bin/env python3
"""
Script centralizado para ejecutar tests de Proyecto J
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path
import time

def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    print(f"\n🔧 {description}")
    print(f"Comando: {command}")
    print("-" * 50)
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ {description} completado exitosamente")
        print(f"⏱️  Tiempo: {end_time - start_time:.2f} segundos")
        if result.stdout:
            print("📋 Salida:")
            print(result.stdout)
    else:
        print(f"❌ {description} falló")
        print(f"⏱️  Tiempo: {end_time - start_time:.2f} segundos")
        if result.stderr:
            print("🚨 Errores:")
            print(result.stderr)
        if result.stdout:
            print("📋 Salida:")
            print(result.stdout)
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Ejecutar tests de Proyecto J")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "e2e", "all"], 
        default="all",
        help="Tipo de tests a ejecutar"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Ejecutar con cobertura de código"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Modo verbose"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Ejecutar tests en paralelo"
    )
    parser.add_argument(
        "--html-report", 
        action="store_true",
        help="Generar reporte HTML"
    )
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="Limpiar archivos temporales antes de ejecutar"
    )
    
    args = parser.parse_args()
    
    # Cambiar al directorio del proyecto
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🧪 Ejecutando Tests de Proyecto J")
    print("=" * 50)
    print(f"📁 Directorio: {project_root}")
    print(f"🔧 Tipo: {args.type}")
    print(f"📊 Cobertura: {args.coverage}")
    print(f"🔍 Verbose: {args.verbose}")
    print(f"⚡ Paralelo: {args.parallel}")
    
    # Limpiar archivos temporales si se solicita
    if args.clean:
        print("\n🧹 Limpiando archivos temporales...")
        clean_commands = [
            "python -c \"import shutil; shutil.rmtree('tests/coverage_html', ignore_errors=True)\"",
            "python -c \"import shutil; shutil.rmtree('tests/reports', ignore_errors=True)\"",
            "python -c \"import os; [os.remove(f) for f in os.listdir('tests') if f.endswith('.xml')]\"" if os.path.exists('tests') else "echo 'No hay archivos XML para eliminar'"
        ]
        
        for cmd in clean_commands:
            subprocess.run(cmd, shell=True)
    
    # Construir comando pytest
    pytest_cmd = ["pytest"]
    
    # Agregar marcadores según tipo
    if args.type == "unit":
        pytest_cmd.append("-m unit")
    elif args.type == "integration":
        pytest_cmd.append("-m integration")
    elif args.type == "e2e":
        pytest_cmd.append("-m e2e")
    
    # Agregar opciones
    if args.verbose:
        pytest_cmd.append("-v")
    
    if args.parallel:
        pytest_cmd.append("-n auto")
    
    if args.coverage:
        pytest_cmd.extend([
            "--cov=proyecto_j",
            "--cov=processing", 
            "--cov=orchestrator",
            "--cov-report=term-missing",
            "--cov-report=html:tests/coverage_html",
            "--cov-report=xml:tests/coverage.xml"
        ])
    
    if args.html_report:
        pytest_cmd.extend([
            "--html=tests/reports/report.html",
            "--self-contained-html",
            "--junitxml=tests/reports/junit.xml"
        ])
    
    # Crear directorios necesarios
    os.makedirs("tests/reports", exist_ok=True)
    os.makedirs("tests/coverage_html", exist_ok=True)
    
    # Ejecutar tests
    command = " ".join(pytest_cmd)
    success = run_command(command, f"Ejecutando tests {args.type}")
    
    # Mostrar resumen
    print("\n" + "=" * 50)
    if success:
        print("🎉 ¡Todos los tests pasaron exitosamente!")
        
        # Mostrar información de cobertura si está disponible
        if args.coverage:
            coverage_file = "tests/coverage_html/index.html"
            if os.path.exists(coverage_file):
                print(f"📊 Reporte de cobertura: {coverage_file}")
        
        if args.html_report:
            report_file = "tests/reports/report.html"
            if os.path.exists(report_file):
                print(f"📋 Reporte HTML: {report_file}")
    else:
        print("❌ Algunos tests fallaron")
        sys.exit(1)

if __name__ == "__main__":
    main() 